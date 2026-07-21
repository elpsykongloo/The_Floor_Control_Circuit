"""WP-E1：探测双角色批处理能否保持逐元素等价。

在 Moshi 独立环境运行：
  <moshi python> scripts/wp_e1_role_batch_probe.py --plan <E1 计划>

同一会话分别执行 agent0/agent1 的 batch=1 流式贪心前向，再把两路合为 batch=2
复算；逐层、逐步、逐元素比较。该工具只读模型和音频，不写激活缓存。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED = REPO_ROOT / "runners" / "_shared"
sys.path.insert(0, str(SHARED))

from moshi_family import (  # noqa: E402
    _assert_streaming_idle,
    build_parallel_codes,
    encode_mimi_stream,
    forward_backbone,
    load_models,
    prepare_teacher_forced_input,
    read_wav_mono,
)


def _run_greedy(lm, codes, layers: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    import torch

    transformer = lm.transformer
    captured: dict[int, object] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[layer] = value.detach()

        return hook

    for layer in layers:
        handles.append(transformer.layers[layer].register_forward_hook(make_hook(layer)))
    model_input = prepare_teacher_forced_input(lm, codes)
    batch_size = int(model_input.shape[0])
    n_steps = int(model_input.shape[2])
    token_rows: list[object] = []
    activation_rows: dict[int, list[object]] = {layer: [] for layer in layers}
    previous = None
    try:
        with torch.no_grad(), transformer.streaming(batch_size=batch_size):
            for step in range(n_steps):
                step_input = model_input[:, :, step : step + 1].clone()
                if previous is not None:
                    step_input[:, 0, 0].copy_(previous)
                captured.clear()
                hidden = forward_backbone(lm, step_input)
                logits = lm.text_linear(hidden)[:, -1].float()
                previous = torch.argmax(logits, dim=-1)
                token_rows.append(previous.detach().cpu())
                for layer in layers:
                    value = captured[layer]
                    activation_rows[layer].append(value[:, -1].to(torch.float16).cpu())
    finally:
        for handle in handles:
            handle.remove()
    tokens = torch.stack(token_rows, dim=0).numpy()  # [T, B]
    activations = {
        layer: torch.stack(rows, dim=0).numpy()  # [T, B, H]
        for layer, rows in activation_rows.items()
    }
    return tokens, activations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="任一 E1 计划 v2 JSON")
    parser.add_argument("--seconds", type=float, default=2.0, help="探测音频前缀秒数")
    parser.add_argument("--device", default="cuda", help="逻辑 CUDA 设备")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    settings = plan["settings"]
    session = plan["sessions"][0]

    import torch

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("批处理等价性探针要求 CUDA")
    if device.index is None:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(int(device.index))
    model_args = SimpleNamespace(
        model_root=str(plan["model_root"]),
        lm_weight=plan.get("lm_weight"),
        mimi_weight=plan.get("mimi_weight"),
        device=str(device),
        n_codebooks=int(settings["n_codebooks"]),
    )
    mimi, lm = load_models(model_args)
    sample_rate = int(getattr(mimi, "sample_rate", 24000))
    wav0 = read_wav_mono(session["audio_ch0"], sample_rate, args.seconds)
    wav1 = read_wav_mono(session["audio_ch1"], sample_rate, args.seconds)
    codes0, _, _ = encode_mimi_stream(
        mimi,
        wav0,
        str(device),
        float(settings["mimi_chunk_seconds"]),
        return_latent=False,
    )
    codes1, _, _ = encode_mimi_stream(
        mimi,
        wav1,
        str(device),
        float(settings["mimi_chunk_seconds"]),
        return_latent=False,
    )
    _assert_streaming_idle(mimi, lm)
    stream_args = SimpleNamespace(stream_order=str(settings["stream_order"]))
    role0, _ = build_parallel_codes(lm, codes0, codes1, stream_args)
    role1, _ = build_parallel_codes(lm, codes1, codes0, stream_args)
    layers = [int(value) for value in settings["layers"]]

    started = time.perf_counter()
    tokens0, acts0 = _run_greedy(lm, role0, layers)
    tokens1, acts1 = _run_greedy(lm, role1, layers)
    sequential_seconds = time.perf_counter() - started
    _assert_streaming_idle(mimi, lm)

    batched_codes = torch.cat((role0, role1), dim=0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    batch_tokens, batch_acts = _run_greedy(lm, batched_codes, layers)
    batched_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    _assert_streaming_idle(mimi, lm)

    comparisons: dict[str, dict] = {}
    all_equal = True
    for role, (expected_tokens, expected_acts) in enumerate(
        ((tokens0[:, 0], acts0), (tokens1[:, 0], acts1))
    ):
        token_equal = bool(np.array_equal(expected_tokens, batch_tokens[:, role]))
        comparisons[f"agent{role}_text_tokens"] = {"equal": token_equal}
        all_equal = all_equal and token_equal
        for layer in layers:
            expected = expected_acts[layer][:, 0]
            actual = batch_acts[layer][:, role]
            equal = bool(np.array_equal(expected, actual))
            delta = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
            comparisons[f"agent{role}_acts_L{layer}"] = {
                "equal": equal,
                "n_mismatch": int(np.count_nonzero(expected != actual)),
                "max_abs": float(delta.max(initial=0.0)),
            }
            all_equal = all_equal and equal

    report = {
        "session_id": session["session_id"],
        "seconds": float(args.seconds),
        "n_steps": int(batch_tokens.shape[0]),
        "layers": layers,
        "sequential_seconds": round(sequential_seconds, 6),
        "batched_seconds": round(batched_seconds, 6),
        "speedup": round(sequential_seconds / max(batched_seconds, 1e-9), 6),
        "batch2_peak_memory_allocated_bytes_short_context": peak_allocated,
        "all_equal": all_equal,
        "comparisons": comparisons,
    }
    output = REPO_ROOT / "reports" / "wp_e1_role_batch_probe.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[report] {output}")
    print(
        f"batch=2 等价性：{'equal' if all_equal else 'different'}；"
        f"短窗加速 {report['speedup']:.3f}x"
    )


if __name__ == "__main__":
    main()
