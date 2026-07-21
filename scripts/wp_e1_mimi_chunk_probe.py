"""WP-E1：探测更大 Mimi 流式块能否保持逐元素等价并降低编码开销。"""

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
    encode_mimi_stream,
    load_models,
    read_wav_mono,
)


def _encode(
    mimi,
    wav: np.ndarray,
    device: str,
    chunk_seconds: float,
    *,
    use_cuda_graph: bool = False,
):
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    codes, latent, source = encode_mimi_stream(
        mimi,
        wav,
        device,
        chunk_seconds,
        return_latent=True,
        use_cuda_graph=use_cuda_graph,
    )
    torch.cuda.synchronize()
    return codes, latent, source, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="任一 E1 计划 v2 JSON")
    parser.add_argument("--seconds", type=float, default=240.0, help="音频前缀秒数")
    parser.add_argument("--baseline", type=float, default=0.08, help="基准块长秒数")
    parser.add_argument(
        "--candidates",
        default="0.64,1.28,5.12,20.48",
        help="候选块长秒数，逗号分隔",
    )
    parser.add_argument("--device", default="cuda", help="逻辑 CUDA 设备")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    settings = plan["settings"]
    session = plan["sessions"][0]

    import torch

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Mimi 块长探针要求 CUDA")
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
    wav = read_wav_mono(session["audio_ch0"], sample_rate, args.seconds)
    baseline_codes, baseline_latent, latent_source, baseline_seconds = _encode(
        mimi,
        wav,
        str(device),
        float(args.baseline),
    )
    _assert_streaming_idle(mimi, lm)
    assert baseline_latent is not None

    results = []
    graph_codes, graph_latent, graph_source, graph_elapsed = _encode(
        mimi,
        wav,
        str(device),
        float(args.baseline),
        use_cuda_graph=True,
    )
    _assert_streaming_idle(mimi, lm)
    assert graph_latent is not None
    graph_delta = np.abs(
        baseline_latent.astype(np.float32) - graph_latent.astype(np.float32)
    )
    graph_codes_equal = bool(torch.equal(baseline_codes, graph_codes))
    graph_latent_equal = bool(np.array_equal(baseline_latent, graph_latent))
    results.append(
        {
            "mode": "cuda_graph",
            "chunk_seconds": float(args.baseline),
            "elapsed_s": round(graph_elapsed, 6),
            "speedup": round(baseline_seconds / max(graph_elapsed, 1e-9), 6),
            "codes_equal": graph_codes_equal,
            "codes_n_mismatch": int(
                torch.count_nonzero(baseline_codes != graph_codes).item()
            ),
            "latent_source_equal": graph_source == latent_source,
            "latent_equal": graph_latent_equal,
            "latent_n_mismatch": int(np.count_nonzero(baseline_latent != graph_latent)),
            "latent_max_abs": float(graph_delta.max(initial=0.0)),
            "all_equal": (
                graph_codes_equal and graph_latent_equal and graph_source == latent_source
            ),
        }
    )
    for candidate in (float(value) for value in args.candidates.split(",")):
        codes, latent, source, elapsed = _encode(
            mimi,
            wav,
            str(device),
            candidate,
        )
        _assert_streaming_idle(mimi, lm)
        assert latent is not None
        codes_equal = bool(torch.equal(baseline_codes, codes))
        latent_equal = bool(np.array_equal(baseline_latent, latent))
        delta = np.abs(baseline_latent.astype(np.float32) - latent.astype(np.float32))
        results.append(
            {
                "mode": "larger_chunk",
                "chunk_seconds": candidate,
                "elapsed_s": round(elapsed, 6),
                "speedup": round(baseline_seconds / max(elapsed, 1e-9), 6),
                "codes_equal": codes_equal,
                "codes_n_mismatch": int(torch.count_nonzero(baseline_codes != codes).item()),
                "latent_source_equal": source == latent_source,
                "latent_equal": latent_equal,
                "latent_n_mismatch": int(np.count_nonzero(baseline_latent != latent)),
                "latent_max_abs": float(delta.max(initial=0.0)),
                "all_equal": codes_equal and latent_equal and source == latent_source,
            }
        )

    report = {
        "session_id": session["session_id"],
        "channel": 0,
        "seconds": float(args.seconds),
        "n_steps": int(baseline_codes.shape[-1]),
        "baseline_chunk_seconds": float(args.baseline),
        "baseline_elapsed_s": round(baseline_seconds, 6),
        "results": results,
    }
    output = REPO_ROOT / "reports" / "wp_e1_mimi_chunk_probe.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[report] {output}")
    for result in results:
        print(
            f"{result['mode']} chunk={result['chunk_seconds']:g}s："
            f"{'equal' if result['all_equal'] else 'different'}，"
            f"{result['speedup']:.3f}x"
        )


if __name__ == "__main__":
    main()
