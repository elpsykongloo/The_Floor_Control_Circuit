"""Moshi E2-lite 方向注入生成 runner（R2 制式 + L{ℓ*} 连续 steering；PREREG #34）。

在 moshi venv 内运行。用户流 = 计划内 wav teacher-forced 编码；agent 流 =
官方 LMGen 采样生成（音频温度 0.8 / 文本 0.7，官方 top-k）。注入：
    h_ℓ ← h_ℓ + α · s_v · v̂        （对该层输出的每个生成步恒等相加）
α=0 时注入向量为全零（基线走同一代码路径，保证跨条件可比）。基线条件可选
缓存指定层激活（fp16），充当 R2 观察性探针数据。

用法（双卡各开一个进程）：
  python runners/moshi/run_steer.py --plan <data_root>/e2_lite/e2_lite.plan.json \
      --model-root C:/.../moshiko-pytorch-bf16 --device cuda:0 --num-shards 2 --shard-id 0

断点续跑：每个运行目录写 manifest.json 作完成标记，重启自动跳过。
--probe-api 转储本地 LMGen 签名后退出（首跑联调）。
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

_SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(_SHARED))

from moshi_family import (  # noqa: E402
    AdapterError,
    load_models,
    log,
    read_wav_mono,
    resolve_code_version,
    sha256_file,
    write_json_atomic,
)

MANIFEST_SCHEMA = "e2_lite_run_v1"


def _load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "e2_lite_plan_v1":
        raise AdapterError(f"未知计划 schema：{plan.get('schema')}")
    return plan


def _load_directions(plan: dict) -> dict[str, np.ndarray]:
    npz_path = Path(plan["directions_npz"])
    if sha256_file(npz_path) != plan["directions_sha256"]:
        raise AdapterError("方向文件摘要与计划不符——先重新生成计划")
    with np.load(npz_path, allow_pickle=False) as payload:
        return {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in payload.files
            if name != "__meta__"
        }


def steer_vector_np(alpha: float, proj_std: float, unit_direction: np.ndarray) -> np.ndarray:
    """与仓库侧 floor_circuit.e1x.core.steer_vector 逐字同语义（runner 解耦副本）。"""
    v = np.asarray(unit_direction, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm == 0:
        raise AdapterError("注入方向为零向量")
    if proj_std <= 0:
        raise AdapterError("proj_std 必须为正")
    return (float(alpha) * float(proj_std)) * (v / norm)


def iter_runs(plan: dict) -> list[dict]:
    runs = []
    for session in plan["sessions"]:
        for condition in plan["conditions"]:
            runs.append(
                {
                    "run_id": f"{session['session_id']}__{condition['name']}",
                    "session": session,
                    "condition": condition,
                }
            )
    return runs


def shard_runs(runs: list[dict], num_shards: int, shard_id: int) -> list[dict]:
    if not (0 <= shard_id < num_shards):
        raise AdapterError("分片编号越界")
    return [r for i, r in enumerate(runs) if i % num_shards == shard_id]


def run_done(out_dir: Path) -> bool:
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("schema") != MANIFEST_SCHEMA or not payload.get("completed"):
        return False
    return (out_dir / "agent.wav").is_file()


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    import wave

    clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def probe_api(args) -> None:
    from moshi.models import LMGen

    signature = str(inspect.signature(LMGen.__init__))
    print(json.dumps({"LMGen.__init__": signature}, ensure_ascii=False, indent=1))


def execute_run(mimi, lm_gen_factory, run: dict, plan: dict, steer_state: dict, device: str) -> dict:
    import torch

    session = run["session"]
    condition = run["condition"]
    sample_rate = int(plan["sample_rate"])
    frame_samples = round(sample_rate / float(plan["frame_hz"]))
    n_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    wav = read_wav_mono(session["user_wav"], sample_rate, max_seconds=float(plan["window_s"]))
    torch.manual_seed(int(session["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(session["seed"]))

    steer_state["vec"].zero_()
    if float(condition["alpha"]) != 0.0:
        vec = steer_vector_np(
            float(condition["alpha"]),
            float(plan["proj_std"][condition["direction"]]),
            steer_state["directions"][condition["direction"]],
        )
        steer_state["vec"].copy_(torch.from_numpy(vec).to(steer_state["vec"].dtype))
    cache_layers = (
        [int(v) for v in plan["cache_layers_baseline"]] if condition.get("cache_acts") else []
    )
    steer_state["captured"] = {layer: [] for layer in cache_layers}
    steer_state["capture_enabled"] = bool(cache_layers)

    lm_gen = lm_gen_factory()
    agent_chunks: list[np.ndarray] = []
    text_tokens: list[int] = []
    first_emitted = None
    started = time.perf_counter()
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for frame_index in range(n_frames):
            chunk = wav[frame_index * frame_samples : (frame_index + 1) * frame_samples]
            if len(chunk) < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
            block = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).to(device)
            codes = mimi.encode(block.reshape(1, 1, -1))
            tokens = lm_gen.step(codes)
            if tokens is None:
                continue
            if first_emitted is None:
                first_emitted = frame_index
            text_tokens.append(int(tokens[0, 0, 0].item()))
            agent_codes = tokens[:, 1:, :]
            agent_wav = mimi.decode(agent_codes)
            agent_chunks.append(agent_wav[0, 0].float().cpu().numpy())
    wall = time.perf_counter() - started
    agent = np.concatenate(agent_chunks) if agent_chunks else np.zeros(0, dtype=np.float32)
    return {
        "agent_wav": agent,
        "text_tokens": np.asarray(text_tokens, dtype=np.int32),
        "first_emitted_frame": first_emitted,
        "n_frames_in": n_frames,
        "wall_s": wall,
        "captured": {
            layer: (np.stack(rows).astype(np.float16) if rows else np.zeros((0,), dtype=np.float16))
            for layer, rows in steer_state["captured"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Moshi E2-lite steering runner")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model-root", required=False)
    parser.add_argument("--lm-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-codebooks", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--only-condition", default=None, help="只跑指定条件名（冒烟用）")
    parser.add_argument("--limit", type=int, default=None, help="至多执行 N 个运行（冒烟用）")
    parser.add_argument("--probe-api", action="store_true")
    parser.add_argument("--code-version", default=None)
    args = parser.parse_args()
    if args.probe_api:
        probe_api(args)
        return
    if not args.model_root:
        raise SystemExit("缺 --model-root")

    # 试点关闭 CUDA Graph：保证 steering hook 逐步在 Python 侧执行、hook_calls 可核验；
    # 速度损失由试点规模吸收（正式 E2 再评估图化路径）。必须在导入 moshi 前设置。
    import os

    os.environ.setdefault("NO_CUDA_GRAPH", "1")
    import torch
    from moshi.models import LMGen

    plan = _load_plan(Path(args.plan))
    directions = _load_directions(plan)
    runs = shard_runs(iter_runs(plan), int(args.num_shards), int(args.shard_id))
    if args.only_condition:
        runs = [r for r in runs if r["condition"]["name"] == args.only_condition]
    if args.limit is not None:
        runs = runs[: int(args.limit)]
    out_root = Path(plan["out_root"]) / "runs"
    pending = [r for r in runs if not run_done(out_root / r["run_id"])]
    log(f"分片 {args.shard_id}/{args.num_shards}：{len(runs)} 运行，其中待执行 {len(pending)}")
    if not pending:
        return

    mimi, lm = load_models(args)
    layer_index = int(plan["layer"])
    transformer = lm.transformer
    layers = getattr(transformer, "layers", None)
    if layers is None or layer_index >= len(layers):
        raise AdapterError(f"lm.transformer.layers 缺失或层号越界（layer={layer_index}）")
    hidden_dim = int(next(iter(directions.values())).shape[0])
    model_dtype = next(lm.parameters()).dtype
    steer_state = {
        "vec": torch.zeros(hidden_dim, dtype=model_dtype, device=args.device),
        "directions": directions,
        "captured": {},
        "capture_enabled": False,
        "hook_calls": 0,
    }

    def steer_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.shape[-1] != hidden_dim:
            raise AdapterError(f"层输出宽度 {tensor.shape[-1]} ≠ 方向维度 {hidden_dim}")
        steer_state["hook_calls"] += 1
        steered = tensor + steer_state["vec"]
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    hook_handles = [layers[layer_index].register_forward_hook(steer_hook)]

    def capture_hook_for(layer: int):
        def capture_hook(_module, _inputs, output):
            if not steer_state["capture_enabled"]:
                return None
            tensor = output[0] if isinstance(output, tuple) else output
            steer_state["captured"][layer].append(
                tensor[0, -1].detach().to(torch.float16).cpu().numpy()
            )
            return None

        return capture_hook

    for layer in {int(v) for v in plan["cache_layers_baseline"]}:
        if layer >= len(layers):
            raise AdapterError(f"缓存层号越界：{layer}")
        hook_handles.append(layers[layer].register_forward_hook(capture_hook_for(layer)))

    lm_gen_kwargs = {
        "temp": float(plan["temperature"]),
        "temp_text": float(plan["text_temperature"]),
        "top_k": int(plan["top_k"]),
        "top_k_text": int(plan["top_k_text"]),
    }
    accepted = inspect.signature(LMGen.__init__).parameters
    lm_gen_kwargs = {k: v for k, v in lm_gen_kwargs.items() if k in accepted}
    if "use_sampling" in accepted:
        lm_gen_kwargs["use_sampling"] = True

    def lm_gen_factory():
        return LMGen(lm, **lm_gen_kwargs)

    code_version = resolve_code_version(args.code_version)
    for index, run in enumerate(pending):
        out_dir = out_root / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        steer_state["hook_calls"] = 0
        result = execute_run(mimi, lm_gen_factory, run, plan, steer_state, args.device)
        _write_wav(out_dir / "agent.wav", result["agent_wav"], int(plan["sample_rate"]))
        np.save(out_dir / "text_tokens.npy", result["text_tokens"], allow_pickle=False)
        for layer, rows in result["captured"].items():
            if rows.size:
                np.save(out_dir / f"acts_L{layer}.npy", rows, allow_pickle=False)
        steps_per_s = result["n_frames_in"] / result["wall_s"] if result["wall_s"] > 0 else None
        write_json_atomic(
            out_dir / "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "completed": True,
                "run_id": run["run_id"],
                "session_id": run["session"]["session_id"],
                "user_wav": run["session"]["user_wav"],
                "user_wav_sha256": run["session"]["user_wav_sha256"],
                "user_channel": run["session"]["user_channel"],
                "seed": run["session"]["seed"],
                "condition": run["condition"],
                "layer": layer_index,
                "scale_rule": plan["scale_rule"],
                "proj_std": plan["proj_std"].get(run["condition"]["direction"]),
                "window_s": plan["window_s"],
                "temperature": plan["temperature"],
                "text_temperature": plan["text_temperature"],
                "top_k": plan["top_k"],
                "top_k_text": plan["top_k_text"],
                "first_emitted_frame": result["first_emitted_frame"],
                "n_frames_in": result["n_frames_in"],
                "n_agent_samples": len(result["agent_wav"]),
                "hook_calls": int(steer_state["hook_calls"]),
                "wall_s": result["wall_s"],
                "steps_per_s": steps_per_s,
                "no_cuda_graph": os.environ.get("NO_CUDA_GRAPH"),
                "code_version": code_version,
                "directions_sha256": plan["directions_sha256"],
            },
        )
        log(
            f"[{index + 1}/{len(pending)}] {run['run_id']} 完成："
            f"{result['wall_s']:.1f} s（{steps_per_s and round(steps_per_s, 2)} 步/秒，"
            f"hook {steer_state['hook_calls']} 次）"
        )
    for handle in hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
