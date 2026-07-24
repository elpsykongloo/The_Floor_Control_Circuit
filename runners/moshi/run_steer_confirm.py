"""Moshi E2 确认臂 runner（PREREG #40(b)）：钳制消融 / 门控注入 / 轴与层特异性。

在 moshi venv 内运行；只提供 eager 参考后端（NO_CUDA_GRAPH=1，逐帧
Mimi 编码 → LMGen 采样 → 逐帧解码），语义与 E2-lite eager_reference 逐字一致，
基线因此可直接复用 E2-lite 的 baseline 运行（同会话同种子）。

逐帧干预（g[t] 为计划预生成的事件门，continuous 条件恒为 1）：
  inject:  h_L ← h_L + g[t] · α · s_v(L) · v̂
  clamp:   h_L ← h_L + g[t] · (μ_v − h_L·v̂) · v̂     （float32 钳制数学）

用法（双卡按会话分片）：
  python runners/moshi/run_steer_confirm.py --plan <data_root>/e2_confirm/e2_confirm.plan.json \
      --model-root C:/.../moshiko-pytorch-bf16 --device cuda:0 --num-shards 2 --shard-id 0
断点续跑：manifest.json 完成标记；--limit/--only-condition 冒烟。
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_steer as base  # 复用 E2-lite runner 的纯函数与 moshi_family 帮手

MANIFEST_SCHEMA = "e2_confirm_run_v1"
MODE_INJECT = "inject"
MODE_CLAMP = "clamp"
GATE_NONE = "none"


def _load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "e2_confirm_plan_v1":
        raise base.AdapterError(f"未知计划 schema：{plan.get('schema')}")
    return plan


def _load_directions(plan: dict) -> dict[str, np.ndarray]:
    npz_path = Path(plan["directions_npz"])
    if base.sha256_file(npz_path) != plan["directions_sha256"]:
        raise base.AdapterError("确认臂方向文件摘要与计划不符——先重新生成计划")
    with np.load(npz_path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name], dtype=np.float64) for name in payload.files if name != "__meta__"}


def _load_gate(plan: dict, session_id: str, gate_name: str, n_frames: int) -> np.ndarray:
    if gate_name == GATE_NONE:
        return np.ones(n_frames, dtype=np.float32)
    entry = plan["gates"].get(session_id)
    if entry is None:
        raise base.AdapterError(f"计划缺会话门文件：{session_id}")
    gate_path = Path(entry["path"])
    if base.sha256_file(gate_path) != entry["sha256"]:
        raise base.AdapterError(f"门文件摘要不符：{gate_path}")
    with np.load(gate_path, allow_pickle=False) as payload:
        if gate_name not in payload.files:
            raise base.AdapterError(f"门文件缺 {gate_name}：{gate_path}")
        gate = np.asarray(payload[gate_name], dtype=np.float32)
    if gate.shape != (n_frames,):
        raise base.AdapterError(f"门长度 {gate.shape} ≠ 帧数 {n_frames}")
    return gate


def run_done(out_dir: Path, expected: dict | None = None) -> bool:
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("schema") != MANIFEST_SCHEMA or not payload.get("completed"):
        return False
    if expected is not None and any(payload.get(key) != value for key, value in expected.items()):
        return False
    return (out_dir / "agent.wav").is_file() and (out_dir / "text_tokens.npy").is_file()


def steer_params_for(condition: dict, plan: dict, directions: dict[str, np.ndarray]) -> dict:
    """计算一条条件的注入/钳制参数（纯 numpy，供测试）。"""
    layer = int(condition["layer"])
    name = str(condition["direction"])
    vector = directions[name]
    unit = vector / float(np.linalg.norm(vector))
    std = float(plan["proj_std"][name][str(layer)])
    mean = float(plan["proj_mean"][name][str(layer)])
    if condition["mode"] == MODE_INJECT:
        vec = base.steer_vector_np(float(condition["alpha"]), std, unit)
        return {"mode": MODE_INJECT, "layer": layer, "vec": vec, "unit": unit, "target": None, "proj_std": std}
    if condition["mode"] == MODE_CLAMP:
        return {
            "mode": MODE_CLAMP,
            "layer": layer,
            "vec": np.zeros_like(unit),
            "unit": unit,
            "target": mean,
            "proj_std": std,
        }
    raise base.AdapterError(f"未知干预模式：{condition['mode']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moshi E2 确认臂 runner")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--lm-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-codebooks", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--only-condition", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--code-version", default=None)
    args = parser.parse_args()

    os.environ["NO_CUDA_GRAPH"] = "1"  # 确认臂只走参考语义；必须先于 moshi 导入
    import torch
    from moshi.models import LMGen

    device_object = torch.device(args.device)
    if device_object.type == "cuda":
        torch.cuda.set_device(device_object)

    plan = _load_plan(Path(args.plan))
    directions = _load_directions(plan)
    runs = base.shard_runs(base.iter_runs(plan), int(args.num_shards), int(args.shard_id))
    if args.only_condition:
        runs = [r for r in runs if r["condition"]["name"] == args.only_condition]
    if args.limit is not None:
        runs = runs[: int(args.limit)]
    out_root = Path(plan["out_root"]) / "runs"
    plan_sha256 = base._canonical_sha256(plan)

    def expected_manifest(run: dict) -> dict:
        return {
            "run_id": run["run_id"],
            "plan_sha256": plan_sha256,
            "directions_sha256": plan["directions_sha256"],
        }

    pending = [r for r in runs if not run_done(out_root / r["run_id"], expected_manifest(r))]
    base.log(f"确认臂分片 {args.shard_id}/{args.num_shards}：{len(runs)} 运行，待执行 {len(pending)}")
    if not pending:
        return

    mimi, lm = base.load_models(args)
    layers = getattr(lm.transformer, "layers", None)
    layers_used = sorted({int(v) for v in plan["layers_used"]})
    if layers is None or max(layers_used) >= len(layers):
        raise base.AdapterError(f"lm.transformer.layers 缺失或层号越界（{layers_used}）")
    hidden_dim = int(next(iter(directions.values())).shape[0])
    model_dtype = next(lm.parameters()).dtype
    state = {
        "active_layer": None,
        "vec": torch.zeros(hidden_dim, dtype=model_dtype, device=args.device),
        "unit32": torch.zeros(hidden_dim, dtype=torch.float32, device=args.device),
        "target": torch.zeros((), dtype=torch.float32, device=args.device),
        "clamp_active": False,
        "gate": torch.ones((), dtype=torch.float32, device=args.device),
        "hook_python_calls": 0,
    }

    def hook_for(layer: int):
        def hook(_module, _inputs, output):
            if state["active_layer"] != layer:
                return None
            tensor = output[0] if isinstance(output, tuple) else output
            if tensor.shape[-1] != hidden_dim:
                raise base.AdapterError(f"层输出宽度 {tensor.shape[-1]} ≠ 方向维度 {hidden_dim}")
            state["hook_python_calls"] += 1
            gate = state["gate"]
            steered = tensor + (gate.to(tensor.dtype) * state["vec"])
            if state["clamp_active"]:
                # float32 钳制数学：把 v̂ 分量钳到 μ_v，再回写模型精度。
                t32 = tensor.to(torch.float32)
                projection = (t32 * state["unit32"]).sum(dim=-1, keepdim=True)
                delta = (state["target"] - projection) * state["unit32"]
                steered = steered + (gate * delta).to(tensor.dtype)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        return hook

    hook_handles = [layers[layer].register_forward_hook(hook_for(layer)) for layer in layers_used]

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

    sample_rate = int(plan["sample_rate"])
    frame_samples = round(sample_rate / float(plan["frame_hz"]))
    n_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    code_version = base.resolve_code_version(args.code_version)

    for index, run in enumerate(pending):
        session = run["session"]
        condition = run["condition"]
        out_dir = out_root / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        params = steer_params_for(condition, plan, directions)
        gate = _load_gate(plan, str(session["session_id"]), str(condition["gate"]), n_frames)
        state["active_layer"] = int(params["layer"])
        state["vec"].copy_(torch.from_numpy(params["vec"]).to(state["vec"].dtype))
        state["unit32"].copy_(torch.from_numpy(params["unit"]).to(torch.float32))
        state["target"].fill_(float(params["target"]) if params["target"] is not None else 0.0)
        state["clamp_active"] = params["mode"] == MODE_CLAMP
        state["hook_python_calls"] = 0

        wav = base.read_wav_mono(session["user_wav"], sample_rate, max_seconds=float(plan["window_s"]))
        torch.manual_seed(int(session["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(session["seed"]))
        gate_device = torch.from_numpy(gate).to(args.device)
        user_audio = torch.from_numpy(wav).to(device=args.device, dtype=torch.float32)
        agent_chunks: list[np.ndarray] = []
        text_tokens: list[int] = []
        first_emitted = None
        lm_gen = LMGen(lm, **lm_gen_kwargs)
        started = time.perf_counter()
        with torch.inference_mode(), mimi.streaming(1), lm_gen.streaming(1):
            for frame_index in range(n_frames):
                state["gate"].copy_(gate_device[frame_index])
                block = user_audio[frame_index * frame_samples : (frame_index + 1) * frame_samples]
                if int(block.numel()) < frame_samples:
                    pad = torch.zeros(frame_samples - int(block.numel()), device=args.device)
                    block = torch.cat([block, pad])
                codes = mimi.encode(block.reshape(1, 1, -1))
                tokens = lm_gen.step(codes)
                if tokens is None:
                    continue
                if first_emitted is None:
                    first_emitted = frame_index
                text_tokens.append(int(tokens[0, 0, 0].item()))
                agent_wav = mimi.decode(tokens[:, 1:, :])
                agent_chunks.append(agent_wav[0, 0].float().cpu().numpy())
        wall = time.perf_counter() - started
        del lm_gen
        gc.collect()
        if state["hook_python_calls"] != n_frames:
            raise base.AdapterError(
                f"干预层执行 {state['hook_python_calls']} 次，预期 {n_frames}（{run['run_id']}）"
            )
        agent = np.concatenate(agent_chunks) if agent_chunks else np.zeros(0, dtype=np.float32)
        base._write_wav(out_dir / "agent.wav", agent, sample_rate)
        np.save(out_dir / "text_tokens.npy", np.asarray(text_tokens, dtype=np.int32), allow_pickle=False)
        steps_per_s = n_frames / wall if wall > 0 else None
        base.write_json_atomic(
            out_dir / "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "completed": True,
                "run_id": run["run_id"],
                "session_id": session["session_id"],
                "user_wav": session["user_wav"],
                "user_wav_sha256": session["user_wav_sha256"],
                "seed": session["seed"],
                "condition": condition,
                "gate_active_frames": int(gate.sum()),
                "proj_std_used": params["proj_std"],
                "clamp_target": params["target"],
                "window_s": plan["window_s"],
                "temperature": plan["temperature"],
                "text_temperature": plan["text_temperature"],
                "top_k": plan["top_k"],
                "top_k_text": plan["top_k_text"],
                "first_emitted_frame": first_emitted,
                "n_frames_in": n_frames,
                "n_agent_samples": len(agent),
                "hook_calls": int(state["hook_python_calls"]),
                "wall_s": wall,
                "steps_per_s": steps_per_s,
                "no_cuda_graph": os.environ.get("NO_CUDA_GRAPH"),
                "plan_sha256": plan_sha256,
                "directions_sha256": plan["directions_sha256"],
                "code_version": code_version,
            },
        )
        base.log(
            f"[{index + 1}/{len(pending)}] {run['run_id']} 完成：{wall:.1f} s"
            f"（{steps_per_s and round(steps_per_s, 2)} 步/秒，门活跃 {int(gate.sum())}/{n_frames} 帧）"
        )
    for handle in hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
