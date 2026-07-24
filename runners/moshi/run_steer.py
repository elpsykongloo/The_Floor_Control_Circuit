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
import gc
import hashlib
import inspect
import json
import math
import os
import sys
import time
from contextlib import contextmanager
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
USER_CODES_SCHEMA = "e2_lite_user_codes_v1"
EXECUTION_BACKENDS = (
    "eager_reference",
    "hybrid_graph",
    "full_graph",
    "full_step_graph",
)


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
        return {name: np.asarray(payload[name], dtype=np.float64) for name in payload.files if name != "__meta__"}


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
    # 会话整体分片：同一段用户音频只需在一张卡上编码一次，也保留条件配对的局部性。
    session_shards: dict[str, int] = {}
    selected = []
    for run in runs:
        session_id = str(run["session"]["session_id"])
        if session_id not in session_shards:
            session_shards[session_id] = len(session_shards) % num_shards
        if session_shards[session_id] == shard_id:
            selected.append(run)
    return selected


def condition_batches(runs: list[dict], batch_size: int) -> list[list[dict]]:
    """同会话非基线条件组成固定小批次；激活基线始终单独执行。"""
    if batch_size < 1:
        raise AdapterError("条件批次必须至少为 1")
    batches: list[list[dict]] = []
    pending: list[dict] = []
    pending_session = None

    def flush() -> None:
        nonlocal pending
        while pending:
            batches.append(pending[:batch_size])
            pending = pending[batch_size:]

    for run in runs:
        session_id = str(run["session"]["session_id"])
        if pending_session is not None and session_id != pending_session:
            flush()
        pending_session = session_id
        if run["condition"].get("cache_acts"):
            flush()
            batches.append([run])
        else:
            pending.append(run)
            if len(pending) == batch_size:
                flush()
    flush()
    return batches


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_fingerprint(plan: dict) -> str:
    """路径原样参与摘要，防止不同数据根或方向包被误判为同一计划。"""
    return _canonical_sha256(plan)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _legacy_reference_matches(payload: dict, expected: dict) -> bool:
    """兼容本轮已经启动的旧清单，使参考后端仍可断点续跑。"""
    if expected.get("execution_backend") != "eager_reference":
        return False
    return (
        str(payload.get("no_cuda_graph")) == "1"
        and payload.get("run_id") == expected.get("run_id")
        and payload.get("directions_sha256") == expected.get("directions_sha256")
    )


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
    if expected is not None:
        strict_match = all(payload.get(key) == value for key, value in expected.items())
        if not strict_match and not _legacy_reference_matches(payload, expected):
            return False
    required = [out_dir / "agent.wav", out_dir / "text_tokens.npy"]
    effective_backend = payload.get("execution_backend", "eager_reference")
    if effective_backend != "eager_reference":
        required.append(out_dir / "agent_codes.npy")
    if payload.get("condition", {}).get("cache_acts"):
        required.extend(out_dir / f"acts_L{int(layer)}.npy" for layer in payload.get("cache_layers", [28, 29, 30, 31]))
    return all(path.is_file() for path in required)


def _shared_noise_sample_token(
    torch,
    logits,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
):
    """批内条件共享一行指数噪声，复现 batch=1 的公共随机数消耗。"""
    if not use_sampling or temp <= 0.0:
        return torch.argmax(logits, dim=-1)
    if top_p > 0.0:
        raise AdapterError("共享随机数批处理暂不支持 top-p")
    probs = torch.softmax(logits / temp, dim=-1)
    if top_k > 0:
        k = min(int(top_k), int(probs.shape[-1]))
        probs, indices = torch.topk(probs, k, dim=-1)
    else:
        indices = None
    noise_shape = list(probs.shape)
    noise_shape[0] = 1
    noise = torch.empty(noise_shape, dtype=probs.dtype, device=probs.device).exponential_(1)
    choice = (probs / noise.expand_as(probs)).argmax(dim=-1, keepdim=True)
    if indices is not None:
        choice = indices.gather(-1, choice)
    return choice[..., 0]


@contextmanager
def common_random_sampling(enabled: bool):
    """在 runner 进程内临时替换 LMGen 的采样函数。"""
    if not enabled:
        yield
        return
    import moshi.models.lm as lm_module
    import torch

    original = lm_module.sample_token

    def sample_token(logits, use_sampling=False, temp=1.0, top_k=0, top_p=0.0):
        return _shared_noise_sample_token(
            torch,
            logits,
            use_sampling=use_sampling,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
        )

    lm_module.sample_token = sample_token
    try:
        yield
    finally:
        lm_module.sample_token = original


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    import wave

    clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with wave.open(str(tmp), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm.tobytes())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def probe_api(args) -> None:
    from moshi.models import LMGen

    signature = str(inspect.signature(LMGen.__init__))
    print(json.dumps({"LMGen.__init__": signature}, ensure_ascii=False, indent=1))


def _release_cuda_graphs(state, names: tuple[str, ...]) -> None:
    """在流式状态销毁前解除图对缓存的引用。"""
    for name in names:
        graph = getattr(state, name, None)
        if graph is not None and callable(getattr(graph, "reset", None)):
            graph.reset()


def _user_code_cache_paths(
    cache_root: Path,
    session: dict,
    plan: dict,
    mimi_sha256: str,
    chunk_frames: int,
    n_codebooks: int,
) -> tuple[Path, Path, dict]:
    metadata = {
        "schema": USER_CODES_SCHEMA,
        "session_id": session["session_id"],
        "user_wav_sha256": session["user_wav_sha256"],
        "mimi_sha256": mimi_sha256,
        "sample_rate": int(plan["sample_rate"]),
        "frame_hz": float(plan["frame_hz"]),
        "window_s": float(plan["window_s"]),
        "chunk_frames": int(chunk_frames),
        "n_codebooks": int(n_codebooks),
    }
    key = _canonical_sha256(metadata)[:16]
    stem = f"{session['session_id']}__{key}"
    return cache_root / f"{stem}.npy", cache_root / f"{stem}.json", metadata


def _load_cached_user_codes(
    torch,
    cache_root: Path,
    session: dict,
    plan: dict,
    mimi_sha256: str,
    chunk_frames: int,
    n_codebooks: int,
    device: str,
):
    npy_path, meta_path, expected = _user_code_cache_paths(
        cache_root,
        session,
        plan,
        mimi_sha256,
        chunk_frames,
        n_codebooks,
    )
    if not npy_path.is_file() or not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        values = np.load(npy_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    expected_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    if (
        values.ndim != 3
        or values.shape[0] != 1
        or values.shape[1] != n_codebooks
        or values.shape[-1] != expected_frames
        or values.dtype.kind not in {"i", "u"}
    ):
        return None
    if metadata.get("codes_sha256") != hashlib.sha256(values.tobytes()).hexdigest():
        return None
    log(f"复用用户码缓存：{npy_path.name}")
    return torch.from_numpy(np.ascontiguousarray(values)).to(device), metadata


def _save_cached_user_codes(
    codes,
    cache_root: Path,
    session: dict,
    plan: dict,
    mimi_sha256: str,
    chunk_frames: int,
    n_codebooks: int,
    encode_wall_s: float,
) -> None:
    npy_path, meta_path, metadata = _user_code_cache_paths(
        cache_root,
        session,
        plan,
        mimi_sha256,
        chunk_frames,
        n_codebooks,
    )
    values = codes.detach().cpu().numpy()
    metadata["shape"] = list(values.shape)
    metadata["dtype"] = str(values.dtype)
    metadata["codes_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    metadata["encode_wall_s"] = float(encode_wall_s)
    _atomic_save_npy(npy_path, values)
    write_json_atomic(meta_path, metadata)


def encode_user_codes(
    mimi,
    session: dict,
    plan: dict,
    device: str,
    *,
    chunk_frames: int = 1,
    cache_root: Path | None = None,
    mimi_sha256: str = "",
    n_codebooks: int = 8,
):
    """有状态分块编码会话前缀，并可跨断点复用离散码。"""
    import torch

    if chunk_frames < 1:
        raise AdapterError("Mimi 编码块帧数必须至少为 1")
    if cache_root is not None:
        cached = _load_cached_user_codes(
            torch,
            cache_root,
            session,
            plan,
            mimi_sha256,
            chunk_frames,
            n_codebooks,
            device,
        )
        if cached is not None:
            codes, metadata = cached
            return codes, {
                "wall_s": 0.0,
                "source_wall_s": float(metadata.get("encode_wall_s", 0.0)),
                "cache_hit": True,
                "cuda_graph": None,
            }

    sample_rate = int(plan["sample_rate"])
    frame_samples = round(sample_rate / float(plan["frame_hz"]))
    n_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    if n_frames % chunk_frames:
        raise AdapterError(f"Mimi 编码块 {chunk_frames} 帧不能整除分析窗 {n_frames} 帧；请选用可整除的块长")
    wav = read_wav_mono(session["user_wav"], sample_rate, max_seconds=float(plan["window_s"]))
    user_audio = torch.from_numpy(wav).to(device=device, dtype=torch.float32).reshape(1, 1, -1)
    all_codes = None
    graph_built = False
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    with torch.inference_mode(), mimi.streaming(1):
        state = mimi._streaming_state
        for frame_start in range(0, n_frames, chunk_frames):
            frame_stop = min(frame_start + chunk_frames, n_frames)
            sample_start = frame_start * frame_samples
            sample_stop = frame_stop * frame_samples
            codes = mimi.encode(user_audio[..., sample_start:sample_stop])
            expected = frame_stop - frame_start
            if codes.shape[-1] != expected:
                raise AdapterError(f"Mimi 编码返回 {codes.shape[-1]} 个码帧，预期 {expected}")
            if all_codes is None:
                all_codes = torch.empty(
                    (*codes.shape[:-1], n_frames),
                    dtype=codes.dtype,
                    device=codes.device,
                )
            all_codes[..., frame_start:frame_stop].copy_(codes)
        if state is not None:
            graph_built = getattr(state.graphed_encoder, "_graph", None) is not None
            _release_cuda_graphs(
                state,
                ("graphed_encoder", "graphed_tr_enc", "graphed_decoder", "graphed_tr_dec"),
            )
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    if all_codes is None:
        raise AdapterError("Mimi 未产生用户码")
    if cache_root is not None:
        _save_cached_user_codes(
            all_codes,
            cache_root,
            session,
            plan,
            mimi_sha256,
            chunk_frames,
            n_codebooks,
            wall,
        )
    return all_codes, {
        "wall_s": wall,
        "source_wall_s": wall,
        "cache_hit": False,
        "cuda_graph": graph_built,
    }


def _steer_matrix(torch, runs: list[dict], plan: dict, steer_state: dict):
    rows = []
    for run in runs:
        condition = run["condition"]
        if float(condition["alpha"]) == 0.0:
            vec = np.zeros(steer_state["hidden_dim"], dtype=np.float64)
        else:
            vec = steer_vector_np(
                float(condition["alpha"]),
                float(plan["proj_std"][condition["direction"]]),
                steer_state["directions"][condition["direction"]],
            )
        rows.append(vec)
    values = np.stack(rows)
    return torch.from_numpy(values).to(
        device=steer_state["device"],
        dtype=steer_state["model_dtype"],
    )


def execute_run_reference(
    mimi,
    lm_gen_factory,
    run: dict,
    plan: dict,
    steer_state: dict,
    device: str,
) -> dict:
    """保留逐帧 Mimi 编码→生成→解码顺序，供既有样本续跑与验收。"""
    import torch

    session = run["session"]
    condition = run["condition"]
    sample_rate = int(plan["sample_rate"])
    frame_samples = round(sample_rate / float(plan["frame_hz"]))
    n_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    wav = read_wav_mono(
        session["user_wav"],
        sample_rate,
        max_seconds=float(plan["window_s"]),
    )
    torch.manual_seed(int(session["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(session["seed"]))

    steer_state["vec"] = _steer_matrix(torch, [run], plan, steer_state)
    steer_state["hook_python_calls"] = 0
    steer_state["hook_device_counter"] = torch.zeros(
        (),
        dtype=torch.int32,
        device=device,
    )
    cache_layers = [int(value) for value in plan["cache_layers_baseline"]] if condition.get("cache_acts") else []
    steer_state["capture_enabled"] = bool(cache_layers)
    steer_state["captured_latest"] = {}
    capture_history = {
        layer: torch.empty(
            (n_frames, steer_state["hidden_dim"]),
            dtype=torch.float16,
            device=device,
        )
        for layer in cache_layers
    }
    text_tokens = torch.empty(n_frames, dtype=torch.int32, device=device)
    agent_codes = torch.empty(
        (int(steer_state["dep_q"]), n_frames),
        dtype=torch.long,
        device=device,
    )
    agent_audio = torch.empty(
        n_frames * frame_samples,
        dtype=torch.float32,
        device=device,
    )
    token_count = 0
    samples_written = 0
    first_emitted = None
    lm_gen = lm_gen_factory()
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    user_audio = torch.from_numpy(wav).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode(), mimi.streaming(1), lm_gen.streaming(1):
        for frame_index in range(n_frames):
            sample_start = frame_index * frame_samples
            block = user_audio[sample_start : sample_start + frame_samples]
            codes = mimi.encode(block.reshape(1, 1, -1))
            tokens = lm_gen.step(codes)
            if cache_layers:
                missing = set(cache_layers) - set(steer_state["captured_latest"])
                if missing:
                    raise AdapterError(f"激活钩子未产生层：{sorted(missing)}")
                for layer in cache_layers:
                    capture_history[layer][frame_index].copy_(steer_state["captured_latest"][layer][0])
            if tokens is None:
                continue
            if first_emitted is None:
                first_emitted = frame_index
            text_tokens[token_count].copy_(tokens[0, 0, 0])
            frame_codes = tokens[:, 1:, :]
            agent_codes[:, token_count].copy_(frame_codes[0, :, 0])
            decoded = mimi.decode(frame_codes)[0, 0].reshape(-1)
            stop = samples_written + int(decoded.numel())
            if stop > agent_audio.numel():
                raise AdapterError(f"Mimi 参考解码样本溢出：{stop} > {agent_audio.numel()}")
            agent_audio[samples_written:stop].copy_(decoded)
            samples_written = stop
            token_count += 1
        torch.cuda.synchronize(device)
        expected_samples = token_count * frame_samples
        if samples_written != expected_samples:
            raise AdapterError(f"Mimi 参考解码仅写出 {samples_written}/{expected_samples} 个样本")
        captured = {layer: rows.cpu().numpy() for layer, rows in capture_history.items()}
        text_cpu = text_tokens[:token_count].cpu().numpy()
        codes_cpu = agent_codes[:, :token_count].cpu().numpy()
        audio_cpu = agent_audio[:samples_written].cpu().numpy()
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    steer_state["captured_latest"].clear()
    del lm_gen
    gc.collect()

    hook_calls = int(steer_state["hook_python_calls"])
    if hook_calls != n_frames:
        raise AdapterError(f"steering 执行 {hook_calls} 次，预期 {n_frames}")
    return {
        "agent_wav": audio_cpu,
        "agent_codes": codes_cpu,
        "text_tokens": text_cpu[None, :],
        "first_emitted_frame": first_emitted,
        "n_frames_in": n_frames,
        "n_frames_out": token_count,
        "generation_wall_s": wall,
        "captured": captured,
        "hook_calls": hook_calls,
        "hook_python_calls": hook_calls,
        "graph_main": False,
        "graph_depth": False,
        "graph_step": False,
    }


def execute_run_batch(
    lm_gen_factory,
    user_codes,
    runs: list[dict],
    plan: dict,
    steer_state: dict,
    execution_backend: str,
) -> dict:
    """一个会话的一组条件联合生成；音频码稍后独立解码。"""
    import torch

    if not runs:
        raise AdapterError("运行批次为空")
    session_ids = {str(run["session"]["session_id"]) for run in runs}
    if len(session_ids) != 1:
        raise AdapterError("一个条件批次只能包含同一会话")
    if sum(bool(run["condition"].get("cache_acts")) for run in runs) > 1:
        raise AdapterError("一个批次至多包含一个激活缓存条件")
    if any(run["condition"].get("cache_acts") for run in runs) and len(runs) != 1:
        raise AdapterError("基线激活条件必须单独运行")

    session = runs[0]["session"]
    batch_size = len(runs)
    n_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    if user_codes.shape[-1] != n_frames:
        raise AdapterError(f"用户码帧数 {user_codes.shape[-1]} ≠ 计划 {n_frames}")
    torch.manual_seed(int(session["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(session["seed"]))

    steer_state["vec"] = _steer_matrix(torch, runs, plan, steer_state)
    steer_state["hook_python_calls"] = 0
    steer_state["hook_device_counter"] = torch.zeros(
        (),
        dtype=torch.int32,
        device=steer_state["device"],
    )
    cache_layers = [int(v) for v in plan["cache_layers_baseline"]] if runs[0]["condition"].get("cache_acts") else []
    steer_state["capture_enabled"] = bool(cache_layers)
    steer_state["captured_latest"] = {}
    capture_history = {
        layer: torch.empty(
            (n_frames, steer_state["hidden_dim"]),
            dtype=torch.float16,
            device=steer_state["device"],
        )
        for layer in cache_layers
    }

    text_tokens = torch.empty(
        (batch_size, n_frames),
        dtype=torch.int32,
        device=steer_state["device"],
    )
    agent_codes = torch.empty(
        (batch_size, int(steer_state["dep_q"]), n_frames),
        dtype=torch.long,
        device=steer_state["device"],
    )
    token_count = 0
    first_emitted = None
    lm_gen = lm_gen_factory()
    graph_main = False
    graph_depth = False
    graph_step = False
    graphed_step = None
    started = time.perf_counter()
    torch.cuda.synchronize(steer_state["device"])
    with torch.inference_mode(), lm_gen.streaming(batch_size):
        state = lm_gen._streaming_state
        if state is None:
            raise AdapterError("LMGen 流式状态未建立")
        if execution_backend == "hybrid_graph":
            state.graphed_main.disable = True
        if execution_backend == "full_step_graph":
            from moshi.utils.compile import CUDAGraphed

            graphed_step = CUDAGraphed(
                lm_gen.step,
                warmup_steps=int(lm_gen.max_delay),
                disable=False,
            )
        batched_codes = user_codes.expand(batch_size, -1, -1)
        for frame_index in range(n_frames):
            frame_codes = batched_codes[..., frame_index : frame_index + 1]
            tokens = graphed_step(frame_codes) if graphed_step is not None else lm_gen.step(frame_codes)
            if cache_layers:
                missing = set(cache_layers) - set(steer_state["captured_latest"])
                if missing:
                    raise AdapterError(f"激活钩子未产生层：{sorted(missing)}")
                for layer in cache_layers:
                    latest = steer_state["captured_latest"][layer]
                    capture_history[layer][frame_index].copy_(latest[0])
            if tokens is None:
                continue
            if first_emitted is None:
                first_emitted = frame_index
            text_tokens[:, token_count].copy_(tokens[:, 0, 0])
            agent_codes[:, :, token_count].copy_(tokens[:, 1:, 0])
            token_count += 1
        torch.cuda.synchronize(steer_state["device"])
        graph_main = getattr(state.graphed_main, "_graph", None) is not None
        graph_depth = state.graphed_depth is not None and getattr(state.graphed_depth, "_graph", None) is not None
        graph_step = graphed_step is not None and getattr(graphed_step, "_graph", None) is not None
        hook_calls = (
            int(steer_state["hook_device_counter"].item())
            if execution_backend in {"full_graph", "full_step_graph"}
            else int(steer_state["hook_python_calls"])
        )
        captured = {layer: rows.cpu().numpy() for layer, rows in capture_history.items()}
        text_cpu = text_tokens[:, :token_count].cpu().numpy()
        if graphed_step is not None:
            graphed_step.reset()
        _release_cuda_graphs(state, ("graphed_main", "graphed_depth"))
        steer_state["captured_latest"].clear()
        torch.cuda.synchronize(steer_state["device"])
    wall = time.perf_counter() - started
    del lm_gen
    gc.collect()

    if hook_calls != n_frames:
        raise AdapterError(f"steering 执行 {hook_calls} 次，预期 {n_frames}")
    if execution_backend == "full_graph" and not graph_main:
        raise AdapterError("主 transformer CUDA 图未成功建立")
    if execution_backend == "full_graph" and not graph_depth:
        raise AdapterError("音频码生成器 CUDA 图未成功建立")
    if execution_backend == "full_step_graph" and not graph_step:
        raise AdapterError("完整 LMGen.step CUDA 图未成功建立")
    return {
        "agent_codes_device": agent_codes[:, :, :token_count],
        "text_tokens": text_cpu,
        "first_emitted_frame": first_emitted,
        "n_frames_in": n_frames,
        "n_frames_out": token_count,
        "generation_wall_s": wall,
        "captured": captured,
        "hook_calls": hook_calls,
        "hook_python_calls": int(steer_state["hook_python_calls"]),
        "graph_main": graph_main,
        "graph_depth": graph_depth,
        "graph_step": graph_step,
    }


def decode_agent_codes(
    mimi,
    codes,
    plan: dict,
    device: str,
    *,
    chunk_frames: int,
) -> dict:
    """在生成完成后用 Mimi 有状态分块解码，尾部填充结果会被裁掉。"""
    import torch

    if chunk_frames < 1:
        raise AdapterError("Mimi 解码块帧数必须至少为 1")
    if codes.ndim != 3 or int(codes.shape[0]) < 1:
        raise AdapterError(f"Mimi 解码码形状无效：{tuple(codes.shape)}")
    batch_size = int(codes.shape[0])
    n_frames = int(codes.shape[-1])
    if n_frames < 1:
        raise AdapterError("Mimi 解码至少需要一个码帧")
    frame_samples = round(int(plan["sample_rate"]) / float(plan["frame_hz"]))
    padded_frames = math.ceil(n_frames / chunk_frames) * chunk_frames
    if padded_frames == n_frames:
        padded = codes
    else:
        padded = torch.empty(
            (*codes.shape[:-1], padded_frames),
            dtype=codes.dtype,
            device=codes.device,
        )
        padded[..., :n_frames].copy_(codes)
        padded[..., n_frames:].copy_(codes[..., -1:].expand(*codes.shape[:-1], padded_frames - n_frames))
    audio = torch.empty(
        (batch_size, padded_frames * frame_samples),
        dtype=torch.float32,
        device=device,
    )
    written = 0
    graph_decoder = False
    graph_transformer = False
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    with torch.inference_mode(), mimi.streaming(batch_size):
        state = mimi._streaming_state
        for start in range(0, padded_frames, chunk_frames):
            decoded = mimi.decode(padded[..., start : start + chunk_frames])[:, 0, :]
            decoded_samples = int(decoded.shape[-1])
            stop = written + decoded_samples
            if stop > audio.shape[-1]:
                raise AdapterError(f"Mimi 解码样本溢出：{stop} > {audio.shape[-1]}")
            audio[:, written:stop].copy_(decoded)
            written = stop
        torch.cuda.synchronize(device)
        if state is not None:
            graph_decoder = getattr(state.graphed_decoder, "_graph", None) is not None
            graph_transformer = (
                state.graphed_tr_dec is not None and getattr(state.graphed_tr_dec, "_graph", None) is not None
            )
            _release_cuda_graphs(
                state,
                ("graphed_encoder", "graphed_tr_enc", "graphed_decoder", "graphed_tr_dec"),
            )
        if written != audio.shape[-1]:
            raise AdapterError(f"Mimi 解码仅写出 {written}/{audio.shape[-1]} 个样本")
        result = audio[:, : n_frames * frame_samples].cpu().numpy()
        torch.cuda.synchronize(device)
    return {
        "agent_wav": result,
        "wall_s": time.perf_counter() - started,
        "cuda_graph_decoder": graph_decoder,
        "cuda_graph_transformer": graph_transformer,
    }


def _resolve_weight_paths(args) -> tuple[Path, Path]:
    root = Path(args.model_root)
    mimi_path = Path(args.mimi_weight) if args.mimi_weight else None
    lm_path = Path(args.lm_weight) if args.lm_weight else None
    if mimi_path is None:
        for pattern in ("tokenizer-*.safetensors", "*mimi*.safetensors"):
            hits = sorted(
                root.glob(pattern),
                key=lambda item: item.stat().st_size,
                reverse=True,
            )
            if hits:
                mimi_path = hits[0]
                break
    if lm_path is None:
        for pattern in ("model.safetensors", "*.safetensors"):
            hits = sorted(
                root.glob(pattern),
                key=lambda item: item.stat().st_size,
                reverse=True,
            )
            if hits:
                lm_path = hits[0]
                break
    if mimi_path is None or not mimi_path.is_file():
        raise AdapterError("无法解析 Mimi 权重")
    if lm_path is None or not lm_path.is_file():
        raise AdapterError("无法解析 Moshi LM 权重")
    if mimi_path.resolve() == lm_path.resolve():
        raise AdapterError("LM 与 Mimi 权重解析到同一文件")
    return mimi_path.resolve(), lm_path.resolve()


def _weight_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _batch_id(batch: list[dict]) -> str:
    session_id = str(batch[0]["session"]["session_id"])
    names = [str(run["condition"]["name"]) for run in batch]
    return f"{session_id}__{'--'.join(names)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Moshi E2-lite steering runner")
    parser.add_argument("--plan")
    parser.add_argument("--model-root", required=False)
    parser.add_argument("--lm-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-codebooks", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--only-condition", default=None, help="只跑指定条件名（冒烟用）")
    parser.add_argument("--limit", type=int, default=None, help="至多执行 N 个运行（冒烟用）")
    parser.add_argument("--out-root", default=None, help="覆盖输出根；其下建立 runs/ 与 user_codes/")
    parser.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        default="eager_reference",
        help="执行后端；参考后端保留旧制式，优化后端启用 CUDA 图",
    )
    parser.add_argument("--condition-batch-size", type=int, default=1)
    parser.add_argument("--encode-chunk-frames", type=int, default=1)
    parser.add_argument("--decode-chunk-frames", type=int, default=1)
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=1,
        help="Mimi 联合解码的条件数；不得超过条件批次",
    )
    parser.add_argument(
        "--persistent-user-code-cache",
        action="store_true",
        help="跨进程复用用户离散码；会改变后续 CUDA 图随机数状态，仅供消融",
    )
    parser.add_argument(
        "--no-user-code-cache",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="已验收的严格等价优化预设：full_graph、单条件、逐帧 Mimi",
    )
    parser.add_argument(
        "--allow-non-equivalent-experimental",
        action="store_true",
        help="允许使用已知可能改变数值轨迹的批处理、分块或整步图实验",
    )
    parser.add_argument("--probe-api", action="store_true")
    parser.add_argument("--code-version", default=None)
    args = parser.parse_args()
    if args.probe_api:
        probe_api(args)
        return
    if not args.plan:
        raise SystemExit("缺 --plan")
    if not args.model_root:
        raise SystemExit("缺 --model-root")

    if args.optimized:
        args.execution_backend = "full_graph"
        args.condition_batch_size = 1
        args.encode_chunk_frames = 1
        args.decode_chunk_frames = 1
        args.decode_batch_size = 1
    if args.condition_batch_size < 1:
        raise SystemExit("--condition-batch-size 必须至少为 1")
    if args.encode_chunk_frames < 1 or args.decode_chunk_frames < 1:
        raise SystemExit("Mimi 编解码块帧数必须至少为 1")
    if args.decode_batch_size < 1:
        raise SystemExit("--decode-batch-size 必须至少为 1")
    if args.decode_batch_size > args.condition_batch_size:
        raise SystemExit("--decode-batch-size 不得超过 --condition-batch-size")
    if args.persistent_user_code_cache and args.no_user_code_cache:
        raise SystemExit("用户码缓存开关互相冲突")
    non_equivalent_features = []
    if args.execution_backend == "full_step_graph":
        non_equivalent_features.append("full_step_graph")
    if args.condition_batch_size > 1:
        non_equivalent_features.append("condition-batch-size>1")
    if args.encode_chunk_frames > 1:
        non_equivalent_features.append("encode-chunk-frames>1")
    if args.decode_chunk_frames > 1:
        non_equivalent_features.append("decode-chunk-frames>1")
    if args.decode_batch_size > 1:
        non_equivalent_features.append("decode-batch-size>1")
    if args.persistent_user_code_cache:
        non_equivalent_features.append("persistent-user-code-cache")
    if non_equivalent_features and not args.allow_non_equivalent_experimental:
        joined = "、".join(non_equivalent_features)
        raise SystemExit(
            f"以下配置可能改变 E2-lite 数值轨迹：{joined}；仅做消融验证时显式传 --allow-non-equivalent-experimental"
        )
    if args.execution_backend == "eager_reference":
        if args.condition_batch_size != 1:
            raise SystemExit("参考后端只允许 condition-batch-size=1")
        if args.encode_chunk_frames != 1 or args.decode_chunk_frames != 1:
            raise SystemExit("参考后端只允许逐帧 Mimi 编解码")
        if args.decode_batch_size != 1:
            raise SystemExit("参考后端只允许 decode-batch-size=1")
        os.environ["NO_CUDA_GRAPH"] = "1"
    else:
        os.environ.pop("NO_CUDA_GRAPH", None)

    import torch
    from moshi.models import LMGen

    device_object = torch.device(args.device)
    if device_object.type == "cuda":
        # CUDA 图使用当前设备的捕获流；多卡进程必须先显式绑定各自设备。
        torch.cuda.set_device(device_object)

    plan = _load_plan(Path(args.plan))
    directions = _load_directions(plan)
    runs = shard_runs(iter_runs(plan), int(args.num_shards), int(args.shard_id))
    if args.only_condition:
        runs = [r for r in runs if r["condition"]["name"] == args.only_condition]
    if args.limit is not None:
        runs = runs[: int(args.limit)]
    plan_sha256 = _plan_fingerprint(plan)
    output_base = Path(args.out_root) if args.out_root else Path(plan["out_root"])
    if args.optimized and args.out_root is None:
        output_base = output_base.with_name(output_base.name + "_optimized")
    out_root = output_base / "runs"
    code_version = resolve_code_version(args.code_version)
    mimi_path, lm_path = _resolve_weight_paths(args)
    execution_profile = {
        "schema": "e2_lite_execution_profile_v1",
        "execution_backend": args.execution_backend,
        "condition_batch_size": int(args.condition_batch_size),
        "encode_chunk_frames": int(args.encode_chunk_frames),
        "decode_chunk_frames": int(args.decode_chunk_frames),
        "decode_batch_size": int(args.decode_batch_size),
        "n_codebooks": int(args.n_codebooks),
        "persistent_user_code_cache": bool(args.persistent_user_code_cache),
        "equivalence_contract": ("experimental_non_equivalent" if non_equivalent_features else "reference_exact"),
        "code_version": code_version,
        "mimi_weight": _weight_identity(mimi_path),
        "lm_weight": _weight_identity(lm_path),
    }
    execution_profile_sha256 = _canonical_sha256(execution_profile)

    def expected_manifest(run: dict, batch: list[dict]) -> dict:
        return {
            "run_id": run["run_id"],
            "execution_backend": args.execution_backend,
            "plan_sha256": plan_sha256,
            "directions_sha256": plan["directions_sha256"],
            "execution_profile_sha256": execution_profile_sha256,
            "group_id": _batch_id(batch),
            "group_size": len(batch),
        }

    all_batches = condition_batches(runs, int(args.condition_batch_size))
    pending_batches = [
        batch
        for batch in all_batches
        if not all(
            run_done(
                out_root / run["run_id"],
                expected_manifest(run, batch),
            )
            for run in batch
        )
    ]
    pending_count = sum(len(batch) for batch in pending_batches)
    log(
        f"分片 {args.shard_id}/{args.num_shards}：{len(runs)} 运行，"
        f"其中待执行 {pending_count}（{len(pending_batches)} 批）"
    )
    if not pending_batches:
        return

    if args.condition_batch_size > 1 and args.execution_backend not in {
        "full_graph",
        "full_step_graph",
    }:
        raise SystemExit("条件批处理仅在 CUDA 图后端启用")

    mimi, lm = load_models(args)
    layer_index = int(plan["layer"])
    transformer = lm.transformer
    layers = getattr(transformer, "layers", None)
    if layers is None or layer_index >= len(layers):
        raise AdapterError(f"lm.transformer.layers 缺失或层号越界（layer={layer_index}）")
    hidden_dim = int(next(iter(directions.values())).shape[0])
    model_dtype = next(lm.parameters()).dtype
    steer_state = {
        "vec": torch.zeros((1, hidden_dim), dtype=model_dtype, device=args.device),
        "directions": directions,
        "hidden_dim": hidden_dim,
        "dep_q": int(lm.dep_q),
        "device": args.device,
        "model_dtype": model_dtype,
        "execution_backend": args.execution_backend,
        "captured_latest": {},
        "capture_enabled": False,
        "hook_python_calls": 0,
        "hook_device_counter": None,
    }

    def steer_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.shape[-1] != hidden_dim:
            raise AdapterError(f"层输出宽度 {tensor.shape[-1]} ≠ 方向维度 {hidden_dim}")
        if int(tensor.shape[0]) != int(steer_state["vec"].shape[0]):
            raise AdapterError(f"层输出批次 {tensor.shape[0]} ≠ steering 批次 {steer_state['vec'].shape[0]}")
        steer_state["hook_python_calls"] += 1
        if steer_state["execution_backend"] in {"full_graph", "full_step_graph"}:
            steer_state["hook_device_counter"].add_(1)
        steered = tensor + steer_state["vec"][:, None, :]
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    hook_handles = [layers[layer_index].register_forward_hook(steer_hook)]

    def capture_hook_for(layer: int):
        def capture_hook(_module, _inputs, output):
            if not steer_state["capture_enabled"]:
                return None
            tensor = output[0] if isinstance(output, tuple) else output
            steer_state["captured_latest"][layer] = tensor[:, -1].detach().to(torch.float16)
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

    mimi_sha256 = sha256_file(mimi_path)
    cache_root = output_base / "user_codes" if args.persistent_user_code_cache and not args.no_user_code_cache else None
    encoded_session_id = None
    user_codes = None
    encode_meta: dict = {}
    completed_count = 0
    try:
        for batch_index, batch in enumerate(pending_batches):
            session_id = str(batch[0]["session"]["session_id"])
            group_id = _batch_id(batch)
            for run in batch:
                out_dir = out_root / run["run_id"]
                out_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(
                    out_dir / "manifest.json",
                    {
                        "schema": MANIFEST_SCHEMA,
                        "completed": False,
                        **expected_manifest(run, batch),
                    },
                )

            if args.execution_backend == "eager_reference":
                encode_meta = {
                    "wall_s": 0.0,
                    "source_wall_s": 0.0,
                    "cache_hit": False,
                    "cuda_graph": False,
                }
                result = execute_run_reference(
                    mimi,
                    lm_gen_factory,
                    batch[0],
                    plan,
                    steer_state,
                    args.device,
                )
                prepared = [
                    {
                        "row_index": 0,
                        "run": batch[0],
                        "agent_wav": result["agent_wav"],
                        "agent_codes": result["agent_codes"],
                        "decode_wall_s": 0.0,
                        "decode_group_wall_s": 0.0,
                        "decode_batch_size": 1,
                        "cuda_graph_decoder": False,
                        "cuda_graph_transformer": False,
                    }
                ]
                decode_wall_total = 0.0
            else:
                if session_id != encoded_session_id:
                    log(f"编码会话 {session_id} 的 {plan['window_s']} 秒用户音频")
                    user_codes, encode_meta = encode_user_codes(
                        mimi,
                        batch[0]["session"],
                        plan,
                        args.device,
                        chunk_frames=int(args.encode_chunk_frames),
                        cache_root=cache_root,
                        mimi_sha256=mimi_sha256,
                        n_codebooks=int(args.n_codebooks),
                    )
                    encoded_session_id = session_id
                if user_codes is None:
                    raise AdapterError("用户码缓存未初始化")
                with common_random_sampling(len(batch) > 1):
                    result = execute_run_batch(
                        lm_gen_factory,
                        user_codes,
                        batch,
                        plan,
                        steer_state,
                        args.execution_backend,
                    )

                prepared = []
                decode_wall_total = 0.0
                for decode_start in range(0, len(batch), int(args.decode_batch_size)):
                    decode_stop = min(
                        decode_start + int(args.decode_batch_size),
                        len(batch),
                    )
                    decode = decode_agent_codes(
                        mimi,
                        result["agent_codes_device"][decode_start:decode_stop],
                        plan,
                        args.device,
                        chunk_frames=int(args.decode_chunk_frames),
                    )
                    decode_size = decode_stop - decode_start
                    decode_wall = float(decode["wall_s"])
                    decode_wall_total += decode_wall
                    for row_index in range(decode_start, decode_stop):
                        local_index = row_index - decode_start
                        prepared.append(
                            {
                                "row_index": row_index,
                                "run": batch[row_index],
                                "agent_wav": decode["agent_wav"][local_index],
                                "agent_codes": (result["agent_codes_device"][row_index].detach().cpu().numpy()),
                                "decode_wall_s": decode_wall / decode_size,
                                "decode_group_wall_s": decode_wall,
                                "decode_batch_size": decode_size,
                                "cuda_graph_decoder": decode["cuda_graph_decoder"],
                                "cuda_graph_transformer": decode["cuda_graph_transformer"],
                            }
                        )

            group_wall = float(result["generation_wall_s"]) + decode_wall_total
            group_steps_per_s = result["n_frames_in"] * len(batch) / group_wall if group_wall > 0 else None

            for item in prepared:
                row_index = int(item["row_index"])
                run = item["run"]
                out_dir = out_root / run["run_id"]
                agent_wav = np.asarray(item["agent_wav"], dtype=np.float32)
                agent_codes_cpu = np.asarray(item["agent_codes"])
                _write_wav(
                    out_dir / "agent.wav",
                    agent_wav,
                    int(plan["sample_rate"]),
                )
                _atomic_save_npy(
                    out_dir / "text_tokens.npy",
                    result["text_tokens"][row_index],
                )
                _atomic_save_npy(out_dir / "agent_codes.npy", agent_codes_cpu)
                for layer, rows in result["captured"].items():
                    _atomic_save_npy(out_dir / f"acts_L{layer}.npy", rows)

                wall_equivalent = float(result["generation_wall_s"]) / len(batch) + float(item["decode_wall_s"])
                steps_per_s = result["n_frames_in"] / wall_equivalent if wall_equivalent > 0 else None
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
                        "cache_layers": [int(value) for value in plan["cache_layers_baseline"]],
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
                        "n_frames_out": result["n_frames_out"],
                        "n_agent_samples": int(agent_wav.size),
                        "agent_wav_sha256": sha256_file(out_dir / "agent.wav"),
                        "agent_codes_shape": list(agent_codes_cpu.shape),
                        "hook_calls": int(result["hook_calls"]),
                        "hook_python_calls": int(result["hook_python_calls"]),
                        "wall_s": wall_equivalent,
                        "steps_per_s": steps_per_s,
                        "group_id": group_id,
                        "group_size": len(batch),
                        "group_wall_s": group_wall,
                        "group_steps_per_s": group_steps_per_s,
                        "generation_wall_s": result["generation_wall_s"],
                        "decode_wall_s": item["decode_wall_s"],
                        "decode_group_wall_s": item["decode_group_wall_s"],
                        "encode_wall_s": encode_meta.get("wall_s"),
                        "encode_source_wall_s": encode_meta.get("source_wall_s"),
                        "user_codes_cache_hit": encode_meta.get("cache_hit"),
                        "execution_backend": args.execution_backend,
                        "execution_profile": execution_profile,
                        "execution_profile_sha256": execution_profile_sha256,
                        "condition_batch_size": len(batch),
                        "condition_batch_size_requested": int(args.condition_batch_size),
                        "shared_random_sampling": len(batch) > 1,
                        "encode_chunk_frames": int(args.encode_chunk_frames),
                        "decode_chunk_frames": int(args.decode_chunk_frames),
                        "decode_batch_size": int(item["decode_batch_size"]),
                        "cuda_graph_main": result["graph_main"],
                        "cuda_graph_depth": result["graph_depth"],
                        "cuda_graph_step": result["graph_step"],
                        "cuda_graph_mimi_encode": encode_meta.get("cuda_graph"),
                        "cuda_graph_mimi_decoder": item["cuda_graph_decoder"],
                        "cuda_graph_mimi_decoder_transformer": item["cuda_graph_transformer"],
                        "no_cuda_graph": os.environ.get("NO_CUDA_GRAPH"),
                        "code_version": code_version,
                        "plan_sha256": plan_sha256,
                        "directions_sha256": plan["directions_sha256"],
                        "mimi_sha256": mimi_sha256,
                    },
                )
                completed_count += 1
                log(
                    f"[{completed_count}/{pending_count}] {run['run_id']} 完成："
                    f"等效 {wall_equivalent:.1f} s（"
                    f"{steps_per_s and round(steps_per_s, 2)} 步/秒，"
                    f"批次 {len(batch)}，hook {result['hook_calls']} 次）"
                )
            del result
            gc.collect()
            log(f"批次 {batch_index + 1}/{len(pending_batches)} 完成：{group_id}，总耗时 {group_wall:.1f} s")
    finally:
        for handle in hook_handles:
            handle.remove()


if __name__ == "__main__":
    main()
