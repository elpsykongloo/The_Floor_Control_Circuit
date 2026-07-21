"""Moshi 系（Moshi / PersonaPlex）R1 teacher-forced 复放 + 逐层激活缓存 runner。

在各自模型 .venv 内运行（依赖：torch、moshi、numpy；音频读取优先 soundfile，缺失退 stdlib wave）。
与主仓库解耦：只按附录 C 契约输出 npy 分片 + manifest.json，仓库侧 wp5_ingest 转 zarr。

运行模式：
  --probe-api          加载模型并转储关键属性/签名 → 首跑联调的依据（出错时把输出整体回传）
  --text-mode greedy   **正式默认**（冻结协议，PREREG #7）：音频流 teacher-forced、
                       文本内心独白流逐步贪心自预测（位置 p 的 text_linear logits 贪心出
                       token，作为位置 p+1 的文本输入）。逐步前向，较慢。
  --text-mode sampled  同上但温度采样（--text-temperature，固定 --seed）：AB 核验用。
  --text-mode pad      文本流全 PAD 的快速分块前向：仅限消融复盘，不得用于正式 G1。

时间对齐声明（写入 manifest.extra.execution.time_alignment，预检核验）：
  序列首位插入初始 token，acts[s] 仅观测 ≤ s·τ 的音频（offset=0）。

已知需要首跑确认的适配点（全部经 _first_attr 探测并写入 manifest.extra）：
  文本 PAD token id、初始填充 token id、流顺序（self_first/other_first）、delays 属性、
  文本输出头 text_linear。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect as _inspect
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BLOCK_STEPS = 4096
DEFAULT_MIMI_CHUNK_SECONDS = 0.08
DEFAULT_FORWARD_CHUNK_STEPS = 128
DEFAULT_TEXT_TEMPERATURE = 0.7
DEFAULT_TEXT_TOP_K = 25  # 官方 LMGen 文本采样默认 top_k_text=25

# 与仓库侧 src/floor_circuit/mve/alignment.py 的 RUNNER_TIME_ALIGNMENT 保持逐字一致
# （runner 与仓库包解耦，不能 import；preflight 会对 manifest 交叉核验该声明）。
TIME_ALIGNMENT = {
    "initial_token_position": 0,
    "acts_observed_through_offset_steps": 0,
}


class AdapterError(RuntimeError):
    """本地 moshi 包 API 与预期不符：报错信息包含实测属性，供回传后修正适配。"""


def log(msg: str) -> None:
    print(f"[moshi-runner] {msg}", flush=True)


def _first_attr(obj, names: list[str], kind: str):
    for n in names:
        if hasattr(obj, n) and getattr(obj, n) is not None:
            return n, getattr(obj, n)
    sample = [a for a in dir(obj) if not a.startswith("__")][:120]
    raise AdapterError(f"未找到{kind}（候选 {names}）。对象 {type(obj).__name__} 属性：{sample}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pcm_prefix_digest(path: str | Path, seconds: float) -> dict:
    """对 WAV 前 N 秒原始 PCM 字节做 sha256——读库无关的输入前缀指纹（PREREG #16(d)）。

    与仓库侧 src/floor_circuit/cachelib/audio_digest.py 逐字对齐（跨环境不 import，
    以单元测试保证两实现在同一文件上输出一致）。
    """
    import wave

    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        n_channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        compression = reader.getcomptype()
        if compression != "NONE":
            raise AdapterError(f"{path} 非 PCM WAV（压缩类型 {compression}），无法计算前缀指纹")
        n_frames = min(int(reader.getnframes()), round(float(seconds) * sample_rate))
        hasher = hashlib.sha256()
        hasher.update(
            f"pcm:{sample_rate}:{n_channels}:{sample_width}:{n_frames}".encode("ascii") + b"\0"
        )
        remaining = n_frames
        while remaining > 0:
            data = reader.readframes(min(remaining, 1 << 16))
            if not data:
                raise AdapterError(f"{path} PCM 数据提前结束：仍缺 {remaining} 帧")
            got, tail = divmod(len(data), n_channels * sample_width)
            if tail:
                raise AdapterError(f"{path} PCM 块字节数 {len(data)} 不是整帧")
            hasher.update(data)
            remaining -= got
    return {
        "sha256": hasher.hexdigest(),
        "n_frames": int(n_frames),
        "sample_rate": int(sample_rate),
        "seconds": float(seconds),
    }


def assert_free_disk(path: str | Path, min_free_bytes: int, *, what: str) -> int:
    """输出卷剩余空间护栏：低于下限立即失败，避免长跑中途写穿磁盘。"""
    import shutil

    probe = Path(path)
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    free = int(shutil.disk_usage(probe).free)
    if min_free_bytes > 0 and free < min_free_bytes:
        raise AdapterError(
            f"{what}：磁盘剩余 {free / 1e9:.1f} GB 低于护栏 {min_free_bytes / 1e9:.1f} GB（{probe}）"
        )
    return free


def resolve_code_version(explicit: str | None) -> str:
    """生成 ``runner 最近提交+内容哈希``，并拒绝执行过期计划。"""
    repo_root = Path(__file__).resolve().parents[2]
    entry = Path(sys.argv[0]).resolve()
    sources = [("shared", Path(__file__).resolve())]
    if entry.is_file() and entry != sources[0][1]:
        sources.append(("entry", entry))
    content = hashlib.sha256()
    for label, path in sources:
        content.update(label.encode("ascii") + b"\0")
        content.update(path.read_bytes())
        content.update(b"\0")
    try:
        commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "-1",
                "--format=%H",
                "--",
                *(str(path.relative_to(repo_root)) for _, path in sources),
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    actual = f"{commit[:7]}+runner.{content.hexdigest()}"
    if explicit is not None and explicit != actual:
        planned_digest = explicit.partition("+runner.")[2]
        actual_digest = actual.partition("+runner.")[2]
        if (
            len(planned_digest) != 64
            or len(actual_digest) != 64
            or planned_digest != actual_digest
        ):
            raise AdapterError(
                f"计划代码版本 {explicit} 与当前 runner {actual} 不一致；"
                "请重新生成缓存计划"
            )
        # 未提交代码正式入库后提交号会变化；内容完全相同时继续执行，
        # 新产物仍记录当前提交对应的 actual，避免伪造旧版本。
    return actual


def write_json_atomic(path: str | Path, payload: dict) -> None:
    """同目录写临时文件后原子替换，避免半截 manifest 被误判为完成。"""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_npy_atomic(path: str | Path, array: np.ndarray) -> None:
    """原子写入 NPY 分片，避免异常中断留下可被误摄取的半截文件。"""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_run_outputs(out_dir: Path) -> None:
    """仅清理 runner 自己的旧产物；保留目录中的其他文件。"""
    paths = [out_dir / "manifest.json", out_dir / "text_tokens.npy"]
    for pattern in ("acts_L*_part*.npy", "acts_part*.npy", "mimi_latent_part*.npy", ".*.tmp"):
        paths.extend(out_dir.glob(pattern))
    removed = 0
    for path in paths:
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        log(f"已清理 {removed} 个旧的 runner 产物")


def read_wav_mono(
    path: str | Path,
    expect_sr: int,
    max_seconds: float | None = None,
) -> np.ndarray:
    """读取单声道波形；给出窗口时只从磁盘读取所需前缀。"""
    max_frames = None
    if max_seconds is not None:
        if float(max_seconds) <= 0:
            raise AdapterError(f"音频窗口必须为正，当前为 {max_seconds}")
        max_frames = round(float(max_seconds) * expect_sr)
    try:
        import soundfile as sf

        read_kwargs = {"frames": max_frames} if max_frames is not None else {}
        wav, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
            **read_kwargs,
        )
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    except ImportError:
        import wave

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n_channels = w.getnchannels()
            n = w.getnframes()
            if max_frames is not None:
                n = min(n, max_frames)
            width = w.getsampwidth()
            raw = w.readframes(n)
        if width == 2:
            wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            wav = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise AdapterError(f"不支持的 wav 位宽 {width * 8}，请先用 ffmpeg 转 16-bit PCM") from None
        if n_channels > 1:
            wav = wav.reshape(-1, n_channels).mean(axis=1)
    if sr != expect_sr:
        raise AdapterError(
            f"{path} 采样率 {sr} ≠ {expect_sr}：请用 scripts/wp2_extract_candor.py 产出的 24 kHz 单声道 wav"
        )
    wav = np.asarray(wav, dtype=np.float32)
    if max_frames is not None and int(wav.shape[0]) != max_frames:
        raise AdapterError(
            f"{path} 读取到 {wav.shape[0]} 帧，短于请求窗口 {max_frames} 帧"
        )
    return wav


def _autodetect_weight(root: Path, patterns: list[str], kind: str) -> Path:
    for pat in patterns:
        hits = sorted(root.glob(pat), key=lambda p: p.stat().st_size, reverse=True)
        if hits:
            return hits[0]
    listing = [p.name for p in root.iterdir()][:60]
    raise AdapterError(f"在 {root} 未找到{kind}（尝试 {patterns}）。目录内容：{listing}")


def load_models(args):
    import torch
    from moshi.models import loaders

    device = args.device
    root = Path(args.model_root)
    mimi_path = (
        Path(args.mimi_weight)
        if args.mimi_weight
        else _autodetect_weight(root, ["tokenizer-*.safetensors", "*mimi*.safetensors"], "Mimi 权重")
    )
    lm_path = (
        Path(args.lm_weight)
        if args.lm_weight
        else _autodetect_weight(root, ["model.safetensors", "*.safetensors"], "LM 权重")
    )
    if lm_path == mimi_path:
        raise AdapterError(f"LM 与 Mimi 权重解析到同一文件 {lm_path}，请显式传 --lm-weight/--mimi-weight")
    log(f"Mimi 权重：{mimi_path}")
    log(f"LM 权重：{lm_path}")
    mimi = loaders.get_mimi(str(mimi_path), device=device)
    mimi.set_num_codebooks(args.n_codebooks)
    lm = loaders.get_moshi_lm(str(lm_path), device=device)
    lm.eval()
    torch.set_grad_enabled(False)
    return mimi, lm


def probe_api(args) -> dict:
    import moshi
    import torch
    from moshi.models import loaders

    report: dict = {
        "moshi_version": getattr(moshi, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "loaders_get_mimi_sig": str(_inspect.signature(loaders.get_mimi)),
        "loaders_get_moshi_lm_sig": str(_inspect.signature(loaders.get_moshi_lm)),
    }
    if args.model_root:
        mimi, lm = load_models(args)
        report["mimi"] = {
            "class": type(mimi).__name__,
            "frame_rate": float(getattr(mimi, "frame_rate", -1)),
            "sample_rate": int(getattr(mimi, "sample_rate", -1)),
            "attrs": sorted(
                a for a in dir(mimi) if "encode" in a.lower() or "latent" in a.lower()
            ),
        }
        interesting = [
            "n_q", "dep_q", "card", "text_card", "delays", "audio_offset",
            "zero_token_id", "text_padding_token_id", "existing_text_padding_id",
            "initial_token_id", "ungenerated_token_id", "num_codebooks",
        ]
        report["lm"] = {
            "class": type(lm).__name__,
            "forward_sig": str(_inspect.signature(lm.forward)),
            "attrs": {k: _jsonable(getattr(lm, k)) for k in interesting if hasattr(lm, k)},
            "has_text_linear": callable(getattr(lm, "text_linear", None)),
            "has_transformer_layers": hasattr(getattr(lm, "transformer", None), "layers"),
            "n_layers": len(lm.transformer.layers) if hasattr(getattr(lm, "transformer", None), "layers") else None,
            "layer_class": type(lm.transformer.layers[0]).__name__
            if hasattr(getattr(lm, "transformer", None), "layers")
            else None,
        }
    return report


def _jsonable(v):
    if isinstance(v, list | tuple):
        return list(v)
    if isinstance(v, int | float | str | bool) or v is None:
        return v
    return repr(v)


def _stream_geometry(mimi, wav: np.ndarray, chunk_seconds: float) -> tuple[int, int, int]:
    """返回每块码帧数、每块采样点数和有效总码帧数。"""
    if wav.size <= 0:
        raise AdapterError("输入音频为空")
    frame_size = int(getattr(mimi, "frame_size", 0))
    frame_rate = float(getattr(mimi, "frame_rate", 0.0))
    if frame_rate <= 0:
        raise AdapterError(
            f"Mimi 缺少有效 frame_rate：{frame_rate}"
        )
    if frame_size <= 0:
        sample_rate = float(getattr(mimi, "sample_rate", 0.0))
        exact_frame_size = sample_rate / frame_rate
        frame_size = round(exact_frame_size)
        if (
            sample_rate <= 0
            or frame_size <= 0
            or abs(exact_frame_size - frame_size) > 1e-6
        ):
            raise AdapterError(
                "Mimi 未暴露 frame_size，且无法从 sample_rate/frame_rate "
                f"精确推导：{sample_rate}/{frame_rate}={exact_frame_size}"
            )
    exact_steps = float(chunk_seconds) * frame_rate
    chunk_steps = round(exact_steps)
    if chunk_seconds <= 0 or chunk_steps <= 0 or abs(exact_steps - chunk_steps) > 1e-6:
        raise AdapterError(
            f"--stream-chunk-seconds={chunk_seconds} 必须对应整数个 Mimi 帧"
            f"（frame_rate={frame_rate}）"
        )
    chunk_samples = chunk_steps * frame_size
    valid_steps = math.ceil(int(wav.size) / frame_size)
    return chunk_steps, chunk_samples, valid_steps


def _iter_padded_wav_chunks(wav: np.ndarray, chunk_samples: int):
    """生成固定长度音频块；末块右侧补零，供 Mimi 流式图复用。"""
    for offset in range(0, int(wav.size), chunk_samples):
        chunk = np.asarray(wav[offset : offset + chunk_samples], dtype=np.float32)
        if chunk.size < chunk_samples:
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[: chunk.size] = chunk
            chunk = padded
        yield chunk


def _unquantized_encoder(mimi):
    """取得量化前编码入口，拒绝把量化后潜表征伪装成连续基线。"""
    fn = getattr(mimi, "encode_to_latent", None)
    if callable(fn) and "quantize" in _inspect.signature(fn).parameters:
        return lambda x: fn(x, quantize=False), "encode_to_latent(quantize=False)"
    fn = getattr(mimi, "_encode_to_unquantized_latent", None)
    if callable(fn):
        return fn, "_encode_to_unquantized_latent"
    raise AdapterError("当前 Mimi 没有可验证的量化前连续潜表征接口")


def encode_mimi_stream(
    mimi,
    wav: np.ndarray,
    device: str,
    chunk_seconds: float,
    *,
    return_latent: bool,
    use_cuda_graph: bool = False,
):
    """单次有状态流式编码，同时产生离散码与可选的量化前连续潜表征。"""
    import torch

    chunk_steps, chunk_samples, valid_steps = _stream_geometry(
        mimi, wav, chunk_seconds
    )
    streaming = getattr(mimi, "streaming", None)
    if not callable(streaming):
        raise AdapterError("当前 Mimi 不支持 streaming()，拒绝退回长音频整段编码")
    quantizer = getattr(mimi, "quantizer", None)
    latent_fn = None
    latent_source = None
    if return_latent:
        latent_fn, latent_source = _unquantized_encoder(mimi)
        if not callable(getattr(quantizer, "encode", None)):
            raise AdapterError("Mimi quantizer 缺少 encode()，无法从连续潜表征生成离散码")
    graphed_step = None
    if use_cuda_graph:
        if torch.device(device).type != "cuda":
            raise AdapterError("Mimi CUDA Graph 只支持 CUDA 设备")
        from moshi.utils.compile import CUDAGraphed

        if latent_fn is None:
            graphed_step = CUDAGraphed(mimi.encode, warmup_steps=1, disable=False)
        else:

            def encode_with_latent(values):
                latent_values = latent_fn(values)
                return quantizer.encode(latent_values), latent_values

            graphed_step = CUDAGraphed(encode_with_latent, warmup_steps=1, disable=False)
    codes_out = None
    latent_out_device = None
    written_steps = 0
    padded_samples = math.ceil(int(wav.size) / chunk_samples) * chunk_samples
    wav_cpu = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    wav_device = torch.zeros(padded_samples, dtype=torch.float32, device=device)
    wav_device[: int(wav.size)].copy_(wav_cpu)
    graph_released = False
    try:
        with torch.no_grad(), streaming(batch_size=1):
            for offset in range(0, padded_samples, chunk_samples):
                x = wav_device[offset : offset + chunk_samples].reshape(1, 1, -1)
                if latent_fn is None:
                    codes = graphed_step(x) if graphed_step is not None else mimi.encode(x)
                elif graphed_step is not None:
                    codes, latent = graphed_step(x)
                else:
                    latent = latent_fn(x)
                    codes = quantizer.encode(latent)
                if latent_fn is not None and int(latent.shape[-1]) != chunk_steps:
                    raise AdapterError(
                        f"Mimi 连续潜表征帧数 {latent.shape[-1]} ≠ 预期 {chunk_steps}"
                    )
                if int(codes.shape[-1]) != chunk_steps:
                    raise AdapterError(
                        f"Mimi 流式码帧数 {codes.shape[-1]} ≠ 预期 {chunk_steps}"
                    )
                take = min(chunk_steps, valid_steps - written_steps)
                if take <= 0:
                    break
                if codes_out is None:
                    codes_out = torch.empty(
                        (int(codes.shape[0]), int(codes.shape[1]), valid_steps),
                        dtype=codes.dtype,
                        device=codes.device,
                    )
                codes_out[..., written_steps : written_steps + take] = codes[..., :take]
                if latent_fn is not None:
                    latent_chunk = latent[0, :, :take].transpose(0, 1)
                    if latent_out_device is None:
                        latent_out_device = torch.empty(
                            (valid_steps, int(latent_chunk.shape[1])),
                            dtype=torch.float16,
                            device=latent.device,
                        )
                    latent_out_device[written_steps : written_steps + take].copy_(latent_chunk)
                written_steps += take
            if graphed_step is not None:
                if getattr(graphed_step, "_graph", None) is None:
                    raise AdapterError("Mimi CUDA Graph 未成功建立")
                torch.cuda.synchronize(torch.device(device))
                graphed_step.reset()
                gc.collect()
                torch.cuda.synchronize(torch.device(device))
                graph_released = True
    finally:
        if graphed_step is not None and not graph_released:
            try:
                torch.cuda.synchronize(torch.device(device))
                graphed_step.reset()
                gc.collect()
                torch.cuda.synchronize(torch.device(device))
            except Exception:
                pass
    if codes_out is None:
        raise AdapterError("Mimi 流式编码没有产生任何码")
    if written_steps != valid_steps:
        raise AdapterError(
            f"Mimi 流式编码仅写出 {written_steps}/{valid_steps} 帧"
        )
    if return_latent and latent_out_device is None:
        raise AdapterError("Mimi 流式编码没有产生连续潜表征")
    latent_out = (
        latent_out_device.cpu().numpy()
        if latent_out_device is not None
        else None
    )
    return codes_out, latent_out, latent_source


def build_parallel_codes(lm, codes_agent, codes_other, args) -> tuple:
    """构造 [1, 1+2*K, T] 的原始 teacher-forced 输入；延迟稍后全局施加一次。"""
    import torch

    _, pad_id = _first_attr(
        lm, ["text_padding_token_id", "existing_text_padding_id"], "文本 PAD token id"
    )
    _, delays = _first_attr(lm, ["delays"], "流延迟表 delays")
    delays = list(delays)
    k_audio = codes_agent.shape[1]
    T = int(min(codes_agent.shape[2], codes_other.shape[2]))
    streams = [None] + (
        [("agent", q) for q in range(k_audio)] + [("other", q) for q in range(k_audio)]
        if args.stream_order == "self_first"
        else [("other", q) for q in range(k_audio)] + [("agent", q) for q in range(k_audio)]
    )
    n_streams = len(streams)
    if len(delays) != n_streams:
        raise AdapterError(f"delays 长度 {len(delays)} 与流数 {n_streams} 不一致")
    codes = torch.empty((1, n_streams, T), dtype=torch.long, device=codes_agent.device)
    codes[0, 0, :] = int(pad_id)
    src_map = {"agent": codes_agent, "other": codes_other}
    for k in range(1, n_streams):
        which, q = streams[k]
        codes[0, k, :] = src_map[which][0, q, :T]
    meta = {
        "text_pad_id": int(pad_id),
        "delays": [int(d) for d in delays[:n_streams]],
        "delay_application": "global_once_before_streaming_forward",
        "stream_order": args.stream_order,
        "n_streams": n_streams,
        "T": T,
    }
    return codes, meta


def prepare_teacher_forced_input(lm, codes):
    """逐句复现 ``LMModel.forward`` 的延迟与首帧插入，供流式 backbone 使用。"""
    import torch

    get_initial = getattr(lm, "_get_initial_token", None)
    if not callable(get_initial):
        raise AdapterError("lm 缺少 _get_initial_token()，无法保证分块前向的时间语义")
    initial = get_initial().to(codes.device)
    batch, n_streams, n_steps = codes.shape
    if initial.shape[1] != n_streams:
        raise AdapterError(
            f"初始 token 流数 {initial.shape[1]} 与输入流数 {n_streams} 不一致"
        )
    initial = initial.expand(batch, -1, -1)
    delays = [int(value) for value in getattr(lm, "delays", [])]
    if len(delays) != n_streams:
        raise AdapterError(f"delays 长度 {len(delays)} 与流数 {n_streams} 不一致")
    delayed = torch.empty_like(codes)
    for stream, delay in enumerate(delays):
        if delay < 0:
            raise AdapterError(f"delays[{stream}]={delay} 小于 0")
        if delay == 0:
            delayed[:, stream] = codes[:, stream]
        elif delay >= n_steps:
            delayed[:, stream] = initial[:, stream, :1]
        else:
            delayed[:, stream, :delay] = initial[:, stream, :1]
            delayed[:, stream, delay:] = codes[:, stream, : n_steps - delay]
    return torch.cat([initial, delayed], dim=2)[:, :, :n_steps]


def forward_backbone(lm, sequence):
    """复现 ``LMModel.forward_text`` 的 embedding→transformer 主干，跳过无关 logits。"""
    n_audio = int(getattr(lm, "num_audio_codebooks", 0))
    audio_offset = int(getattr(lm, "audio_offset", 1))
    embeddings = getattr(lm, "emb", None)
    text_embedding = getattr(lm, "text_emb", None)
    if n_audio <= 0 or embeddings is None or not callable(text_embedding):
        raise AdapterError("lm 缺少 forward_text 主干所需的 embedding 属性")
    if getattr(lm, "fuser", None) is not None:
        raise AdapterError("当前模型带条件 fuser，尚未验证无条件流式主干语义")
    hidden = None
    for codebook in range(n_audio):
        value = embeddings[codebook](sequence[:, codebook + audio_offset])
        hidden = value if hidden is None else hidden + value
    text_value = text_embedding(sequence[:, 0])
    hidden = text_value if hidden is None else hidden + text_value
    hidden = lm.transformer(hidden)
    if getattr(lm, "out_norm", None):
        hidden = lm.out_norm(hidden)
    return hidden


def forward_capture(
    lm,
    codes,
    layers: list[int],
    out_dir: Path,
    chunk_steps: int,
) -> dict:
    """保持 transformer 状态的有界分块前向，并逐块原子写出残差流。"""
    import torch

    if chunk_steps <= 0:
        raise AdapterError(f"chunk_steps 必须大于 0，当前为 {chunk_steps}")
    tf = lm.transformer
    if not hasattr(tf, "layers"):
        raise AdapterError(f"lm.transformer 无 .layers；类型 {type(tf).__name__}")
    streaming = getattr(tf, "streaming", None)
    if not callable(streaming):
        raise AdapterError("lm.transformer 不支持 streaming()，拒绝退回整段前向")
    n_layers = len(tf.layers)
    context = getattr(lm, "context", None)
    if context is not None and int(context) > 0 and chunk_steps > int(context):
        raise AdapterError(
            f"forward 分块 {chunk_steps} 超过 transformer context={int(context)}"
        )
    bad = [ell for ell in layers if ell < 0 or ell >= n_layers]
    if bad:
        raise AdapterError(f"层号越界 {bad}（共 {n_layers} 层，0 起）")
    captured: dict[int, list] = {ell: [] for ell in layers}
    handles = []

    def mk_hook(ell):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[ell].append(h.detach()[0].to(torch.float16).cpu())

        return hook

    for ell in layers:
        handles.append(tf.layers[ell].register_forward_hook(mk_hook(ell)))
    t0 = time.time()
    model_input = prepare_teacher_forced_input(lm, codes)
    n_steps = int(model_input.shape[2])
    n_parts = math.ceil(n_steps / chunk_steps)
    hidden_dim = None
    try:
        with torch.no_grad(), streaming(batch_size=int(model_input.shape[0])):
            for part_index, start in enumerate(range(0, n_steps, chunk_steps)):
                stop = min(start + chunk_steps, n_steps)
                for values in captured.values():
                    values.clear()
                result = forward_backbone(lm, model_input[:, :, start:stop])
                del result
                for ell in layers:
                    if not captured[ell]:
                        raise AdapterError(
                            f"层 {ell} 在分块 {part_index} 未捕获输出"
                        )
                    acts = torch.cat(captured[ell], dim=0).numpy()
                    if acts.shape[0] != stop - start:
                        raise AdapterError(
                            f"层 {ell} 分块 {part_index} 帧数 {acts.shape[0]}"
                            f" ≠ {stop - start}"
                        )
                    current_dim = int(acts.shape[1])
                    if hidden_dim is None:
                        hidden_dim = current_dim
                    elif hidden_dim != current_dim:
                        raise AdapterError(
                            f"隐藏维度在分块间变化：{hidden_dim} → {current_dim}"
                        )
                    write_npy_atomic(
                        out_dir / f"acts_L{ell}_part{part_index:05d}.npy",
                        acts,
                    )
                log(
                    f"前向分块 {part_index + 1}/{n_parts} 完成"
                    f"（帧 {start}:{stop}）"
                )
    finally:
        for hd in handles:
            hd.remove()
    if hidden_dim is None:
        raise AdapterError("流式前向没有捕获任何隐藏状态")
    log(f"流式前向完成：T={n_steps}，耗时 {time.time() - t0:.1f}s")
    return {
        "hidden_dim": hidden_dim,
        "n_steps": n_steps,
        "n_parts": n_parts,
        "forward_chunk_steps": chunk_steps,
        "transformer_context": int(context) if context is not None else None,
    }


class _ActivationDoubleBufferWriter:
    """以双缓冲异步回传激活，并由单线程保持分片写出顺序。

    layout=per_layer_v1：每层每分片一个 ``acts_L{ell}_part*.npy``（MVE 历史契约）。
    layout=stacked_tlh_v2：每分片一个 ``acts_part*.npy``，形状 [rows, L, H]，层轴顺序
    与 ``layers`` 列表一致（PREREG #16(d)，机械盘小文件治理）。
    """

    def __init__(
        self,
        torch,
        layers: list[int],
        chunk_steps: int,
        hidden_dim: int,
        device,
        out_dir: Path,
        layout: str = "per_layer_v1",
    ):
        if layout not in ("per_layer_v1", "stacked_tlh_v2"):
            raise AdapterError(f"未知激活布局：{layout!r}")
        self.torch = torch
        self.layers = layers
        self.chunk_steps = chunk_steps
        self.out_dir = out_dir
        self.layout = layout
        if layout == "per_layer_v1":
            self.gpu_buffers = [
                {
                    ell: torch.empty(
                        (chunk_steps, hidden_dim),
                        dtype=torch.float16,
                        device=device,
                    )
                    for ell in layers
                }
                for _ in range(2)
            ]
            self.cpu_buffers = [
                {
                    ell: torch.empty(
                        (chunk_steps, hidden_dim),
                        dtype=torch.float16,
                        device="cpu",
                        pin_memory=True,
                    )
                    for ell in layers
                }
                for _ in range(2)
            ]
        else:
            # 时间主序连续布局：逐步合并 32 层，整块只需一次异步 D2H。
            self.gpu_buffers = [
                torch.empty(
                    (chunk_steps, len(layers), hidden_dim),
                    dtype=torch.float16,
                    device=device,
                )
                for _ in range(2)
            ]
            self.cpu_buffers = [
                torch.empty(
                    (chunk_steps, len(layers), hidden_dim),
                    dtype=torch.float16,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="moshi-activation-writer",
        )
        self.pending: list[Future | None] = [None, None]

    def write_row(self, slot: int, row: int, captured: dict[int, object]) -> None:
        """把当前时钟步的选层激活写入设备缓冲。"""
        if self.layout == "per_layer_v1":
            for ell in self.layers:
                self.gpu_buffers[slot][ell][row].copy_(captured[ell][-1])
            return
        self.torch.stack(
            tuple(captured[ell][-1] for ell in self.layers),
            dim=0,
            out=self.gpu_buffers[slot][row],
        )

    def wait_slot(self, slot: int) -> None:
        future = self.pending[slot]
        if future is not None:
            future.result()
            self.pending[slot] = None

    def submit(self, slot: int, rows: int, part_index: int) -> None:
        if rows <= 0 or rows > self.chunk_steps:
            raise AdapterError(f"激活异步写出行数非法：{rows}")
        self.wait_slot(slot)
        current_stream = self.torch.cuda.current_stream()
        with self.torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_stream(current_stream)
            if self.layout == "per_layer_v1":
                for ell in self.layers:
                    dst = self.cpu_buffers[slot][ell][:rows]
                    dst.copy_(
                        self.gpu_buffers[slot][ell][:rows],
                        non_blocking=True,
                    )
            else:
                self.cpu_buffers[slot][:rows].copy_(
                    self.gpu_buffers[slot][:rows],
                    non_blocking=True,
                )
            ready = self.torch.cuda.Event()
            ready.record(self.transfer_stream)
        self.pending[slot] = self.executor.submit(
            self._write_after_transfer,
            ready,
            slot,
            rows,
            part_index,
        )

    def _write_after_transfer(
        self,
        ready,
        slot: int,
        rows: int,
        part_index: int,
    ) -> None:
        ready.synchronize()
        if self.layout == "per_layer_v1":
            for ell in self.layers:
                values = self.cpu_buffers[slot][ell][:rows].numpy()
                write_npy_atomic(
                    self.out_dir / f"acts_L{ell}_part{part_index:05d}.npy",
                    values,
                )
        else:
            write_npy_atomic(
                self.out_dir / f"acts_part{part_index:05d}.npy",
                self.cpu_buffers[slot][:rows].numpy(),
            )

    def close(self) -> None:
        try:
            for slot in range(2):
                self.wait_slot(slot)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)


def _forward_capture_selftext_cuda(
    lm,
    codes,
    layers: list[int],
    out_dir: Path,
    chunk_steps: int,
    *,
    layout: str = "per_layer_v1",
) -> dict:
    """CUDA 上保持单步贪心语义，并消除逐步主机同步。"""
    import torch
    from moshi.utils.compile import CUDAGraphed

    tf = lm.transformer
    if not hasattr(tf, "layers"):
        raise AdapterError(f"lm.transformer 无 .layers；类型 {type(tf).__name__}")
    streaming = getattr(tf, "streaming", None)
    if not callable(streaming):
        raise AdapterError("lm.transformer 不支持 streaming()，拒绝退回整段前向")
    n_layers = len(tf.layers)
    bad = [ell for ell in layers if ell < 0 or ell >= n_layers]
    if bad:
        raise AdapterError(f"层号越界 {bad}（共 {n_layers} 层，0 起）")
    context = getattr(lm, "context", None)
    head_name, text_head = _first_attr(
        lm,
        ["text_linear"],
        "文本输出头 text_linear",
    )
    if not callable(text_head):
        raise AdapterError(f"lm.{head_name} 不可调用，无法产生文本 logits")
    _, pad_id = _first_attr(
        lm,
        ["text_padding_token_id", "existing_text_padding_id"],
        "文本 PAD token id",
    )

    captured: dict[int, object] = {}
    handles = []

    def mk_hook(ell):
        def hook(_mod, _inp, out):
            value = out[0] if isinstance(out, tuple) else out
            captured[ell] = value.detach()[0].to(torch.float16)

        return hook

    for ell in layers:
        handles.append(tf.layers[ell].register_forward_hook(mk_hook(ell)))

    model_input = prepare_teacher_forced_input(lm, codes)
    if int(model_input.shape[0]) != 1:
        raise AdapterError("自预测文本流仅支持 batch=1（文本反馈按单序列实现）")
    n_steps = int(model_input.shape[2])
    if n_steps < 2:
        raise AdapterError("CUDA Graph 贪心前向至少需要 2 个时钟步")
    text_tokens_device = torch.empty(
        n_steps,
        dtype=torch.long,
        device=model_input.device,
    )
    step_input = torch.empty_like(model_input[:, :, :1])
    writer = None
    hidden_dim = None
    part_index = 0
    active_slot = 0
    row_in_part = 0
    prev_token = None
    device_index = (
        model_input.device.index
        if model_input.device.index is not None
        else torch.cuda.current_device()
    )
    t0 = time.time()

    def graph_step(values):
        hidden = forward_backbone(lm, values)
        logits = text_head(hidden)[0, -1].float()
        return torch.argmax(logits)

    graphed_step = CUDAGraphed(graph_step, warmup_steps=1, disable=False)
    graph_released = False
    try:
        with torch.no_grad(), streaming(batch_size=1):
            for p in range(n_steps):
                step_input.copy_(model_input[:, :, p : p + 1])
                if prev_token is not None:
                    step_input[:, 0, 0].copy_(prev_token.reshape(1))
                token = graphed_step(step_input)
                if p == 1 and getattr(graphed_step, "_graph", None) is None:
                    raise AdapterError("单步 CUDA Graph 未成功建立")
                text_tokens_device[p].copy_(token)
                prev_token = token

                if set(captured) != set(layers):
                    missing = sorted(set(layers) - set(captured))
                    raise AdapterError(f"CUDA Graph 前向未捕获层：{missing}")
                if writer is None:
                    hidden_dim = int(captured[layers[0]].shape[-1])
                    for ell in layers:
                        if int(captured[ell].shape[-1]) != hidden_dim:
                            raise AdapterError("选定层的隐藏维度不一致")
                    writer = _ActivationDoubleBufferWriter(
                        torch,
                        layers,
                        chunk_steps,
                        hidden_dim,
                        model_input.device,
                        out_dir,
                        layout=layout,
                    )
                    writer.wait_slot(active_slot)

                writer.write_row(active_slot, row_in_part, captured)
                row_in_part += 1
                is_full = row_in_part == chunk_steps
                is_last = p + 1 == n_steps
                if is_full or is_last:
                    writer.submit(active_slot, row_in_part, part_index)
                    part_index += 1
                    row_in_part = 0
                    active_slot = 1 - active_slot
                    if not is_last:
                        writer.wait_slot(active_slot)
                    if part_index % 10 == 0:
                        rate = (p + 1) / max(time.time() - t0, 1e-9)
                        log(f"自预测前向 {p + 1}/{n_steps} 步（{rate:.1f} 步/s）")
            # 图引用流式 KV 状态；须在 streaming 上下文释放状态前销毁，
            # 否则下一角色重新捕获时可能触发 stream capture invalidated。
            torch.cuda.synchronize(model_input.device)
            captured.clear()
            graphed_step.reset()
            gc.collect()
            torch.cuda.synchronize(model_input.device)
            graph_released = True
        if writer is None or hidden_dim is None:
            raise AdapterError("CUDA Graph 前向没有产生激活")
        writer.close()
        writer = None
    finally:
        if not graph_released:
            try:
                torch.cuda.synchronize(model_input.device)
                captured.clear()
                graphed_step.reset()
                gc.collect()
                torch.cuda.synchronize(model_input.device)
            except Exception:
                # 保留原始前向异常；异常路径由进程退出并从原子分片断点恢复。
                pass
        if writer is not None:
            writer.close()
        for handle in handles:
            handle.remove()

    text_tokens = text_tokens_device.cpu().numpy()
    write_npy_atomic(out_dir / "text_tokens.npy", text_tokens)
    pad_fraction = float(np.mean(text_tokens == int(pad_id)))
    elapsed = time.time() - t0
    log(
        f"自预测前向完成：T={n_steps}，PAD 占比 {pad_fraction:.3f}，"
        f"耗时 {elapsed:.1f}s"
    )
    return {
        "hidden_dim": hidden_dim,
        "n_steps": n_steps,
        "n_parts": part_index,
        "forward_chunk_steps": chunk_steps,
        "transformer_context": int(context) if context is not None else None,
        "text_head_source": head_name,
        "text_token_pad_fraction": pad_fraction,
        "activation_device_double_buffered": True,
        "greedy_token_device_resident": True,
        "cuda_graph": True,
        "activation_layout": layout,
        "telemetry": {
            "forward_elapsed_s": round(elapsed, 3),
            "steps_per_second": round(n_steps / max(elapsed, 1e-9), 3),
            "device_index": int(device_index),
        },
    }


def forward_capture_selftext(
    lm,
    codes,
    layers: list[int],
    out_dir: Path,
    chunk_steps: int,
    mode: str,
    temperature: float | None,
    seed: int,
    top_k: int = DEFAULT_TEXT_TOP_K,
    *,
    layout: str = "per_layer_v1",
) -> dict:
    """逐步自预测文本流的 teacher-forced 前向（PREREG #7 冻结协议）。

    音频码本按官方延迟表 teacher-forced；文本流与官方 LMGen 同语义：位置 p 的
    主干输出经 text_linear 得到文本 logits，贪心 argmax（或温度采样）出的 token
    作为位置 p+1 的文本输入，位置 0 输入初始 token。逐步前向保持 transformer
    流式状态；每累计 chunk_steps 行原子写出一个 npy 分片（与 pad 路径同布局）。
    """
    import torch

    if chunk_steps <= 0:
        raise AdapterError(f"chunk_steps 必须大于 0，当前为 {chunk_steps}")
    if mode not in ("greedy", "sampled"):
        raise AdapterError(f"未知自预测文本模式 {mode!r}")
    if mode == "greedy" and codes.device.type == "cuda":
        return _forward_capture_selftext_cuda(
            lm,
            codes,
            layers,
            out_dir,
            chunk_steps,
            layout=layout,
        )
    if layout != "per_layer_v1":
        raise AdapterError(
            f"激活布局 {layout!r} 仅在 CUDA greedy 路径实现；当前 mode={mode!r}，"
            f"device={codes.device.type!r}"
        )
    tf = lm.transformer
    if not hasattr(tf, "layers"):
        raise AdapterError(f"lm.transformer 无 .layers；类型 {type(tf).__name__}")
    streaming = getattr(tf, "streaming", None)
    if not callable(streaming):
        raise AdapterError("lm.transformer 不支持 streaming()，拒绝退回整段前向")
    n_layers = len(tf.layers)
    context = getattr(lm, "context", None)
    bad = [ell for ell in layers if ell < 0 or ell >= n_layers]
    if bad:
        raise AdapterError(f"层号越界 {bad}（共 {n_layers} 层，0 起）")
    head_name, text_head = _first_attr(lm, ["text_linear"], "文本输出头 text_linear")
    if not callable(text_head):
        raise AdapterError(f"lm.{head_name} 不可调用，无法产生文本 logits")
    _, pad_id = _first_attr(
        lm, ["text_padding_token_id", "existing_text_padding_id"], "文本 PAD token id"
    )
    generator = None
    if mode == "sampled":
        if temperature is None or float(temperature) <= 0:
            raise AdapterError("sampled 文本模式需要正的 --text-temperature")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    captured: dict[int, list] = {ell: [] for ell in layers}
    handles = []

    def mk_hook(ell):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[ell].append(h.detach()[0].to(torch.float16).cpu())

        return hook

    for ell in layers:
        handles.append(tf.layers[ell].register_forward_hook(mk_hook(ell)))

    t0 = time.time()
    model_input = prepare_teacher_forced_input(lm, codes)
    if int(model_input.shape[0]) != 1:
        raise AdapterError("自预测文本流仅支持 batch=1（文本反馈按单序列实现）")
    n_steps = int(model_input.shape[2])
    buffers: dict[int, list] = {ell: [] for ell in layers}
    text_tokens = np.empty(n_steps, dtype=np.int64)
    hidden_dim = None
    part_index = 0
    written = 0

    def flush(part_rows: int) -> None:
        nonlocal part_index, written
        for ell in layers:
            acts = torch.cat(buffers[ell], dim=0).numpy()
            if acts.shape[0] != part_rows:
                raise AdapterError(
                    f"层 {ell} 分片 {part_index} 帧数 {acts.shape[0]} ≠ {part_rows}"
                )
            write_npy_atomic(out_dir / f"acts_L{ell}_part{part_index:05d}.npy", acts)
            buffers[ell].clear()
        part_index += 1
        written += part_rows

    prev_token: int | None = None
    try:
        with torch.no_grad(), streaming(batch_size=int(model_input.shape[0])):
            for p in range(n_steps):
                step_input = model_input[:, :, p : p + 1].clone()
                if p > 0:
                    step_input[:, 0, 0] = int(prev_token)
                for values in captured.values():
                    values.clear()
                hidden = forward_backbone(lm, step_input)
                logits = text_head(hidden)[0, -1].float()
                if mode == "greedy":
                    token = int(torch.argmax(logits).item())
                else:
                    # 与官方 LMGen 文本采样同口径：top-k 截断后温度采样
                    k = min(int(top_k), int(logits.shape[-1])) if top_k else int(logits.shape[-1])
                    top_values, top_indices = torch.topk(logits, k)
                    probs = torch.softmax(top_values / float(temperature), dim=-1)
                    pick = int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
                    token = int(top_indices[pick].item())
                text_tokens[p] = token
                prev_token = token
                del hidden, logits
                for ell in layers:
                    if not captured[ell]:
                        raise AdapterError(f"层 {ell} 在步 {p} 未捕获输出")
                    h = torch.cat(captured[ell], dim=0)
                    if h.shape[0] != 1:
                        raise AdapterError(f"层 {ell} 步 {p} 捕获 {h.shape[0]} 行，期望 1")
                    current_dim = int(h.shape[1])
                    if hidden_dim is None:
                        hidden_dim = current_dim
                    elif hidden_dim != current_dim:
                        raise AdapterError(f"隐藏维度在步间变化：{hidden_dim} → {current_dim}")
                    buffers[ell].append(h)
                if (p + 1) % chunk_steps == 0:
                    flush(chunk_steps)
                    if part_index % 10 == 0:
                        rate = (p + 1) / max(time.time() - t0, 1e-9)
                        log(f"自预测前向 {p + 1}/{n_steps} 步（{rate:.1f} 步/s）")
            if n_steps % chunk_steps:
                flush(n_steps % chunk_steps)
    finally:
        for hd in handles:
            hd.remove()
    if hidden_dim is None:
        raise AdapterError("自预测前向没有捕获任何隐藏状态")
    if written != n_steps:
        raise AdapterError(f"自预测前向仅写出 {written}/{n_steps} 步")
    write_npy_atomic(out_dir / "text_tokens.npy", text_tokens)
    pad_fraction = float(np.mean(text_tokens == int(pad_id))) if n_steps else 0.0
    log(
        f"自预测前向完成：T={n_steps}，PAD 占比 {pad_fraction:.3f}，"
        f"耗时 {time.time() - t0:.1f}s"
    )
    return {
        "hidden_dim": hidden_dim,
        "n_steps": n_steps,
        "n_parts": part_index,
        "forward_chunk_steps": chunk_steps,
        "transformer_context": int(context) if context is not None else None,
        "text_head_source": head_name,
        "text_token_pad_fraction": pad_fraction,
        "activation_device_double_buffered": False,
        "greedy_token_device_resident": False,
        "cuda_graph": False,
    }


def _assert_streaming_idle(mimi, lm) -> None:
    """每个声道和角色结束后确认流式状态已经显式释放。"""
    active = []
    if bool(getattr(mimi, "is_streaming", False)):
        active.append("mimi")
    transformer = getattr(lm, "transformer", None)
    if bool(getattr(transformer, "is_streaming", False)):
        active.append("transformer")
    if active:
        raise AdapterError(f"流式状态未释放：{active}")


def _write_encoded_run(
    args,
    mimi,
    lm,
    codes_agent,
    codes_other,
    latent: np.ndarray | None,
    latent_source: str | None,
    code_version: str,
    source_audio: dict[str, str],
    engineering: dict,
) -> None:
    """写出一个角色缓存；模型与 Mimi 编码结果可由持久进程复用。"""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_run_outputs(out_dir)
    log(f"文本流模式：{args.text_mode}")
    plan_v2 = getattr(args, "plan_v2", None)
    if plan_v2 is not None:
        if args.text_mode != "greedy":
            raise AdapterError("计划 v2 只允许 greedy 文本流（PREREG #7/#16）")
        import torch

        torch.cuda.reset_peak_memory_stats()
    codes, build_meta = build_parallel_codes(lm, codes_agent, codes_other, args)
    if args.text_mode == "pad":
        stats = forward_capture(
            lm, codes, args.layers, out_dir, args.forward_chunk_steps
        )
    else:
        stats = forward_capture_selftext(
            lm,
            codes,
            args.layers,
            out_dir,
            args.forward_chunk_steps,
            args.text_mode,
            args.text_temperature,
            args.seed,
            top_k=args.text_top_k,
            layout=(
                str(plan_v2["activation_layout"]) if plan_v2 is not None else "per_layer_v1"
            ),
        )
    if plan_v2 is not None:
        expected_steps = int(plan_v2["expected_steps"])
        if int(stats["n_steps"]) != expected_steps:
            raise AdapterError(
                f"步数 {stats['n_steps']} ≠ 计划 expected_steps={expected_steps}"
                f"（会话 {args.session_id}）——输入窗口或音频异常，拒绝写出"
            )
        expected_hidden = plan_v2.get("expected_hidden_dim")
        if expected_hidden is not None and int(stats["hidden_dim"]) != int(expected_hidden):
            raise AdapterError(
                f"隐藏维度 {stats['hidden_dim']} ≠ 计划 expected_hidden_dim={expected_hidden}"
            )
        expected_parts = int(plan_v2["expected_parts"])
        if int(stats["n_parts"]) != expected_parts:
            raise AdapterError(
                f"分片数 {stats['n_parts']} ≠ 计划 expected_parts={expected_parts}"
            )
    mimi_latent_ok = latent is not None
    if latent is not None:
        for pi, i0 in enumerate(range(0, latent.shape[0], BLOCK_STEPS)):
            write_npy_atomic(
                out_dir / f"mimi_latent_part{pi:05d}.npy",
                latent[i0 : i0 + BLOCK_STEPS],
            )
    execution = {
        "forward_mode": "streaming_teacher_forced_backbone",
        "text_mode": args.text_mode,
        "mimi_chunk_seconds": float(args.mimi_chunk_seconds),
        "mimi_chunk_frames": round(
            float(args.mimi_chunk_seconds) * float(getattr(mimi, "frame_rate", 0.0))
        ),
        "forward_chunk_steps": stats["forward_chunk_steps"],
        "n_parts": stats["n_parts"],
        "transformer_context": stats["transformer_context"],
        "transformer_state_preserved": True,
        "mimi_state_preserved": True,
        "depformer_skipped": True,
        "max_seconds": float(args.max_seconds) if args.max_seconds else None,
        "latent_kind": (
            "pre_quantization_continuous"
            if latent is not None
            else None
        ),
        "latent_source": latent_source,
        "time_alignment": dict(TIME_ALIGNMENT),
    }
    execution.update(engineering)
    for key in (
        "activation_device_double_buffered",
        "greedy_token_device_resident",
        "cuda_graph",
    ):
        if key in stats:
            execution[key] = bool(stats[key])
    build_meta["execution"] = execution
    if args.text_mode != "pad":
        build_meta["execution"]["text_head_source"] = stats["text_head_source"]
        build_meta["execution"]["text_token_pad_fraction"] = stats["text_token_pad_fraction"]
        if args.text_mode == "sampled":
            build_meta["execution"]["text_temperature"] = float(args.text_temperature)
            build_meta["execution"]["text_top_k"] = int(args.text_top_k)
            build_meta["execution"]["text_seed"] = int(args.seed)
    manifest = {
        "schema_version": 1 if plan_v2 is None else 2,
        "model": args.model_name,
        "mode": "R1",
        "session_id": args.session_id,
        "layers": list(args.layers),
        "hidden_dim": stats["hidden_dim"],
        "clock_hz": float(getattr(mimi, "frame_rate", 12.5)),
        "n_steps": stats["n_steps"],
        "seed": args.seed,
        "temperature": float(args.text_temperature) if args.text_mode == "sampled" else None,
        "text_mode": args.text_mode,
        "source_audio": source_audio,
        "mimi_latent": mimi_latent_ok,
        "code_version": code_version,
        "extra": build_meta,
    }
    if plan_v2 is not None:
        import torch

        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        telemetry = dict(stats.get("telemetry", {}))
        telemetry.update(
            {
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "gpu_name": str(device_props.name),
                "gpu_uuid": str(getattr(device_props, "uuid", "unknown")),
                "gpu_total_memory_bytes": int(device_props.total_memory),
            }
        )
        manifest["extra"]["e1"] = {
            "plan_id": str(plan_v2["plan_id"]),
            "prereg_tag": str(plan_v2.get("prereg_tag", "")),
            "experiment": "E1",
            "cohort": str(plan_v2["cohort"]),
            "agent_channel": int(plan_v2["agent_channel"]),
            "expected_steps": int(plan_v2["expected_steps"]),
            "expected_parts": int(plan_v2["expected_parts"]),
            "activation_layout": str(plan_v2["activation_layout"]),
            "analysis_max_label_step": int(plan_v2["analysis_max_label_step"]),
            "common_window_steps": int(plan_v2["common_window_steps"]),
            "input_prefix": plan_v2["input_prefix"],
            "shard_id": plan_v2.get("shard_id"),
            "telemetry": telemetry,
        }
    manifest["extra"]["output_files"] = {
        path.name: path.stat().st_size for path in sorted(out_dir.glob("*.npy"))
    }
    if plan_v2 is not None:
        manifest["extra"]["e1"]["telemetry"]["output_bytes"] = int(
            sum(manifest["extra"]["output_files"].values())
        )
    write_json_atomic(out_dir / "manifest.json", manifest)
    log(f"完成：{out_dir}（layers={args.layers}，T={stats['n_steps']}，H={stats['hidden_dim']}）")


def run(args) -> None:
    """单路兼容入口；正式长跑使用持久会话批处理入口。"""
    code_version = resolve_code_version(args.code_version)
    mimi, lm = load_models(args)
    sr = int(getattr(mimi, "sample_rate", 24000))
    wav_agent = read_wav_mono(args.audio_agent, sr, args.max_seconds)
    wav_other = read_wav_mono(args.audio_other, sr, args.max_seconds)
    log(f"Mimi 流式编码 agent（块长 {args.mimi_chunk_seconds:g}s）")
    codes_agent, latent, latent_source = encode_mimi_stream(
        mimi,
        wav_agent,
        args.device,
        args.mimi_chunk_seconds,
        return_latent=args.dump_mimi_latent,
    )
    log(f"Mimi 流式编码 other（块长 {args.mimi_chunk_seconds:g}s）")
    codes_other, _, _ = encode_mimi_stream(
        mimi,
        wav_other,
        args.device,
        args.mimi_chunk_seconds,
        return_latent=False,
    )
    _assert_streaming_idle(mimi, lm)
    _write_encoded_run(
        args,
        mimi,
        lm,
        codes_agent,
        codes_other,
        latent,
        latent_source,
        code_version,
        {
            str(args.audio_agent): sha256_file(args.audio_agent),
            str(args.audio_other): sha256_file(args.audio_other),
        },
        {
            "engineering_mode": "single_run_device_buffered_v1",
            "persistent_worker": False,
            "session_pair_reuse": False,
            "mimi_device_buffered": True,
        },
    )
    _assert_streaming_idle(mimi, lm)


def cached_run_is_valid(
    out_dir: str | Path,
    *,
    accepted_code_versions: set[str],
    layers: list[int],
    max_seconds: float,
    mimi_chunk_seconds: float,
    forward_chunk_steps: int,
    text_mode: str,
) -> bool:
    """断点续跑只接受完整且符合当前冻结执行契约的缓存。"""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        execution = payload["extra"]["execution"]
        output_files = payload["extra"]["output_files"]
        if payload.get("code_version") not in accepted_code_versions:
            return False
        if payload.get("text_mode") != text_mode:
            return False
        if payload.get("layers") != layers or payload.get("mimi_latent") is not True:
            return False
        if execution.get("time_alignment") != TIME_ALIGNMENT:
            return False
        if execution.get("latent_kind") != "pre_quantization_continuous":
            return False
        if float(execution.get("max_seconds")) != float(max_seconds):
            return False
        if float(execution.get("mimi_chunk_seconds")) != float(mimi_chunk_seconds):
            return False
        if int(execution.get("forward_chunk_steps")) != int(forward_chunk_steps):
            return False
        if not isinstance(output_files, dict) or not output_files:
            return False
        if len(payload.get("source_audio", {})) != 2:
            return False
        for name, expected_size in output_files.items():
            path = out_dir / name
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _role_args(base_args, session: dict, channel: int, extra: dict | None = None):
    values = vars(base_args).copy()
    other = 1 - channel
    values.update(
        {
            "session_id": str(session["session_id"]),
            "audio_agent": str(session[f"audio_ch{channel}"]),
            "audio_other": str(session[f"audio_ch{other}"]),
            "out": str(session[f"out_agent{channel}"]),
        }
    )
    if extra:
        values.update(extra)
    return SimpleNamespace(**values)


def _planned_audio_hashes(session: dict) -> dict[str, str]:
    """计划 v2 已核验完整摘要；旧计划缺字段时才回退全文件散列。"""
    hashes: dict[str, str] = {}
    for channel in (0, 1):
        path = str(session[f"audio_ch{channel}"])
        planned = session.get(f"audio_sha256_ch{channel}")
        if planned is None:
            hashes[path] = sha256_file(path)
            continue
        digest = str(planned).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise AdapterError(f"会话 {session['session_id']} ch{channel} 的计划摘要格式非法")
        hashes[path] = digest
    return hashes


def run_session_pair(
    base_args,
    session: dict,
    mimi,
    lm,
    code_version: str,
    *,
    run_agent0: bool,
    run_agent1: bool,
    role_extras: tuple[dict | None, dict | None] = (None, None),
    engineering_mode: str = "persistent_session_pair_cuda_graph_v1",
    preloaded_audio: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """一次读取并编码会话双声道，随后依次执行两个角色。"""
    if not run_agent0 and not run_agent1:
        return 0
    sr = int(getattr(mimi, "sample_rate", 24000))
    audio0 = str(session["audio_ch0"])
    audio1 = str(session["audio_ch1"])
    if preloaded_audio is None:
        wav0 = read_wav_mono(audio0, sr, base_args.max_seconds)
        wav1 = read_wav_mono(audio1, sr, base_args.max_seconds)
    else:
        wav0, wav1 = preloaded_audio

    log(
        f"会话 {session['session_id']}：Mimi 编码 ch0/ch1"
        f"（块长 {base_args.mimi_chunk_seconds:g}s）"
    )
    codes0, latent0, latent_source0 = encode_mimi_stream(
        mimi,
        wav0,
        base_args.device,
        base_args.mimi_chunk_seconds,
        return_latent=base_args.dump_mimi_latent,
        use_cuda_graph=bool(getattr(base_args, "mimi_cuda_graph", False)),
    )
    codes1, latent1, latent_source1 = encode_mimi_stream(
        mimi,
        wav1,
        base_args.device,
        base_args.mimi_chunk_seconds,
        return_latent=base_args.dump_mimi_latent,
        use_cuda_graph=bool(getattr(base_args, "mimi_cuda_graph", False)),
    )
    _assert_streaming_idle(mimi, lm)
    audio_hashes = _planned_audio_hashes(session)
    engineering = {
        "engineering_mode": engineering_mode,
        "persistent_worker": True,
        "session_pair_reuse": True,
        "mimi_device_buffered": True,
        "mimi_cuda_graph": bool(getattr(base_args, "mimi_cuda_graph", False)),
        "audio_prefix_only": bool(base_args.max_seconds),
        "audio_prefetched": preloaded_audio is not None,
        "planned_full_audio_digest_reused": all(
            session.get(f"audio_sha256_ch{channel}") is not None for channel in (0, 1)
        ),
    }
    completed = 0
    if run_agent0:
        args0 = _role_args(base_args, session, 0, role_extras[0])
        _write_encoded_run(
            args0,
            mimi,
            lm,
            codes0,
            codes1,
            latent0,
            latent_source0,
            code_version,
            {audio0: audio_hashes[audio0], audio1: audio_hashes[audio1]},
            engineering,
        )
        _assert_streaming_idle(mimi, lm)
        completed += 1
    if run_agent1:
        args1 = _role_args(base_args, session, 1, role_extras[1])
        _write_encoded_run(
            args1,
            mimi,
            lm,
            codes1,
            codes0,
            latent1,
            latent_source1,
            code_version,
            {audio1: audio_hashes[audio1], audio0: audio_hashes[audio0]},
            engineering,
        )
        _assert_streaming_idle(mimi, lm)
        completed += 1
    return completed


def cached_run_is_valid_v2(
    out_dir: str | Path,
    *,
    plan: dict,
    session: dict,
    channel: int,
    accepted_code_versions: set[str],
) -> bool:
    """E1 计划 v2 的断点续跑校验：契约、指纹、分片计数与文件完整性全对才复用。"""
    out_dir = Path(out_dir)
    settings = plan["settings"]
    try:
        payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        execution = payload["extra"]["execution"]
        output_files = payload["extra"]["output_files"]
        e1 = payload["extra"]["e1"]
        if int(payload.get("schema_version", 0)) != 2:
            return False
        if payload.get("code_version") not in accepted_code_versions:
            return False
        if payload.get("text_mode") != "greedy" or payload.get("mimi_latent") is not True:
            return False
        if payload.get("layers") != [int(value) for value in settings["layers"]]:
            return False
        if int(payload.get("n_steps", -1)) != int(settings["expected_steps"]):
            return False
        if execution.get("time_alignment") != TIME_ALIGNMENT:
            return False
        if execution.get("latent_kind") != "pre_quantization_continuous":
            return False
        if float(execution.get("max_seconds")) != float(settings["window_seconds"]):
            return False
        if float(execution.get("mimi_chunk_seconds")) != float(settings["mimi_chunk_seconds"]):
            return False
        if bool(execution.get("mimi_cuda_graph")) != bool(settings["mimi_cuda_graph"]):
            return False
        if int(execution.get("forward_chunk_steps")) != int(settings["forward_chunk_steps"]):
            return False
        if str(e1.get("plan_id")) != str(plan["plan_id"]):
            return False
        if str(e1.get("cohort")) != str(session["cohort"]):
            return False
        if int(e1.get("agent_channel", -1)) != int(channel):
            return False
        if str(e1.get("activation_layout")) != str(settings["activation_layout"]):
            return False
        if int(e1.get("expected_parts", -1)) != int(settings["expected_parts"]):
            return False
        expected_prefix = {
            "ch0": dict(session["prefix_ch0"]),
            "ch1": dict(session["prefix_ch1"]),
        }
        if e1.get("input_prefix") != expected_prefix:
            return False
        if not isinstance(output_files, dict) or not output_files:
            return False
        n_stacked = sum(
            1 for name in output_files if name.startswith("acts_part") and name.endswith(".npy")
        )
        if n_stacked != int(settings["expected_parts"]):
            return False
        for name, expected_size in output_files.items():
            path = out_dir / name
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _prepare_e1_session_audio(
    session: dict,
    sample_rate: int,
    window_seconds: float,
) -> tuple[dict[str, dict], np.ndarray, np.ndarray]:
    """校验输入前缀并只读取计算所需窗口，供后台预取线程调用。"""
    input_prefix: dict[str, dict] = {}
    waves: list[np.ndarray] = []
    for channel in (0, 1):
        path = session[f"audio_ch{channel}"]
        digest = pcm_prefix_digest(path, window_seconds)
        expected = dict(session[f"prefix_ch{channel}"])
        if digest != expected:
            raise AdapterError(
                f"会话 {session['session_id']} ch{channel} 输入前缀指纹与计划不符"
                f"（实测 {digest['sha256'][:12]}… vs 计划 {expected.get('sha256', '')[:12]}…）"
                "——音频文件在计划生成后被改动，拒绝执行"
            )
        input_prefix[f"ch{channel}"] = digest
        waves.append(read_wav_mono(path, sample_rate, window_seconds))
    return input_prefix, waves[0], waves[1]


def run_batch_plan_v2(plan: dict, device: str = "cuda") -> None:
    """E1 缓存计划 v2：单卡持久进程、连续堆叠分片与音频预取。"""
    import torch

    if not torch.cuda.is_available():
        raise AdapterError("计划 v2 需要 CUDA（CUDA Graph 贪心路径）")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise AdapterError(f"计划 v2 只支持 cuda 设备，收到 {device!r}")
    if torch_device.index is None:
        torch_device = torch.device("cuda", 0)
    torch.cuda.set_device(int(torch_device.index))
    settings = dict(plan["settings"])
    resources = dict(plan.get("resources", {}))
    code_version = resolve_code_version(str(plan["code_version"]))
    accepted = {str(value) for value in plan.get("accepted_code_versions", [])}
    accepted.add(code_version)
    if str(settings["text_mode"]) != "greedy":
        raise AdapterError("计划 v2 只允许 greedy 文本流（PREREG #7/#16）")
    if str(settings["activation_layout"]) != "stacked_tlh_v2":
        raise AdapterError(f"计划 v2 未知激活布局：{settings['activation_layout']!r}")
    layers = [int(value) for value in settings["layers"]]
    if not layers or layers != sorted(set(layers)) or layers[0] < 0:
        raise AdapterError(f"计划层号必须为严格递增的非负列表：{layers[:5]}…")
    shard_id = (plan.get("sharding") or {}).get("shard_id")
    base_args = SimpleNamespace(
        model_root=str(plan["model_root"]),
        lm_weight=plan.get("lm_weight"),
        mimi_weight=plan.get("mimi_weight"),
        device=str(torch_device),
        n_codebooks=int(settings["n_codebooks"]),
        layers=layers,
        max_seconds=float(settings["window_seconds"]),
        mimi_chunk_seconds=float(settings["mimi_chunk_seconds"]),
        mimi_cuda_graph=bool(settings["mimi_cuda_graph"]),
        forward_chunk_steps=int(settings["forward_chunk_steps"]),
        text_mode="greedy",
        text_temperature=DEFAULT_TEXT_TEMPERATURE,
        text_top_k=DEFAULT_TEXT_TOP_K,
        stream_order=str(settings["stream_order"]),
        dump_mimi_latent=True,
        seed=int(settings["seed"]),
        model_name=str(plan.get("model_name", "moshi")),
    )
    min_free = int(resources.get("min_free_disk_bytes", 0))
    window_seconds = float(settings["window_seconds"])
    sessions = list(plan["sessions"])
    log(
        f"E1 计划 v2：{plan['plan_id']}，shard={shard_id}，会话 {len(sessions)}，"
        f"层 {layers[0]}..{layers[-1]}，窗 {window_seconds:g}s"
    )
    skipped = 0
    work_items: list[tuple[int, dict, list[bool]]] = []
    for index, session in enumerate(sessions, start=1):
        valid = [
            cached_run_is_valid_v2(
                session[f"out_agent{ch}"],
                plan=plan,
                session=session,
                channel=ch,
                accepted_code_versions=accepted,
            )
            for ch in (0, 1)
        ]
        skipped += sum(valid)
        if all(valid):
            log(f"会话 {index}/{len(sessions)} 已完整，跳过 {session['session_id']}")
            continue
        work_items.append((index, session, valid))

    if not work_items:
        log(f"E1 持久批处理完成：新增 0 路，复用 {skipped} 路（未加载模型）")
        return

    assert_free_disk(
        work_items[0][1]["out_agent0"],
        min_free,
        what=f"会话 {work_items[0][1]['session_id']} 开跑前",
    )
    log(f"预检完成：待计算 {len(work_items)} 个会话，开始加载模型")
    mimi, lm = load_models(base_args)
    n_layers_model = len(lm.transformer.layers)
    if layers[-1] >= n_layers_model:
        raise AdapterError(f"计划层号 {layers[-1]} 超出模型层数 {n_layers_model}")
    sample_rate = int(getattr(mimi, "sample_rate", 24000))
    completed = 0
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="moshi-audio-prefetch") as executor:
        future = executor.submit(
            _prepare_e1_session_audio,
            work_items[0][1],
            sample_rate,
            window_seconds,
        )
        for position, (index, session, valid) in enumerate(work_items):
            input_prefix, wav0, wav1 = future.result()
            next_future = None
            if position + 1 < len(work_items):
                next_future = executor.submit(
                    _prepare_e1_session_audio,
                    work_items[position + 1][1],
                    sample_rate,
                    window_seconds,
                )
            assert_free_disk(
                session["out_agent0"],
                min_free,
                what=f"会话 {session['session_id']} 开跑前",
            )
            role_extras = tuple(
                {
                    "plan_v2": {
                        "plan_id": plan["plan_id"],
                        "prereg_tag": plan.get("prereg_tag", ""),
                        "cohort": session["cohort"],
                        "agent_channel": channel,
                        "expected_steps": int(settings["expected_steps"]),
                        "expected_parts": int(settings["expected_parts"]),
                        "expected_hidden_dim": settings.get("expected_hidden_dim"),
                        "activation_layout": str(settings["activation_layout"]),
                        "analysis_max_label_step": int(settings["analysis_max_label_step"]),
                        "common_window_steps": int(settings["common_window_steps"]),
                        "input_prefix": input_prefix,
                        "shard_id": shard_id,
                    }
                }
                for channel in (0, 1)
            )
            completed += run_session_pair(
                base_args,
                session,
                mimi,
                lm,
                code_version,
                run_agent0=not valid[0],
                run_agent1=not valid[1],
                role_extras=role_extras,
                engineering_mode="persistent_session_pair_cuda_graph_v3_io_prefetch",
                preloaded_audio=(wav0, wav1),
            )
            log(
                f"E1 持久批处理进度：会话 {index}/{len(sessions)}，"
                f"本进程新增 {completed} 路、复用 {skipped} 路"
            )
            if next_future is not None:
                future = next_future
    log(f"E1 持久批处理完成：新增 {completed} 路，复用 {skipped} 路")


def run_batch_plan(plan_path: str | Path, device: str = "cuda") -> None:
    """每张卡加载一次模型，并按会话级计划持续补齐缓存。"""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    schema_version = int(plan.get("schema_version", 0))
    if schema_version == 2:
        run_batch_plan_v2(plan, device)
        return
    if schema_version != 1:
        raise AdapterError(f"未知持久批处理计划版本：{plan.get('schema_version')}")
    code_version = resolve_code_version(str(plan["code_version"]))
    settings = dict(plan["settings"])
    base_args = SimpleNamespace(
        model_root=str(plan["model_root"]),
        lm_weight=plan.get("lm_weight"),
        mimi_weight=plan.get("mimi_weight"),
        device=device,
        n_codebooks=int(settings["n_codebooks"]),
        layers=[int(value) for value in settings["layers"]],
        max_seconds=float(settings["max_seconds"]),
        mimi_chunk_seconds=float(settings["mimi_chunk_seconds"]),
        forward_chunk_steps=int(settings["forward_chunk_steps"]),
        text_mode=str(settings["text_mode"]),
        text_temperature=float(settings["text_temperature"]),
        text_top_k=int(settings["text_top_k"]),
        stream_order=str(settings["stream_order"]),
        dump_mimi_latent=True,
        seed=int(settings["seed"]),
        model_name=str(plan.get("model_name", "moshi")),
    )
    if base_args.text_mode != "greedy":
        raise AdapterError("持久正式批处理只允许 greedy 文本流")
    accepted = {
        str(value)
        for value in plan.get("accepted_code_versions", [])
    }
    accepted.add(code_version)
    mimi, lm = load_models(base_args)
    completed = 0
    skipped = 0
    sessions = list(plan["sessions"])
    for index, session in enumerate(sessions, start=1):
        valid0 = cached_run_is_valid(
            session["out_agent0"],
            accepted_code_versions=accepted,
            layers=base_args.layers,
            max_seconds=base_args.max_seconds,
            mimi_chunk_seconds=base_args.mimi_chunk_seconds,
            forward_chunk_steps=base_args.forward_chunk_steps,
            text_mode=base_args.text_mode,
        )
        valid1 = cached_run_is_valid(
            session["out_agent1"],
            accepted_code_versions=accepted,
            layers=base_args.layers,
            max_seconds=base_args.max_seconds,
            mimi_chunk_seconds=base_args.mimi_chunk_seconds,
            forward_chunk_steps=base_args.forward_chunk_steps,
            text_mode=base_args.text_mode,
        )
        skipped += int(valid0) + int(valid1)
        if valid0 and valid1:
            log(f"会话 {index}/{len(sessions)} 已完整，跳过 {session['session_id']}")
            continue
        completed += run_session_pair(
            base_args,
            session,
            mimi,
            lm,
            code_version,
            run_agent0=not valid0,
            run_agent1=not valid1,
        )
        log(
            f"持久批处理进度：会话 {index}/{len(sessions)}，"
            f"本进程新增 {completed} 路、复用 {skipped} 路"
        )
    log(f"持久批处理完成：新增 {completed} 路，复用 {skipped} 路")


def build_parser(default_model: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{default_model} R1 复放缓存 runner")
    ap.add_argument("--probe-api", action="store_true", help="转储本地 moshi API 关键信息后退出")
    ap.add_argument("--model-root", help="权重目录（如 moshiko-pytorch-bf16）")
    ap.add_argument("--lm-weight")
    ap.add_argument("--mimi-weight")
    ap.add_argument("--audio-agent", help="agent 通道 24 kHz 单声道 wav")
    ap.add_argument("--audio-other", help="对方通道 24 kHz 单声道 wav")
    ap.add_argument("--session-id", default="unknown")
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[4, 12, 20, 28])
    ap.add_argument("--out", help="run 输出目录（npy 分片 + manifest.json）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-codebooks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument(
        "--mimi-chunk-seconds",
        type=float,
        default=DEFAULT_MIMI_CHUNK_SECONDS,
        help="Mimi 有状态编码块长；默认按官方在线路径每次 0.08 秒（1 帧）",
    )
    ap.add_argument(
        "--forward-chunk-steps",
        type=int,
        default=DEFAULT_FORWARD_CHUNK_STEPS,
        help="transformer 有状态前向块长（Moshi 帧）；默认 128 步",
    )
    ap.add_argument(
        "--text-mode",
        choices=["greedy", "sampled", "pad"],
        default="greedy",
        help="冻结协议默认 greedy；sampled 仅 AB 核验；pad 仅消融复盘",
    )
    ap.add_argument(
        "--text-temperature",
        type=float,
        default=DEFAULT_TEXT_TEMPERATURE,
        help="sampled 文本模式的采样温度（greedy/pad 模式忽略）",
    )
    ap.add_argument(
        "--text-top-k",
        type=int,
        default=DEFAULT_TEXT_TOP_K,
        help="sampled 文本模式的 top-k 截断（官方 LMGen 默认 25；greedy/pad 忽略）",
    )
    ap.add_argument("--stream-order", choices=["self_first", "other_first"], default="self_first")
    ap.add_argument("--dump-mimi-latent", action="store_true", default=True)
    ap.add_argument("--no-dump-mimi-latent", dest="dump_mimi_latent", action="store_false")
    ap.add_argument("--code-version", default=None)
    ap.add_argument("--model-name", default=default_model)
    return ap


def main(default_model: str) -> None:
    args = build_parser(default_model).parse_args()
    if args.probe_api:
        report = probe_api(args)
        text = json.dumps(report, ensure_ascii=False, indent=1)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return
    required = ["model_root", "audio_agent", "audio_other", "out"]
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        print(f"缺少参数：{missing}（或用 --probe-api 做首跑自检）", file=sys.stderr)
        sys.exit(2)
    run(args)


def batch_main(default_model: str) -> None:
    """持久批处理入口。"""
    parser = argparse.ArgumentParser(description=f"{default_model} 持久会话缓存 runner")
    parser.add_argument("--plan", required=True, help="会话级 JSON 批处理计划")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_batch_plan(args.plan, args.device)
