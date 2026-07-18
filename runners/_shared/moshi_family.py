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
import hashlib
import inspect as _inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

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
        raise AdapterError(
            f"计划代码版本 {explicit} 与当前 runner {actual} 不一致；请重新生成缓存计划"
        )
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
    for pattern in ("acts_L*_part*.npy", "mimi_latent_part*.npy", ".*.tmp"):
        paths.extend(out_dir.glob(pattern))
    removed = 0
    for path in paths:
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        log(f"已清理 {removed} 个旧的 runner 产物")


def read_wav_mono(path: str | Path, expect_sr: int) -> np.ndarray:
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    except ImportError:
        import wave

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            width = w.getsampwidth()
            raw = w.readframes(n)
        if width == 2:
            wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            wav = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise AdapterError(f"不支持的 wav 位宽 {width * 8}，请先用 ffmpeg 转 16-bit PCM") from None
        if w.getnchannels() > 1:  # type: ignore[union-attr]
            wav = wav.reshape(-1, w.getnchannels()).mean(axis=1)  # type: ignore[union-attr]
    if sr != expect_sr:
        raise AdapterError(
            f"{path} 采样率 {sr} ≠ {expect_sr}：请用 scripts/wp2_extract_candor.py 产出的 24 kHz 单声道 wav"
        )
    return np.asarray(wav, dtype=np.float32)


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
    if frame_size <= 0 or frame_rate <= 0:
        raise AdapterError(
            f"Mimi 缺少有效 frame_size/frame_rate：{frame_size}/{frame_rate}"
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
    codes_out = None
    latent_out = None
    written_steps = 0
    with torch.no_grad(), streaming(batch_size=1):
        for chunk in _iter_padded_wav_chunks(wav, chunk_samples):
            x = torch.from_numpy(chunk)[None, None, :].to(device)
            if latent_fn is None:
                codes = mimi.encode(x)
            else:
                latent = latent_fn(x)
                codes = quantizer.encode(latent)
                if int(latent.shape[-1]) != chunk_steps:
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
                latent_chunk = (
                    latent[0, :, :take]
                    .transpose(0, 1)
                    .to(torch.float16)
                    .cpu()
                    .numpy()
                )
                if latent_out is None:
                    latent_out = np.empty(
                        (valid_steps, int(latent_chunk.shape[1])),
                        dtype=np.float16,
                    )
                latent_out[written_steps : written_steps + take] = latent_chunk
            written_steps += take
    if codes_out is None:
        raise AdapterError("Mimi 流式编码没有产生任何码")
    if written_steps != valid_steps:
        raise AdapterError(
            f"Mimi 流式编码仅写出 {written_steps}/{valid_steps} 帧"
        )
    if return_latent and latent_out is None:
        raise AdapterError("Mimi 流式编码没有产生连续潜表征")
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
    }


def run(args) -> None:
    code_version = resolve_code_version(args.code_version)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_run_outputs(out_dir)
    mimi, lm = load_models(args)
    sr = int(getattr(mimi, "sample_rate", 24000))
    wav_agent = read_wav_mono(args.audio_agent, sr)
    wav_other = read_wav_mono(args.audio_other, sr)
    if args.max_seconds:
        n = int(args.max_seconds * sr)
        wav_agent, wav_other = wav_agent[:n], wav_other[:n]
    log(f"文本流模式：{args.text_mode}")
    log(f"Mimi 流式编码 agent（块长 {args.mimi_chunk_seconds:g}s）")
    codes_agent, z, latent_source = encode_mimi_stream(
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
        )
    mimi_latent_ok = z is not None
    if z is not None:
        for pi, i0 in enumerate(range(0, z.shape[0], BLOCK_STEPS)):
            write_npy_atomic(
                out_dir / f"mimi_latent_part{pi:05d}.npy",
                z[i0 : i0 + BLOCK_STEPS],
            )
    build_meta["execution"] = {
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
        "latent_kind": "pre_quantization_continuous" if z is not None else None,
        "latent_source": latent_source,
        "time_alignment": dict(TIME_ALIGNMENT),
    }
    if args.text_mode != "pad":
        build_meta["execution"]["text_head_source"] = stats["text_head_source"]
        build_meta["execution"]["text_token_pad_fraction"] = stats["text_token_pad_fraction"]
        if args.text_mode == "sampled":
            build_meta["execution"]["text_temperature"] = float(args.text_temperature)
            build_meta["execution"]["text_top_k"] = int(args.text_top_k)
            build_meta["execution"]["text_seed"] = int(args.seed)
    manifest = {
        "schema_version": 1,
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
        "source_audio": {
            str(args.audio_agent): sha256_file(args.audio_agent),
            str(args.audio_other): sha256_file(args.audio_other),
        },
        "mimi_latent": mimi_latent_ok,
        "code_version": code_version,
        "extra": build_meta,
    }
    manifest["extra"]["output_files"] = {
        path.name: path.stat().st_size for path in sorted(out_dir.glob("*.npy"))
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    log(f"完成：{out_dir}（layers={args.layers}，T={stats['n_steps']}，H={stats['hidden_dim']}）")


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
