"""Moshi 系（Moshi / PersonaPlex）R1 teacher-forced 复放 + 逐层激活缓存 runner。

在各自模型 .venv 内运行（依赖：torch、moshi、numpy；音频读取优先 soundfile，缺失退 stdlib wave）。
与主仓库解耦：只按附录 C 契约输出 npy 分片 + manifest.json，仓库侧 wp5_ingest 转 zarr。

三种模式：
  --probe-api      加载模型并转储关键属性/签名 → 首跑联调的依据（出错时把输出整体回传）
  --text-mode pad     并行 teacher-forced 前向（默认，快）：双音频流强制、文本流全 PAD。
                      因果注意力下与逐步流式在数学上等价，是 R1 缓存的生产路径。
  --text-mode greedy  逐步贪心自预测文本流（实验性，慢）：hook sanity 的 AB 对照用。

已知需要首跑确认的适配点（全部经 _first_attr 探测并写入 manifest.extra）：
  文本 PAD token id、初始填充 token id、流顺序（self_first/other_first）、delays 属性。
"""

from __future__ import annotations

import argparse
import hashlib
import inspect as _inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

BLOCK_STEPS = 4096


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


def encode_codes(mimi, wav: np.ndarray, device: str):
    import torch

    x = torch.from_numpy(wav)[None, None, :].to(device)
    with torch.no_grad():
        codes = mimi.encode(x)
    return codes  # [1, K, T]


def encode_mimi_latent(mimi, wav: np.ndarray, device: str) -> np.ndarray | None:
    """量化前连续潜表征（Mimi 基线用）。方法名跨版本探测，找不到则返回 None 并记录。"""
    import torch

    x = torch.from_numpy(wav)[None, None, :].to(device)
    for name in ("encode_to_latent", "_encode_to_unquantized_latent", "encode_latent"):
        fn = getattr(mimi, name, None)
        if fn is None:
            continue
        try:
            with torch.no_grad():
                z = fn(x)
            z = z[0].transpose(0, 1).float().cpu().numpy()  # [T, D]
            return z.astype(np.float16)
        except Exception as e:
            log(f"mimi.{name} 失败：{e!r}")
    log("警告：未获得 Mimi 连续潜表征（基线将退用码本嵌入路径，见报告）")
    return None


def build_parallel_codes(lm, codes_agent, codes_other, args) -> tuple:
    """构造 [1, 1+2*K, T] 的 teacher-forced 输入（文本=PAD），按 lm.delays 施加流延迟。"""
    import torch

    _, pad_id = _first_attr(
        lm, ["text_padding_token_id", "existing_text_padding_id"], "文本 PAD token id"
    )
    try:
        _, init_id = _first_attr(lm, ["initial_token_id", "zero_token_id"], "初始填充 token id")
    except AdapterError:
        init_id = int(pad_id)
        log("未找到初始填充 token id，退用文本 PAD id 填充音频延迟位（首跑核对）")
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
    if len(delays) < n_streams:
        raise AdapterError(f"delays 长度 {len(delays)} < 流数 {n_streams}")
    codes = torch.full((1, n_streams, T), int(init_id), dtype=torch.long, device=codes_agent.device)
    codes[0, 0, :] = int(pad_id)
    src_map = {"agent": codes_agent, "other": codes_other}
    for k in range(1, n_streams):
        which, q = streams[k]
        d = int(delays[k])
        if d > 0:
            codes[0, k, d:T] = src_map[which][0, q, : T - d]
        else:
            codes[0, k, :] = src_map[which][0, q, :T]
    meta = {
        "text_pad_id": int(pad_id),
        "init_fill_id": int(init_id),
        "delays": [int(d) for d in delays[:n_streams]],
        "stream_order": args.stream_order,
        "n_streams": n_streams,
        "T": T,
    }
    return codes, meta


def forward_capture(lm, codes, layers: list[int], out_dir: Path, args) -> dict:
    """并行前向 + 残差流 hook 捕获，按 BLOCK_STEPS 写 npy 分片。"""
    import torch

    tf = lm.transformer
    if not hasattr(tf, "layers"):
        raise AdapterError(f"lm.transformer 无 .layers；类型 {type(tf).__name__}")
    n_layers = len(tf.layers)
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
    try:
        try:
            _ = lm(codes)
        except TypeError:
            sig = _inspect.signature(lm.forward)
            kwargs = {}
            if "condition_tensors" in sig.parameters:
                kwargs["condition_tensors"] = None
            _ = lm(codes, **kwargs)
    finally:
        for hd in handles:
            hd.remove()
    log(f"前向完成：T={codes.shape[2]}，耗时 {time.time() - t0:.1f}s")
    hidden_dim = None
    for ell in layers:
        if not captured[ell]:
            raise AdapterError(f"层 {ell} 的 hook 未捕获输出（layer 模块可能非逐层调用）")
        acts = torch.cat(captured[ell], dim=0).numpy()  # [T, H]
        hidden_dim = int(acts.shape[1])
        for pi, i0 in enumerate(range(0, acts.shape[0], BLOCK_STEPS)):
            np.save(out_dir / f"acts_L{ell}_part{pi:05d}.npy", acts[i0 : i0 + BLOCK_STEPS])
    return {"hidden_dim": hidden_dim, "n_steps": int(codes.shape[2])}


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mimi, lm = load_models(args)
    sr = int(getattr(mimi, "sample_rate", 24000))
    wav_agent = read_wav_mono(args.audio_agent, sr)
    wav_other = read_wav_mono(args.audio_other, sr)
    if args.max_seconds:
        n = int(args.max_seconds * sr)
        wav_agent, wav_other = wav_agent[:n], wav_other[:n]
    if args.text_mode != "pad":
        raise AdapterError("greedy 文本模式为实验项，待 pad 路径首跑通过后在本机联调（hook sanity AB 用）")
    codes_agent = encode_codes(mimi, wav_agent, args.device)
    codes_other = encode_codes(mimi, wav_other, args.device)
    codes, build_meta = build_parallel_codes(lm, codes_agent, codes_other, args)
    stats = forward_capture(lm, codes, args.layers, out_dir, args)
    mimi_latent_ok = False
    if args.dump_mimi_latent:
        z = encode_mimi_latent(mimi, wav_agent, args.device)
        if z is not None:
            for pi, i0 in enumerate(range(0, z.shape[0], BLOCK_STEPS)):
                np.save(out_dir / f"mimi_latent_part{pi:05d}.npy", z[i0 : i0 + BLOCK_STEPS])
            mimi_latent_ok = True
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
        "temperature": None,
        "text_mode": args.text_mode,
        "source_audio": {
            str(args.audio_agent): sha256_file(args.audio_agent),
            str(args.audio_other): sha256_file(args.audio_other),
        },
        "mimi_latent": mimi_latent_ok,
        "code_version": args.code_version,
        "extra": build_meta,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
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
    ap.add_argument("--text-mode", choices=["pad", "greedy"], default="pad")
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
