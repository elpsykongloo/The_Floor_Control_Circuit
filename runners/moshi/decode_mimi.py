"""Mimi 离散码 → 24 kHz wav 解码（在 Moshi_family/.venv 内运行；G0 音源生产）。

用法：
  单文件：<moshi python> runners/moshi/decode_mimi.py --model-root <moshiko 目录> \
          --codes codes_ch0.npy --out audio_ch0.wav
  批量：  <moshi python> runners/moshi/decode_mimi.py --model-root <moshiko 目录> \
          --batch-root <data_root>/dualturn_prep
（批量模式对每个 <sid>/codes_ch{0,1}.npy 产出同目录 audio_ch{0,1}.wav，已存在则跳过。）

校验点：codes npy 形状 [T, 8]（帧主序假定）；解码后时长应 ≈ T/12.5 s；
请人工听任一会话 5 秒——正常语音 = reshape 正确，杂音 = 需回报（改码本主序）。
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from moshi_family import AdapterError, _autodetect_weight, log


def load_mimi(args):
    from moshi.models import loaders

    root = Path(args.model_root)
    mimi_path = (
        Path(args.mimi_weight)
        if args.mimi_weight
        else _autodetect_weight(root, ["tokenizer-*.safetensors", "*mimi*.safetensors"], "Mimi 权重")
    )
    log(f"Mimi 权重：{mimi_path}")
    mimi = loaders.get_mimi(str(mimi_path), device=args.device)
    mimi.set_num_codebooks(args.n_codebooks)
    return mimi


def write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    """原子写：先写 .tmp 再改名，保证断点续跑的 exists() 判断只见到完整文件。"""
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def decode_one(mimi, codes_path: Path, out_path: Path, device: str) -> dict:
    import torch

    codes = np.load(codes_path, allow_pickle=False)
    if codes.ndim != 2 or codes.shape[1] != 8:
        raise AdapterError(f"{codes_path} 形状 {codes.shape}，期望 [T, 8]")
    x = torch.from_numpy(codes.astype(np.int64).T)[None].to(device)  # [1, 8, T]
    with torch.no_grad():
        wav = mimi.decode(x)
    wav_np = wav[0, 0].float().cpu().numpy()
    sr = int(getattr(mimi, "sample_rate", 24000))
    write_wav(out_path, wav_np, sr)
    return {
        "codes": str(codes_path),
        "out": str(out_path),
        "frames": int(codes.shape[0]),
        "decoded_s": round(len(wav_np) / sr, 2),
        "expected_s": round(codes.shape[0] / 12.5, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Mimi 码解码为 wav")
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--mimi-weight")
    ap.add_argument("--codes")
    ap.add_argument("--out")
    ap.add_argument("--batch-root", help="逐个处理 <sid>/codes_ch*.npy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-codebooks", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    mimi = load_mimi(args)
    results = []
    if args.batch_root:
        dirs = sorted(p for p in Path(args.batch_root).iterdir() if (p / "codes_ch0.npy").exists())
        if args.limit:
            dirs = dirs[: args.limit]
        for sdir in dirs:
            for ch in (0, 1):
                out = sdir / f"audio_ch{ch}.wav"
                if out.exists():
                    continue
                try:
                    results.append(decode_one(mimi, sdir / f"codes_ch{ch}.npy", out, args.device))
                    log(f"{sdir.name} ch{ch}: {results[-1]['decoded_s']}s（期望 {results[-1]['expected_s']}s）")
                except Exception as e:
                    log(f"失败 {sdir.name} ch{ch}: {e!r}")
                    results.append({"codes": str(sdir / f'codes_ch{ch}.npy'), "error": repr(e)})
    elif args.codes and args.out:
        results.append(decode_one(mimi, Path(args.codes), Path(args.out), args.device))
    else:
        ap.error("需要 --codes/--out 或 --batch-root")
    n_ok = sum(1 for r in results if "error" not in r)
    log(f"解码完成 {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
