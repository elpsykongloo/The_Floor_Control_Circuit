"""Mimi 离散码 → 24 kHz wav 解码（在 Moshi_family/.venv 内运行；G0 音源生产）。

用法：
  单文件：<moshi python> runners/moshi/decode_mimi.py --model-root <moshiko 目录> \
          --codes codes_ch0.npy --out audio_ch0.wav
  批量：  <moshi python> runners/moshi/decode_mimi.py --model-root <moshiko 目录> \
          --batch-root <data_root>/dualturn_prep [--split val]
  双卡：  分别启动两个批量进程，并传 --shard-count 2 --shard-index 0/1 与对应 cuda:0/1。
（批量模式对每个 <sid>/codes_ch{0,1}.npy 产出同目录 audio_ch{0,1}.wav，已存在则跳过。）

校验点：codes npy 形状 [T, 8]（帧主序假定）；解码后时长应 ≈ T/12.5 s；
请人工听任一会话 5 秒——正常语音 = reshape 正确，杂音 = 需回报（改码本主序）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from moshi_family import AdapterError, _autodetect_weight, log


def select_batch_dirs(batch_root: str | Path, split: str | None) -> list[Path]:
    """筛选待解码会话；指定划分时要求每个候选目录的元数据可读。"""
    dirs = sorted(p for p in Path(batch_root).iterdir() if (p / "codes_ch0.npy").exists())
    if split is None:
        return dirs
    selected = []
    for sdir in dirs:
        meta_path = sdir / "meta.json"
        try:
            split_name = json.loads(meta_path.read_text(encoding="utf-8")).get("split")
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(f"{meta_path} 不可读，拒绝在 --split 模式静默跳过") from exc
        if split_name == split:
            selected.append(sdir)
    return selected


def shard_batch_dirs(dirs: list[Path], shard_count: int, shard_index: int) -> list[Path]:
    """按排序后位置做互斥步进分片；所有分片的并集严格覆盖原目录列表。"""

    if shard_count < 1:
        raise AdapterError("--shard-count 必须至少为 1")
    if not 0 <= shard_index < shard_count:
        raise AdapterError(
            f"--shard-index 必须位于 [0, {shard_count})，当前为 {shard_index}"
        )
    return dirs[shard_index::shard_count]


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
    """原子写：使用唯一临时文件，完成后替换目标。"""
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_existing_wav(
    codes_path: Path,
    out_path: Path,
    sample_rate: int,
    frame_rate: float,
) -> dict | None:
    """验证既有 WAV 的容器、格式和时长；任何异常都要求重新解码。"""
    try:
        if sample_rate <= 0 or frame_rate <= 0:
            return None
        codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
        if codes.ndim != 2 or codes.shape[1] != 8:
            return None
        expected_samples = round(int(codes.shape[0]) / frame_rate * sample_rate)
        with wave.open(str(out_path), "rb") as wav:
            n_frames = wav.getnframes()
            valid = (
                wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getframerate() == sample_rate
                and wav.getcomptype() == "NONE"
                and n_frames > 0
                and abs(n_frames - expected_samples) <= round(sample_rate / frame_rate)
            )
        if not valid:
            return None
        return {
            "codes": str(codes_path),
            "out": str(out_path),
            "frames": int(codes.shape[0]),
            "decoded_s": round(n_frames / sample_rate, 2),
            "expected_s": round(int(codes.shape[0]) / frame_rate, 2),
            "skipped": True,
        }
    except (OSError, EOFError, ValueError, wave.Error):
        return None


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
    ap.add_argument("--split", help="批量模式按 meta.json 的 split 字段过滤（如 val）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-codebooks", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=1, help="批量模式的确定性分片总数")
    ap.add_argument("--shard-index", type=int, default=0, help="当前分片编号，从 0 开始")
    args = ap.parse_args()
    mimi = load_mimi(args)
    sample_rate = int(getattr(mimi, "sample_rate", 24000))
    frame_rate = float(getattr(mimi, "frame_rate", 12.5))
    results = []
    if args.batch_root:
        dirs = select_batch_dirs(args.batch_root, args.split)
        if args.split:
            log(f"按 split={args.split} 选中 {len(dirs)} 个会话")
            if not dirs:
                raise AdapterError(f"批量目录中没有 split={args.split} 的会话")
        dirs = shard_batch_dirs(dirs, args.shard_count, args.shard_index)
        log(
            f"分片 {args.shard_index}/{args.shard_count}：选中 {len(dirs)} 个会话"
        )
        if args.limit:
            dirs = dirs[: args.limit]
        for sdir in dirs:
            for ch in (0, 1):
                out = sdir / f"audio_ch{ch}.wav"
                try:
                    codes_path = sdir / f"codes_ch{ch}.npy"
                    if out.exists():
                        cached = validate_existing_wav(
                            codes_path,
                            out,
                            sample_rate,
                            frame_rate,
                        )
                        if cached is not None:
                            results.append(cached)
                            continue
                        log(f"{sdir.name} ch{ch}: 既有 WAV 校验失败，重新解码")
                    results.append(decode_one(mimi, codes_path, out, args.device))
                    log(f"{sdir.name} ch{ch}: {results[-1]['decoded_s']}s（期望 {results[-1]['expected_s']}s）")
                except Exception as e:
                    log(f"失败 {sdir.name} ch{ch}: {e!r}")
                    results.append({"codes": str(sdir / f'codes_ch{ch}.npy'), "error": repr(e)})
    elif args.codes and args.out:
        results.append(decode_one(mimi, Path(args.codes), Path(args.out), args.device))
    else:
        ap.error("需要 --codes/--out 或 --batch-root")
    n_ok = sum(1 for r in results if "error" not in r)
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_errors = len(results) - n_ok
    log(f"解码完成 {n_ok}/{len(results)}（已存在跳过 {n_skipped}，失败 {n_errors}）")
    if n_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
