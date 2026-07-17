"""在短音频上比较 Mimi 流式块长与离线参考，选择有界生产块长。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from moshi_family import (
    AdapterError,
    _autodetect_weight,
    encode_mimi_stream,
    read_wav_mono,
    write_json_atomic,
)

MAX_SECONDS = 30.0


def _compare_codes(reference, candidate) -> dict:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise AdapterError(
            f"码形状不一致：{tuple(reference.shape)} 与 {tuple(candidate.shape)}"
        )
    mismatch = reference.ne(candidate)
    return {
        "exact": not bool(mismatch.any()),
        "n_values": int(mismatch.numel()),
        "n_mismatched_values": int(mismatch.sum().item()),
        "mismatched_value_rate": float(mismatch.float().mean().item()),
        "n_mismatched_frames": int(mismatch.any(dim=1).sum().item()),
    }


def _compare_latent(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise AdapterError(f"潜表征形状不一致：{reference.shape} 与 {candidate.shape}")
    ref = reference.astype(np.float64).reshape(-1)
    cand = candidate.astype(np.float64).reshape(-1)
    delta = cand - ref
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    return {
        "shape": list(reference.shape),
        "max_abs_error": float(np.max(np.abs(delta), initial=0.0)),
        "relative_rmse": float(
            np.sqrt(np.mean(delta * delta))
            / max(np.sqrt(np.mean(ref * ref)), 1e-12)
        ),
        "cosine_similarity": float(np.dot(ref, cand) / denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--chunks",
        default="0.08,0.16,0.32,0.64,1.28,2.56,4.0",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-codebooks", type=int, default=8)
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > MAX_SECONDS:
        parser.error(f"--seconds 必须在 (0, {MAX_SECONDS:g}] 内")

    import torch
    from moshi.models import loaders

    root = Path(args.model_root)
    weight = _autodetect_weight(
        root,
        ["tokenizer-*.safetensors", "*mimi*.safetensors"],
        "Mimi 权重",
    )
    mimi = loaders.get_mimi(str(weight), device=args.device)
    mimi.set_num_codebooks(args.n_codebooks)
    wav = read_wav_mono(args.audio, int(mimi.sample_rate))
    wav = wav[: round(args.seconds * mimi.sample_rate)]

    x = torch.from_numpy(wav)[None, None].to(args.device)
    with torch.no_grad():
        latent_ref = mimi.encode_to_latent(x, quantize=False)
        codes_ref = mimi.quantizer.encode(latent_ref)
    latent_ref_np = (
        latent_ref[0].transpose(0, 1).to(torch.float16).cpu().numpy()
    )

    rows = []
    for chunk_seconds in [float(value) for value in args.chunks.split(",")]:
        started = time.perf_counter()
        codes, latent, source = encode_mimi_stream(
            mimi,
            wav,
            args.device,
            chunk_seconds,
            return_latent=True,
        )
        elapsed = time.perf_counter() - started
        if latent is None:
            raise AdapterError("流式路径没有连续潜表征")
        rows.append(
            {
                "chunk_seconds": chunk_seconds,
                "elapsed_seconds": elapsed,
                "realtime_factor": elapsed / args.seconds,
                "latent_source": source,
                "codes": _compare_codes(codes_ref, codes),
                "latent": _compare_latent(latent_ref_np, latent),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))

    write_json_atomic(
        Path(args.out),
        {
            "schema_version": 1,
            "seconds": args.seconds,
            "audio": str(Path(args.audio)),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
