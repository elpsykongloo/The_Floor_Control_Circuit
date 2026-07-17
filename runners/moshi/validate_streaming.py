"""Moshi 分块缓存路径的短音频等价性验证。

仅允许最多 30 秒音频。参考侧调用官方 ``forward_text``，不会调用 depformer；
候选侧调用生产用有状态分块主干，并对拍 Mimi 码、连续潜表征和四层残差流。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from moshi_family import (
    AdapterError,
    build_parallel_codes,
    encode_mimi_stream,
    forward_capture,
    load_models,
    prepare_teacher_forced_input,
    read_wav_mono,
    write_json_atomic,
)

MAX_VALIDATION_SECONDS = 30.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise AdapterError(
            f"对拍形状不一致：参考 {reference.shape}，分块 {candidate.shape}"
        )
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    cand = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = cand - ref
    ref_rms = float(np.sqrt(np.mean(ref * ref)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = float(np.dot(ref, cand) / denominator) if denominator > 0 else 1.0
    result = {
        "shape": list(reference.shape),
        "max_abs_error": float(np.max(np.abs(delta), initial=0.0)),
        "rmse": rmse,
        "relative_rmse": rmse / max(ref_rms, 1e-12),
        "cosine_similarity": cosine,
    }
    if reference.ndim == 2 and reference.shape[0] > 0:
        frame_rmse = np.sqrt(np.mean(
            (candidate.astype(np.float64) - reference.astype(np.float64)) ** 2,
            axis=1,
        ))
        bad = np.flatnonzero(frame_rmse > 1e-3)
        result["frame_error"] = {
            "threshold_rmse": 1e-3,
            "n_above_threshold": int(bad.size),
            "first_above_threshold": int(bad[0]) if bad.size else None,
            "last_above_threshold": int(bad[-1]) if bad.size else None,
            "worst_frame": int(np.argmax(frame_rmse)),
            "worst_frame_rmse": float(np.max(frame_rmse)),
        }
    return result


def _code_metrics(reference, candidate) -> dict:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise AdapterError(
            f"码形状不一致：参考 {tuple(reference.shape)}，候选 {tuple(candidate.shape)}"
        )
    mismatch = reference.ne(candidate)
    n_values = int(mismatch.numel())
    n_mismatched = int(mismatch.sum().item())
    frame_mismatch = mismatch.any(dim=1)
    return {
        "shape": list(reference.shape),
        "exact": n_mismatched == 0,
        "n_values": n_values,
        "n_mismatched_values": n_mismatched,
        "mismatched_value_rate": n_mismatched / max(n_values, 1),
        "n_mismatched_frames": int(frame_mismatch.sum().item()),
    }


def _capture_reference(lm, sequence, layers: list[int]) -> dict[int, np.ndarray]:
    """捕获官方整段 ``forward_text`` 的层输出；短序列专用于等价性取证。"""
    import torch

    captured: dict[int, list] = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer].append(
                hidden.detach()[0].to(torch.float16).cpu().numpy()
            )

        return hook

    for layer in layers:
        handles.append(lm.transformer.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.no_grad():
            transformer_out, text_logits = lm.forward_text(sequence)
            del transformer_out, text_logits
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer in layers:
        if not captured[layer]:
            raise AdapterError(f"参考路径未捕获层 {layer}")
        result[layer] = np.concatenate(captured[layer], axis=0)
    return result


def _load_parts(root: Path, prefix: str) -> np.ndarray:
    paths = sorted(root.glob(f"{prefix}_part*.npy"))
    if not paths:
        raise AdapterError(f"分块路径没有生成 {prefix}")
    return np.concatenate([np.load(path, allow_pickle=False) for path in paths], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--audio-agent", required=True)
    parser.add_argument("--audio-other", required=True)
    parser.add_argument("--out", required=True, help="验证报告 JSON")
    parser.add_argument("--work-dir", required=True, help="临时层输出目录")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--mimi-chunk-seconds", type=float, default=0.08)
    parser.add_argument("--forward-chunk-steps", type=int, default=128)
    parser.add_argument("--layers", default="4,12,20,28")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-codebooks", type=int, default=8)
    args = parser.parse_args()

    if args.seconds <= 0 or args.seconds > MAX_VALIDATION_SECONDS:
        parser.error(f"--seconds 必须在 (0, {MAX_VALIDATION_SECONDS:g}] 内")
    layers = [int(value) for value in args.layers.split(",")]
    args.lm_weight = None
    args.mimi_weight = None

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    mimi, lm = load_models(args)
    sample_rate = int(mimi.sample_rate)
    n_samples = round(args.seconds * sample_rate)
    agent_path = Path(args.audio_agent)
    other_path = Path(args.audio_other)
    wav_agent = read_wav_mono(agent_path, sample_rate)[:n_samples]
    wav_other = read_wav_mono(other_path, sample_rate)[:n_samples]
    if len(wav_agent) < n_samples or len(wav_other) < n_samples:
        raise AdapterError("验证音频短于请求时长")

    import torch

    x_agent = torch.from_numpy(wav_agent)[None, None].to(args.device)
    x_other = torch.from_numpy(wav_other)[None, None].to(args.device)
    with torch.no_grad():
        latent_agent_ref = mimi.encode_to_latent(x_agent, quantize=False)
        codes_agent_ref = mimi.quantizer.encode(latent_agent_ref)
        latent_other_ref = mimi.encode_to_latent(x_other, quantize=False)
        codes_other_ref = mimi.quantizer.encode(latent_other_ref)
    latent_agent_ref_np = (
        latent_agent_ref[0].transpose(0, 1).to(torch.float16).cpu().numpy()
    )

    codes_agent_stream, latent_agent_stream, latent_source = encode_mimi_stream(
        mimi,
        wav_agent,
        args.device,
        args.mimi_chunk_seconds,
        return_latent=True,
    )
    codes_other_stream, _, _ = encode_mimi_stream(
        mimi,
        wav_other,
        args.device,
        args.mimi_chunk_seconds,
        return_latent=False,
    )
    agent_code_metrics = _code_metrics(codes_agent_ref, codes_agent_stream)
    other_code_metrics = _code_metrics(codes_other_ref, codes_other_stream)
    agent_codes_equal = bool(agent_code_metrics["exact"])
    other_codes_equal = bool(other_code_metrics["exact"])
    if latent_agent_stream is None:
        raise AdapterError("分块路径没有返回量化前连续潜表征")
    latent_metrics = _metrics(latent_agent_ref_np, latent_agent_stream)

    code_args = SimpleNamespace(stream_order="self_first")
    codes, code_meta = build_parallel_codes(
        lm, codes_agent_stream, codes_other_stream, code_args
    )
    sequence = prepare_teacher_forced_input(lm, codes)
    reference_layers = _capture_reference(lm, sequence, layers)
    stream_stats = forward_capture(
        lm, codes, layers, work_dir, args.forward_chunk_steps
    )
    layer_metrics = {
        str(layer): _metrics(
            reference_layers[layer],
            _load_parts(work_dir, f"acts_L{layer}"),
        )
        for layer in layers
    }

    cosine_floor = 0.9999
    code_mismatch_ceiling = 0.05
    passed = (
        latent_metrics["cosine_similarity"] >= cosine_floor
        and agent_code_metrics["mismatched_value_rate"] <= code_mismatch_ceiling
        and other_code_metrics["mismatched_value_rate"] <= code_mismatch_ceiling
        and all(
            metrics["cosine_similarity"] >= cosine_floor
            for metrics in layer_metrics.values()
        )
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "scope": {
            "seconds": args.seconds,
            "maximum_allowed_seconds": MAX_VALIDATION_SECONDS,
            "sample_rate": sample_rate,
            "mimi_chunk_seconds": args.mimi_chunk_seconds,
            "mimi_mode": "official_frame_streaming",
            "forward_chunk_steps": args.forward_chunk_steps,
            "layers": layers,
            "reference_path": "mimi_offline_and_lm_forward_text_without_depformer",
            "candidate_path": "official_frame_streaming_mimi_and_stateful_transformer_backbone",
            "semantic_authority": "official_mimi_frame_streaming",
        },
        "source_audio": {
            str(agent_path): _sha256(agent_path),
            str(other_path): _sha256(other_path),
        },
        "mimi": {
            "agent_codes_exact": agent_codes_equal,
            "other_codes_exact": other_codes_equal,
            "agent_codes": agent_code_metrics,
            "other_codes": other_code_metrics,
            "latent_source": latent_source,
            "latent": latent_metrics,
        },
        "transformer": {
            "code_meta": code_meta,
            "stream_stats": stream_stats,
            "cosine_floor": cosine_floor,
            "offline_code_mismatch_ceiling": code_mismatch_ceiling,
            "layers": layer_metrics,
        },
    }
    write_json_atomic(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
