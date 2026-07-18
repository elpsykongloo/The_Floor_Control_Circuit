"""WP7：R1 文本流 AB 核验（文档/02 WP4 冻结要求 + PREREG #7）。

对同一段音频、同一组层的两个 runner 输出目录（npy 分片，ingest 前）逐层逐步对比
残差流，量化文本流处理方式带来的中层表征差异：
  1) greedy vs sampled —— 差异可忽略则冻结 greedy（协议规定的 AB 核验）；
  2) pad vs greedy —— 量化被撤回的 PAD 变体的伪影大小（诊断用）。

用法（本机，Moshi runner 分别以 --text-mode greedy / sampled / pad 跑同一会话后）：
  uv run python scripts/wp7_text_mode_ab.py --run-a <dir_greedy> --run-b <dir_sampled> \
      --label greedy_vs_sampled
产出：reports/text_mode_ab_<label>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{run_dir} 缺少 manifest.json（runner 未完成？）")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_layer(run_dir: Path, layer: int) -> np.ndarray:
    parts = sorted(run_dir.glob(f"acts_L{layer}_part*.npy"))
    if not parts:
        raise SystemExit(f"{run_dir} 缺少层 {layer} 的 npy 分片")
    return np.concatenate([np.load(p, allow_pickle=False) for p in parts], axis=0)


def _layer_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise SystemExit(f"两个 run 的激活形状不一致：{a.shape} vs {b.shape}")
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    norm_a = np.linalg.norm(a64, axis=1)
    norm_b = np.linalg.norm(b64, axis=1)
    denominator = np.maximum(norm_a * norm_b, 1e-12)
    cosine = np.sum(a64 * b64, axis=1) / denominator
    rel_l2 = np.linalg.norm(a64 - b64, axis=1) / np.maximum(norm_a, 1e-12)
    return {
        "n_steps": int(a.shape[0]),
        "hidden_dim": int(a.shape[1]),
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "cosine_min": float(cosine.min()),
        "rel_l2_mean": float(rel_l2.mean()),
        "rel_l2_p95": float(np.percentile(rel_l2, 95)),
        "rel_l2_max": float(rel_l2.max()),
        "exact_equal": bool(np.array_equal(a, b)),
    }


def _text_token_stats(run_dir: Path, manifest: dict) -> dict | None:
    path = run_dir / "text_tokens.npy"
    if not path.is_file():
        return None
    tokens = np.load(path, allow_pickle=False)
    pad_id = manifest.get("extra", {}).get("text_pad_id")
    stats = {
        "n_tokens": len(tokens),
        "n_unique": len(np.unique(tokens)),
    }
    if pad_id is not None:
        stats["pad_fraction"] = float(np.mean(tokens == int(pad_id)))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="runner 输出目录 A（如 greedy）")
    ap.add_argument("--run-b", required=True, help="runner 输出目录 B（如 sampled 或 pad）")
    ap.add_argument("--label", required=True, help="报告标签，如 greedy_vs_sampled")
    args = ap.parse_args()
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    manifest_a, manifest_b = _load_manifest(run_a), _load_manifest(run_b)

    if manifest_a.get("source_audio") != manifest_b.get("source_audio"):
        raise SystemExit("两个 run 的源音频哈希不一致，AB 对比无意义")
    layers_a = list(manifest_a.get("layers", []))
    if layers_a != list(manifest_b.get("layers", [])):
        raise SystemExit(f"层列表不一致：{layers_a} vs {manifest_b.get('layers')}")

    per_layer = {}
    for layer in layers_a:
        per_layer[f"L{layer}"] = _layer_diff_stats(
            _load_layer(run_a, int(layer)),
            _load_layer(run_b, int(layer)),
        )
    report = {
        "label": args.label,
        "run_a": {
            "path": str(run_a),
            "text_mode": manifest_a.get("text_mode"),
            "code_version": manifest_a.get("code_version"),
            "text_tokens": _text_token_stats(run_a, manifest_a),
        },
        "run_b": {
            "path": str(run_b),
            "text_mode": manifest_b.get("text_mode"),
            "code_version": manifest_b.get("code_version"),
            "text_tokens": _text_token_stats(run_b, manifest_b),
        },
        "per_layer": per_layer,
    }
    write_report_json(f"text_mode_ab_{args.label}.json", report)
    worst = min(stats["cosine_p05"] for stats in per_layer.values())
    print(f"AB 完成（{args.label}）：各层 cosine P5 最小值 {worst:.6f}")


if __name__ == "__main__":
    main()
