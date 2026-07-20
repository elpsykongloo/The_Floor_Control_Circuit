"""Gate G2 判据评估（文档/00 §7；操作化解释冻结于 PREREG #15）。

三条件：
  1. 跨 3 种子 top-3 层重叠：|∩_{种子} top3(种子)| ≥ top3_overlap_min（=2）；
     top-3 按各种子的层 AUC 降序取，AUC 并列时取较小层号（与 G1 选层同规）。
  2. 有效秩 ≤ effective_rank_max（=16）；有效秩由 rank-k 投影性能曲线另行测得后传入。
  3. 跨种子方向余弦：所有种子对 |cos| 的最小值 ≥ direction_cosine_min（=0.8）。
"""

from __future__ import annotations

import numpy as np


def top_layers_by_auc(auc_by_layer: dict[int, float], k: int = 3) -> list[int]:
    """按 AUC 降序取前 k 层；并列取较小层号（确定性）。"""
    ordered = sorted(auc_by_layer.items(), key=lambda item: (-float(item[1]), int(item[0])))
    return [int(layer) for layer, _auc in ordered[:k]]


def pairwise_abs_cosines(directions: dict[int, np.ndarray]) -> dict[str, float]:
    """种子 → 方向向量（同一层位），返回所有种子对的 |cos|。"""
    seeds = sorted(directions)
    out: dict[str, float] = {}
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            a = np.asarray(directions[seed_a], dtype=np.float64).ravel()
            b = np.asarray(directions[seed_b], dtype=np.float64).ravel()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator == 0.0 or a.shape != b.shape:
                raise ValueError(f"种子 {seed_a}/{seed_b} 方向向量为零或形状不一致")
            out[f"{seed_a}-{seed_b}"] = float(abs(np.dot(a, b)) / denominator)
    return out


def evaluate_g2(
    *,
    auc_by_seed_layer: dict[int, dict[int, float]],
    effective_rank: float,
    direction_cosines: dict[str, float],
    g2_cfg: dict,
) -> dict:
    """返回逐条件结果与总裁决 verdict ∈ {pass, fail}。"""

    if len(auc_by_seed_layer) < 3:
        raise ValueError("G2 需要 3 个种子的层 AUC 表")
    if not direction_cosines:
        raise ValueError("G2 需要至少一对跨种子方向余弦")
    top3_by_seed = {
        seed: top_layers_by_auc(layers, k=3) for seed, layers in sorted(auc_by_seed_layer.items())
    }
    overlap = set.intersection(*(set(layers) for layers in top3_by_seed.values()))
    overlap_pass = len(overlap) >= int(g2_cfg["top3_overlap_min"])
    rank_pass = float(effective_rank) <= float(g2_cfg["effective_rank_max"])
    min_cosine = min(float(value) for value in direction_cosines.values())
    cosine_pass = min_cosine >= float(g2_cfg["direction_cosine_min"])
    verdict = "pass" if (overlap_pass and rank_pass and cosine_pass) else "fail"
    return {
        "verdict": verdict,
        "conditions": {
            "top3_overlap": {
                "top3_by_seed": {seed: layers for seed, layers in top3_by_seed.items()},
                "overlap": sorted(overlap),
                "required_min": int(g2_cfg["top3_overlap_min"]),
                "passed": overlap_pass,
            },
            "effective_rank": {
                "value": float(effective_rank),
                "required_max": float(g2_cfg["effective_rank_max"]),
                "passed": rank_pass,
            },
            "direction_cosine": {
                "pairwise_abs": direction_cosines,
                "min": min_cosine,
                "required_min": float(g2_cfg["direction_cosine_min"]),
                "passed": cosine_pass,
            },
        },
    }
