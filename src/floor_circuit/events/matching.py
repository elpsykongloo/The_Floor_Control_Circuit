"""稀疏事件时刻的一对一最大匹配（时间域，秒）。

与 events/g0.py 的帧域 `_match_sparse` 同算法：双方排序后，金标依次取"最早的
可行预测"——一维区间二部图上该贪心即最优（反例 pred [0,3]/gold [2,4]/tol 2 已在
2026-07-17 修复并测试）。本模块供 T4 标签审计等时间域用途复用。
"""

from __future__ import annotations

import numpy as np


def match_sparse_times(pred_times, gold_times, tol_s: float) -> int:
    """±tol_s 秒内一对一最大匹配数。"""
    if tol_s < 0:
        raise ValueError("tol_s 不能为负")
    pred = sorted(float(value) for value in pred_times)
    gold = sorted(float(value) for value in gold_times)
    i = hits = 0
    for g in gold:
        while i < len(pred) and pred[i] < g - tol_s:
            i += 1
        if i < len(pred) and pred[i] <= g + tol_s:
            hits += 1
            i += 1
    return hits


def precision_recall_f1(hits: int, n_pred: int, n_gold: int) -> dict:
    precision = hits / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    recall = hits / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_pred": int(n_pred),
        "n_gold": int(n_gold),
        "n_matched": int(hits),
    }


def macro_f1(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row["f1"] for row in rows]))
