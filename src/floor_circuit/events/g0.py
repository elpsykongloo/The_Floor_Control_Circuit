"""G0 校准：我方事件 → DualTurn-SWB 12.5 Hz 帧级四类 {eot, hold, bot, bc} 与 F1 评分。

评分协议：稀疏类（eot/bot）用 ±tolerance_frames 容差的一对一贪心匹配；
密集类（hold/bc）逐帧计分。最终判据 = 四类 macro-F1 ≥ 0.85（configs/events.yaml g0 节）。
官方标签语义以 V2 检查结论为准，映射表在其后冻结。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.schemas import Event

G0_CLASSES = ("eot", "hold", "bot", "bc")


def events_to_frames(
    events: list[Event], n_frames: int, hz: float, mapping: dict, channel: int
) -> np.ndarray:
    """指定通道的事件序列 → 帧级类别数组（默认类 mapping['default']）。"""
    inverse = {v: k for k, v in mapping.items() if k != "default"}
    frames = np.full(n_frames, mapping["default"], dtype=object)
    spans, points = [], []
    for e in events:
        if e.channel != channel:
            continue
        cls = inverse.get(e.kind.value)
        if cls is None:
            continue
        if e.t_end is not None:
            spans.append((cls, e.t, e.t_end))
        else:
            points.append((cls, e.t))
    for cls, t0, t1 in spans:  # 跨度类（bc）先铺
        i0 = max(0, int(np.floor(t0 * hz)))
        i1 = min(n_frames, max(i0 + 1, int(np.ceil(t1 * hz))))
        frames[i0:i1] = cls
    for cls, t in points:  # 点类（eot/bot）后铺，优先级更高
        i = round(t * hz)
        if 0 <= i < n_frames:
            frames[i] = cls
    return frames


def _match_sparse(pred_idx: np.ndarray, gold_idx: np.ndarray, tol: int) -> int:
    """±tol 帧内一对一贪心匹配，返回命中数。"""
    used = np.zeros(len(pred_idx), dtype=bool)
    hits = 0
    for g in gold_idx:
        best, best_d = -1, tol + 1
        for j, p in enumerate(pred_idx):
            if used[j]:
                continue
            d = abs(int(p) - int(g))
            if d <= tol and d < best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            hits += 1
    return hits


def _prf(hits: int, n_pred: int, n_gold: int) -> dict:
    p = hits / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = hits / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "n_pred": n_pred, "n_gold": n_gold}


def f1_report(
    pred: np.ndarray, gold: np.ndarray, tolerance_frames: int, sparse=("eot", "bot")
) -> dict:
    """逐类 P/R/F1 + macro-F1。pred/gold 为等长帧级类别数组。"""
    if len(pred) != len(gold):
        n = min(len(pred), len(gold))
        pred, gold = pred[:n], gold[:n]
    report: dict = {"per_class": {}}
    for cls in G0_CLASSES:
        p_idx = np.nonzero(pred == cls)[0]
        g_idx = np.nonzero(gold == cls)[0]
        if cls in sparse:
            hits = _match_sparse(p_idx, g_idx, tolerance_frames)
            report["per_class"][cls] = _prf(hits, len(p_idx), len(g_idx))
        else:
            hits = int(np.sum((pred == cls) & (gold == cls)))
            report["per_class"][cls] = _prf(hits, len(p_idx), len(g_idx))
    report["macro_f1"] = float(np.mean([report["per_class"][c]["f1"] for c in G0_CLASSES]))
    return report
