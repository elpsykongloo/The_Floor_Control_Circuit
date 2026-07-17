"""G0 校准：我方事件 → DualTurn-SWB 12.5 Hz 帧级 {eot, hold, bot, bc} 与 F1 评分。

V2 盘点结论（2026-07-17）：官方金标是**每类一条二值轨**（eot/hold/bot/bc_ch{0,1}，int8 帧级），
不是单一四分类列。据此评分协议冻结为：
- 逐类二值轨对比；eot/bot/hold 为稀疏点事件 → ±tolerance_frames 容差一对一贪心匹配；
  bc 为跨度 → 逐帧计分；
- 语义映射：eot = 非-bc IPU 的 TURNEND；bot = 非-bc IPU 的合格 ONSET；
  hold = 非-bc IPU 末且非 TURNEND（说话人保持话轮的停顿边界）；bc = BC 事件跨度。
  该映射与官方 README 语义的最终核对由 wp1_g0_prepare 抄录 README 后确认。
- 判据 = 四类 macro-F1 ≥ 0.85（configs/events.yaml g0 节），语料级 micro 汇聚。

早期的单列四分类接口（events_to_frames / f1_report）保留，供自测与单元测试使用。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.schemas import Event, EventKind, Seg

G0_CLASSES = ("eot", "hold", "bot", "bc")
SPARSE_CLASSES = ("eot", "bot", "hold")


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
        if i == n_frames and t * hz < n_frames + 0.5:
            i = n_frames - 1  # 末帧钳位，与 build_pred_tracks 一致
        if 0 <= i < n_frames:
            frames[i] = cls
    return frames


def _match_sparse(pred_idx: np.ndarray, gold_idx: np.ndarray, tol: int) -> int:
    """±tol 帧内一对一**最大**匹配数。

    双方排序后金标依次取"最早的可行预测"——一维区间二部图上该贪心即最优
    （2026-07-17 修复：旧的按距离贪心不保证最大匹配，如 pred [0,3]、gold [2,4]、tol 2）。
    """
    pred = sorted(int(p) for p in pred_idx)
    gold = sorted(int(g) for g in gold_idx)
    i = hits = 0
    for g in gold:
        while i < len(pred) and pred[i] < g - tol:
            i += 1
        if i < len(pred) and pred[i] <= g + tol:
            hits += 1
            i += 1
    return hits


def _prf(hits: int, n_pred: int, n_gold: int) -> dict:
    p = hits / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = hits / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "n_pred": n_pred, "n_gold": n_gold}


def build_pred_tracks(
    ipus: list[Seg],
    events: list[Event],
    channel: int,
    n_frames: int,
    hz: float,
    dt_tol: float = 0.02,
) -> dict[str, np.ndarray]:
    """通道级预测二值轨 {eot, hold, bot, bc}（映射语义见模块 docstring）。

    末帧钳位：音频末端的事件（如延伸到片段末尾的 IPU 末）round 后恰为 n_frames，
    在半帧容差内钳到 n_frames-1，避免片段截断处的 eot/hold 被系统性丢弃。
    """

    def frame(t: float) -> int:
        i = round(t * hz)
        if i == n_frames and t * hz < n_frames + 0.5:
            return n_frames - 1
        return i

    bc_spans = [
        (e.t, e.t_end if e.t_end is not None else e.t)
        for e in events
        if e.kind == EventKind.BC and e.channel == channel
    ]
    turnend_ts = [e.t for e in events if e.kind == EventKind.TURNEND and e.channel == channel]
    onset_ts = [e.t for e in events if e.kind == EventKind.ONSET and e.channel == channel]

    def is_bc_ipu(ipu: Seg) -> bool:
        return any(abs(ipu.start - s) <= dt_tol for s, _ in bc_spans)

    tracks = {name: np.zeros(n_frames, dtype=np.int8) for name in G0_CLASSES}
    for s, t_end in bc_spans:
        i0 = max(0, int(np.floor(s * hz)))
        i1 = min(n_frames, max(i0 + 1, int(np.ceil(t_end * hz))))
        tracks["bc"][i0:i1] = 1
    non_bc_ipus = [ipu for ipu in ipus if not is_bc_ipu(ipu)]
    for t in onset_ts:
        at_non_bc_start = any(abs(t - ipu.start) <= dt_tol for ipu in non_bc_ipus)
        if at_non_bc_start and 0 <= frame(t) < n_frames:
            tracks["bot"][frame(t)] = 1
    for ipu in non_bc_ipus:
        t_e = ipu.end
        i = frame(t_e)
        if not (0 <= i < n_frames):
            continue
        if any(abs(t_e - t) <= dt_tol for t in turnend_ts):
            tracks["eot"][i] = 1
        else:
            tracks["hold"][i] = 1
    return tracks


def score_binary_tracks(
    pred: dict[str, np.ndarray],
    gold: dict[str, np.ndarray],
    tolerance_frames: int,
    sparse: tuple[str, ...] = SPARSE_CLASSES,
) -> dict:
    """逐类二值轨评分（G0 生产协议）。pred/gold 各含四类等长 0/1 数组。"""
    report: dict = {"per_class": {}}
    for cls in G0_CLASSES:
        p, g = np.asarray(pred[cls]), np.asarray(gold[cls])
        n = min(len(p), len(g))
        p, g = p[:n] > 0, g[:n] > 0
        p_idx, g_idx = np.nonzero(p)[0], np.nonzero(g)[0]
        if cls in sparse:
            hits = _match_sparse(p_idx, g_idx, tolerance_frames)
            report["per_class"][cls] = _prf(hits, len(p_idx), len(g_idx))
        else:
            hits = int(np.sum(p & g))
            report["per_class"][cls] = _prf(hits, len(p_idx), len(g_idx))
    report["macro_f1"] = float(np.mean([report["per_class"][c]["f1"] for c in G0_CLASSES]))
    return report


def accumulate_counts(
    total: dict[str, dict[str, int]] | None,
    pred: dict[str, np.ndarray],
    gold: dict[str, np.ndarray],
    tolerance_frames: int,
    sparse: tuple[str, ...] = SPARSE_CLASSES,
) -> dict[str, dict[str, int]]:
    """语料级 micro 汇聚：逐会话累计 hits/n_pred/n_gold，最后 finalize_counts 出 F1。"""
    if total is None:
        total = {c: {"hits": 0, "n_pred": 0, "n_gold": 0} for c in G0_CLASSES}
    rep = score_binary_tracks(pred, gold, tolerance_frames, sparse)
    for c in G0_CLASSES:
        cell = rep["per_class"][c]
        hits = round(cell["precision"] * cell["n_pred"]) if cell["n_pred"] else 0
        total[c]["hits"] += int(hits)
        total[c]["n_pred"] += int(cell["n_pred"])
        total[c]["n_gold"] += int(cell["n_gold"])
    return total


def finalize_counts(total: dict[str, dict[str, int]]) -> dict:
    report: dict = {"per_class": {}}
    for c in G0_CLASSES:
        report["per_class"][c] = _prf(total[c]["hits"], total[c]["n_pred"], total[c]["n_gold"])
    report["macro_f1"] = float(np.mean([report["per_class"][c]["f1"] for c in G0_CLASSES]))
    return report


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
