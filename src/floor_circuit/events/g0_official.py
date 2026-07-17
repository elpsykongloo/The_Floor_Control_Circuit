"""DualTurn 官方标签算法的本地复现（G0 重构层 1/3 的核心，2026-07-17）。

依据（官方 README 的 Label definitions + 用户对 relabel_context_aware.py 的复核）：
- EOT：非-BC 语音段末，4 s 内**对方**先恢复（对方段须为有效段，≥ ~1 s）；
- HOLD：非-BC 语音段末，4 s 内**本人**先恢复（无交接）；
- BOT：段长 ≥ ~1 s 的语音段起点，且过去 4 s 内最近的有效发言者是对方；
- BC：段长 ≤ ~1 s、前后各 ≥ ~1 s 静音、附近存在对方有效话轮；BC 覆盖整段跨度。

边界细节（帧取整、1 s = 12/13 帧、事件落在段末语音帧还是首静音帧、重叠中的对方是否算
"立即接管"、本人恢复是否要求有效段）无法从描述唯一确定——全部参数化进 OfficialParams，
由 protocol_check 的网格搜索在"官方金标 VAD → 本算法 vs 官方金标标签"上收敛到逐帧全等后冻结。
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product

import numpy as np

from floor_circuit.schemas import Seg

FRAME_HZ = 12.5
OFFICIAL_CLASSES = ("eot", "hold", "bot", "bc")


@dataclass(frozen=True)
class OfficialParams:
    lookahead_f: int = 50  # 4 s 前瞻（EOT/HOLD）
    lookback_f: int = 50  # 4 s 回看（BOT）
    min_valid_f: int = 13  # "有效"段最短帧数（≥ ~1 s）
    bot_min_f: int = 13  # BOT 段最短帧数
    bc_max_f: int = 12  # BC 段最长帧数（≤ ~1 s）
    bc_gap_f: int = 13  # BC 前后静音最短帧数
    bc_context_f: int = 50  # BC "附近对方有效话轮"窗口
    self_resume_valid_only: bool = False  # 本人恢复是否要求有效段
    other_ongoing_counts: bool = True  # 段末时对方正处有效段中 → 视为立即接管
    eot_at_last_speech: bool = True  # 事件帧 = 段末最后语音帧（False = 段末首静音帧）

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def vad_segments(vad: np.ndarray) -> list[tuple[int, int]]:
    """二值轨 → [start, end) 帧段列表。"""
    v = np.asarray(vad).astype(bool)
    if v.size == 0:
        return []
    padded = np.concatenate([[False], v, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _bc_flags(
    segs_self: list[tuple[int, int]],
    valid_other: list[tuple[int, int]],
    p: OfficialParams,
) -> list[bool]:
    flags = []
    for i, (s, e) in enumerate(segs_self):
        dur = e - s
        if dur > p.bc_max_f:
            flags.append(False)
            continue
        gap_before = s - segs_self[i - 1][1] if i > 0 else 10**9
        gap_after = segs_self[i + 1][0] - e if i + 1 < len(segs_self) else 10**9
        if gap_before < p.bc_gap_f or gap_after < p.bc_gap_f:
            flags.append(False)
            continue
        near = any(
            os_ < e + p.bc_context_f and oe > s - p.bc_context_f for os_, oe in valid_other
        )
        flags.append(near)
    return flags


def official_tracks(
    vad_self: np.ndarray, vad_other: np.ndarray, p: OfficialParams | None = None
) -> dict[str, np.ndarray]:
    """单通道官方四类轨。vad_* 为 12.5 Hz 二值轨（等长）。"""
    p = p or OfficialParams()
    n = int(min(len(vad_self), len(vad_other)))
    segs_s = vad_segments(np.asarray(vad_self)[:n])
    segs_o = vad_segments(np.asarray(vad_other)[:n])
    valid_o = [(s, e) for s, e in segs_o if e - s >= p.min_valid_f]
    bc = _bc_flags(segs_s, valid_o, p)
    non_bc = [seg for seg, is_bc in zip(segs_s, bc, strict=True) if not is_bc]
    resume_self = (
        [(s, e) for s, e in non_bc if e - s >= p.min_valid_f] if p.self_resume_valid_only else non_bc
    )
    valid_self = [(s, e) for s, e in non_bc if e - s >= p.min_valid_f]

    tracks = {name: np.zeros(n, dtype=np.int8) for name in OFFICIAL_CLASSES}
    for (s, e), is_bc in zip(segs_s, bc, strict=True):
        if is_bc:
            tracks["bc"][s:e] = 1

    # BOT：非-BC 且段长达标的段起点；过去 lookback 窗内最近的有效发言者是对方
    for s, e in non_bc:
        if e - s < p.bot_min_f:
            continue
        lo = max(0, s - p.lookback_f)
        last_other = max(
            (min(oe, s) - 1 for os_, oe in valid_o if os_ < s and min(oe, s) - 1 >= lo),
            default=None,
        )
        last_self = max(
            (min(se_, s) - 1 for ss, se_ in valid_self if ss < s and (ss, se_) != (s, e) and min(se_, s) - 1 >= lo),
            default=None,
        )
        if last_other is not None and (last_self is None or last_other > last_self):
            tracks["bot"][s] = 1

    # EOT/HOLD：非-BC 段末，比较 4 s 内谁先恢复
    for _s, e in non_bc:
        ev = e - 1 if p.eot_at_last_speech else min(e, n - 1)
        next_self = min(
            (ss for ss, _ in resume_self if e <= ss <= e + p.lookahead_f), default=None
        )
        if p.other_ongoing_counts and any(os_ < e < oe for os_, oe in valid_o):
            next_other: int | None = e  # 对方此刻正处有效段中 → 立即接管
        else:
            next_other = min(
                (os_ for os_, _ in valid_o if e <= os_ <= e + p.lookahead_f), default=None
            )
        if next_other is None and next_self is None:
            continue
        if next_other is not None and (next_self is None or next_other < next_self):
            tracks["eot"][ev] = 1
        else:
            tracks["hold"][ev] = 1
    return tracks


def segments_to_frame_track(
    segs: list[Seg], n_frames: int, hz: float = FRAME_HZ, rule: str = "majority"
) -> np.ndarray:
    """秒域 VAD 段 → 12.5 Hz 二值轨。majority：帧内活跃占比 ≥ 0.5；any：有任何活跃。"""
    track = np.zeros(n_frames, dtype=np.int8)
    frame_len = 1.0 / hz
    for seg in segs:
        f0 = max(0, int(np.floor(seg.start * hz)))
        f1 = min(n_frames - 1, int(np.ceil(seg.end * hz)))
        for f in range(f0, f1 + 1):
            t0, t1 = f * frame_len, (f + 1) * frame_len
            overlap = max(0.0, min(seg.end, t1) - max(seg.start, t0))
            # 浮点容差：恰好半帧的覆盖按多数计入
            if (rule == "majority" and overlap >= 0.5 * frame_len - 1e-9) or (rule == "any" and overlap > 1e-12):
                track[f] = 1
    return track


def track_prf(pred: np.ndarray, gold: np.ndarray) -> dict:
    """帧级二值 P/R/F1（VAD 一致性层用）。"""
    n = min(len(pred), len(gold))
    p, g = np.asarray(pred[:n]) > 0, np.asarray(gold[:n]) > 0
    tp = int(np.sum(p & g))
    prec = tp / int(p.sum()) if p.sum() else (1.0 if not g.sum() else 0.0)
    rec = tp / int(g.sum()) if g.sum() else (1.0 if not p.sum() else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_pred": int(p.sum()), "n_gold": int(g.sum())}


def exact_mismatches(
    pred: dict[str, np.ndarray], gold: dict[str, np.ndarray]
) -> dict[str, int]:
    """逐帧不等的帧数（协议正确性层：目标全零）。"""
    out = {}
    for cls in OFFICIAL_CLASSES:
        n = min(len(pred[cls]), len(gold[cls]))
        out[cls] = int(np.sum((np.asarray(pred[cls][:n]) > 0) != (np.asarray(gold[cls][:n]) > 0)))
    return out


def param_grid() -> list[OfficialParams]:
    """协议收敛网格（64 组合）：帧取整 × 事件落点 × 重叠接管 × 本人恢复过滤。"""
    base = OfficialParams()
    combos = []
    for mv, bcm, bcg, last, ongoing, srv in product(
        (12, 13), (12, 13), (12, 13), (True, False), (True, False), (False, True)
    ):
        combos.append(
            replace(
                base,
                min_valid_f=mv,
                bot_min_f=mv,
                bc_max_f=bcm,
                bc_gap_f=bcg,
                eot_at_last_speech=last,
                other_ongoing_counts=ongoing,
                self_resume_valid_only=srv,
            )
        )
    return combos
