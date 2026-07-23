"""X5：T2 视野扫描（步栅变体）与 T4 错误解剖（分桶优势）。

T2h 步栅变体的诚实声明：正式 T2 标签用 dt=10 ms 栅格上的合格 OFFSET
（后继静默 ≥400 ms）；本变体在 80 ms 决策步栅上以"agent 状态离开发声态"
近似，属探索性描述。h=5（=400 ms）档与正式标签的逐行一致率随结果登记。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.schemas import State

_AGENT_SPEAKING = (
    State.SPEAK.value,
    State.OVERLAP_YIELD.value,
    State.OVERLAP_HOLD.value,
    State.OVERLAP_UNRESOLVED.value,
)


def t2_horizon_labels(states: np.ndarray, anchor_steps: np.ndarray, horizon_steps: int) -> np.ndarray:
    """步栅 T2 变体：锚点后 ≤h 步内 agent 状态离开发声态 → 1，否则 0。

    锚点步本身处于重叠（agent 发声中），从 anchor+1 起查 h 步。序列尾部不足
    h 步的锚点直接硬失败——调用方应先按 usable 步数过滤。
    """
    states = np.asarray(states)
    out = np.zeros(len(anchor_steps), dtype=np.int64)
    for i, anchor in enumerate(np.asarray(anchor_steps, dtype=np.int64)):
        stop = anchor + int(horizon_steps) + 1
        if stop > len(states):
            raise ValueError(f"锚点 {int(anchor)} 之后不足 {horizon_steps} 步状态")
        window = states[anchor + 1 : stop]
        out[i] = int(bool((~np.isin(window, _AGENT_SPEAKING)).any()))
    return out


def label_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError("一致率要求同形非空数组")
    return float((a == b).mean())


# ---------------------------------------------------------------------------
# T4 错误解剖：行级桶掩码 + 桶内 AUC
# ---------------------------------------------------------------------------


def f0_slope(acoustic: np.ndarray, anchor: int, tail_steps: int, f0_column: int = 1) -> float | None:
    """锚点前 tail_steps 步（含锚点）内 voiced 帧 F0 的一阶斜率；voiced <3 帧返回 None。"""
    lo = max(0, int(anchor) - int(tail_steps) + 1)
    segment = np.asarray(acoustic[lo : int(anchor) + 1, f0_column], dtype=np.float64)
    voiced = np.flatnonzero(segment > 0)
    if len(voiced) < 3:
        return None
    coeffs = np.polyfit(voiced.astype(np.float64), segment[voiced], deg=1)
    return float(coeffs[0])


def tercile_buckets(values: list[float | None]) -> tuple[np.ndarray, dict]:
    """把连续值分为三分位桶 {0,1,2}；None → −1（无效桶）。返回 (bucket, 边界)。"""
    array = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    valid = ~np.isnan(array)
    if valid.sum() < 9:
        raise ValueError("三分位分桶需要至少 9 个有效值")
    q1, q2 = np.percentile(array[valid], [100 / 3, 200 / 3])
    bucket = np.full(len(array), -1, dtype=np.int64)
    bucket[valid & (array <= q1)] = 0
    bucket[valid & (array > q1) & (array <= q2)] = 1
    bucket[valid & (array > q2)] = 2
    return bucket, {"q1": float(q1), "q2": float(q2), "n_valid": int(valid.sum())}


def filter_cells_by_mask(
    cells: dict[str, tuple[np.ndarray, np.ndarray]],
    row_masks: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """按逐会话行掩码过滤格子分数；长度不匹配硬失败（顺序契约破坏的哨兵）。"""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid, (y, p) in cells.items():
        if sid not in row_masks:
            raise ValueError(f"会话 {sid} 缺少行掩码")
        mask = np.asarray(row_masks[sid], dtype=bool)
        if len(mask) != len(y):
            raise ValueError(f"会话 {sid} 行掩码长度 {len(mask)} ≠ 格子行数 {len(y)}")
        if mask.any():
            out[sid] = (np.asarray(y)[mask], np.asarray(p)[mask])
    return out
