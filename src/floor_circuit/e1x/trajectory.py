"""X2：事件对齐的决策变量轨迹（探索性）。

把探针 logit 视为逐步决策变量：z[s] = w·(acts[s+1]−μ)/σ + b（行映射沿 #8）。
围绕 T4 锚点（对方 IPU 末步）按标签分组对齐平均，量化两类轨迹的分叉时刻，
并与 Mimi 决策变量的分叉时刻对比得到"内部证据领先量"。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.schemas import State


def aligned_matrix(
    series_by_role: dict[tuple[str, int], np.ndarray],
    anchors: list[tuple[str, int, int]],
    window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按锚点对齐切窗：返回 (matrix [n, 2w+1], session_index [n])；越界填 NaN。

    series_by_role[key][s] 必须已经是"标签步 s 的决策变量"（行映射由调用方完成）。
    anchors 元素为 (session_id, channel, step)。session_index 供会话级 bootstrap。
    """
    width = 2 * int(window_steps) + 1
    matrix = np.full((len(anchors), width), np.nan, dtype=np.float64)
    sessions = sorted({sid for sid, _ch, _s in anchors})
    sid_code = {sid: i for i, sid in enumerate(sessions)}
    session_index = np.empty(len(anchors), dtype=np.int64)
    for row, (sid, channel, step) in enumerate(anchors):
        series = series_by_role[(sid, channel)]
        lo = step - window_steps
        hi = step + window_steps + 1
        src_lo = max(lo, 0)
        src_hi = min(hi, len(series))
        matrix[row, src_lo - lo : src_hi - lo] = series[src_lo:src_hi]
        session_index[row] = sid_code[sid]
    return matrix, session_index


def group_mean_curves(
    matrix: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """两类（0/1）的逐偏移 NaN 均值曲线。"""
    labels = np.asarray(labels)
    return {
        "mean_label1": np.nanmean(matrix[labels == 1], axis=0),
        "mean_label0": np.nanmean(matrix[labels == 0], axis=0),
        "n_label1": int((labels == 1).sum()),
        "n_label0": int((labels == 0).sum()),
    }


def divergence_step(
    matrix: np.ndarray,
    labels: np.ndarray,
    session_index: np.ndarray,
    *,
    n_boot: int = 1000,
    min_consecutive: int = 3,
    seed: int = 20260723,
) -> dict:
    """两类均值差的会话级 bootstrap CI 首次持续排除 0 的最早偏移。

    返回 offsets 上的差值、CI 下/上界，以及 divergence_offset（相对锚点的步偏移，
    负值 = 在锚点之前已分叉；None = 从未持续显著）。
    """
    labels = np.asarray(labels)
    session_index = np.asarray(session_index)
    if not ((labels == 1).any() and (labels == 0).any()):
        raise ValueError("分叉分析需要两类锚点")
    width = matrix.shape[1]
    n_sessions = int(session_index.max()) + 1
    rng = np.random.default_rng(seed)
    point = np.nanmean(matrix[labels == 1], axis=0) - np.nanmean(matrix[labels == 0], axis=0)
    draws = rng.integers(0, n_sessions, size=(int(n_boot), n_sessions))
    samples = np.full((int(n_boot), width), np.nan, dtype=np.float64)
    # 会话重采样：按会话计数加权的两类均值差（NaN 感知）。
    ones = matrix.copy()
    ones[np.isnan(matrix)] = 0.0
    counts_valid = (~np.isnan(matrix)).astype(np.float64)
    per_session_sum: dict[int, dict[int, np.ndarray]] = {}
    per_session_cnt: dict[int, dict[int, np.ndarray]] = {}
    for cls in (0, 1):
        per_session_sum[cls] = {}
        per_session_cnt[cls] = {}
        for s in range(n_sessions):
            mask = (session_index == s) & (labels == cls)
            per_session_sum[cls][s] = ones[mask].sum(axis=0)
            per_session_cnt[cls][s] = counts_valid[mask].sum(axis=0)
    for b in range(int(n_boot)):
        weights = np.bincount(draws[b], minlength=n_sessions).astype(np.float64)
        curves = {}
        for cls in (0, 1):
            total = np.zeros(width)
            cnt = np.zeros(width)
            for s in range(n_sessions):
                if weights[s] == 0:
                    continue
                total += weights[s] * per_session_sum[cls][s]
                cnt += weights[s] * per_session_cnt[cls][s]
            curves[cls] = np.divide(total, cnt, out=np.full(width, np.nan), where=cnt > 0)
        samples[b] = curves[1] - curves[0]
    lo = np.nanpercentile(samples, 2.5, axis=0)
    hi = np.nanpercentile(samples, 97.5, axis=0)
    significant = (lo > 0) | (hi < 0)
    divergence = None
    run = 0
    for offset in range(width):
        run = run + 1 if significant[offset] else 0
        if run >= int(min_consecutive):
            divergence = offset - (min_consecutive - 1)
            break
    half = (width - 1) // 2
    return {
        "diff": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "divergence_offset": None if divergence is None else int(divergence - half),
    }


def next_state_step(states: np.ndarray, anchor: int, target_state: int, max_steps: int) -> int | None:
    """锚点之后（不含锚点步）首次进入目标状态的步距；超过 max_steps 返回 None。"""
    stop = min(len(states), anchor + int(max_steps) + 1)
    for step in range(anchor + 1, stop):
        if int(states[step]) == int(target_state):
            return step - anchor
    return None


def onset_steps_from_states(states: np.ndarray) -> np.ndarray:
    """状态序列中 agent 发声 onset 的步号：从非说话态进入 {SPEAK, OVERLAP_*}。"""
    speaking = np.isin(
        states,
        [
            State.SPEAK.value,
            State.OVERLAP_YIELD.value,
            State.OVERLAP_HOLD.value,
            State.OVERLAP_UNRESOLVED.value,
        ],
    )
    rising = speaking & ~np.concatenate([[False], speaking[:-1]])
    return np.flatnonzero(rising)
