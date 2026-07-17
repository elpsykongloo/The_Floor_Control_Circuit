"""T1–T5 探针标签生成（文档/00 §2.3），按目标模型决策时钟栅格化。

时钟约定：步 s 覆盖 [s·τ, (s+1)·τ)；步 s 的表征编码到 (s+1)·τ 为止的音频。
T1 的 δ 前瞻实现为整步平移：y[s] = "合格 ONSET_agent 落在步 s + δ/τ 内"，
即 δ=0 表示 onset 在当前步内，δ=240 ms（τ=80 ms）表示恰好 3 步之后。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from floor_circuit.events.detect import (
    ChannelContext,
    overlap_episodes,
    plain_risings,
    qualified_offsets,
    qualified_onsets,
)
from floor_circuit.schemas import Event, EventKind, State


def clock_rasterize(mask: np.ndarray, dt: float, step_s: float, n_steps: int) -> np.ndarray:
    """dt 栅格 → 时钟步栅格（步内任一 dt 帧活跃即活跃）。"""
    out = np.zeros(n_steps, dtype=bool)
    per = step_s / dt
    for s in range(n_steps):
        i0 = round(s * per)
        i1 = round((s + 1) * per)
        out[s] = mask[i0 : max(i1, i0 + 1)].any()
    return out


def _overlap_runs(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    both = a & b
    runs = []
    i = 0
    n = len(both)
    while i < n:
        if both[i]:
            j = i
            while j < n and both[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def agent_overlap_resolution(
    mask_agent: np.ndarray, mask_other: np.ndarray, dt: float, ev_cfg: dict
) -> list[dict]:
    """每个重叠连通段的 agent 侧结局：agent 在重叠开始后 ≤1.0 s 合格 OFFSET → yield；
    继续 ≥1.5 s → hold；否则 unresolved。与谁先闯入无关，纯 agent 视角（供 T5）。"""
    agent_offsets = qualified_offsets(mask_agent, dt, ev_cfg["offset_post_silence_s"])
    yield_max = float(ev_cfg["yield_max_s"])
    hold_min = float(ev_cfg["hold_min_s"])
    n = len(mask_agent)
    out = []
    for i0, i1 in _overlap_runs(mask_agent, mask_other):
        t0 = i0 * dt
        if any(t0 < t <= t0 + yield_max for t in agent_offsets):
            res = "yield"
        else:
            idx = round((t0 + hold_min) / dt)
            no_off = not any(t0 < t <= t0 + hold_min for t in agent_offsets)
            res = "hold" if (no_off and idx < n and mask_agent[idx]) else "unresolved"
        out.append({"i0": i0, "i1": i1, "t0": t0, "resolution": res})
    return out


def t5_states(
    mask_agent: np.ndarray,
    mask_other: np.ndarray,
    dt: float,
    step_s: float,
    n_steps: int,
    ev_cfg: dict,
) -> np.ndarray:
    a = clock_rasterize(mask_agent, dt, step_s, n_steps)
    b = clock_rasterize(mask_other, dt, step_s, n_steps)
    states = np.full(n_steps, State.GAP.value, dtype=np.int8)
    states[a & ~b] = State.SPEAK.value
    states[~a & b] = State.LISTEN.value
    runs = agent_overlap_resolution(mask_agent, mask_other, dt, ev_cfg)
    res_map = {
        "yield": State.OVERLAP_YIELD.value,
        "hold": State.OVERLAP_HOLD.value,
        "unresolved": State.OVERLAP_UNRESOLVED.value,
    }
    both = a & b
    per = step_s / dt
    for s in np.nonzero(both)[0]:
        mid = (s + 0.5) * per
        state = State.OVERLAP_UNRESOLVED.value
        for run in runs:
            if run["i0"] <= mid < run["i1"]:
                state = res_map[run["resolution"]]
                break
        states[s] = state
    return states


def build_labels(
    agent_ch: int,
    ctx_agent: ChannelContext,
    ctx_other: ChannelContext,
    events: list[Event],
    dt: float,
    step_s: float,
    total_dur: float,
    cfg: dict,
    t1_deltas_ms: list[int],
) -> pd.DataFrame:
    """单个 agent 视角的全部探针标签。返回列：
    [agent_channel, target, step, t, label, delta_ms]，t = 步末时刻 (s+1)·τ。"""
    ev_cfg = cfg["events"]
    n_steps = int(np.floor(total_dur / step_s))
    rows: list[dict] = []

    states = t5_states(ctx_agent.mask, ctx_other.mask, dt, step_s, n_steps, ev_cfg)
    for s in range(n_steps):
        rows.append(_row(agent_ch, "T5", s, step_s, int(states[s])))

    # T1：{LISTEN, GAP} 态步，合格 ONSET_agent 的整步平移
    onset_steps = np.zeros(n_steps, dtype=bool)
    for t in qualified_onsets(ctx_agent.mask, dt, ev_cfg["onset_pre_silence_s"]):
        s = int(np.floor(t / step_s))
        if 0 <= s < n_steps:
            onset_steps[s] = True
    eligible = np.isin(states, [State.LISTEN.value, State.GAP.value])
    for delta_ms in t1_deltas_ms:
        d = round((delta_ms / 1000.0) / step_s)
        shifted = np.zeros(n_steps, dtype=bool)
        if d < n_steps:
            shifted[: n_steps - d] = onset_steps[d:]
        for s in np.nonzero(eligible)[0]:
            if s + d >= n_steps:
                continue  # 尾部无监督信号，剔除
            rows.append(_row(agent_ch, "T1", int(s), step_s, int(shifted[s]), delta_ms))

    # T2：对方在 agent 说话中 onset 后、重叠持续期间的各步；标签 = agent 于 400 ms 内合格 OFFSET
    agent_offsets = qualified_offsets(ctx_agent.mask, dt, ev_cfg["offset_post_silence_s"])
    t2_win = float(ev_cfg["t2_offset_window_s"])
    for ep in overlap_episodes(ctx_agent.mask, ctx_other.mask, dt, ev_cfg):
        t_trig = ep["trigger_t"]
        t_stop = ep["resolve_t"] if ep["resolution"] == "yield" else t_trig + float(
            ev_cfg["yield_max_s"]
        )
        s0 = int(np.ceil(t_trig / step_s))
        s1 = min(n_steps, int(np.ceil(t_stop / step_s)))
        for s in range(s0, s1):
            t_e = (s + 1) * step_s
            label = int(any(t_e < t <= t_e + t2_win for t in agent_offsets))
            rows.append(_row(agent_ch, "T2", s, step_s, label))

    # T3：对方 onset + 2 时钟步处三分类 {0: backchannel, 1: 抢话, 2: 其他}
    other_ch = 1 - agent_ch
    bc_ts = {e.t for e in events if e.kind == EventKind.BC and e.channel == other_ch}
    grab_ts = {e.t for e in events if e.kind == EventKind.GRAB and e.channel == other_ch}
    for t_j in plain_risings(ctx_other.mask, dt):
        s = int(np.floor(t_j / step_s)) + 2
        if s >= n_steps:
            continue
        if _close_in(t_j, bc_ts, dt):
            label = 0
        elif _close_in(t_j, grab_ts, dt):
            label = 1
        else:
            label = 2
        rows.append(_row(agent_ch, "T3", s, step_s, label))

    # T4：对方 IPU/金标段末步二分类 {1: complete, 0: incomplete}
    turnend_ts = {e.t for e in events if e.kind == EventKind.TURNEND and e.channel == other_ch}
    if ctx_other.gold and ctx_other.turns is not None:
        ends = [
            (turn.end, 1 if (turn.label or "").lower() == "complete" else 0)
            for turn in ctx_other.turns
            if (turn.label or "").lower() in ("complete", "incomplete")
        ]
    else:
        ends = [(seg.end, int(_close_in(seg.end, turnend_ts, dt))) for seg in ctx_other.ipus]
    for t_e, label in ends:
        s = min(n_steps - 1, int(np.floor(max(t_e - 1e-9, 0.0) / step_s)))
        if s < 0:
            continue
        rows.append(_row(agent_ch, "T4", s, step_s, label))

    df = pd.DataFrame(rows, columns=["agent_channel", "target", "step", "t", "label", "delta_ms"])
    return df.sort_values(["target", "delta_ms", "step"], kind="stable").reset_index(drop=True)


def _row(ch: int, target: str, step: int, step_s: float, label: int, delta_ms=None) -> dict:
    return {
        "agent_channel": ch,
        "target": target,
        "step": int(step),
        "t": float((step + 1) * step_s),
        "label": int(label),
        "delta_ms": delta_ms,
    }


def _close_in(t: float, ts: set[float], tol: float) -> bool:
    return any(abs(t - x) <= tol + 1e-9 for x in ts)
