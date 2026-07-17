"""事件检测状态机（文档/00 §2.2，阈值冻结于 configs/events.yaml 的 events 节）。

约定：
- 一切检测在 IPU 级掩码（<180 ms 间隙已合并）与 dt 内部栅格上进行；
- Event.channel = 事件主体通道：ONSET/OFFSET/YIELD/HOLD 归 X（说话方），
  BC/GRAB/PAUSE/TURNEND 归 Y（来话方 / 被投射方）；
- 录音起点前与终点后按静默处理（窗口越界时以可见部分判定）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from floor_circuit.schemas import Event, EventKind, Seg, Turn


@dataclass
class ChannelContext:
    """单通道的检测输入。turns 可为 EN 启发式或 ZH 金标（gold=True 时用金标规则）。"""

    mask: np.ndarray  # IPU 级布尔栅格
    ipus: list[Seg]
    bc_flags: list[bool] = field(default_factory=list)
    turns: list[Turn] | None = None
    gold: bool = False


def _win_all_false(mask: np.ndarray, i0: int, i1: int) -> bool:
    i0, i1 = max(0, i0), min(len(mask), i1)
    return not mask[i0:i1].any()


def qualified_onsets(mask: np.ndarray, dt: float, pre_silence_s: float) -> list[float]:
    """VA 0→1 且此前静默 ≥ pre_silence_s 的时刻。"""
    k = round(pre_silence_s / dt)
    out = []
    prev = False
    for i, v in enumerate(mask):
        if v and not prev and _win_all_false(mask, i - k, i):
            out.append(i * dt)
        prev = bool(v)
    return out


def qualified_offsets(mask: np.ndarray, dt: float, post_silence_s: float) -> list[float]:
    """VA 1→0 且此后静默 ≥ post_silence_s 的时刻。"""
    k = round(post_silence_s / dt)
    n = len(mask)
    out = []
    for i in range(1, n + 1):
        was = mask[i - 1]
        now = mask[i] if i < n else False
        if was and not now and _win_all_false(mask, i, i + k):
            out.append(i * dt)
    return out


def plain_risings(mask: np.ndarray, dt: float) -> list[float]:
    """普通 VA 0→1 时刻（无 400 ms 前置静默要求）。"""
    prev = False
    out = []
    for i, v in enumerate(mask):
        if v and not prev:
            out.append(i * dt)
        prev = bool(v)
    return out


def overlap_episodes(
    mask_x: np.ndarray, mask_y: np.ndarray, dt: float, ev_cfg: dict
) -> list[dict]:
    """Y 在 X 说话中 onset 触发的重叠事件。

    resolution ∈ {"yield", "hold", "unresolved"}：
    - yield：X 在触发后 ≤ yield_max_s 内出现合格 OFFSET；
    - hold：yield 窗内无合格 OFFSET，且 X 在触发后 hold_min_s 处仍活跃、期间无合格 OFFSET；
    - unresolved：两者皆非（含录音尾部截断），不产生事件，仅供 T5 审计。
    """
    x_offsets = qualified_offsets(mask_x, dt, ev_cfg["offset_post_silence_s"])
    yield_max = float(ev_cfg["yield_max_s"])
    hold_min = float(ev_cfg["hold_min_s"])
    n = len(mask_x)
    episodes = []
    for t_j in plain_risings(mask_y, dt):
        j = round(t_j / dt)
        if j >= n or not mask_x[j]:
            continue  # 非重叠 onset
        off_in_yield = [t for t in x_offsets if t_j < t <= t_j + yield_max]
        if off_in_yield:
            episodes.append(
                {"trigger_t": t_j, "resolution": "yield", "resolve_t": off_in_yield[0]}
            )
            continue
        off_in_hold = [t for t in x_offsets if t_j < t <= t_j + hold_min]
        idx_hold = round((t_j + hold_min) / dt)
        if not off_in_hold and idx_hold < n and mask_x[idx_hold]:
            episodes.append(
                {"trigger_t": t_j, "resolution": "hold", "resolve_t": t_j + hold_min}
            )
        else:
            episodes.append({"trigger_t": t_j, "resolution": "unresolved", "resolve_t": None})
    return episodes


def _bc_events(ctx_y: ChannelContext, mask_x: np.ndarray, dt: float, ev_cfg: dict, y_ch: int):
    """BC_Y：Y 的 IPU ≤ bc_max_s、命中 backchannel 判据、X 的 VA 未中断。"""
    out = []
    n = len(mask_x)
    flags = ctx_y.bc_flags or [False] * len(ctx_y.ipus)
    for seg, bc in zip(ctx_y.ipus, flags, strict=True):
        if not bc or seg.dur > float(ev_cfg["bc_max_s"]):
            continue
        i0 = max(0, int(np.floor(seg.start / dt)))
        i1 = min(n, max(i0 + 1, int(np.ceil(seg.end / dt))))
        if mask_x[i0:i1].all():
            out.append(Event(EventKind.BC, y_ch, seg.start, t_end=seg.end))
    return out


def _pause_events(ctx_y: ChannelContext, dt: float, ev_cfg: dict, y_ch: int) -> list[Event]:
    """PAUSE_Y：turn 内静默 ∈ [pause_min_s, pause_max_s]。

    EN 启发式：turn 内相邻 IPU 间隙。ZH 金标：incomplete 段末静默（至该通道下次发声）。
    """
    lo, hi = float(ev_cfg["pause_min_s"]), float(ev_cfg["pause_max_s"])
    out = []
    if ctx_y.turns is None:
        return out
    if not ctx_y.gold:
        for turn in ctx_y.turns:
            for a, b in zip(turn.ipu_indices[:-1], turn.ipu_indices[1:], strict=False):
                gap0, gap1 = ctx_y.ipus[a].end, ctx_y.ipus[b].start
                if lo <= gap1 - gap0 <= hi:
                    out.append(Event(EventKind.PAUSE, y_ch, gap0, t_end=gap1))
    else:
        next_speech = sorted(s.start for s in ctx_y.ipus)
        for turn in ctx_y.turns:
            if (turn.label or "").lower() != "incomplete":
                continue
            following = [t for t in next_speech if t > turn.end + 1e-9]
            gap_end = following[0] if following else turn.end + hi
            if lo <= gap_end - turn.end <= hi:
                out.append(Event(EventKind.PAUSE, y_ch, turn.end, t_end=gap_end))
    return out


def _turnend_events(
    ctx_y: ChannelContext, mask_x: np.ndarray, dt: float, ev_cfg: dict, y_ch: int
) -> list[Event]:
    """TURNEND_Y。EN：IPU 末且 Y ≥ turnend_no_continue_s 无续说，或 X ≤ turnend_other_onset_s 内
    onset。ZH：complete 金标段末。"""
    out = []
    if ctx_y.gold and ctx_y.turns is not None:
        for turn in ctx_y.turns:
            if (turn.label or "").lower() == "complete":
                out.append(Event(EventKind.TURNEND, y_ch, turn.end, aux={"rule": "gold"}))
        return out
    no_cont = float(ev_cfg["turnend_no_continue_s"])
    other_win = float(ev_cfg["turnend_other_onset_s"])
    x_risings = plain_risings(mask_x, dt)
    for seg in ctx_y.ipus:
        t_e = seg.end
        y_continues = any(s.start > t_e and s.start <= t_e + no_cont for s in ctx_y.ipus)
        silence_rule = not y_continues
        onset_rule = any(t_e < t <= t_e + other_win for t in x_risings)
        if silence_rule or onset_rule:
            rule = "silence" if silence_rule else "other_onset"
            out.append(Event(EventKind.TURNEND, y_ch, t_e, aux={"rule": rule}))
    return out


def detect_all(ctx0: ChannelContext, ctx1: ChannelContext, dt: float, cfg: dict) -> list[Event]:
    """两通道对称检测全部事件（每通道各当一次 X 与一次 Y）。"""
    ev_cfg = cfg["events"]
    contexts = {0: ctx0, 1: ctx1}
    events: list[Event] = []
    for ch, ctx in contexts.items():
        for t in qualified_onsets(ctx.mask, dt, ev_cfg["onset_pre_silence_s"]):
            events.append(Event(EventKind.ONSET, ch, t))
        for t in qualified_offsets(ctx.mask, dt, ev_cfg["offset_post_silence_s"]):
            events.append(Event(EventKind.OFFSET, ch, t))
    for x_ch, y_ch in ((0, 1), (1, 0)):
        ctx_x, ctx_y = contexts[x_ch], contexts[y_ch]
        for ep in overlap_episodes(ctx_x.mask, ctx_y.mask, dt, ev_cfg):
            if ep["resolution"] == "yield":
                events.append(
                    Event(
                        EventKind.YIELD,
                        x_ch,
                        ep["resolve_t"],
                        aux={"trigger_t": ep["trigger_t"]},
                    )
                )
                events.append(
                    Event(
                        EventKind.GRAB, y_ch, ep["trigger_t"], aux={"yield_t": ep["resolve_t"]}
                    )
                )
            elif ep["resolution"] == "hold":
                events.append(
                    Event(EventKind.HOLD, x_ch, ep["trigger_t"], aux={"held_until": ep["resolve_t"]})
                )
        events.extend(_bc_events(ctx_y, ctx_x.mask, dt, ev_cfg, y_ch))
        events.extend(_pause_events(ctx_y, dt, ev_cfg, y_ch))
        events.extend(_turnend_events(ctx_y, ctx_x.mask, dt, ev_cfg, y_ch))
    events.sort(key=lambda e: (e.t, e.kind.value, e.channel))
    return events
