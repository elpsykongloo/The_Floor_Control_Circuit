"""会话级事件管线：VAD 段/金标 → IPU → turn → 事件 → 标签，一次装配。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from floor_circuit.events.detect import ChannelContext, detect_all
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.labels import build_labels
from floor_circuit.events.turns import build_turns_en, is_backchannel
from floor_circuit.events.vad import rasterize
from floor_circuit.schemas import Event, Seg, Turn


@dataclass
class SessionChannel:
    """单通道输入：VAD 段（或由调用方从金标铺出的段）+ 可选转录与金标 turn。"""

    va_segs: list[Seg]
    transcript: list[tuple[Seg, str]] | None = None  # [(时间段, 文本)]
    gold_turns: list[Turn] | None = None  # ZH 金标；提供则 turn/TURNEND/PAUSE 走金标


def _ipu_texts(ipus: list[Seg], transcript: list[tuple[Seg, str]] | None) -> list[str | None]:
    if transcript is None:
        return [None] * len(ipus)
    out: list[str | None] = []
    for ipu in ipus:
        hits = [txt for seg, txt in transcript if seg.start < ipu.end and seg.end > ipu.start]
        out.append(" ".join(hits) if hits else None)
    return out


def process_session(
    ch0: SessionChannel,
    ch1: SessionChannel,
    total_dur: float,
    cfg: dict,
    lang: str = "en",
) -> tuple[list[Event], dict[int, ChannelContext], float]:
    """两通道 → 事件列表 + 检测上下文。dt 取 cfg['grid_dt_s']。"""
    dt = float(cfg["grid_dt_s"])
    lexicon = cfg["backchannel_lexicon"][lang]
    bc_max = float(cfg["turn_en"]["bc_max_s"])
    ctxs: dict[int, ChannelContext] = {}
    ipus_all, flags_all = {}, {}
    for ch, sc in ((0, ch0), (1, ch1)):
        ipus = build_ipus(sc.va_segs, float(cfg["ipu"]["merge_gap_s"]))
        texts = _ipu_texts(ipus, sc.transcript)
        flags = [is_backchannel(ipu, txt, lexicon, bc_max) for ipu, txt in zip(ipus, texts, strict=True)]
        ipus_all[ch], flags_all[ch] = ipus, flags
    for ch, sc in ((0, ch0), (1, ch1)):
        other = 1 - ch
        if sc.gold_turns is not None:
            turns, gold = sc.gold_turns, True
        else:
            turns = build_turns_en(
                ipus_all[ch], ipus_all[other], flags_all[other], float(cfg["turn_en"]["merge_gap_s"])
            )
            for t in turns:
                t.channel = ch
            gold = False
        ctxs[ch] = ChannelContext(
            mask=rasterize(ipus_all[ch], dt, total_dur),
            ipus=ipus_all[ch],
            bc_flags=flags_all[ch],
            turns=turns,
            gold=gold,
        )
    events = detect_all(ctxs[0], ctxs[1], dt, cfg)
    return events, ctxs, dt


def labels_both_roles(
    events: list[Event],
    ctxs: dict[int, ChannelContext],
    dt: float,
    total_dur: float,
    cfg: dict,
    step_s: float,
    t1_deltas_ms: list[int],
) -> pd.DataFrame:
    """双通道对称各作一次 agent，拼接 T1–T5 标签表。"""
    parts = []
    for agent_ch in (0, 1):
        parts.append(
            build_labels(
                agent_ch,
                ctxs[agent_ch],
                ctxs[1 - agent_ch],
                events,
                dt,
                step_s,
                total_dur,
                cfg,
                t1_deltas_ms,
            )
        )
    return pd.concat(parts, ignore_index=True)


def gold_turns_to_va(turns: list[Turn]) -> list[Seg]:
    """无独立 VAD 时，把金标段铺成 VA 段（ZH 解析器可用它直接驱动管线做纯金标分析）。"""
    return [Seg(t.start, t.end) for t in turns]


def masks_summary(ctxs: dict[int, ChannelContext], dt: float) -> dict:
    out = {}
    for ch, ctx in ctxs.items():
        out[f"ch{ch}"] = {
            "n_ipus": len(ctx.ipus),
            "speech_s": float(np.sum(ctx.mask) * dt),
            "n_turns": len(ctx.turns or []),
            "gold": ctx.gold,
        }
    return out
