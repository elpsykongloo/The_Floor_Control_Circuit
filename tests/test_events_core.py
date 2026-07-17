"""事件层单测：IPU 合并、backchannel 判据、EN turn 启发式、事件状态机（§2.2 全事件）。"""

from __future__ import annotations

from floor_circuit.events.detect import (
    ChannelContext,
    detect_all,
    overlap_episodes,
    qualified_offsets,
    qualified_onsets,
)
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.turns import build_turns_en, is_backchannel
from floor_circuit.events.vad import mask_to_segments, rasterize
from floor_circuit.schemas import EventKind, Seg, Turn

DT = 0.01
TOTAL = 10.0


def ctx(segs: list[Seg], bc: list[bool] | None = None, turns=None, gold=False) -> ChannelContext:
    ipus = build_ipus(segs, 0.18)
    return ChannelContext(
        mask=rasterize(ipus, DT, TOTAL),
        ipus=ipus,
        bc_flags=bc if bc is not None else [False] * len(ipus),
        turns=turns,
        gold=gold,
    )


def kinds(events, kind: EventKind, channel: int | None = None):
    return [e for e in events if e.kind == kind and (channel is None or e.channel == channel)]


class TestIpu:
    def test_merge_small_gap(self):
        out = build_ipus([Seg(0.0, 1.0), Seg(1.1, 2.0)], 0.18)
        assert out == [Seg(0.0, 2.0)]

    def test_keep_large_gap(self):
        out = build_ipus([Seg(0.0, 1.0), Seg(1.3, 2.0)], 0.18)
        assert len(out) == 2

    def test_exact_threshold_not_merged(self):
        out = build_ipus([Seg(0.0, 1.0), Seg(1.18, 2.0)], 0.18)
        assert len(out) == 2  # 间隙 < 180 ms 才合并，等于不合并


class TestRaster:
    def test_roundtrip(self):
        segs = [Seg(0.5, 1.2), Seg(3.0, 3.4)]
        mask = rasterize(segs, DT, TOTAL)
        assert mask_to_segments(mask, DT) == [Seg(0.5, 1.2), Seg(3.0, 3.4)]


class TestBackchannel:
    def test_lexicon_hit(self, ev_cfg):
        lex = ev_cfg["backchannel_lexicon"]["en"]
        assert is_backchannel(Seg(0, 0.5), "Yeah!", lex, 1.0)
        assert is_backchannel(Seg(0, 0.9), "yeah yeah", lex, 1.0)
        assert is_backchannel(Seg(0, 0.5), "I see", lex, 1.0)
        assert not is_backchannel(Seg(0, 0.5), "no way", lex, 1.0)

    def test_zh_lexicon(self, ev_cfg):
        lex = ev_cfg["backchannel_lexicon"]["zh"]
        assert is_backchannel(Seg(0, 0.4), "嗯嗯", lex, 1.0)
        assert not is_backchannel(Seg(0, 0.4), "不对吧", lex, 1.0)

    def test_duration_gate(self, ev_cfg):
        lex = ev_cfg["backchannel_lexicon"]["en"]
        assert not is_backchannel(Seg(0, 1.5), "yeah", lex, 1.0)

    def test_no_transcript_proxy(self, ev_cfg):
        lex = ev_cfg["backchannel_lexicon"]["en"]
        assert is_backchannel(Seg(0, 0.5), None, lex, 1.0)


class TestTurnsEn:
    def test_merge_and_break(self):
        x = [Seg(1.0, 2.0), Seg(2.5, 4.0), Seg(6.0, 7.0)]
        turns = build_turns_en(x, [], [], 1.0)
        assert [(t.start, t.end) for t in turns] == [(1.0, 4.0), (6.0, 7.0)]

    def test_non_bc_other_breaks(self):
        x = [Seg(1.0, 2.0), Seg(2.5, 4.0)]
        y = [Seg(2.1, 2.4)]
        turns = build_turns_en(x, y, [False], 1.0)
        assert len(turns) == 2

    def test_bc_other_does_not_break(self):
        x = [Seg(1.0, 2.0), Seg(2.5, 4.0)]
        y = [Seg(2.1, 2.4)]
        turns = build_turns_en(x, y, [True], 1.0)
        assert len(turns) == 1


class TestQualifiedEdges:
    def test_onset_needs_pre_silence(self):
        mask = rasterize([Seg(0.5, 1.0), Seg(1.2, 2.0)], DT, TOTAL)
        onsets = qualified_onsets(mask, DT, 0.4)
        assert onsets == [0.5]  # 1.2 前仅 0.2 s 静默，不合格

    def test_offset_needs_post_silence(self):
        mask = rasterize([Seg(0.5, 1.0), Seg(1.2, 2.0)], DT, TOTAL)
        offsets = qualified_offsets(mask, DT, 0.4)
        assert offsets == [2.0]

    def test_start_of_recording_counts_as_silence(self):
        mask = rasterize([Seg(0.1, 1.0)], DT, TOTAL)
        assert qualified_onsets(mask, DT, 0.4) == [0.1]


class TestDetect:
    def test_yield_and_grab(self, ev_cfg):
        c0 = ctx([Seg(1.0, 3.5)])
        c1 = ctx([Seg(3.0, 6.0)])
        events = detect_all(c0, c1, DT, ev_cfg)
        y = kinds(events, EventKind.YIELD, 0)
        g = kinds(events, EventKind.GRAB, 1)
        assert len(y) == 1 and abs(y[0].t - 3.5) < 1e-6 and abs(y[0].aux["trigger_t"] - 3.0) < 1e-6
        assert len(g) == 1 and abs(g[0].t - 3.0) < 1e-6

    def test_hold(self, ev_cfg):
        c0 = ctx([Seg(1.0, 6.0)])
        c1 = ctx([Seg(3.0, 3.8)], bc=[False])
        events = detect_all(c0, c1, DT, ev_cfg)
        h = kinds(events, EventKind.HOLD, 0)
        assert len(h) == 1 and abs(h[0].t - 3.0) < 1e-6
        assert not kinds(events, EventKind.YIELD, 0)

    def test_bc_requires_x_uninterrupted(self, ev_cfg):
        c0 = ctx([Seg(1.0, 6.0)])
        c1 = ctx([Seg(3.0, 3.8)], bc=[True])
        events = detect_all(c0, c1, DT, ev_cfg)
        assert len(kinds(events, EventKind.BC, 1)) == 1
        # X 中断的情形不算 BC
        c0b = ctx([Seg(1.0, 3.2)])
        events_b = detect_all(c0b, c1, DT, ev_cfg)
        assert not kinds(events_b, EventKind.BC, 1)

    def test_pause_within_turn(self, ev_cfg):
        x_segs = [Seg(1.0, 2.0), Seg(2.5, 4.0)]
        ipus = build_ipus(x_segs, 0.18)
        turns = build_turns_en(ipus, [], [], 1.0)
        for t in turns:
            t.channel = 0
        c0 = ChannelContext(
            mask=rasterize(ipus, DT, TOTAL), ipus=ipus, bc_flags=[False] * len(ipus), turns=turns
        )
        c1 = ctx([])
        events = detect_all(c0, c1, DT, ev_cfg)
        p = kinds(events, EventKind.PAUSE, 0)
        assert len(p) == 1 and abs(p[0].t - 2.0) < 1e-6 and abs(p[0].t_end - 2.5) < 1e-6

    def test_turnend_en_silence_and_other_onset(self, ev_cfg):
        c0 = ctx([Seg(1.0, 2.0)])
        c1 = ctx([Seg(2.5, 4.0)])
        events = detect_all(c0, c1, DT, ev_cfg)
        te0 = kinds(events, EventKind.TURNEND, 0)
        assert len(te0) == 1 and te0[0].aux["rule"] in ("silence", "other_onset")

    def test_turnend_zh_gold(self, ev_cfg):
        gold = [
            Turn(channel=1, start=1.0, end=2.0, label="complete"),
            Turn(channel=1, start=3.0, end=4.0, label="incomplete"),
        ]
        c0 = ctx([])
        c1 = ctx([Seg(1.0, 2.0), Seg(3.0, 4.0), Seg(5.0, 6.0)], turns=gold, gold=True)
        events = detect_all(c0, c1, DT, ev_cfg)
        te = kinds(events, EventKind.TURNEND, 1)
        assert len(te) == 1 and abs(te[0].t - 2.0) < 1e-6 and te[0].aux["rule"] == "gold"
        p = kinds(events, EventKind.PAUSE, 1)  # incomplete 段末静默 4.0→5.0 = 1.0 s ∈ [0.3, 2.0]
        assert len(p) == 1 and abs(p[0].t - 4.0) < 1e-6

    def test_unresolved_overlap_no_event(self, ev_cfg):
        # X 在触发后 1.2 s 停（>1.0 yield 窗）且未持续到 1.5 s → 无 YIELD/HOLD
        c0 = ctx([Seg(1.0, 4.2)])
        c1 = ctx([Seg(3.0, 6.0)], bc=[False])
        eps = overlap_episodes(c0.mask, c1.mask, DT, ev_cfg["events"])
        assert eps[0]["resolution"] == "unresolved"
        events = detect_all(c0, c1, DT, ev_cfg)
        assert not kinds(events, EventKind.YIELD, 0) and not kinds(events, EventKind.HOLD, 0)
