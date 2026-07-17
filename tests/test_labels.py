"""T1–T5 标签生成单测（Moshi 时钟 τ=80 ms）。"""

from __future__ import annotations

import numpy as np

from floor_circuit.events.detect import ChannelContext, detect_all
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.labels import build_labels, t5_states
from floor_circuit.events.vad import rasterize
from floor_circuit.schemas import Seg, State

DT = 0.01
STEP = 0.08
TOTAL = 12.0


def make_ctx(segs, bc=None, turns=None, gold=False):
    ipus = build_ipus(segs, 0.18)
    return ChannelContext(
        mask=rasterize(ipus, DT, TOTAL),
        ipus=ipus,
        bc_flags=bc if bc is not None else [False] * len(ipus),
        turns=turns,
        gold=gold,
    )


def labels_for(ev_cfg, agent_segs, other_segs, deltas=(240,), **kw):
    ca, co = make_ctx(agent_segs, **kw.get("agent_kw", {})), make_ctx(other_segs, **kw.get("other_kw", {}))
    events = detect_all(ca, co, DT, ev_cfg)
    return build_labels(0, ca, co, events, DT, STEP, TOTAL, ev_cfg, list(deltas)), events


class TestT1:
    def test_delta_shift_is_three_steps(self, ev_cfg):
        df, _ = labels_for(ev_cfg, [Seg(4.0, 6.0)], [])
        t1 = df[df["target"] == "T1"].set_index("step")["label"]
        onset_step = int(np.floor(4.0 / STEP))  # 50
        assert t1.loc[onset_step - 3] == 1  # δ=240ms = 3 步
        assert t1.loc[onset_step - 4] == 0
        assert onset_step - 2 not in t1.index or t1.loc[onset_step - 2] == 0

    def test_only_listen_gap_steps(self, ev_cfg):
        df, _ = labels_for(ev_cfg, [Seg(4.0, 6.0)], [])
        t1_steps = set(df[df["target"] == "T1"]["step"])
        speak_step = int(np.floor(5.0 / STEP))
        assert speak_step not in t1_steps  # SPEAK 态不入 T1


class TestT5:
    def test_states(self, ev_cfg):
        ca, co = make_ctx([Seg(1.0, 3.5)]), make_ctx([Seg(3.0, 6.0)])
        states = t5_states(ca.mask, co.mask, DT, STEP, int(TOTAL / STEP), ev_cfg["events"])
        assert states[int(2.0 / STEP)] == State.SPEAK.value
        assert states[int(4.5 / STEP)] == State.LISTEN.value
        assert states[int(8.0 / STEP)] == State.GAP.value
        assert states[int(3.2 / STEP)] == State.OVERLAP_YIELD.value

    def test_hold_state(self, ev_cfg):
        ca, co = make_ctx([Seg(1.0, 6.0)]), make_ctx([Seg(3.0, 3.8)])
        states = t5_states(ca.mask, co.mask, DT, STEP, int(TOTAL / STEP), ev_cfg["events"])
        assert states[int(3.4 / STEP)] == State.OVERLAP_HOLD.value


class TestT2:
    def test_yield_window_positive(self, ev_cfg):
        df, _ = labels_for(ev_cfg, [Seg(1.0, 3.5)], [Seg(3.0, 6.0)])
        t2 = df[df["target"] == "T2"].set_index("step")["label"]
        assert len(t2) >= 1
        # OFFSET 在 3.5：步 t_e ∈ (3.1, 3.5) 内的步应标 1（3.5 - t_e ≤ 0.4）
        pos_steps = [s for s, v in t2.items() if v == 1]
        assert pos_steps, f"T2 应有正例，实际 {dict(t2)}"
        for s in pos_steps:
            t_e = (s + 1) * STEP
            assert t_e < 3.5 <= t_e + 0.4 + 1e-9


class TestT3:
    def test_three_way(self, ev_cfg):
        # bc 来话：other 短 IPU 落在 agent 说话中
        df_bc, _ = labels_for(
            ev_cfg, [Seg(1.0, 6.0)], [Seg(3.0, 3.8)], other_kw={"bc": [True]}
        )
        t3 = df_bc[df_bc["target"] == "T3"]
        assert list(t3["label"]) == [0]
        # 抢话：agent 让位
        df_grab, _ = labels_for(ev_cfg, [Seg(1.0, 3.5)], [Seg(3.0, 6.0)])
        t3g = df_grab[df_grab["target"] == "T3"]
        assert list(t3g["label"]) == [1]
        # 其他：安静期来话
        df_other, _ = labels_for(ev_cfg, [], [Seg(3.0, 6.0)])
        t3o = df_other[df_other["target"] == "T3"]
        assert list(t3o["label"]) == [2]

    def test_step_position(self, ev_cfg):
        df, _ = labels_for(ev_cfg, [], [Seg(3.0, 6.0)])
        t3 = df[df["target"] == "T3"]
        assert int(t3.iloc[0]["step"]) == int(np.floor(3.0 / STEP)) + 2


class TestT4:
    def test_en_heuristic(self, ev_cfg):
        # other 两段：2.0 结束后 0.5s 续说（incomplete），4.0 结束后长静默（complete）
        df, _ = labels_for(ev_cfg, [], [Seg(1.0, 2.0), Seg(2.5, 4.0)])
        t4 = df[df["target"] == "T4"].sort_values("step")
        assert list(t4["label"]) == [0, 1]

    def test_zh_gold(self, ev_cfg):
        from floor_circuit.schemas import Turn

        gold = [
            Turn(channel=1, start=1.0, end=2.0, label="complete"),
            Turn(channel=1, start=3.0, end=4.0, label="incomplete"),
            Turn(channel=1, start=5.0, end=5.4, label="backchannel"),
        ]
        df, _ = labels_for(
            ev_cfg, [], [Seg(1.0, 2.0), Seg(3.0, 4.0), Seg(5.0, 5.4)],
            other_kw={"turns": gold, "gold": True},
        )
        t4 = df[df["target"] == "T4"].sort_values("step")
        assert list(t4["label"]) == [1, 0]  # backchannel/wait 段不入 T4
