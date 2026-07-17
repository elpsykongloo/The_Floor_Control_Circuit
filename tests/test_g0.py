"""G0 映射与评分单测。"""

from __future__ import annotations

import numpy as np

from floor_circuit.events.g0 import events_to_frames, f1_report
from floor_circuit.schemas import Event, EventKind

HZ = 12.5
MAPPING = {"eot": "TURNEND", "bot": "ONSET", "bc": "BC", "default": "hold"}


def test_events_to_frames_mapping():
    events = [
        Event(EventKind.ONSET, 0, 1.0),
        Event(EventKind.BC, 0, 2.0, t_end=2.4),
        Event(EventKind.TURNEND, 0, 4.0),
        Event(EventKind.ONSET, 1, 5.0),  # 其他通道不应写入
    ]
    frames = events_to_frames(events, int(8 * HZ), HZ, MAPPING, channel=0)
    assert frames[round(1.0 * HZ)] == "bot"
    assert frames[round(4.0 * HZ)] == "eot"
    bc_span = frames[int(np.floor(2.0 * HZ)) : int(np.ceil(2.4 * HZ))]
    assert all(f == "bc" for f in bc_span)
    assert frames[round(5.0 * HZ)] == "hold"
    assert frames[0] == "hold"


def test_f1_perfect():
    events = [Event(EventKind.ONSET, 0, 1.0), Event(EventKind.TURNEND, 0, 4.0)]
    frames = events_to_frames(events, 100, HZ, MAPPING, channel=0)
    rep = f1_report(frames, frames.copy(), tolerance_frames=2)
    assert rep["macro_f1"] == 1.0


def test_f1_tolerance_for_sparse():
    gold = np.array(["hold"] * 100, dtype=object)
    pred = gold.copy()
    gold[50] = "eot"
    pred[52] = "eot"  # 偏移 2 帧，容差内
    rep = f1_report(pred, gold, tolerance_frames=2)
    assert rep["per_class"]["eot"]["f1"] == 1.0
    pred2 = gold.copy()
    pred2[50] = "hold"
    pred2[54] = "eot"  # 偏移 4 帧，容差外
    rep2 = f1_report(pred2, gold, tolerance_frames=2)
    assert rep2["per_class"]["eot"]["f1"] == 0.0


def test_f1_dense_hold():
    gold = np.array(["hold"] * 10 + ["bc"] * 5, dtype=object)
    pred = np.array(["hold"] * 12 + ["bc"] * 3, dtype=object)
    rep = f1_report(pred, gold, tolerance_frames=2)
    assert rep["per_class"]["bc"]["recall"] == 3 / 5
    assert rep["per_class"]["hold"]["recall"] == 1.0
    assert 0 < rep["macro_f1"] < 1
