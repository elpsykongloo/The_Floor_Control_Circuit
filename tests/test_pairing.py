"""S1-A 确证臂配对单测。"""

from __future__ import annotations

from floor_circuit.stimuli.pairing import StimulusClip, greedy_duration_pairing


def clip(id_: str, dur: float, speech: float | None = None) -> StimulusClip:
    return StimulusClip(id=id_, duration_s=dur, speech_s=speech)


class TestGreedyPairing:
    def test_basic_matching_within_tol(self):
        completes = [clip("c1", 3.00), clip("c2", 4.00), clip("c3", 5.00)]
        incompletes = [clip("i1", 3.05), clip("i2", 3.95), clip("i3", 8.00)]
        pairs, stats = greedy_duration_pairing(completes, incompletes, tol_pct=5.0)
        matched = {(p["complete_id"], p["incomplete_id"]) for p in pairs}
        assert matched == {("c1", "i1"), ("c2", "i2")}
        assert stats["n_pairs"] == 2

    def test_forbid_same_id(self):
        completes = [clip("x", 3.00)]
        incompletes = [clip("x", 3.00)]
        pairs, _ = greedy_duration_pairing(completes, incompletes, tol_pct=5.0)
        assert pairs == []
        pairs2, _ = greedy_duration_pairing(completes, incompletes, tol_pct=5.0, forbid_same_id=False)
        assert len(pairs2) == 1

    def test_one_to_one_no_reuse(self):
        completes = [clip("c1", 3.00)]
        incompletes = [clip("i1", 3.00), clip("i2", 3.01)]
        pairs, _ = greedy_duration_pairing(completes, incompletes, tol_pct=5.0)
        assert len(pairs) == 1  # 单个 complete 只能配一次

    def test_speech_filter_drops(self):
        completes = [clip("c1", 3.00, speech=2.8), clip("c2", 4.00, speech=3.9)]
        incompletes = [clip("i1", 3.02, speech=1.0), clip("i2", 4.02, speech=3.8)]
        pairs, stats = greedy_duration_pairing(completes, incompletes, tol_pct=5.0)
        assert {(p["complete_id"], p["incomplete_id"]) for p in pairs} == {("c2", "i2")}
        assert stats["n_speech_dropped"] == 1
        assert "speech_diff_pct" in pairs[0]

    def test_maximum_matching_order(self):
        # 贪心最早可行：i(3.0) 应配 c(2.9) 而把 c(3.1) 留给 i(3.2)
        completes = [clip("c_a", 2.9), clip("c_b", 3.1)]
        incompletes = [clip("i_a", 3.0), clip("i_b", 3.2)]
        pairs, _ = greedy_duration_pairing(completes, incompletes, tol_pct=10.0)
        assert len(pairs) == 2
