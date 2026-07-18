"""G0 冻结门槛（PREREG #9）评估逻辑单测：等价带 + 尾部护栏 + 层1/层2 条件。"""

from __future__ import annotations

import numpy as np
import pytest

from floor_circuit.events.g0_gate import evaluate_g0_gate

GATE_CFG = {
    "layer1_exact_macro_f1": 1.0,
    "layer2_vad_f1_min": 0.90,
    "layer3_corpus_band": [0.48, 0.56],
    "layer3_session_p10_min": 0.30,
    "exploration_sessions": 20,
    "confirmation_expected_sessions": 118,
}


def _evaluate(
    *,
    layer1: float = 1.0,
    per_class: dict | None = None,
    layer2: dict | None = None,
    corpus: float = 0.52,
    sessions: list[float] | None = None,
) -> dict:
    return evaluate_g0_gate(
        layer1_macro_f1=layer1,
        layer1_per_class_f1=per_class or {"eot": 1.0, "hold": 1.0, "bot": 1.0, "bc": 1.0},
        layer2_f1_by_channel=layer2 or {"ch0": 0.92, "ch1": 0.91},
        layer3_corpus_macro_f1=corpus,
        layer3_session_macro_f1s=sessions if sessions is not None else [0.5] * 118,
        gate_cfg=GATE_CFG,
    )


class TestG0Gate:
    def test_all_conditions_pass(self):
        result = _evaluate()
        assert result["verdict"] == "pass"
        assert result["hard_failures"] == []
        assert result["conditions"]["layer3_corpus_band"]["status"] == "in_band"

    def test_below_band_fails(self):
        result = _evaluate(corpus=0.4799)
        assert result["verdict"] == "fail"
        assert "layer3_below_band" in result["hard_failures"]

    def test_above_band_is_red_flag_not_pass(self):
        """方案 2 上界红旗：异常偏高必须触发调查，不得自动通过。"""
        result = _evaluate(corpus=0.5601)
        assert result["verdict"] == "red_flag_investigate"
        assert result["hard_failures"] == []
        assert result["conditions"]["layer3_corpus_band"]["status"] == "above_band"

    def test_band_edges_are_inclusive(self):
        assert _evaluate(corpus=0.48)["verdict"] == "pass"
        assert _evaluate(corpus=0.56)["verdict"] == "pass"

    def test_session_p10_tail_guard(self):
        # 105 个 0.55 + 13 个 0.05：linear 分位下 P10 落在坏尾内（0.05 < 0.30）
        sessions = [0.55] * 105 + [0.05] * 13
        p10 = float(np.percentile(np.asarray(sessions), 10, method="linear"))
        result = _evaluate(sessions=sessions)
        assert result["conditions"]["layer3_session_p10"]["p10"] == pytest.approx(p10)
        assert result["verdict"] == "fail"
        assert "layer3_session_p10" in result["hard_failures"]

    def test_layer1_inexact_fails_even_if_macro_rounds_to_one(self):
        result = _evaluate(per_class={"eot": 1.0, "hold": 1.0, "bot": 1.0, "bc": 0.9999})
        assert result["verdict"] == "fail"
        assert "layer1_exact" in result["hard_failures"]

    def test_layer2_single_channel_below_min_fails(self):
        result = _evaluate(layer2={"ch0": 0.95, "ch1": 0.8999})
        assert result["verdict"] == "fail"
        assert "layer2_vad_f1" in result["hard_failures"]

    def test_hard_failure_dominates_above_band(self):
        """硬失败与上界红旗并存时，裁决必须是 fail。"""
        result = _evaluate(corpus=0.5601, layer2={"ch0": 0.5, "ch1": 0.5})
        assert result["verdict"] == "fail"

    def test_empty_sessions_rejected(self):
        with pytest.raises(ValueError, match="P10"):
            _evaluate(sessions=[])
