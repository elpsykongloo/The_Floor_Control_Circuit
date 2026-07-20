"""E1 基础件单测：冻结会话切片、G2 判据、G0 train roster 抽样与门槛改写契约。"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.e1.g2 import evaluate_g2, pairwise_abs_cosines, top_layers_by_auc
from floor_circuit.e1.sets import E1_N_TRAIN, e1_sessions

REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload(n_train: int = 994, n_val: int = 248) -> dict:
    return {
        "splits": {
            "probe_train": [f"tr-{index:04d}" for index in range(n_train)],
            "probe_val": [f"va-{index:04d}" for index in range(n_val)],
            "causal_eval": [f"ce-{index:04d}" for index in range(414)],
        }
    }


class TestE1Sessions:
    def test_frozen_slices(self):
        sets = e1_sessions(_payload())
        assert len(sets.train) == E1_N_TRAIN == 400
        assert len(sets.eval) == 100
        assert len(sets.mve_hist) == 40
        assert sets.n_reserve == 108
        assert sets.train[0] == "tr-0000" and sets.train[-1] == "tr-0399"
        assert sets.eval[0] == "va-0040" and sets.eval[-1] == "va-0139"  # 跳过 MVE 历史 40
        assert sets.mve_hist == tuple(f"va-{index:04d}" for index in range(40))
        assert not set(sets.eval) & set(sets.mve_hist)
        assert not set(sets.train) & set(sets.eval)

    def test_insufficient_pools_fail(self):
        with pytest.raises(ValueError, match="probe_train"):
            e1_sessions(_payload(n_train=399))
        with pytest.raises(ValueError, match="probe_val"):
            e1_sessions(_payload(n_val=139))

    def test_broken_split_overlap_fails(self):
        payload = _payload()
        payload["splits"]["probe_val"][41] = payload["splits"]["probe_train"][0]
        with pytest.raises(ValueError, match="重叠"):
            e1_sessions(payload)


class TestG2:
    @staticmethod
    def _aucs(order: list[int]) -> dict[int, float]:
        return {layer: 0.9 - 0.01 * rank for rank, layer in enumerate(order)}

    def test_top_layers_tie_breaks_to_smaller_layer(self):
        assert top_layers_by_auc({4: 0.8, 12: 0.8, 20: 0.7, 28: 0.9}) == [28, 4, 12]

    def test_pass_case(self):
        result = evaluate_g2(
            auc_by_seed_layer={
                0: self._aucs([20, 28, 12, 4]),
                1: self._aucs([28, 20, 12, 4]),
                2: self._aucs([20, 12, 28, 4]),
            },
            effective_rank=8,
            direction_cosines={"0-1": 0.92, "0-2": 0.88, "1-2": 0.85},
            g2_cfg={"top3_overlap_min": 2, "effective_rank_max": 16, "direction_cosine_min": 0.8},
        )
        assert result["verdict"] == "pass"
        assert result["conditions"]["top3_overlap"]["overlap"] == [12, 20, 28]

    def test_each_condition_can_fail(self):
        base = dict(
            auc_by_seed_layer={
                0: self._aucs([20, 28, 12, 4]),
                1: self._aucs([28, 20, 12, 4]),
                2: self._aucs([20, 12, 28, 4]),
            },
            effective_rank=8,
            direction_cosines={"0-1": 0.92, "0-2": 0.88, "1-2": 0.85},
            g2_cfg={"top3_overlap_min": 2, "effective_rank_max": 16, "direction_cosine_min": 0.8},
        )
        overlap_broken = dict(base)
        overlap_broken["auc_by_seed_layer"] = {
            0: self._aucs([4, 12, 20, 28]),
            1: self._aucs([20, 28, 4, 12]),
            2: self._aucs([28, 20, 12, 4]),
        }
        assert evaluate_g2(**overlap_broken)["verdict"] == "fail"  # 交集 {20} 只有 1 层
        rank_broken = dict(base, effective_rank=17)
        assert evaluate_g2(**rank_broken)["verdict"] == "fail"
        cosine_broken = dict(base, direction_cosines={"0-1": 0.92, "0-2": 0.79, "1-2": 0.85})
        assert evaluate_g2(**cosine_broken)["verdict"] == "fail"

    def test_pairwise_cosines_are_sign_invariant(self):
        rng = np.random.default_rng(0)
        vector = rng.normal(size=16)
        cosines = pairwise_abs_cosines({0: vector, 1: -vector, 2: vector * 2.0})
        assert all(np.isclose(value, 1.0) for value in cosines.values())
        with pytest.raises(ValueError, match="方向向量"):
            pairwise_abs_cosines({0: vector, 1: np.zeros_like(vector)})


class TestTrainRosterSampling:
    def setup_method(self):
        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp1_g0_train_roster_test", scripts / "wp1_g0_train_roster.py"
        )
        assert spec and spec.loader
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_deterministic_sorted_sample(self):
        candidates = [f"s{index:04d}" for index in range(1986)]
        first = self.mod.sample_roster(list(reversed(candidates)), seed=20260717, k=300)
        second = self.mod.sample_roster(candidates, seed=20260717, k=300)
        assert first == second  # 输入顺序无关（内部先排序）
        assert first == sorted(first) and len(set(first)) == 300
        different = self.mod.sample_roster(candidates, seed=1, k=300)
        assert different != first

    def test_insufficient_pool_fails(self):
        with pytest.raises(ValueError, match="不足"):
            self.mod.sample_roster(["a", "b"], seed=0, k=3)
        with pytest.raises(ValueError, match="重复"):
            self.mod.sample_roster(["a", "a", "b"], seed=0, k=2)


class TestGateRewriteContract:
    """--derive-gate --apply 的正则必须恰好命中 events.yaml gate 两行（文件格式契约）。"""

    def test_patterns_match_exactly_once(self):
        text = (REPO_ROOT / "configs" / "events.yaml").read_text(encoding="utf-8")
        assert len(re.findall(r"layer3_corpus_band: \[[^\]]*\]", text)) == 1
        assert len(re.findall(r"layer3_session_p10_min: [0-9.]+", text)) == 1
        assert len(re.findall(r"decoded_vad_threshold: 0\.4", text)) == 1
