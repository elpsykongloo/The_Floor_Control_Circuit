"""划分冻结 + 探针/统计/G1 裁决单测。"""

from __future__ import annotations

import functools
import operator

import numpy as np
import pytest

from floor_circuit.data.splits import CANDOR_RATIOS, freeze_split, load_split, write_split
from floor_circuit.probes.linear import downsample_negatives, fit_probe, score_sessions
from floor_circuit.probes.stats import (
    cluster_bootstrap_auc,
    g1_verdict,
    paired_advantage_bootstrap,
    pooled_metrics,
    shuffle_labels_within_session,
)


class TestSplits:
    def test_deterministic_and_disjoint(self):
        ids = [f"s{i:03d}" for i in range(100)]
        a = freeze_split(ids, CANDOR_RATIOS, seed=7)
        b = freeze_split(ids, CANDOR_RATIOS, seed=7)
        assert a == b
        assert freeze_split(ids, CANDOR_RATIOS, seed=8) != a
        all_ids = functools.reduce(operator.iadd, (v for v in a.values()), [])
        assert sorted(all_ids) == sorted(ids)
        assert len(a["probe_train"]) == 60 and len(a["probe_val"]) == 15 and len(a["causal_eval"]) == 25

    def test_write_load_and_freeze_protection(self, tmp_path):
        ids = [f"s{i}" for i in range(10)]
        splits = freeze_split(ids, {"train": 0.7, "eval": 0.3}, seed=1)
        path = tmp_path / "x.json"
        write_split(path, "x", splits, 1, {"train": 0.7, "eval": 0.3})
        payload = load_split(path)
        assert payload["counts"]["train"] == 7
        with pytest.raises(FileExistsError):
            write_split(path, "x", splits, 1, {"train": 0.7, "eval": 0.3})

    def test_tamper_detection(self, tmp_path):
        ids = [f"s{i}" for i in range(10)]
        splits = freeze_split(ids, {"train": 1.0}, seed=1)
        path = tmp_path / "y.json"
        write_split(path, "y", splits, 1, {"train": 1.0})
        text = path.read_text(encoding="utf-8").replace("s3", "s3_hacked")
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="sha256"):
            load_split(path)


def synth_sessions(n_sessions=12, n=300, d=8, effect=1.6, seed=0):
    """dim0 携带信号的可分数据；其余维为噪声。"""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_sessions):
        y = (rng.random(n) < 0.15).astype(np.int64)
        X = rng.normal(0, 1, (n, d)).astype(np.float32)
        X[:, 0] += effect * y
        data[f"sess{i:02d}"] = (X, y)
    return data


class TestProbeProtocol:
    def test_downsample_ratio(self):
        rng = np.random.default_rng(0)
        X = np.zeros((1000, 2), dtype=np.float32)
        y = np.zeros(1000, dtype=np.int64)
        y[:50] = 1
        _X2, y2 = downsample_negatives(X, y, 5, rng)
        assert y2.sum() == 50 and (y2 == 0).sum() == 250

    def test_probe_beats_noise_and_scores_sessions(self):
        data = synth_sessions()
        sids = sorted(data)
        train, evals = sids[:8], sids[8:]
        fit = fit_probe(data, train, evals, [0.01, 0.1, 1.0], seed=0)
        per = score_sessions(fit, data, evals)
        m = pooled_metrics(per)
        assert m["auc"] > 0.8 and m["n_sessions"] == 4

    def test_seeds_reproducible(self):
        data = synth_sessions()
        sids = sorted(data)
        f1 = fit_probe(data, sids[:8], sids[8:], [0.1], seed=3)
        f2 = fit_probe(data, sids[:8], sids[8:], [0.1], seed=3)
        assert np.allclose(f1.model.coef_, f2.model.coef_)


class TestStats:
    def make_per_session(self, auc_shift, seed=0, n_sessions=10):
        rng = np.random.default_rng(seed)
        per = {}
        for i in range(n_sessions):
            y = (rng.random(200) < 0.2).astype(np.int64)
            scores = rng.normal(0, 1, 200) + auc_shift * y
            per[f"s{i}"] = (y, scores)
        return per

    def test_bootstrap_ci_brackets_point(self):
        per = self.make_per_session(1.5)
        res = cluster_bootstrap_auc(per, n_boot=200, seed=0)
        assert res["ci_lo"] <= res["point"] <= res["ci_hi"]
        assert res["point"] > 0.75

    def test_paired_advantage_positive(self):
        strong = self.make_per_session(1.5, seed=1)
        weak = {k: (v[0], np.random.default_rng(2).normal(0, 1, len(v[0]))) for k, v in strong.items()}
        adv = paired_advantage_bootstrap(strong, {"noise": weak}, n_boot=200, seed=0)
        assert adv["advantage_point"] > 0.15 and adv["ci_lo"] > 0

    def test_shuffled_labels_near_chance(self):
        per = self.make_per_session(2.0, seed=3)
        shuffled = shuffle_labels_within_session(per, seed=0)
        assert abs(pooled_metrics(shuffled)["auc"] - 0.5) < 0.06

    def test_g1_branches(self):
        assert g1_verdict(0.08, 0.02, 0.05, 0.02) == "full_e1"
        assert g1_verdict(0.08, -0.01, 0.05, 0.02) == "backup_mve"  # 点估计过线但 CI 下界 ≤ 0
        assert g1_verdict(0.03, 0.01, 0.05, 0.02) == "backup_mve"
        assert g1_verdict(0.01, -0.02, 0.05, 0.02) == "n1"


class TestHazardFeatures:
    def test_shapes_and_duration(self):
        from floor_circuit.probes.baselines import hazard_features

        states = np.array([4, 4, 1, 1, 1, 0, 0, 4], dtype=np.int8)
        X = hazard_features(states, step_s=0.08)
        assert X.shape == (8, 8)
        assert X[1, 0] == np.float32(0.08) and X[4, 0] == np.float32(0.16)
        assert X[2, 0] == 0.0  # 切换步时长归零
