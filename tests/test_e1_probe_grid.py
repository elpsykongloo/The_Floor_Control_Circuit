"""E1 探针网格（PREREG #18）护栏：训练器 sklearn 等价、抽样、多分类指标、装配对齐。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg

REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE_CFG = {
    "c_grid": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "seeds": [0, 1, 2],
    "inner_val_sessions": 80,
    "seed_subsample_pool": [80, 400],
    "seed_subsample_n": 288,
    "neg_ratio_t1": 5,
    "t5_step_stride": 4,
}


class TestTrainerParity:
    def _make_binary(self, n=600, d=12, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d)).astype(np.float32)
        w = rng.standard_normal(d)
        y = (x @ w + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
        return x, y

    @pytest.mark.parametrize("c_value", [0.001, 0.1, 1.0])
    def test_torch_matches_sklearn_binary(self, c_value):
        x, y = self._make_binary()
        x_eval, y_eval = self._make_binary(seed=7)
        probe = pg.fit_linear_probe(x, y, 2, c_value, device="cpu")
        auc_torch = pg.primary_metric(y_eval, probe.predict_proba(x_eval), 2)
        weight_ref, prob_ref = pg.sklearn_reference_fit(x, y, c_value, seed=0)
        auc_ref = pg.primary_metric(y_eval, prob_ref(x_eval), 2)
        assert abs(auc_torch - auc_ref) <= 1e-3
        cosine = abs(
            float(
                np.dot(probe.direction(), weight_ref / np.linalg.norm(weight_ref))
            )
        )
        assert cosine >= 0.999

    def test_multinomial_beats_chance_and_probs_normalize(self):
        rng = np.random.default_rng(3)
        centers = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64)
        y = rng.integers(0, 3, 900)
        x = (centers[y] + rng.standard_normal((900, 3))).astype(np.float32)
        probe = pg.fit_linear_probe(x, y, 3, 1.0, device="cpu")
        probs = probe.predict_proba(x)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        auc, detail = pg.macro_ovr_auc(y, probs, 3)
        assert auc > 0.9 and detail["n_classes_present"] == 3

    def test_macro_auc_skips_missing_class(self):
        y = np.array([0, 0, 1, 1])
        probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
        auc, detail = pg.macro_ovr_auc(y, probs, 3)
        assert detail["per_class_auc"]["2"] is None and auc == 1.0

    def test_binary_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, 500)
        scores = rng.standard_normal(500)
        scores[y == 1] += 0.4
        assert abs(pg._binary_auc(y, scores) - roc_auc_score(y, scores)) < 1e-12


class TestSampling:
    def test_seed_pool_composition(self):
        sessions = [f"s{i:04d}" for i in range(400)]
        pool0 = g.seed_train_sessions(sessions, PROBE_CFG, 0)
        pool1 = g.seed_train_sessions(sessions, PROBE_CFG, 1)
        assert pool0[:80] == sessions[:80] == pool1[:80]
        assert len(pool0) == 80 + 288 == len(set(pool0))
        assert pool0 != pool1  # 种子扰动生效
        assert set(pool0[80:]) <= set(sessions[80:400])
        assert pool0 == g.seed_train_sessions(sessions, PROBE_CFG, 0)  # 确定性

    def test_expand_specs_frozen_ten(self):
        specs = g.expand_specs(PROBE_CFG)
        names = [s.name for s in specs]
        assert names == [
            "T1_d0", "T1_d80", "T1_d160", "T1_d240", "T1_d400", "T1_d800",
            "T2", "T3", "T4", "T5",
        ]
        assert [s.n_classes for s in specs] == [2] * 7 + [3, 2, 5]

    def test_build_rows_t1_downsample_and_t5_stride(self, tmp_path):
        import pandas as pd

        rows = []
        for step in range(100):
            t_end = 0.08 * (step + 1)
            rows.append(
                {"agent_channel": 0, "target": "T5", "step": step, "t": t_end, "label": step % 5, "delta_ms": None}
            )
            rows.append(
                {
                    "agent_channel": 0,
                    "target": "T1",
                    "step": step,
                    "t": t_end,
                    "label": int(step % 25 == 0),
                    "delta_ms": 400,
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_parquet(tmp_path / "sess.parquet")
        n_steps = {("sess", 0): 100, ("sess", 1): 100}
        spec_t1 = next(s for s in g.expand_specs(PROBE_CFG) if s.name == "T1_d400")
        got = g.build_rows(tmp_path, ["sess"], n_steps, spec_t1, PROBE_CFG, 0, downsample=True)
        y = np.concatenate([r.labels for r in got])
        n_pos = int((y == 1).sum())
        assert n_pos >= 1 and (y == 0).sum() <= 5 * n_pos
        spec_t5 = next(s for s in g.expand_specs(PROBE_CFG) if s.name == "T5")
        got5 = g.build_rows(tmp_path, ["sess"], n_steps, spec_t5, PROBE_CFG, 0, downsample=False)
        steps5 = np.concatenate([r.steps for r in got5])
        assert (steps5 % 4 == 0).all() and steps5.max() < g.usable_label_steps(100)


class TestAssembly:
    def test_alignment_row_mapping(self):
        store = {("sess", 0): np.arange(50, dtype=np.float16).reshape(10, 5)}
        roles = [g.RoleRows("sess", 0, np.array([2, 4]), np.array([0, 1]))]
        x, y, _sid = g.assemble(roles, "acts", store)
        # acts 读行 s+1（PREREG #8）
        assert np.array_equal(np.asarray(x[:, 0], dtype=np.float64), [15.0, 25.0])
        assert np.array_equal(y, [0, 1])

    def test_mimi_concat_self_other(self):
        store = {
            ("sess", 0): np.ones((6, 2), dtype=np.float16),
            ("sess", 1): np.full((6, 2), 2.0, dtype=np.float16),
        }
        roles = [g.RoleRows("sess", 0, np.array([1]), np.array([1]))]
        x, _y, _sid = g.assemble(roles, "mimi", store)
        assert x.shape == (1, 4)
        assert np.array_equal(np.asarray(x[0], np.float64), [1, 1, 2, 2])

    def test_t5_state_array_requires_full_coverage(self, tmp_path):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {"agent_channel": 0, "target": "T5", "step": s, "t": 0.0, "label": 1, "delta_ms": None}
                for s in range(4)
            ]
        )
        states = g.t5_state_array(frame, 0, 4)
        assert np.array_equal(states, [1, 1, 1, 1])
        with pytest.raises(ValueError, match="覆盖缺"):
            g.t5_state_array(frame, 0, 6)


class TestEffectiveRank:
    def test_low_rank_signal_detected(self):
        rng = np.random.default_rng(5)
        direction = np.zeros(16)
        direction[0] = 1.0
        y = rng.integers(0, 2, 800)
        x = rng.standard_normal((800, 16)) * 0.3
        x[:, 0] += (2 * y - 1) * 2.0
        result = pg.effective_rank(
            x.astype(np.float32), y, x.astype(np.float32), y, 2, 1.0,
            [1, 2, 4, 8, 16], 0.95, device="cpu",
        )
        assert result["effective_rank"] == 1
        assert result["auc_full"] > 0.95


class TestEngineScript:
    def test_module_loads_and_cell_roundtrip(self, tmp_path, monkeypatch):
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_under_test", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scores = {
            "sid-a": (np.array([0, 1, 1]), np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])),
        }
        path = tmp_path / "cell.npz"
        module._save_cell(path, scores, {"chosen_c": 0.01}, np.array([1.0, 2.0]))
        loaded, meta, weight = module._load_cell(path)
        assert meta["chosen_c"] == 0.01
        assert np.allclose(weight, [1.0, 2.0])
        assert np.array_equal(loaded["sid-a"][0], [0, 1, 1])
        assert np.allclose(loaded["sid-a"][1], scores["sid-a"][1])

    def test_bootstrap_adv_sign(self, tmp_path):
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_boot", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(0)
        probe_cells, base_cells = [], []
        for _seed in range(3):
            cell_p, cell_b = {}, {}
            for i in range(12):
                y = rng.integers(0, 2, 60)
                good = y + 0.4 * rng.standard_normal(60)
                bad = y + 1.5 * rng.standard_normal(60)
                cell_p[f"s{i}"] = (y, np.stack([-good, good], axis=1))
                cell_b[f"s{i}"] = (y, np.stack([-bad, bad], axis=1))
            probe_cells.append(cell_p)
            base_cells.append(cell_b)
        out = module._bootstrap_adv(probe_cells, base_cells, 2, 200)
        assert out["advantage"] > 0 and out["ci95"][0] > 0
