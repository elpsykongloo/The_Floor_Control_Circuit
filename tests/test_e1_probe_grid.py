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

    def test_block_preparation_matches_single_matrix(self):
        x, y = self._make_binary(n=720, d=10, seed=13)
        reference = pg.fit_linear_probe(x, y, 2, 0.1, device="cpu")
        first_x = x[:111].copy()
        first_y = y[:111].copy()
        first_x.setflags(write=False)
        first_y.setflags(write=False)
        prepared = pg.prepare_linear_probe_blocks(
            [(first_x, first_y), (x[111:503], y[111:503]), (x[503:], y[503:])],
            len(x),
            x.shape[1],
            2,
            device="cpu",
        )
        blocked = prepared.fit(0.1)
        assert np.allclose(blocked.mean, reference.mean, atol=1e-6)
        assert np.allclose(blocked.scale, reference.scale, atol=1e-6)
        assert np.allclose(
            blocked.predict_proba(x), reference.predict_proba(x), atol=2e-5
        )

    def test_batch_predictor_reuses_feature_block(self):
        x, y = self._make_binary(n=500, d=8, seed=17)
        probes = [
            pg.fit_linear_probe(x, y, 2, c_value, device="cpu")
            for c_value in (0.01, 0.1, 1.0)
        ]
        predictor = pg.LinearProbeBatchPredictor(probes, device="cpu")
        outputs = predictor.predict_proba(x)
        for probe, output in zip(probes, outputs, strict=True):
            assert np.allclose(output, probe.predict_proba(x), atol=2e-6)


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
                {
                    "agent_channel": 0,
                    "target": "T5",
                    "step": step,
                    "t": t_end,
                    "label": 5 if step == 8 else step % 5,
                    "delta_ms": None,
                }
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
        assert 8 not in steps5
        assert max(np.concatenate([r.labels for r in got5])) < 5

    def test_t5_unknown_state_still_fails(self, tmp_path):
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "agent_channel": 0,
                    "target": "T5",
                    "step": 0,
                    "t": 0.08,
                    "label": 6,
                    "delta_ms": None,
                }
            ]
        ).to_parquet(tmp_path / "sess.parquet")
        spec_t5 = next(s for s in g.expand_specs(PROBE_CFG) if s.name == "T5")
        with pytest.raises(ValueError, match="标签越界"):
            g.build_rows(
                tmp_path,
                ["sess"],
                {("sess", 0): 2, ("sess", 1): 2},
                spec_t5,
                PROBE_CFG,
                0,
                downsample=False,
            )

    def test_multi_spec_reads_each_label_file_once(self, tmp_path, monkeypatch):
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "agent_channel": channel,
                    "target": target,
                    "step": step,
                    "t": 0.08 * (step + 1),
                    "label": step % (5 if target == "T5" else 2),
                    "delta_ms": 400 if target == "T1" else None,
                }
                for channel in (0, 1)
                for target in ("T1", "T5")
                for step in range(12)
            ]
        )
        frame.to_parquet(tmp_path / "sess.parquet")
        calls = 0
        original = g.pd.read_parquet

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(g.pd, "read_parquet", counted)
        specs = [
            next(s for s in g.expand_specs(PROBE_CFG) if s.name == "T1_d400"),
            next(s for s in g.expand_specs(PROBE_CFG) if s.name == "T5"),
        ]
        result = g.build_rows_multi(
            tmp_path,
            ["sess"],
            {("sess", 0): 12, ("sess", 1): 12},
            specs,
            PROBE_CFG,
        )
        assert calls == 1
        assert set(result) == {"T1_d400", "T5"}


class TestAssembly:
    def test_alignment_row_mapping(self):
        store = {("sess", 0): np.arange(50, dtype=np.float16).reshape(10, 5)}
        roles = [g.RoleRows("sess", 0, np.array([2, 4]), np.array([0, 1]))]
        x, y, _sid = g.assemble(roles, "acts", store)
        # acts 读行 s+1（PREREG #8）
        assert np.array_equal(np.asarray(x[:, 0], dtype=np.float64), [15.0, 25.0])
        assert np.array_equal(y, [0, 1])

    def test_compact_layer_assembly_matches_full_and_missing_row_fails(self):
        full = np.arange(60, dtype=np.float16).reshape(10, 6)
        roles = [
            g.RoleRows(
                "sess",
                0,
                np.array([1, 3, 6], dtype=np.int64),
                np.array([0, 1, 0], dtype=np.int64),
            )
        ]
        required = g.required_layer_rows([roles], n_rows=10)
        rows = required[("sess", 0)]
        compact = g.IndexedLayerArray(rows, np.ascontiguousarray(full[rows]))

        expected = g.assemble(roles, "acts", {("sess", 0): full})
        actual = g.assemble(roles, "acts", {("sess", 0): compact})
        for left, right in zip(expected, actual, strict=True):
            assert np.array_equal(left, right)

        incomplete = g.IndexedLayerArray(rows[:-1], np.ascontiguousarray(full[rows[:-1]]))
        with pytest.raises(ValueError, match="压紧层缺少请求行"):
            g.assemble(roles, "acts", {("sess", 0): incomplete})

    def test_preload_layer_keeps_only_requested_rows(self, tmp_path):
        import zarr

        role_root = tmp_path / "sess_agent0"
        full = np.arange(48, dtype=np.float16).reshape(8, 6)
        group = zarr.open_group(str(role_root), mode="w")
        group.create_array("acts_L2", data=full, chunks=full.shape)
        selected = np.array([1, 4, 7], dtype=np.int32)

        store = g.preload_layer(
            tmp_path,
            [("sess", 0)],
            2,
            row_indices={("sess", 0): selected},
        )

        compact = store[("sess", 0)]
        assert isinstance(compact, g.IndexedLayerArray)
        assert np.array_equal(compact.rows, selected)
        assert np.array_equal(compact.values, full[selected])

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
        t1_spec = g.ProbeSpec("T1_d400", "T1", 400, 2, "neg5")
        t5_spec = g.ProbeSpec("T5", "T5", None, 5, "stride")
        train_rows = {
            ("T1_d400", seed): [
                g.RoleRows(
                    "session-a",
                    0,
                    np.arange(20, dtype=np.int64),
                    np.zeros(20, dtype=np.int64),
                )
            ]
            for seed in (0, 1)
        }
        train_rows.update(
            {
                ("T5", seed): [
                    g.RoleRows(
                        "session-a",
                        0,
                        np.arange(10, dtype=np.int64),
                        np.zeros(10, dtype=np.int64),
                    )
                ]
                for seed in (0, 1, 2)
            }
        )
        fit_tasks = module._ordered_fit_tasks(
            [(t1_spec, [0, 1]), (t5_spec, [0, 1, 2])], train_rows
        )
        assert [(item[0].name, item[1]) for item in fit_tasks[:3]] == [
            ("T5", 0),
            ("T5", 1),
            ("T5", 2),
        ]
        assert {(item[0].name, item[1]) for item in fit_tasks} == {
            ("T1_d400", 0),
            ("T1_d400", 1),
            ("T5", 0),
            ("T5", 1),
            ("T5", 2),
        }
        assert module._event_labels_path(
            {"events": tmp_path}, "session-a"
        ).name == "session-a.labels.parquet"
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

        probe = pg.LinearProbe(
            mean=np.array([0.1, 0.2], dtype=np.float32),
            scale=np.array([1.0, 2.0], dtype=np.float32),
            weight=np.array([[0.3, -0.4]], dtype=np.float32),
            bias=np.array([0.5], dtype=np.float32),
            n_classes=2,
            c_value=0.01,
            converged=True,
        )
        fit_path = tmp_path / "fit.npz"
        module._save_fit(
            fit_path,
            probe,
            {"chosen_c": 0.01, "converged": True, "n_classes": 2},
        )
        loaded_probe, loaded_meta = module._load_fit(fit_path)
        assert loaded_meta["chosen_c"] == 0.01
        assert np.array_equal(loaded_probe.weight, probe.weight)
        assert np.array_equal(loaded_probe.mean, probe.mean)

        row_spec = g.ProbeSpec("T4", "T4", None, 2, "none")
        row_roles = [
            g.RoleRows(
                "session-a",
                1,
                np.array([2, 7], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
            )
        ]
        row_path = tmp_path / "rows.npz"
        module._save_row_plan(
            row_path,
            "signature",
            [row_spec],
            [0],
            {("T4", 0): row_roles},
            {"T4": row_roles},
        )
        cached = module._load_row_plan(
            row_path, "signature", [row_spec], [0]
        )
        assert cached is not None
        cached_train, cached_eval = cached
        assert cached_train[("T4", 0)][0].session_id == "session-a"
        assert np.array_equal(cached_eval["T4"][0].steps, [2, 7])

    def test_acoustic_prefix_is_bounded_and_output_is_atomic(self, tmp_path):
        import sys

        import soundfile as sf

        from floor_circuit.probes.baselines import acoustic_frames

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_acoustic", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        sample_rate = 24_000
        seconds = 2.0
        timeline = np.arange(int(sample_rate * seconds)) / sample_rate
        wav = (0.1 * np.sin(2 * np.pi * 180.0 * timeline)).astype(np.float32)
        audio_path = tmp_path / "audio.wav"
        output_path = tmp_path / "features.npy"
        sf.write(audio_path, wav, sample_rate, subtype="FLOAT")

        window_s = 1.0
        n_steps = 12
        full, sr = sf.read(audio_path, dtype="float32")
        expected = acoustic_frames(full[: int(window_s * sr)], sr)[:n_steps]
        module._extract_acoustic_role(
            (str(audio_path), str(output_path), n_steps, window_s)
        )

        actual = np.load(output_path, allow_pickle=False)
        assert np.array_equal(actual, expected.astype(np.float32))
        assert module._valid_acoustic_output(output_path, n_steps)
        assert not output_path.with_suffix(".tmp.npy").exists()

        tasks = [
            ("input", str(tmp_path / f"out-{index}.npy"), n_steps, window_s)
            for index in range(6)
        ]
        shard0 = module._pending_acoustic_tasks(tasks, 2, 0, force=False)
        shard1 = module._pending_acoustic_tasks(tasks, 2, 1, force=False)
        assert shard0 == tasks[0::2]
        assert shard1 == tasks[1::2]
        assert set(shard0).isdisjoint(shard1)

    def test_parity_can_use_frozen_partial_inputs(self, tmp_path, monkeypatch):
        import json
        import shutil
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_early_parity", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        roots = {
            "events": tmp_path / "events",
            "labels": tmp_path / "labels",
            "runs": tmp_path / "runs",
            "work": tmp_path / "work",
        }
        roots["events"].mkdir()
        source = roots["events"] / "session-a.labels.parquet"
        source.write_bytes(b"labels")
        for channel in (0, 1):
            role_dir = roots["runs"] / f"session-a_agent{channel}"
            role_dir.mkdir(parents=True)
            (role_dir / "manifest.json").write_text(
                json.dumps({"n_steps": 3000}), encoding="utf-8"
            )
        monkeypatch.setattr(module, "_accepted_label_fingerprints", lambda: set())
        monkeypatch.setattr(
            module,
            "_validated_label_record",
            lambda _roots, sid, _accepted: ([sid], None),
        )
        monkeypatch.setattr(module, "_frozen_window_and_steps", lambda: (240.0, 3000))
        monkeypatch.setattr(shutil, "copy2", shutil.copyfile)

        steps = module._prepare_parity_inputs(roots, ["session-a"])

        assert steps == {("session-a", 0): 3000, ("session-a", 1): 3000}
        assert (roots["labels"] / "session-a.parquet").read_bytes() == b"labels"

    def test_early_baseline_inputs_do_not_require_zarr(self, tmp_path, monkeypatch):
        import json
        import shutil
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_early_baselines", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        roots = {
            "events": tmp_path / "events",
            "labels": tmp_path / "labels",
            "runs": tmp_path / "missing-zarr-runs",
            "work": tmp_path / "work",
        }
        roots["events"].mkdir()
        source = roots["events"] / "session-a.labels.parquet"
        source.write_bytes(b"labels")
        monkeypatch.setattr(module, "_accepted_label_fingerprints", lambda: set())
        monkeypatch.setattr(
            module,
            "_validated_label_record",
            lambda _roots, sid, _accepted: ([sid], None),
        )
        monkeypatch.setattr(module, "_frozen_window_and_steps", lambda: (240.0, 3000))
        monkeypatch.setattr(shutil, "copy2", shutil.copyfile)

        _labels, roles = module._prepare_label_inputs(
            roots, ["session-a"], verify_run_manifests=False
        )

        assert roles == [["session-a", 0, 3000], ["session-a", 1, 3000]]
        payload = json.loads((roots["work"] / "run_specs.json").read_text("utf-8"))
        assert payload["roles"] == roles
        assert not roots["runs"].exists()

    def test_label_record_validates_marker_hash_and_audio(self, tmp_path):
        import hashlib
        import json
        import sys
        import wave

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_label_record", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        roots = {
            "events": tmp_path / "events",
            "audio": tmp_path / "audio",
        }
        roots["events"].mkdir()
        session_audio = roots["audio"] / "session-a"
        session_audio.mkdir(parents=True)
        source_audio = {}
        for channel in (0, 1):
            path = session_audio / f"audio_ch{channel}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24_000)
                handle.writeframes(np.zeros(32, dtype="<i2").tobytes())
            stat = path.stat()
            source_audio[path.name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        label_path = roots["events"] / "session-a.labels.parquet"
        label_path.write_bytes(b"label-payload")
        marker = {
            "schema_version": 1,
            "session": "session-a",
            "input": {
                "event_pipeline_code_sha256": "code",
                "settings_sha256": "settings",
                "source_audio": source_audio,
            },
            "outputs": {
                "labels": {
                    "name": label_path.name,
                    "size": label_path.stat().st_size,
                    "sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                }
            },
        }
        (roots["events"] / "session-a.complete.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )

        record, reason = module._validated_label_record(
            roots, "session-a", {("code", "settings")}
        )

        assert reason is None and record is not None
        label_path.write_bytes(b"tampered")
        record, reason = module._validated_label_record(
            roots, "session-a", {("code", "settings")}
        )
        assert record is None and reason in {"size", "sha256"}

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

    def test_fast_bootstrap_matches_literal_resampling(self):
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_boot_exact", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(23)
        probe_cells, base_cells = [], []
        for seed in range(2):
            probe_cell, base_cell = {}, {}
            for sid_index in range(5):
                y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
                good = y + rng.normal(0, 0.25 + 0.02 * seed, len(y))
                weak = y + rng.normal(0, 0.9 + 0.03 * sid_index, len(y))
                probe_cell[f"s{sid_index}"] = (y, np.stack([-good, good], axis=1))
                base_cell[f"s{sid_index}"] = (y, np.stack([-weak, weak], axis=1))
            probe_cells.append(probe_cell)
            base_cells.append(base_cell)

        def literal(take):
            def mean_metric(cells):
                values = []
                for cell in cells:
                    y = np.concatenate([cell[sid][0] for sid in take])
                    p = np.concatenate([cell[sid][1] for sid in take])
                    values.append(pg.primary_metric(y, p, 2))
                return float(np.mean(values))

            return mean_metric(probe_cells) - mean_metric(base_cells)

        sids = sorted(probe_cells[0])
        expected_point = literal(sids)
        boot_rng = np.random.default_rng(20260717)
        expected_samples = [
            literal([sids[i] for i in boot_rng.integers(0, len(sids), len(sids))])
            for _ in range(120)
        ]
        expected_ci = np.percentile(expected_samples, [2.5, 97.5])
        result = module._bootstrap_adv(
            probe_cells, base_cells, 2, 120, seed=20260717
        )
        assert result["advantage"] == pytest.approx(expected_point, abs=1e-12)
        assert result["ci95"] == pytest.approx(expected_ci, abs=1e-12)

    def test_fast_bootstrap_matches_multiclass_resampling(self):
        import sys

        scripts = REPO_ROOT / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "wp_e1_probe_grid_boot_multi", scripts / "wp_e1_probe_grid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(31)
        probe_cell, base_cell = {}, {}
        for sid_index in range(6):
            y = np.tile(np.arange(3, dtype=np.int64), 3)
            good_logits = rng.normal(0, 0.5, (len(y), 3))
            weak_logits = rng.normal(0, 1.0, (len(y), 3))
            good_logits[np.arange(len(y)), y] += 1.5
            weak_logits[np.arange(len(y)), y] += 0.4

            def softmax(logits):
                expv = np.exp(logits - logits.max(axis=1, keepdims=True))
                return expv / expv.sum(axis=1, keepdims=True)

            probe_cell[f"s{sid_index}"] = (y, softmax(good_logits))
            base_cell[f"s{sid_index}"] = (y, softmax(weak_logits))
        sids = sorted(probe_cell)

        def literal(take):
            y_probe = np.concatenate([probe_cell[sid][0] for sid in take])
            p_probe = np.concatenate([probe_cell[sid][1] for sid in take])
            y_base = np.concatenate([base_cell[sid][0] for sid in take])
            p_base = np.concatenate([base_cell[sid][1] for sid in take])
            return pg.primary_metric(y_probe, p_probe, 3) - pg.primary_metric(
                y_base, p_base, 3
            )

        boot_rng = np.random.default_rng(77)
        samples = [
            literal([sids[i] for i in boot_rng.integers(0, len(sids), len(sids))])
            for _ in range(80)
        ]
        result = module._bootstrap_adv(
            [probe_cell], [base_cell], 3, 80, seed=77
        )
        assert result["advantage"] == pytest.approx(literal(sids), abs=1e-12)
        assert result["ci95"] == pytest.approx(
            np.percentile(samples, [2.5, 97.5]), abs=1e-12
        )
