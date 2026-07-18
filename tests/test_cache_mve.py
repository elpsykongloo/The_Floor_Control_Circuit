"""缓存层（manifest / npy ingest / zarr 回环）与 MVE 编排单测。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from floor_circuit.cachelib.manifest import RunManifest, load_manifest, save_manifest
from floor_circuit.cachelib.zarr_io import (
    ingest_npy_run,
    read_acts,
    roundtrip_check,
    write_acts_direct,
)
from floor_circuit.mve.dataset import build_session_data, run_dir_for
from floor_circuit.mve.run import (
    ProbeCell,
    average_over_seeds,
    evaluate_target,
    overall_g1,
    probe_grid,
    render_report,
)
from floor_circuit.probes.stats import pooled_metrics


class TestManifest:
    def test_roundtrip(self, tmp_path):
        m = RunManifest(model="moshi", mode="R1", session_id="s1", layers=[4, 12], hidden_dim=8,
                        clock_hz=12.5, source_audio={"a.wav": "deadbeef"})
        save_manifest(tmp_path, m)
        back = load_manifest(tmp_path)
        assert back == m

    def test_mode_validation(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunManifest(model="moshi", mode="R9", session_id="s1")


class TestIngest:
    def make_run(self, tmp_path, sid="sessX", t=100, h=8, layers=(4, 12)):
        src = tmp_path / f"{sid}_src"
        src.mkdir()
        ref = {}
        rng = np.random.default_rng(0)
        for layer in layers:
            acts = rng.normal(0, 1, (t, h)).astype(np.float16)
            np.save(src / f"acts_L{layer}_part00000.npy", acts[:60])
            np.save(src / f"acts_L{layer}_part00001.npy", acts[60:])
            ref[layer] = acts
        manifest = {"schema_version": 1, "model": "moshi", "mode": "R1", "session_id": sid,
                    "layers": list(layers), "hidden_dim": h, "clock_hz": 12.5, "seed": 0,
                    "source_audio": {}}
        (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return src, ref

    def test_ingest_and_roundtrip(self, tmp_path):
        src, ref = self.make_run(tmp_path)
        dest = tmp_path / "dest"
        m = ingest_npy_run(src, dest)
        assert m.n_steps == 100
        for layer, acts in ref.items():
            assert roundtrip_check(dest, layer, acts)
        assert read_acts(dest, 4).dtype == np.float16

    def test_missing_layer_raises(self, tmp_path):
        src, _ = self.make_run(tmp_path, layers=(4,))
        manifest = json.loads((src / "manifest.json").read_text())
        manifest["layers"] = [4, 20]
        (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            ingest_npy_run(src, tmp_path / "d2")


def make_world(tmp_path, sids, t=120, h=8, layers=(4, 12), effect=2.0):
    """合成 MVE 小世界：zarr 激活（dim0 携带 T1 信号）+ 标签 parquet。"""
    runs_root = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    rng = np.random.default_rng(0)
    for sid in sids:
        rows = []
        for ch in (0, 1):
            rd = run_dir_for(runs_root, sid, ch)
            y = (rng.random(t) < 0.2).astype(np.int64)
            base = rng.normal(0, 1, (t, h)).astype(np.float32)
            base[:, 0] += effect * y
            for layer in layers:
                noise = base if layer == 12 else rng.normal(0, 1, (t, h)).astype(np.float32)
                write_acts_direct(rd, layer, noise.astype(np.float16))
            manifest = RunManifest(model="moshi", mode="R1", session_id=sid, layers=list(layers),
                                   hidden_dim=h, clock_hz=12.5, n_steps=t)
            save_manifest(rd, manifest)
            for s in range(t):
                rows.append({"agent_channel": ch, "target": "T1", "step": s, "t": (s + 1) * 0.08,
                             "label": int(y[s]), "delta_ms": 240})
                rows.append({"agent_channel": ch, "target": "T4", "step": s, "t": (s + 1) * 0.08,
                             "label": int(y[s]), "delta_ms": None})
        pd.DataFrame(rows).to_parquet(labels_root / f"{sid}.parquet")
    return runs_root, labels_root


class TestMveDataset:
    def test_build_session_data(self, tmp_path):
        sids = [f"s{i}" for i in range(3)]
        runs_root, labels_root = make_world(tmp_path, sids)
        data = build_session_data(runs_root, labels_root, sids, layer=12, target="T1", delta_ms=240)
        assert set(data) == set(sids)
        X, y = data["s0"]
        assert X.shape == (240, 8) and len(y) == 240  # 两角色 × 120 步


class TestMveOrchestration:
    def test_grid_report_and_g1(self, tmp_path):
        sids = [f"s{i}" for i in range(8)]
        runs_root, labels_root = make_world(tmp_path, sids)
        train, evals = sids[:5], sids[5:]
        data_by_layer = {
            layer: build_session_data(runs_root, labels_root, sids, layer, "T1", 240)
            for layer in (4, 12)
        }
        cells = probe_grid(data_by_layer, train, evals, seeds=[0, 1], c_grid=[0.1, 1.0],
                           neg_ratio=5, target="T1")
        assert len(cells) == 4
        summary = average_over_seeds(cells)
        assert summary[12]["auc_mean"] > summary[4]["auc_mean"]  # 信号在 L12
        reference = summary[12]["per_seed"][0]
        rng = np.random.default_rng(1)
        noise_baseline = {
            sid: (reference[sid][0], rng.normal(0, 1, len(reference[sid][0])))
            for sid in evals
        }
        res = evaluate_target(summary, {"noise": noise_baseline}, n_boot=100,
                              full_thr=0.05, backup_thr=0.02)
        assert res["best_layer"] == 12 and res["verdict"] == "full_e1"
        assert set(res["baseline_metrics"]["noise"]) >= {"auc", "auprc", "balanced_acc"}
        assert abs(res["shuffled_auc"] - 0.5) < 0.12
        overall = overall_g1({"T1": res}, 0.05, 0.02)
        report = render_report({"T1": res}, overall, {"layers": [4, 12], "seeds": [0, 1],
                                                      "bootstrap_n": 100,
                                                      "n_train_sessions": 5, "n_eval_sessions": 3})
        assert "full_e1" in report and "| L12 |" in report
        assert "| noise |" in report and "AUPRC" in report

    def test_seed_mean_point_bootstrap_and_report_use_same_estimand(self):
        """点估计与 bootstrap 都应先算各 seed AUC，再取 seed 均值。"""

        y = np.array([0, 0, 1, 1])
        probe_scores = {
            0: np.array([0.1, 0.2, 0.8, 0.9]),  # AUC 1.00
            1: np.array([0.1, 0.8, 0.9, 0.2]),  # AUC 0.75
            2: np.array([0.9, 0.8, 0.2, 0.1]),  # AUC 0.00
        }

        def sessions(scores):
            return {
                f"s{index}": (y.copy(), scores.copy())
                for index in range(4)
            }

        cells = []
        for seed, scores in probe_scores.items():
            per_session = sessions(scores)
            metrics = pooled_metrics(per_session) | {"best_c": 0.1}
            cells.append(ProbeCell(12, "T1", seed, metrics, per_session))
        summary = average_over_seeds(cells)

        hazard = sessions(np.array([0.9, 0.7, 0.3, 0.1]))
        mimi = {
            0: sessions(np.array([0.9, 0.1, 0.8, 0.2])),
            1: sessions(np.array([0.2, 0.8, 0.7, 0.1])),
            2: sessions(np.array([0.8, 0.6, 0.4, 0.2])),
        }
        baselines = {"hazard": hazard, "mimi": mimi}
        result = evaluate_target(
            summary,
            baselines,
            n_boot=100,
            full_thr=0.05,
            backup_thr=0.02,
        )

        expected_probe = float(
            np.mean([pooled_metrics(sessions(scores))["auc"] for scores in probe_scores.values()])
        )
        expected_mimi = float(
            np.mean([pooled_metrics(scores)["auc"] for scores in mimi.values()])
        )
        expected_hazard = pooled_metrics(hazard)["auc"]
        expected_advantage = expected_probe - max(expected_hazard, expected_mimi)
        assert summary[12]["auc_mean"] == pytest.approx(expected_probe)
        assert result["advantage"]["probe_auc"] == pytest.approx(expected_probe)
        assert result["probe_ci"]["point"] == pytest.approx(expected_probe)
        assert result["probe_ci"]["ci_lo"] == pytest.approx(expected_probe)
        assert result["probe_ci"]["ci_hi"] == pytest.approx(expected_probe)
        assert result["advantage"]["advantage_point"] == pytest.approx(expected_advantage)
        assert result["advantage"]["ci_lo"] == pytest.approx(expected_advantage)
        assert result["advantage"]["ci_hi"] == pytest.approx(expected_advantage)
        assert result["baseline_metrics"]["mimi"]["n_seeds"] == 3

        overall = overall_g1({"T1": result}, 0.05, 0.02)
        report = render_report(
            {"T1": result},
            overall,
            {
                "layers": [12],
                "seeds": [0, 1, 2],
                "bootstrap_n": 100,
                "n_train_sessions": 4,
                "n_eval_sessions": 4,
            },
        )
        assert "不覆盖模型选择不确定性" in report
        assert "3 个种子" in report
        assert f"探针 AUC {summary[12]['auc_mean']:.4f}" in report
