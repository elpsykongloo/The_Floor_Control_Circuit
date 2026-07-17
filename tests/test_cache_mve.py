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
    average_over_seeds,
    evaluate_target,
    overall_g1,
    probe_grid,
    render_report,
)


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
        rng = np.random.default_rng(1)
        noise_baseline = {
            sid: (summary[12]["rep_per_session"][sid][0],
                  rng.normal(0, 1, len(summary[12]["rep_per_session"][sid][0])))
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
