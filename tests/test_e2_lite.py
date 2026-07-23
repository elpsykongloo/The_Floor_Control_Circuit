"""E2-lite（PREREG #34）的定向测试：行为指标、计划网格、runner 纯函数。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.config import load_config
from floor_circuit.e1x.behavior import bootstrap_mean_ci, floor_metrics, paired_deltas

REPO = Path(__file__).resolve().parents[1]


def _mask(total_s: float, dt: float, segments: list[tuple[float, float]]) -> np.ndarray:
    n = round(total_s / dt)
    mask = np.zeros(n, dtype=bool)
    for start, end in segments:
        mask[round(start / dt) : round(end / dt)] = True
    return mask


EV_CFG = {
    "onset_pre_silence_s": 0.4,
    "offset_post_silence_s": 0.4,
    "yield_max_s": 1.0,
    "hold_min_s": 1.5,
}


def test_floor_metrics_latency_response_and_interruption():
    dt = 0.01
    # 用户：说 [1,3)；agent：说 [3.5,5)（对用户 offset=3.0 的响应，延迟 0.5 s）
    mask_user = _mask(10.0, dt, [(1.0, 3.0)])
    mask_agent = _mask(10.0, dt, [(3.5, 5.0)])
    metrics = floor_metrics(
        mask_user, mask_agent, dt, EV_CFG,
        response_window_s=1.2, latency_max_s=2.5, yield_windows_s=[0.4, 1.0],
    )
    assert metrics["median_response_latency_s"] == pytest.approx(0.5, abs=0.02)
    assert metrics["response_rate"] == pytest.approx(1.0)
    assert metrics["interruption_rate_per_min_user_speech"] == pytest.approx(0.0)
    assert metrics["n_user_incursions"] == 0
    assert metrics["yield_rate_400ms"] is None  # 无重叠闯入 → 分母为空


def test_floor_metrics_interruption_and_yield():
    dt = 0.01
    # agent 说 [1,4)；用户在 2.0 闯入并持续 → agent 于 2.3 合格 offset（让位 ≤400 ms）
    mask_agent = _mask(10.0, dt, [(1.0, 2.3), (6.0, 7.0)])
    mask_user = _mask(10.0, dt, [(2.0, 4.0), (5.5, 6.5)])
    # agent 第二段 onset=6.0 时用户仍在发声（5.5–6.5）→ 抢话 1 次
    metrics = floor_metrics(
        mask_user, mask_agent, dt, EV_CFG,
        response_window_s=1.2, latency_max_s=2.5, yield_windows_s=[0.4, 1.0],
    )
    assert metrics["n_user_incursions"] == 1
    assert metrics["yield_rate_400ms"] == pytest.approx(1.0)
    assert metrics["interruption_rate_per_min_user_speech"] > 0


def test_paired_deltas_and_bootstrap_ci():
    per_session = {
        "s1": {"baseline": {"m": 0.1}, "steer": {"m": 0.3}},
        "s2": {"baseline": {"m": 0.2}, "steer": {"m": 0.5}},
        "s3": {"baseline": {"m": None}, "steer": {"m": 0.4}},  # 缺基线 → 剔除
    }
    deltas = paired_deltas(per_session, "steer", "baseline", "m")
    assert deltas == pytest.approx([0.2, 0.3])
    stats = bootstrap_mean_ci(deltas, n_boot=500)
    assert stats["n"] == 2
    assert stats["n_pos"] == 2
    assert stats["ci95"][0] <= stats["mean"] <= stats["ci95"][1]
    assert bootstrap_mean_ci([])["mean"] is None


def test_build_conditions_grid_unique_and_baseline():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from wp_e2_lite_plan import build_conditions
    finally:
        sys.path.pop(0)
    cfg = load_config("grids")["e1"]["e2_lite"]
    names = ["probe_meanseed", "diffmeans", "random_r0", "random_r1", "random_r2"]
    conditions = build_conditions(cfg, names)
    labels = [c["name"] for c in conditions]
    assert "baseline" in labels
    assert len(labels) == len(set(labels))
    baseline = next(c for c in conditions if c["name"] == "baseline")
    assert baseline["alpha"] == 0.0 and baseline["cache_acts"]
    n_expected = (
        len(cfg["alphas_primary"])
        + len(cfg["alphas_diffmeans"])
        + 3 * len(cfg["alphas_random"])
    )
    assert len(conditions) == n_expected


def _load_run_steer():
    spec = importlib.util.spec_from_file_location(
        "run_steer_test", REPO / "runners" / "moshi" / "run_steer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_steer_pure_functions(tmp_path):
    rs = _load_run_steer()
    vec = rs.steer_vector_np(2.0, 1.5, np.array([3.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(vec, [3.0, 0.0, 0.0, 0.0])  # 2·1.5·单位向量
    plan = {
        "sessions": [{"session_id": "a"}, {"session_id": "b"}],
        "conditions": [{"name": "c1"}, {"name": "c2"}, {"name": "c3"}],
    }
    runs = rs.iter_runs(plan)
    assert [r["run_id"] for r in runs][:3] == ["a__c1", "a__c2", "a__c3"]
    shard0 = rs.shard_runs(runs, 2, 0)
    shard1 = rs.shard_runs(runs, 2, 1)
    assert len(shard0) + len(shard1) == len(runs)
    assert not {r["run_id"] for r in shard0} & {r["run_id"] for r in shard1}
    # 完成标记：schema + completed + agent.wav 三者齐备才算完成
    run_dir = tmp_path / "a__c1"
    run_dir.mkdir()
    assert not rs.run_done(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema": rs.MANIFEST_SCHEMA, "completed": True}), encoding="utf-8"
    )
    assert not rs.run_done(run_dir)
    (run_dir / "agent.wav").write_bytes(b"RIFF")
    assert rs.run_done(run_dir)


def test_e2_lite_config_frozen_shape():
    cfg = load_config("grids")["e1"]["e2_lite"]
    assert cfg["layer"] == 29
    assert 0.0 in [float(a) for a in cfg["alphas_primary"]]
    assert cfg["scale"] == "proj_std"
    assert float(cfg["temperature"]) == pytest.approx(0.8)
    e1x_cfg = load_config("grids")["e1"]["e1x"]
    assert 0 in [int(k) for k in e1x_cfg["t4_shift_steps"]]
    assert int(e1x_cfg["layer_primary"]) == 29
