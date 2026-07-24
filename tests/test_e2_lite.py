"""E2-lite（PREREG #34）的定向测试：行为指标、计划网格、runner 纯函数。"""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
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
        mask_user,
        mask_agent,
        dt,
        EV_CFG,
        response_window_s=1.2,
        latency_max_s=2.5,
        yield_windows_s=[0.4, 1.0],
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
        mask_user,
        mask_agent,
        dt,
        EV_CFG,
        response_window_s=1.2,
        latency_max_s=2.5,
        yield_windows_s=[0.4, 1.0],
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
    assert labels[0] == "baseline"
    assert len(labels) == len(set(labels))
    baseline = next(c for c in conditions if c["name"] == "baseline")
    assert baseline["alpha"] == 0.0 and baseline["cache_acts"]
    n_expected = len(cfg["alphas_primary"]) + len(cfg["alphas_diffmeans"]) + 3 * len(cfg["alphas_random"])
    assert len(conditions) == n_expected


def _load_run_steer():
    spec = importlib.util.spec_from_file_location("run_steer_test", REPO / "runners" / "moshi" / "run_steer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validator():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "wp_e2_lite_validate_test",
            REPO / "scripts" / "wp_e2_lite_validate_optimized.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _load_analyzer():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "wp_e2_lite_analyze_test",
            REPO / "scripts" / "wp_e2_lite_analyze.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


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
    assert {r["session"]["session_id"] for r in shard0} == {"a"}
    assert {r["session"]["session_id"] for r in shard1} == {"b"}
    batches = rs.condition_batches(runs, 2)
    assert [[r["run_id"] for r in batch] for batch in batches] == [
        ["a__c1", "a__c2"],
        ["a__c3"],
        ["b__c1", "b__c2"],
        ["b__c3"],
    ]
    # 完成标记还要求音频和文本令牌文件齐备。
    run_dir = tmp_path / "a__c1"
    run_dir.mkdir()
    assert not rs.run_done(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema": rs.MANIFEST_SCHEMA, "completed": True}), encoding="utf-8"
    )
    assert not rs.run_done(run_dir)
    (run_dir / "agent.wav").write_bytes(b"RIFF")
    assert not rs.run_done(run_dir)
    np.save(run_dir / "text_tokens.npy", np.array([1], dtype=np.int32))
    assert rs.run_done(run_dir)


def test_optimization_validator_array_and_wav(tmp_path):
    validator = _load_validator()
    reference_array = tmp_path / "reference.npy"
    candidate_array = tmp_path / "candidate.npy"
    np.save(reference_array, np.array([1.0, 2.0, 3.0], dtype=np.float16))
    np.save(candidate_array, np.array([1.0, 2.0, 3.0], dtype=np.float16))
    exact = validator._compare_array(reference_array, candidate_array)
    assert exact["equal"] and exact["equivalent"]

    np.save(candidate_array, np.array([1.0, 2.0, 3.01], dtype=np.float16))
    strict = validator._compare_array(reference_array, candidate_array)
    tolerant = validator._compare_array(
        reference_array,
        candidate_array,
        atol=0.02,
    )
    assert not strict["equal"] and not strict["equivalent"]
    assert not tolerant["equal"] and tolerant["equivalent"]

    pcm = np.array([0, 12, -12, 32767, -32768], dtype="<i2")
    wav_paths = [tmp_path / "reference.wav", tmp_path / "candidate.wav"]
    for path in wav_paths:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(pcm.tobytes())
    wav_result = validator._compare_wav(*wav_paths)
    assert wav_result["equal"] and wav_result["file_equal"]

    manifest = {
        "run_id": "s__baseline",
        "seed": 7,
        "n_frames_in": 3000,
        "execution_profile": {"equivalence_contract": "reference_exact"},
    }
    contract = validator._compare_manifest(
        manifest,
        dict(manifest),
        {"window_s": 240.0, "frame_hz": 12.5},
    )
    assert contract["equal"]
    changed = dict(manifest)
    changed["seed"] = 8
    assert not validator._compare_manifest(
        manifest,
        changed,
        {"window_s": 240.0, "frame_hz": 12.5},
    )["equal"]


def test_user_vad_mask_cache_contract(tmp_path):
    analyzer = _load_analyzer()
    manifest = {
        "session_id": "s1",
        "user_wav_sha256": "abc123",
    }
    mask = np.array([True, False, True], dtype=bool)
    analyzer._save_user_mask(tmp_path, manifest, "analysis-a", mask)
    loaded = analyzer._load_user_mask(
        tmp_path,
        manifest,
        "analysis-a",
        expected_steps=3,
    )
    np.testing.assert_array_equal(loaded, mask)
    assert (
        analyzer._load_user_mask(
            tmp_path,
            manifest,
            "analysis-b",
            expected_steps=3,
        )
        is None
    )


def test_condition_batches_keep_cache_run_single():
    rs = _load_run_steer()
    session = {"session_id": "s"}
    runs = [
        {"run_id": "s__baseline", "session": session, "condition": {"cache_acts": True}},
        {"run_id": "s__a", "session": session, "condition": {"cache_acts": False}},
        {"run_id": "s__b", "session": session, "condition": {"cache_acts": False}},
        {"run_id": "s__c", "session": session, "condition": {"cache_acts": False}},
    ]
    assert [[r["run_id"] for r in batch] for batch in rs.condition_batches(runs, 2)] == [
        ["s__baseline"],
        ["s__a", "s__b"],
        ["s__c"],
    ]


def test_shared_noise_sampling_uses_same_draw_across_batch():
    torch = pytest.importorskip("torch")
    rs = _load_run_steer()
    logits = torch.tensor(
        [
            [[1.0, 0.5, -1.0, 2.0]],
            [[1.0, 0.5, -1.0, 2.0]],
        ]
    )
    torch.manual_seed(7)
    sampled = rs._shared_noise_sample_token(
        torch,
        logits,
        use_sampling=True,
        temp=0.8,
        top_k=4,
    )
    assert sampled.shape == (2, 1)
    assert torch.equal(sampled[0], sampled[1])


def test_shared_noise_sampling_matches_repeated_batch1_rng():
    torch = pytest.importorskip("torch")
    rs = _load_run_steer()
    logits = torch.tensor(
        [
            [[2.0, 1.0, 0.0, -1.0]],
            [[-1.0, 0.0, 1.0, 2.0]],
        ]
    )
    torch.manual_seed(19)
    batched = rs._shared_noise_sample_token(torch, logits, use_sampling=True, temp=0.7, top_k=4)
    next_after_batch = torch.rand(1)
    expected = []
    for row in logits:
        torch.manual_seed(19)
        expected.append(
            rs._shared_noise_sample_token(
                torch,
                row.unsqueeze(0),
                use_sampling=True,
                temp=0.7,
                top_k=4,
            )[0]
        )
    torch.manual_seed(19)
    rs._shared_noise_sample_token(
        torch,
        logits[:1],
        use_sampling=True,
        temp=0.7,
        top_k=4,
    )
    next_after_single = torch.rand(1)
    assert torch.equal(batched, torch.stack(expected))
    assert torch.equal(next_after_batch, next_after_single)


def test_e2_lite_config_frozen_shape():
    cfg = load_config("grids")["e1"]["e2_lite"]
    assert cfg["layer"] == 29
    assert 0.0 in [float(a) for a in cfg["alphas_primary"]]
    assert cfg["scale"] == "proj_std"
    assert float(cfg["temperature"]) == pytest.approx(0.8)
    e1x_cfg = load_config("grids")["e1"]["e1x"]
    assert 0 in [int(k) for k in e1x_cfg["t4_shift_steps"]]
    assert int(e1x_cfg["layer_primary"]) == 29
