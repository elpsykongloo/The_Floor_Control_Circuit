"""E2 确认臂（PREREG #40(b)）的定向测试：条件网格、事件门、钳制数学、runner 纯函数。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.config import load_config
from floor_circuit.e1x import confirm as cf

REPO = Path(__file__).resolve().parents[1]

EV_CFG = {
    "onset_pre_silence_s": 0.4,
    "offset_post_silence_s": 0.4,
}


def _mask(total_s: float, dt: float, segments: list[tuple[float, float]]) -> np.ndarray:
    n = round(total_s / dt)
    mask = np.zeros(n, dtype=bool)
    for start, end in segments:
        mask[round(start / dt) : round(end / dt)] = True
    return mask


def test_build_confirm_conditions_grid():
    cfg = load_config("grids")["e1"]["e2_confirm"]
    conditions = cf.build_confirm_conditions(cfg)
    names = [c["name"] for c in conditions]
    assert len(names) == len(set(names))
    # 1 clamp + 2 门 × 2α + 2 轴 × 2α + 2 错层 × 2α = 13
    n_alphas = len(cfg["alphas"])
    expected = 1 + 2 * n_alphas + len(cfg["axis_directions"]) * n_alphas + len(cfg["wrong_layers"]) * n_alphas
    assert len(conditions) == expected
    clamp = next(c for c in conditions if c["mode"] == "clamp")
    assert clamp["layer"] == int(cfg["layer"]) and clamp["gate"] == "none"
    assert all(c["alpha"] != 0.0 for c in conditions if c["mode"] == "inject")
    wrong = [c for c in conditions if c["layer"] != int(cfg["layer"])]
    assert {c["layer"] for c in wrong} == {int(v) for v in cfg["wrong_layers"]}


def test_build_confirm_conditions_rejects_zero_alpha_and_main_layer_in_wrong():
    cfg = {
        "layer": 29,
        "wrong_layers": [20],
        "alphas": [0.0, 4.0],
        "axis_directions": [],
        "clamp": True,
    }
    with pytest.raises(ValueError, match="α"):
        cf.build_confirm_conditions(cfg)
    cfg2 = {**cfg, "alphas": [4.0], "wrong_layers": [29]}
    with pytest.raises(ValueError, match="主层"):
        cf.build_confirm_conditions(cfg2)


def test_respond_and_user_speech_gate_frames():
    dt = 0.01
    frame_s = 0.08
    # 用户说 [1.0, 2.0)：合格 offset 在 2.0；respond 门 = [2.0, 3.2) → 帧 25..39
    mask_user = _mask(8.0, dt, [(1.0, 2.0)])
    respond = cf.respond_gate_frames(
        mask_user, dt, EV_CFG, window_s=1.2, n_frames=100, frame_s=frame_s
    )
    active = np.flatnonzero(respond)
    assert active.min() == 25 and active.max() == 39
    speech = cf.user_speech_gate_frames(mask_user, dt, n_frames=100, frame_s=frame_s)
    speech_active = np.flatnonzero(speech)
    # 发声帧 = [1.0,2.0) 覆盖的帧 12..24（12.5 帧起点在帧 12 内）
    assert speech_active.min() == 12 and speech_active.max() == 24
    assert not (respond & speech).any()  # 两门互斥（offset 后用户已静默）


def test_apply_clamp_reference_math():
    unit = np.array([1.0, 0.0, 0.0])
    hidden = np.array([[3.0, 2.0, -1.0], [-5.0, 0.5, 4.0]])
    clamped = cf.apply_clamp(hidden, unit, target=1.5)
    np.testing.assert_allclose(clamped[:, 0], [1.5, 1.5])  # v̂ 分量被钳到 μ
    np.testing.assert_allclose(clamped[:, 1:], hidden[:, 1:])  # 正交部分不变
    untouched = cf.apply_clamp(hidden, unit, target=1.5, gate=0.0)
    np.testing.assert_allclose(untouched, hidden)  # 门关闭 → 恒等
    with pytest.raises(ValueError, match="单位方向"):
        cf.apply_clamp(hidden, np.array([2.0, 0.0, 0.0]), target=0.0)


def _load_confirm_runner():
    spec = importlib.util.spec_from_file_location(
        "run_steer_confirm_test", REPO / "runners" / "moshi" / "run_steer_confirm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirm_runner_steer_params_and_done(tmp_path):
    rc = _load_confirm_runner()
    directions = {"probe_meanseed": np.array([0.0, 2.0, 0.0])}
    plan = {
        "proj_std": {"probe_meanseed": {"29": 0.9, "20": 2.5}},
        "proj_mean": {"probe_meanseed": {"29": 0.1, "20": -0.3}},
    }
    inject = rc.steer_params_for(
        {"mode": "inject", "direction": "probe_meanseed", "alpha": -4.0, "layer": 20, "gate": "none"},
        plan,
        directions,
    )
    np.testing.assert_allclose(inject["vec"], [0.0, -4.0 * 2.5, 0.0])  # 错层用该层 σ 重标定
    assert inject["target"] is None
    clamp = rc.steer_params_for(
        {"mode": "clamp", "direction": "probe_meanseed", "alpha": 0.0, "layer": 29, "gate": "none"},
        plan,
        directions,
    )
    assert np.allclose(clamp["vec"], 0.0)
    assert clamp["target"] == pytest.approx(0.1)
    np.testing.assert_allclose(np.linalg.norm(clamp["unit"]), 1.0)

    run_dir = tmp_path / "s__c"
    run_dir.mkdir()
    assert not rc.run_done(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema": rc.MANIFEST_SCHEMA, "completed": True, "run_id": "s__c"}),
        encoding="utf-8",
    )
    (run_dir / "agent.wav").write_bytes(b"RIFF")
    np.save(run_dir / "text_tokens.npy", np.array([1], dtype=np.int32))
    assert rc.run_done(run_dir)
    assert not rc.run_done(run_dir, {"run_id": "other"})


def test_confirm_gate_loader_contract(tmp_path):
    rc = _load_confirm_runner()
    gate_path = tmp_path / "sid.npz"
    respond = np.zeros(10, dtype=np.uint8)
    respond[3:5] = 1
    np.savez(gate_path, respond=respond, user_speech=np.ones(10, dtype=np.uint8))
    import hashlib

    sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    plan = {"gates": {"sid": {"path": str(gate_path), "sha256": sha}}}
    loaded = rc._load_gate(plan, "sid", "respond", 10)
    np.testing.assert_allclose(loaded, respond.astype(np.float32))
    ones = rc._load_gate(plan, "sid", "none", 10)
    np.testing.assert_allclose(ones, np.ones(10, dtype=np.float32))
    plan_bad = {"gates": {"sid": {"path": str(gate_path), "sha256": "0" * 64}}}
    with pytest.raises(Exception, match="摘要"):
        rc._load_gate(plan_bad, "sid", "respond", 10)


def test_sdt_gate_and_clamp_composition_smoke():
    """门 × 钳制的组合语义：门=1 时钳制生效、门=0 时恒等（数值参考）。"""
    unit = np.array([0.6, 0.8])
    hidden = np.array([[1.0, 1.0]])
    on = cf.apply_clamp(hidden, unit, target=0.0, gate=1.0)
    off = cf.apply_clamp(hidden, unit, target=0.0, gate=0.0)
    assert abs(float((on @ unit)[0])) < 1e-9
    np.testing.assert_allclose(off, hidden)
