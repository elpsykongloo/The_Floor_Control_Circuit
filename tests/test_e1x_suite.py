"""E1-X 探索套件库（PREREG #33）的定向测试：全部合成数据、CPU 可跑。"""

from __future__ import annotations

import numpy as np
import pytest

from floor_circuit.e1.grid import RoleRows
from floor_circuit.e1.probe_gpu import fit_linear_probe
from floor_circuit.e1x import anatomy as ax
from floor_circuit.e1x import core as cx
from floor_circuit.e1x import trajectory as tx
from floor_circuit.mve.alignment import feature_row_indices
from floor_circuit.schemas import State


def _role(sid: str, steps, labels, channel: int = 0) -> RoleRows:
    return RoleRows(sid, channel, np.asarray(steps, dtype=np.int64), np.asarray(labels, dtype=np.int64))


# ---------------------------------------------------------------------------
# X1：平移行域
# ---------------------------------------------------------------------------


def test_shift_roles_alignment_and_restriction():
    roles = [_role("s1", [10, 30, 50], [1, 0, 1])]
    restricted = cx.restrict_min_step(roles, 25)
    assert restricted[0].steps.tolist() == [30, 50]
    shifted = cx.shift_roles(restricted, 25)
    assert shifted[0].steps.tolist() == [5, 25]
    assert shifted[0].labels.tolist() == [0, 1]  # 标签跟随锚点，不随特征行平移
    # acts 行映射沿 #8：平移后步 s−k 读行 s−k+1
    assert feature_row_indices("acts", shifted[0].steps).tolist() == [6, 26]


def test_shift_roles_rejects_negative_steps():
    roles = [_role("s1", [3], [1])]
    with pytest.raises(ValueError, match="restrict_min_step"):
        cx.shift_roles(roles, 5)


def test_shifted_event_set_constant_across_shifts():
    roles = [_role("s1", [10, 40, 80], [1, 0, 1])]
    restricted = cx.restrict_min_step(roles, 20)
    for shift in (0, 5, 20):
        shifted = cx.shift_roles(restricted, shift)
        assert len(shifted[0].labels) == 2  # 锚点集合恒定
        assert shifted[0].labels.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# 嵌套 C 拟合
# ---------------------------------------------------------------------------


def test_nested_probe_fit_selects_reasonable_c_and_refits_full_pool():
    rng = np.random.default_rng(0)
    direction = np.array([1.0, -1.0, 0.5, 0.0])
    x = rng.standard_normal((600, 4)).astype(np.float32)
    y = (x @ direction + 0.1 * rng.standard_normal(600) > 0).astype(np.int64)
    inner = np.zeros(600, dtype=bool)
    inner[:120] = True
    result = cx.nested_probe_fit(x, y, inner, 2, [1e-4, 1e-2, 1.0], device="cpu")
    assert str(result.chosen_c) in result.inner_curve
    assert result.inner_curve[str(result.chosen_c)] == max(result.inner_curve.values())
    probs = result.probe.predict_proba(x)
    assert probs.shape == (600, 2)
    assert result.inner_metric > 0.9


def test_nested_probe_fit_rejects_degenerate_inner_mask():
    x = np.zeros((10, 2), dtype=np.float32)
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError):
        cx.nested_probe_fit(x, y, np.ones(10, dtype=bool), 2, [1.0])


# ---------------------------------------------------------------------------
# X3：岭回归残差化
# ---------------------------------------------------------------------------


def test_fit_ridge_matches_sklearn_on_standardized_space():
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 5)).astype(np.float32)
    weight = rng.standard_normal((5, 3))
    y = (x @ weight + 0.05 * rng.standard_normal((300, 3))).astype(np.float32)
    lam = 7.5
    model = cx.fit_ridge(x, y, lam)
    xs = (x - model.x_mean) / model.x_scale
    ys = (y - model.y_mean) / model.y_scale
    reference = sklearn.Ridge(alpha=lam, fit_intercept=True).fit(xs, ys)
    np.testing.assert_allclose(model.weight, reference.coef_.T, atol=1e-4)
    np.testing.assert_allclose(model.predict_std(x), reference.predict(xs), atol=1e-4)


def test_ridge_residual_removes_linear_component():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((500, 4)).astype(np.float32)
    y = (x @ rng.standard_normal((4, 6))).astype(np.float32)  # 纯线性可解释
    model = cx.fit_ridge(x, y, 1e-3)
    residual = model.residual_std(x, y)
    assert float(np.abs(residual).mean()) < 1e-2
    assert model.r_squared(x, y) > 0.999


# ---------------------------------------------------------------------------
# X4：方向几何
# ---------------------------------------------------------------------------


def test_raw_direction_recovers_planted_axis_under_anisotropic_scale():
    rng = np.random.default_rng(3)
    planted = np.array([0.8, -0.6, 0.0, 0.0])
    raw = rng.standard_normal((2000, 4))
    scale = np.array([5.0, 0.2, 3.0, 1.0])
    features = (raw * scale).astype(np.float32)  # 原始空间各维尺度悬殊
    y = ((raw @ planted) > 0).astype(np.int64)
    probe = fit_linear_probe(features, y, 2, 1.0, device="cpu")
    direction = cx.raw_direction(probe)
    target = (planted / scale) / np.linalg.norm(planted / scale)
    assert abs(float(direction @ target)) > 0.95


def test_diff_means_and_projection_std_and_steer_vector():
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((400, 3)) + np.array([0.0, 0.0, 0.0])
    x1 = rng.standard_normal((400, 3)) + np.array([2.0, 0.0, 0.0])
    features = np.concatenate([x0, x1]).astype(np.float32)
    labels = np.concatenate([np.zeros(400), np.ones(400)]).astype(np.int64)
    direction = cx.diff_means_direction(features, labels)
    assert direction[0] > 0.9  # 指向类 1
    s_v = cx.projection_std(features, direction)
    assert 1.0 < s_v < 2.0  # 双峰混合的投影标准差
    steer = cx.steer_vector(-2.0, s_v, direction)
    np.testing.assert_allclose(np.linalg.norm(steer), 2.0 * s_v, rtol=1e-6)
    assert np.allclose(cx.steer_vector(0.0, s_v, direction), 0.0)


def test_cosine_matrix_and_pc_mass_spectrum():
    dirs = {"a": np.array([1.0, 0.0]), "b": np.array([-1.0, 0.0]), "c": np.array([0.0, 2.0])}
    matrix = cx.cosine_matrix(dirs)
    assert matrix["a"]["b"] == pytest.approx(-1.0)
    assert matrix["a"]["c"] == pytest.approx(0.0)
    basis = np.eye(4)
    spectrum = cx.pc_mass_spectrum(basis, np.array([1.0, 0.0, 0.0, 0.0]))
    assert spectrum["participation_ratio"] == pytest.approx(1.0)
    assert spectrum["cumulative"][0] == pytest.approx(1.0)
    spread = cx.pc_mass_spectrum(basis, np.ones(4) / 2.0)
    assert spread["participation_ratio"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# X2：轨迹
# ---------------------------------------------------------------------------


def test_aligned_matrix_windows_and_nan_padding():
    series = {("s1", 0): np.arange(10, dtype=np.float64)}
    matrix, session_index = tx.aligned_matrix(series, [("s1", 0, 1), ("s1", 0, 8)], 2)
    np.testing.assert_allclose(matrix[0], [np.nan, 0, 1, 2, 3], equal_nan=True)
    np.testing.assert_allclose(matrix[1], [6, 7, 8, 9, np.nan], equal_nan=True)
    assert session_index.tolist() == [0, 0]


def test_divergence_step_detects_separation_onset():
    rng = np.random.default_rng(5)
    n_per = 40
    width = 11  # window=5，偏移 −5..+5
    anchors = []
    series = {}
    labels = []
    for i in range(20):
        sid = f"s{i}"
        base = rng.standard_normal(40) * 0.05
        # 类 1 角色：从绝对步 22（相对锚点 20 的偏移 +2）起抬升 1.0
        lifted = base.copy()
        lifted[22:] += 1.0
        series[(sid, 0)] = lifted
        series[(sid, 1)] = base
        anchors.append((sid, 0, 20))
        labels.append(1)
        anchors.append((sid, 1, 20))
        labels.append(0)
    matrix, session_index = tx.aligned_matrix(series, anchors, 5)
    assert matrix.shape == (n_per, width)
    result = tx.divergence_step(matrix, np.asarray(labels), session_index, n_boot=200, min_consecutive=2)
    assert result["divergence_offset"] == 2


def test_next_state_step_and_onset_extraction():
    states = np.array(
        [
            State.LISTEN.value,
            State.LISTEN.value,
            State.GAP.value,
            State.SPEAK.value,
            State.SPEAK.value,
            State.LISTEN.value,
            State.OVERLAP_HOLD.value,
        ]
    )
    assert tx.next_state_step(states, 0, State.SPEAK.value, 10) == 3
    assert tx.next_state_step(states, 3, State.GAP.value, 2) is None
    assert tx.onset_steps_from_states(states).tolist() == [3, 6]


# ---------------------------------------------------------------------------
# X5：T2 视野与错误解剖
# ---------------------------------------------------------------------------


def test_t2_horizon_labels_step_grid_variant():
    speak = State.SPEAK.value
    listen = State.LISTEN.value
    hold = State.OVERLAP_HOLD.value
    states = np.array([speak, hold, hold, listen, listen, speak, speak, speak, speak])
    anchors = np.array([1, 5])
    # 锚点 1：步 2 仍发声、步 3 离开发声态 → h=1 内未离开、h=2 内离开
    assert ax.t2_horizon_labels(states, anchors, 1).tolist() == [0, 0]
    assert ax.t2_horizon_labels(states, anchors, 2).tolist() == [1, 0]
    with pytest.raises(ValueError, match="不足"):
        ax.t2_horizon_labels(states, np.array([7]), 5)
    assert ax.label_agreement(np.array([1, 0, 1]), np.array([1, 1, 1])) == pytest.approx(2 / 3)


def test_f0_slope_and_tercile_buckets():
    acoustic = np.zeros((20, 4), dtype=np.float32)
    acoustic[10:16, 1] = np.array([200, 190, 180, 170, 160, 150])  # 下降 10 Hz/步
    slope = ax.f0_slope(acoustic, anchor=15, tail_steps=6)
    assert slope == pytest.approx(-10.0, abs=1e-6)
    assert ax.f0_slope(acoustic, anchor=5, tail_steps=4) is None  # voiced 不足
    values = [float(v) for v in range(12)] + [None]
    bucket, edges = ax.tercile_buckets(values)
    assert bucket[-1] == -1
    assert edges["n_valid"] == 12
    assert sorted(np.unique(bucket[:-1]).tolist()) == [0, 1, 2]


def test_filter_cells_by_mask_enforces_row_contract():
    cells = {"s1": (np.array([1, 0, 1]), np.full((3, 2), 0.5))}
    filtered = ax.filter_cells_by_mask(cells, {"s1": np.array([True, False, True])})
    assert filtered["s1"][0].tolist() == [1, 1]
    with pytest.raises(ValueError, match="行掩码长度"):
        ax.filter_cells_by_mask(cells, {"s1": np.array([True, False])})
    with pytest.raises(ValueError, match="缺少行掩码"):
        ax.filter_cells_by_mask(cells, {})
