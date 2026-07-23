"""E1 几何解剖脚本（PREREG #32/#35/#36/#37）的合成数据测试。

#37 核心：#36 的两个替代量本身仍有缺陷（云端实证）——
  - feature_split 仍受类内协方差混淆：判别均值方向稠密均匀分布、但高 SNR 依赖跨
    坐标噪声抵消的信号，半切后保留率仅 0.12，会被误判"局部化"（本文件反例固化）。
    → 降为描述量；坐标分布性判读改协方差无关的 participation ratio（PR）。
  - 白化残差塌缩是 mean-projection tautology（随机标签也塌缩），且白化子空间漏
    信号时仍 collapsed=True（无 full-auc 复现门）→ 撤销；秩-1 改解析陈述。
另：转向包 proj_std 改去重训练并集口径，与 E1-X 一致（本文件 dedup 测试固化）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = {"lbfgs_max_iter": 300, "lbfgs_tolerance_grad": 1e-6}
C_GRID = [0.001, 0.01, 0.1, 1.0]


def _load_module():
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "wp_e1_geometry_autopsy_under_test",
        scripts / "wp_e1_geometry_autopsy.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fit_fn(x_block, y_block):
    return pg.fit_linear_probe(x_block, y_block, 2, 0.1, device="cpu")


def _misaligned_dataset(n_rows: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    n_dims = 32
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = np.zeros((n_rows, n_dims), dtype=np.float64)
    x[:, :4] = rng.standard_normal((n_rows, 4)) * np.array([10.0, 9.0, 8.0, 7.0])
    x[:, 4] = (2.0 * y - 1.0) * 1.0 + rng.standard_normal(n_rows) * 0.3
    x[:, 5:] = rng.standard_normal((n_rows, n_dims - 5)) * 0.2
    return x.astype(np.float32), y


def _dense_covariance_trap(n_rows: int = 8000, seed: int = 0):
    """#37 反例：判别均值方向均匀分布于全部坐标，类内协方差沿该方向方差极小
    （高 SNR 依赖跨坐标噪声抵消）。feature_split 误判局部化；PR 正确判分布式。"""
    rng = np.random.default_rng(seed)
    D = 32
    d = np.ones(D) / np.sqrt(D)
    cov = 0.01 * np.outer(d, d) + 1.0 * (np.eye(D) - np.outer(d, d))
    chol = np.linalg.cholesky(cov)
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = ((rng.standard_normal((n_rows, D)) @ chol.T) + np.outer((2 * y - 1) * 0.12, d)).astype(np.float32)
    return x, y


def _localized_dataset(n_rows: int = 4000, seed: int = 1):
    rng = np.random.default_rng(seed)
    n_dims = 32
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = rng.standard_normal((n_rows, n_dims)) * 0.5
    x[:, 0] += (2.0 * y - 1.0) * 1.0
    return x.astype(np.float32), y


def _pca(x: np.ndarray):
    x64 = np.asarray(x, dtype=np.float64)
    center = x64.mean(axis=0)
    _, _, vt = np.linalg.svd(x64 - center, full_matrices=False)
    return center, vt


# --------------------------------------------------------------------------- #
# 分析支柱
# --------------------------------------------------------------------------- #


def test_identity_projection_auc_equals_full_probe_auc():
    module = _load_module()
    x, y = _misaligned_dataset()
    probe = pg.fit_linear_probe(x, y, 2, 0.1, device="cpu")
    auc_full = pg.primary_metric(y, probe.predict_proba(x), 2)
    v_star = module.orig_space_direction(probe.weight[0], probe.scale)
    auc_projection = module.projection_auc(y, np.asarray(x, dtype=np.float64) @ v_star)
    assert auc_full > 0.9
    assert abs(auc_full - auc_projection) < 1e-6


def test_energy_profile_localizes_misaligned_signal_direction():
    module = _load_module()
    x, y = _misaligned_dataset()
    _center, vt = _pca(x)
    probe = pg.fit_linear_probe(x, y, 2, 0.1, device="cpu")
    v_star = module.orig_space_direction(probe.weight[0], probe.scale)
    cumulative = module.energy_profile(vt, v_star)["cumulative"]
    assert cumulative[3] < 0.10
    assert cumulative[4] > 0.90
    assert module.alignment_rank(cumulative, 0.5) == 5
    assert module.alignment_rank(np.array([0.2, 0.4]), 0.95) is None


def test_top_pc_removal_keeps_misaligned_signal():
    module = _load_module()
    x, y = _misaligned_dataset()
    center, vt = _pca(x)
    x_no_top = module.remove_top_pcs(x, center, vt, 4)
    probe = pg.fit_linear_probe(x_no_top, y, 2, 0.1, device="cpu")
    assert pg.primary_metric(y, probe.predict_proba(x_no_top), 2) > 0.9


# --------------------------------------------------------------------------- #
# #37：participation ratio 是协方差无关的坐标分布度量
# --------------------------------------------------------------------------- #


def test_participation_ratio_dense_vs_sparse():
    module = _load_module()
    D = 64
    dense = module.coordinate_concentration(np.ones(D))
    assert dense["participation_ratio"] == pytest.approx(D)
    assert dense["participation_fraction"] == pytest.approx(1.0)
    sparse = np.zeros(D)
    sparse[0] = 1.0
    conc = module.coordinate_concentration(sparse)
    assert conc["participation_ratio"] == pytest.approx(1.0)
    assert conc["participation_fraction"] == pytest.approx(1.0 / D)
    assert conc["top16_coord_mass"] == pytest.approx(1.0)


def test_participation_ratio_solves_covariance_trap_where_feature_split_fails():
    """核心 #37：稠密均匀信号 + 低方差判别方向。feature_split 误判局部化（<0.25），
    而 PR(v*)/D 正确判分布式（高）。固化"为什么改用 participation ratio"。"""
    module = _load_module()
    x, y = _dense_covariance_trap()
    xtr, xte, ytr, yte = x[:4000], x[4000:], y[:4000], y[4000:]
    # feature_split（描述量）被协方差混淆：稠密信号却给低保留率。
    fs = module.feature_split_redundancy(xtr, ytr, xte, yte, C_GRID, TRAINER, "cpu", folds=4, seed=0)
    assert fs["full_auc"] > 0.85
    assert fs["median_retention"] < 0.25  # 会被旧 α/β 误判"局部化"
    # participation ratio（协方差无关）正确判分布式。
    probe = pg.fit_linear_probe(xtr, ytr, 2, 0.1, device="cpu")
    v_star = module.orig_space_direction(probe.weight[0], probe.scale)
    conc = module.coordinate_concentration(v_star)
    assert conc["participation_fraction"] >= 0.5  # 稠密读出


def test_participation_ratio_calls_localized_sparse():
    module = _load_module()
    x, y = _localized_dataset()
    probe = pg.fit_linear_probe(x, y, 2, 0.1, device="cpu")
    v_star = module.orig_space_direction(probe.weight[0], probe.scale)
    conc = module.coordinate_concentration(v_star)
    assert conc["participation_fraction"] < 0.1  # 集中于少数神经元


# --------------------------------------------------------------------------- #
# 均值投影自检（数学必然，仅 sanity）
# --------------------------------------------------------------------------- #


def test_mean_projection_forces_train_chance():
    module = _load_module()
    for maker in (_localized_dataset, _dense_covariance_trap):
        x, y = maker()
        check = module.mean_projection_check(x, y, x, y, fit_fn=_fit_fn)
        assert check["train_mean_gap_after_projection"] < 1e-4
        assert abs(check["train_auc"] - 0.5) < module.MEAN_PROJECTION_TOL


def test_refit_auc_grid_reports_all_c_and_max():
    module = _load_module()
    x, y = _misaligned_dataset(n_rows=1500)
    result = module.refit_auc_grid(x, y, x, y, [0.001, 0.1], TRAINER, "cpu")
    assert set(result["auc_by_c"]) == {"0.001", "0.1"}
    assert result["max_auc"] == pytest.approx(max(result["auc_by_c"].values()))
    assert result["max_auc"] > 0.9
    assert result["nonconverged"] == 0


# --------------------------------------------------------------------------- #
# #37：转向包 proj_std 去重并集口径（与 E1-X 一致）
# --------------------------------------------------------------------------- #


def test_union_train_roles_dedups_by_session_channel_step():
    module = _load_module()
    train_rows = {
        ("T4", 0): [g.RoleRows("s1", 0, np.array([1, 2, 3]), np.array([0, 1, 0]))],
        ("T4", 1): [g.RoleRows("s1", 0, np.array([2, 3, 4]), np.array([1, 0, 1]))],
        ("T4", 2): [g.RoleRows("s1", 0, np.array([3, 5]), np.array([0, 1]))],
    }
    union = module._union_train_roles(train_rows, [0, 1, 2])
    assert len(union) == 1
    role = union[0]
    assert role.session_id == "s1" and role.agent_channel == 0
    # 唯一 step 集合 = {1,2,3,4,5}，标签取首次出现口径（step 3 首见 seed0 label 0）。
    assert role.steps.tolist() == [1, 2, 3, 4, 5]
    assert role.labels.tolist() == [0, 1, 0, 1, 1]


def test_projection_std_on_matches_numpy():
    module = _load_module()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 8)).astype(np.float32)
    v = rng.standard_normal(8)
    v /= np.linalg.norm(v)
    result = module._projection_std_on(x, {"v": v})
    assert result["v"] == pytest.approx(float((x.astype(np.float64) @ v).std()), rel=1e-4)


# --------------------------------------------------------------------------- #
# 方向工具 / 符号规范
# --------------------------------------------------------------------------- #


def test_diff_in_means_aligns_with_planted_direction():
    module = _load_module()
    x, y = _misaligned_dataset()
    planted = np.zeros(32)
    planted[4] = 1.0
    d_unit = module.unit(module.diff_in_means(x, y))
    assert module.abs_cosine(d_unit, planted) > 0.95
    with pytest.raises(ValueError):
        module.diff_in_means(x[y == 1], y[y == 1])


def test_mean_direction_label1_verifies_orientation():
    module = _load_module()
    v = np.array([1.0, 0.0])
    good = {"mean_pos": 1.0, "mean_neg": -1.0}
    bad = {"mean_pos": -1.0, "mean_neg": 1.0}
    mean = module.mean_direction_label1([v, v], [good, good])
    assert module.abs_cosine(mean, v) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        module.mean_direction_label1([v, v], [good, bad])


def test_probe_direction_is_canonically_label1_oriented():
    module = _load_module()
    x, y = _misaligned_dataset()
    dirs = [
        module.orig_space_direction(
            pg.fit_linear_probe(x, y, 2, c, device="cpu").weight[0],
            pg.fit_linear_probe(x, y, 2, c, device="cpu").scale,
        )
        for c in (0.05, 0.1, 0.2)
    ]
    signed = [float(np.dot(module.unit(dirs[i]), module.unit(dirs[j]))) for i in range(3) for j in range(i + 1, 3)]
    assert min(signed) > 0


def test_orig_space_direction_validates_inputs():
    module = _load_module()
    with pytest.raises(ValueError):
        module.orig_space_direction(np.ones(4), np.array([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        module.orig_space_direction(np.ones(4), np.ones(3))
    with pytest.raises(ValueError):
        module.unit(np.zeros(4))


def test_mimi_ridge_separates_predictable_and_random_targets():
    module = _load_module()
    rng = np.random.default_rng(11)
    ft = rng.standard_normal((600, 20))
    fe = rng.standard_normal((300, 20))
    beta = rng.standard_normal(20)
    tt = ft @ beta + rng.standard_normal(600) * 0.05
    te = fe @ beta + rng.standard_normal(300) * 0.05
    ridge = module.MimiRidge(ft, 1e-4)
    r2 = ridge.r2(
        np.column_stack([tt, rng.standard_normal(600)]),
        fe,
        np.column_stack([te, rng.standard_normal(300)]),
    )
    assert r2[0] > 0.95
    assert abs(r2[1]) < 0.2


# --------------------------------------------------------------------------- #
# 轨迹统计：sup-t 同时带 + 持续显著
# --------------------------------------------------------------------------- #


def test_extract_event_windows_drops_boundary_events():
    module = _load_module()
    projections = np.arange(40, dtype=np.float64)[:, None]
    windows, keep = module.extract_event_windows(projections, np.array([2, 10, 38]), 3, 38)
    assert keep.tolist() == [False, True, False]
    assert windows.shape == (1, 7, 1)
    assert windows[0, :, 0].tolist() == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]


def test_sustained_onset_requires_consecutive_run():
    module = _load_module()
    flags = np.array([False, True, False, True, True, True, False])
    assert module.sustained_onset(flags, 3) == 3
    assert module.sustained_onset(flags, 4) is None
    assert module.sustained_onset(np.zeros(5, dtype=bool), 3) is None


def test_cluster_bootstrap_sup_t_controls_null_false_positives():
    module = _load_module()
    rng = np.random.default_rng(21)
    n_sessions, n_offsets = 30, 51
    count_pos = rng.integers(3, 8, size=n_sessions).astype(np.float64)
    count_neg = rng.integers(3, 8, size=n_sessions).astype(np.float64)
    sum_pos = count_pos[:, None] * rng.standard_normal((n_sessions, n_offsets)) * 0.3
    sum_neg = count_neg[:, None] * rng.standard_normal((n_sessions, n_offsets)) * 0.3
    result = module.cluster_bootstrap_separation(
        sum_pos, count_pos, sum_neg, count_neg, 300, 5, min_consecutive=3
    )
    assert result["onset_index_sustained"] is None
    assert result["sup_t_quantile_95"] > 1.96


def test_cluster_bootstrap_detects_planted_sustained_separation():
    module = _load_module()
    rng = np.random.default_rng(5)
    n_sessions, n_offsets = 30, 9
    count_pos = rng.integers(4, 9, size=n_sessions).astype(np.float64)
    count_neg = rng.integers(4, 9, size=n_sessions).astype(np.float64)
    shift = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    sum_pos = count_pos[:, None] * (shift[None, :] + rng.standard_normal((n_sessions, n_offsets)) * 0.05)
    sum_neg = count_neg[:, None] * (rng.standard_normal((n_sessions, n_offsets)) * 0.05)
    result = module.cluster_bootstrap_separation(
        sum_pos, count_pos, sum_neg, count_neg, 300, 3, min_consecutive=3
    )
    assert result["onset_index_sustained"] == 3
    assert result["simultaneous_lower"][3] > 0.5
    with pytest.raises(ValueError):
        module.cluster_bootstrap_separation(sum_pos, count_pos * 0, sum_neg, count_neg, 10, 3)


def test_projection_stats_reports_both_classes():
    module = _load_module()
    stats = module.projection_stats(np.array([1.0, 2.0, -1.0, -2.0]), np.array([1, 1, 0, 0]))
    assert stats["mean_pos"] == pytest.approx(1.5)
    assert stats["mean_neg"] == pytest.approx(-1.5)


def test_verdict_interval_bands():
    module = _load_module()
    assert module._verdict_interval(0.4, 0.5, 0.8) == "alpha"
    assert module._verdict_interval(0.6, 0.5, 0.8) == "indeterminate"
    assert module._verdict_interval(0.9, 0.5, 0.8) == "beta"
