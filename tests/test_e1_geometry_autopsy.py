"""E1 几何解剖脚本（PREREG #32/#35/#36）的合成数据测试。

#36 核心：方向剔除法（INLP r₁）受协方差旋转混淆——严格单均值信号在相关噪声下
会被误判为"厚方向束"。改用两个协方差无关/纠正的度量：
  - feature_split_redundancy：特征子集冗余，区分"分布式承载于众多神经元"vs"集中"；
  - whitened_rank_check：白化后剔除唯一判别方向，残差→0.5 确认二分类线性读出秩=1，
    且不受 Σ⁻¹d 相对 d 的旋转混淆（单均值信号正确塌缩）。
均值投影自检（Mean Projection，非 LEACE）仅作实现 sanity，其一轮归零是数学必然。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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
    """前 4 维大方差噪声，信号方向 = e4（中排序主轴），其余小噪声。"""
    rng = np.random.default_rng(seed)
    n_dims = 32
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = np.zeros((n_rows, n_dims), dtype=np.float64)
    x[:, :4] = rng.standard_normal((n_rows, 4)) * np.array([10.0, 9.0, 8.0, 7.0])
    x[:, 4] = (2.0 * y - 1.0) * 1.0 + rng.standard_normal(n_rows) * 0.3
    x[:, 5:] = rng.standard_normal((n_rows, n_dims - 5)) * 0.2
    return x.astype(np.float32), y


def _single_mean_correlated_noise(n_rows: int = 6000, seed: int = 0):
    """#36 反例：标签均值严格沿 e0，其余为标签无关的相关高斯噪声。

    最优判别方向 ≈ Σ⁻¹e0 偏离 e0；未白化方向剔除后 e0 分量仍可读（误判厚方向束），
    白化后剔除唯一判别方向则正确塌缩到 ~0.5。
    """
    rng = np.random.default_rng(seed)
    n_dims = 24
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    a = rng.standard_normal((n_dims, n_dims))
    cov = a @ a.T / n_dims + 0.1 * np.eye(n_dims)
    noise = rng.standard_normal((n_rows, n_dims)) @ np.linalg.cholesky(cov).T
    d0 = np.zeros(n_dims)
    d0[0] = 1.0
    x = (noise + np.outer((2.0 * y - 1.0) * 1.2, d0)).astype(np.float32)
    return x, y


def _localized_dataset(n_rows: int = 4000, seed: int = 1):
    rng = np.random.default_rng(seed)
    n_dims = 32
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = rng.standard_normal((n_rows, n_dims)) * 0.5
    x[:, 0] += (2.0 * y - 1.0) * 1.0
    return x.astype(np.float32), y


def _distributed_dataset(n_rows: int = 4000, seed: int = 2):
    rng = np.random.default_rng(seed)
    n_dims = 32
    u = rng.standard_normal(n_dims)
    u /= np.linalg.norm(u)
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = rng.standard_normal((n_rows, n_dims)) * 0.5 + np.outer((2.0 * y - 1.0) * 1.0, u)
    return x.astype(np.float32), y


def _pca(x: np.ndarray):
    x64 = np.asarray(x, dtype=np.float64)
    center = x64.mean(axis=0)
    _, _, vt = np.linalg.svd(x64 - center, full_matrices=False)
    return center, vt


# --------------------------------------------------------------------------- #
# 分析支柱：一维投影恒等式
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
    profile = module.energy_profile(vt, v_star)
    cumulative = profile["cumulative"]
    assert cumulative[3] < 0.10
    assert cumulative[4] > 0.90
    assert module.alignment_rank(cumulative, 0.5) == 5
    assert module.alignment_rank(np.array([0.2, 0.4]), 0.95) is None


def test_energy_at_ks_caps_beyond_spectrum_length():
    module = _load_module()
    table = module.energy_at_ks(np.array([0.3, 0.6, 0.9]), (1, 2, 16))
    assert table["1"] == pytest.approx(0.3)
    assert table["16"] == pytest.approx(0.9)


def test_top_pc_removal_keeps_misaligned_signal():
    module = _load_module()
    x, y = _misaligned_dataset()
    center, vt = _pca(x)
    x_no_top = module.remove_top_pcs(x, center, vt, 4)
    probe = pg.fit_linear_probe(x_no_top, y, 2, 0.1, device="cpu")
    assert pg.primary_metric(y, probe.predict_proba(x_no_top), 2) > 0.9


# --------------------------------------------------------------------------- #
# #36：feature-split 分布式判据（协方差无关）
# --------------------------------------------------------------------------- #


def test_feature_split_separates_localized_from_distributed():
    module = _load_module()
    xl, yl = _localized_dataset()
    localized = module.feature_split_redundancy(
        xl[:2000], yl[:2000], xl[2000:], yl[2000:], C_GRID, TRAINER, "cpu", folds=4, seed=0
    )
    assert localized["full_auc"] > 0.9
    assert localized["median_retention"] < 0.25  # 集中 → β

    xd, yd = _distributed_dataset()
    distributed = module.feature_split_redundancy(
        xd[:2000], yd[:2000], xd[2000:], yd[2000:], C_GRID, TRAINER, "cpu", folds=4, seed=0
    )
    assert distributed["full_auc"] > 0.9
    assert distributed["min_retention"] >= 0.5  # 分布式 → α
    assert distributed["nonconverged"] == 0


# --------------------------------------------------------------------------- #
# #36：白化秩-1 确认解决协方差混淆
# --------------------------------------------------------------------------- #


def test_whitened_check_collapses_single_mean_that_direction_removal_misjudges():
    module = _load_module()
    x, y = _single_mean_correlated_noise()
    xtr, xte, ytr, yte = x[:3000], x[3000:], y[:3000], y[3000:]
    ctr, vt_tr = _pca(xtr)
    result = module.whitened_rank_check(
        xtr, ytr, xte, yte, ctr, vt_tr, C_GRID, TRAINER, "cpu", k=24, shrinkage=1e-2
    )
    assert result["whitened_full_auc"] > 0.9
    # 关键：严格单均值信号白化后残差塌缩（未白化方向剔除会误判为厚方向束）。
    assert result["residual_auc"] <= module.WHITEN_COLLAPSE_MAX
    assert result["collapsed"]


def test_whitened_check_collapses_isotropic_and_distributed():
    module = _load_module()
    for maker in (_localized_dataset, _distributed_dataset):
        x, y = maker()
        ctr, vt = _pca(x[:2000])
        result = module.whitened_rank_check(
            x[:2000], y[:2000], x[2000:], y[2000:], ctr, vt, C_GRID, TRAINER, "cpu",
            k=min(31, x.shape[1] - 1), shrinkage=1e-2,
        )
        assert result["residual_auc"] <= module.WHITEN_COLLAPSE_MAX


# --------------------------------------------------------------------------- #
# 均值投影自检（数学必然，仅 sanity）
# --------------------------------------------------------------------------- #


def test_mean_projection_forces_train_chance():
    module = _load_module()
    for maker in (_localized_dataset, _distributed_dataset, _single_mean_correlated_noise):
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
    """探针 w/σ 天然指向 label=1，跨种子同向：directions 阶段直接平均无需翻转。"""
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
    assert min(signed) > 0  # 全部同向（正号）——无需 sign_aligned_mean


def test_sign_aligned_mean_flips_opposite_directions():
    module = _load_module()
    base = module.unit(np.array([1.0, 2.0, 3.0]))
    mean = module.sign_aligned_mean([base, -base, base])
    assert module.abs_cosine(mean, base) == pytest.approx(1.0)


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
    assert result["separation"][3] == pytest.approx(1.0, abs=0.1)
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
