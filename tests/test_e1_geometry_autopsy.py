"""E1 几何解剖脚本（PREREG #32）的合成数据测试。

核心覆盖：一维投影恒等式（分析支柱）、错向几何下的能量谱/对齐秩、
方向剔除与主轴剔除、diff-in-means、Mimi 岭回归、事件窗与会话级 bootstrap。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.e1 import probe_gpu as pg

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _misaligned_dataset(n_rows: int = 4000, seed: int = 7):
    """已知几何：前 4 维大方差噪声，信号方向 = e4（中排序主轴），其余小噪声。"""
    rng = np.random.default_rng(seed)
    n_dims = 32
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    x = np.zeros((n_rows, n_dims), dtype=np.float64)
    x[:, :4] = rng.standard_normal((n_rows, 4)) * np.array([10.0, 9.0, 8.0, 7.0])
    x[:, 4] = (2.0 * y - 1.0) * 1.0 + rng.standard_normal(n_rows) * 0.3
    x[:, 5:] = rng.standard_normal((n_rows, n_dims - 5)) * 0.2
    return x.astype(np.float32), y


def _pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    center = x64.mean(axis=0)
    _, _, vt = np.linalg.svd(x64 - center, full_matrices=False)
    return center, vt


def test_identity_projection_auc_equals_full_probe_auc():
    """分析支柱：全维二分类探针 AUC == 原始空间单方向投影 AUC。"""
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
    # 信号方向与前 4 个大方差主轴近正交，能量集中在第 5 主轴。
    assert cumulative[3] < 0.10
    assert cumulative[4] > 0.90
    assert module.alignment_rank(cumulative, 0.5) == 5
    assert module.alignment_rank(cumulative, 0.95) >= 5
    assert profile["span_fraction"] == pytest.approx(1.0, abs=1e-9)
    assert module.alignment_rank(np.array([0.2, 0.4]), 0.95) is None


def test_energy_at_ks_caps_beyond_spectrum_length():
    module = _load_module()
    cumulative = np.array([0.3, 0.6, 0.9])
    table = module.energy_at_ks(cumulative, (1, 2, 16))
    assert table["1"] == pytest.approx(0.3)
    assert table["16"] == pytest.approx(0.9)


def test_top_pc_removal_keeps_misaligned_signal():
    module = _load_module()
    x, y = _misaligned_dataset()
    center, vt = _pca(x)
    probe = pg.fit_linear_probe(x, y, 2, 0.1, device="cpu")
    v_star = module.orig_space_direction(probe.weight[0], probe.scale)

    x_no_top = module.remove_top_pcs(x, center, vt, 4)
    probe_no_top = pg.fit_linear_probe(x_no_top, y, 2, 0.1, device="cpu")
    auc_no_top = pg.primary_metric(y, probe_no_top.predict_proba(x_no_top), 2)
    assert auc_no_top > 0.9
    # 剔除后的矩阵与被剔方向正交（f32 容差）。
    x_no_dir = module.remove_directions(x, v_star)
    assert float(np.abs(np.asarray(x_no_dir, np.float64) @ v_star).max()) < 1e-3


def test_iterative_nulling_collapses_planted_one_dim_signal_quickly():
    """单次剔除对方向估计误差不稳健（残留分量可被重训重新聚焦），
    迭代剔除必须在少数轮内把一维信号打到崩塌线以下。"""
    module = _load_module()
    x, y = _misaligned_dataset()

    def fit_fn(x_block, y_block):
        return pg.fit_linear_probe(x_block, y_block, 2, 0.1, device="cpu")

    result = module.iterative_nulling(
        x, y, x, y, fit_fn=fit_fn, max_directions=6, stop_auc=0.55
    )
    assert result["collapsed"]
    assert result["rounds_to_collapse"] is not None
    assert result["rounds_to_collapse"] <= 3
    assert result["auc_sequence"][0] > 0.9
    # 序列单调不增的大趋势：末端显著低于首端。
    assert result["auc_sequence"][-1] <= 0.55


def test_diff_in_means_aligns_with_planted_direction():
    module = _load_module()
    x, y = _misaligned_dataset()
    planted = np.zeros(32)
    planted[4] = 1.0
    d_unit = module.unit(module.diff_in_means(x, y))
    assert module.abs_cosine(d_unit, planted) > 0.95
    with pytest.raises(ValueError):
        module.diff_in_means(x[y == 1], y[y == 1])


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
    features_train = rng.standard_normal((600, 20))
    features_eval = rng.standard_normal((300, 20))
    beta = rng.standard_normal(20)
    target_train = features_train @ beta + rng.standard_normal(600) * 0.05
    target_eval = features_eval @ beta + rng.standard_normal(300) * 0.05
    noise_train = rng.standard_normal(600)
    noise_eval = rng.standard_normal(300)
    ridge = module.MimiRidge(features_train, 1e-4)
    r2 = ridge.r2(
        np.column_stack([target_train, noise_train]),
        features_eval,
        np.column_stack([target_eval, noise_eval]),
    )
    assert r2[0] > 0.95
    assert abs(r2[1]) < 0.2


def test_extract_event_windows_drops_boundary_events():
    module = _load_module()
    projections = np.arange(40, dtype=np.float64)[:, None]
    event_rows = np.array([2, 10, 38])
    windows, keep = module.extract_event_windows(projections, event_rows, 3, 38)
    assert keep.tolist() == [False, True, False]
    assert windows.shape == (1, 7, 1)
    assert windows[0, :, 0].tolist() == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]


def test_cluster_bootstrap_detects_planted_separation():
    module = _load_module()
    rng = np.random.default_rng(5)
    n_sessions, n_offsets = 30, 7
    count_pos = rng.integers(3, 8, size=n_sessions).astype(np.float64)
    count_neg = rng.integers(3, 8, size=n_sessions).astype(np.float64)
    shift = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    sum_pos = count_pos[:, None] * (shift[None, :] + rng.standard_normal((n_sessions, n_offsets)) * 0.05)
    sum_neg = count_neg[:, None] * (rng.standard_normal((n_sessions, n_offsets)) * 0.05)
    result = module.cluster_bootstrap_separation(sum_pos, count_pos, sum_neg, count_neg, 200, 3)
    assert result["first_offset_index_ci_excludes_zero"] == 3
    assert result["separation"][3] == pytest.approx(1.0, abs=0.1)
    assert result["ci_lower"][3] > 0.5
    with pytest.raises(ValueError):
        module.cluster_bootstrap_separation(sum_pos, count_pos * 0, sum_neg, count_neg, 10, 3)


def test_projection_stats_reports_both_classes():
    module = _load_module()
    scores = np.array([1.0, 2.0, -1.0, -2.0])
    labels = np.array([1, 1, 0, 0])
    stats = module.projection_stats(scores, labels)
    assert stats["mean_pos"] == pytest.approx(1.5)
    assert stats["mean_neg"] == pytest.approx(-1.5)
    assert stats["pooled_mean"] == pytest.approx(0.0)


def test_verdict_interval_bands():
    module = _load_module()
    assert module._verdict_interval(0.4, 0.5, 0.8) == "alpha"
    assert module._verdict_interval(0.6, 0.5, 0.8) == "indeterminate"
    assert module._verdict_interval(0.9, 0.5, 0.8) == "beta"
