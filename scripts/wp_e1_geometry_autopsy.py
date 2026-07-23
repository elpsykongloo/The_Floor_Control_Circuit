"""E1 几何解剖第一梯队（PREREG #32/#35/#36，探索性、严格非裁决）。

对 G2 有效秩失败做机制解剖（文档/04 §2）：T4 的线性读出在数学上是一维方向
v* = (w/σ)/‖w/σ‖（AUC 对单调变换不变），本脚本量化该方向与激活方差主轴的
错向程度、其空间分布性，并给出 E2 方向注入所需的转向向量包。四个 stage：

  directions  零激活载入：8 个二分类规格 × 32 层 × 3 种子的原始空间方向族
              （来自 work/fits 断点），层间传播、跨种子稳定性、T1×T4 方向关系；
  spectrum    L29/L30/L31 × 3 种子：PCA 能量谱与对齐秩、特征折冗余、白化秩-1
              确认、均值投影自检、补空间重训、Mimi 主轴鉴定、转向包（e1x schema）；
  trajectory  评估集事件锁定轨迹：对方 IPU 末步 ±25 步窗内方向投影按
              complete/incomplete 分组的会话级 bootstrap 均值带（sup-t 同时带）；
  finalize    汇总判读矩阵（文档/04 §2.2）并写 reports/。

冗余度量的两条数学约束（#36 修订，均由合成测试固化）：
  (1) 均值信号的线性擦除秩恒为 1（LEACE guardedness）：擦除 diff-in-means 后两类
      质心精确相等，凸损失下零权重最优，一轮归零对 1/k 维均值信号一视同仁——
      "擦除轮数"不度量维数；本脚本仅保留为 Mean Projection 实现自检。
  (2) 方向剔除法（INLP r₁）受协方差旋转混淆：最优判别方向 ≈ Σ⁻¹d 偏离均值方向 d，
      剔除后 d 的正交分量仍可读，单均值信号被误判为厚方向束（#36 实证 r₁=0.78）——
      已撤销。冗余改协方差无关的**特征折**（不相交半空间各自重训，判"分布式 vs
      局部化"）；秩由**白化秩-1 确认**（白化后剔除唯一判别方向，残差→0.5）刻画。

一切结果不回写正式汇总，不改变 G2=fail；断点按协议哈希（v2：绑定 script/probe_gpu/
grid/engine/alignment 源码摘要、行域签名、正式汇总摘要，续跑另比对逐任务 fit SHA）
隔离，重复启动只算缺失点。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import wp_e1_probe_grid as engine
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import load_config
from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg
from floor_circuit.mve import alignment as _alignment
from floor_circuit.mve.alignment import ANALYSIS_MAX_LABEL_STEP

SCHEMA_VERSION = 2
PROTOCOL_NAME = "geometry-autopsy-v2"
TARGET = "T4"
SPECTRUM_LAYERS = (29, 30, 31)
BINARY_SPECS = ("T1_d0", "T1_d80", "T1_d160", "T1_d240", "T1_d400", "T1_d800", "T2", "T4")
N_LAYERS = 32
REMOVE_TOP_PC_KS = (1, 8, 16, 32, 64, 128)
ENERGY_KS = (1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512)
ALIGN_RHOS = (0.5, 0.8, 0.95)
MIMI_TOP_PCS = 32
RIDGE_LAMBDA_SCALE = 1e-3
FEATURE_SPLIT_FOLDS = 4
FEATURE_SPLIT_SEED = 20260723
# 坐标分布性阈值（暂定，#37）：参照随机高斯方向 PR/D≈1/3；α 门为其约 1/3，β 门≤~41 神经元。
PR_ALPHA_MIN = 0.10
PR_BETA_MAX = 0.01
MEAN_PROJECTION_TOL = 0.05
TRAJ_HALF_WINDOW = 25
TRAJ_N_BOOT = 1000
TRAJ_SEED = 20260723
TRAJ_MIN_CONSECUTIVE = 3
IDENTITY_AUC_TOLERANCE = 1e-3
FULL_REFIT_TOLERANCE = 5e-3


# ---------------------------------------------------------------------------
# 纯函数（tests/test_e1_geometry_autopsy.py 直接覆盖）
# ---------------------------------------------------------------------------


def unit(vector: np.ndarray) -> np.ndarray:
    """float64 单位化；零向量硬失败。"""
    v = np.asarray(vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("方向向量范数为零或非有限")
    return v / norm


def orig_space_direction(weight_std: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """标准化空间权重 → 原始激活空间等效判别方向 (w/σ)，单位化。

    探针分数 s(x) = w·((x−μ)/σ) + b 是 (w/σ)·x 的单调变换，AUC 由后者唯一决定。
    """
    w = np.asarray(weight_std, dtype=np.float64).ravel()
    sigma = np.asarray(scale, dtype=np.float64).ravel()
    if w.shape != sigma.shape:
        raise ValueError("权重与标准化 σ 形状不一致")
    if np.any(sigma <= 0):
        raise ValueError("标准化 σ 必须为正")
    return unit(w / sigma)


def abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.dot(unit(a), unit(b))))


def sign_aligned_mean(directions: list[np.ndarray]) -> np.ndarray:
    """以第一个方向为符号参考，翻转反向者后平均并单位化。

    仅用于**无自然符号**的方向（PCA 主轴等）。有标签语义的方向（探针方向、
    diff-means）必须走 mean_direction_label1，保证 +v 指向 label=1。
    """
    if not directions:
        raise ValueError("至少需要一个方向")
    reference = unit(directions[0])
    stacked = [reference]
    for vector in directions[1:]:
        v = unit(vector)
        stacked.append(-v if float(np.dot(reference, v)) < 0 else v)
    return unit(np.mean(np.stack(stacked), axis=0))


def mean_direction_label1(directions: list[np.ndarray], stats: list[dict]) -> np.ndarray:
    """对已按 label=1 取向的方向组做平均单位化；成员取向错误则硬失败。

    E2 注入依赖符号约定"+v 指向 T4 label=1"；这里用各成员的训练集类条件投影
    统计（mean_pos/mean_neg）显式核验取向，不做静默翻转。
    """
    if len(directions) != len(stats) or not directions:
        raise ValueError("方向与统计数量不一致或为空")
    for index, stat in enumerate(stats):
        if not float(stat["mean_pos"]) > float(stat["mean_neg"]):
            raise ValueError(f"方向 {index} 未指向 label=1（mean_pos ≤ mean_neg）")
    return unit(np.mean(np.stack([unit(v) for v in directions]), axis=0))


def projection_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """一维投影分数的 AUC（正类 = label 1）；口径与正式 primary_metric 相同。"""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    probs = np.column_stack([-scores, scores])
    return pg.primary_metric(np.asarray(labels), probs, 2)


def energy_profile(vt: np.ndarray, direction: np.ndarray) -> dict:
    """方向在 PCA 基（vt 行 = 主轴，按方差降序）上的能量累计。"""
    basis = np.asarray(vt, dtype=np.float64)
    v = unit(direction)
    if basis.ndim != 2 or basis.shape[1] != v.shape[0]:
        raise ValueError("PCA 基与方向维度不一致")
    coefficients = basis @ v
    energy = np.square(coefficients)
    cumulative = np.cumsum(energy)
    return {
        "cumulative": cumulative,
        "span_fraction": float(cumulative[-1]),
        "n_components": int(basis.shape[0]),
    }


def energy_at_ks(cumulative: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    """E(k) 表；k 超出谱长时取谱内全量并按谱长键控（capped）。"""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    out: dict[str, float] = {}
    for k in ks:
        index = min(int(k), len(cumulative))
        out[str(k)] = float(cumulative[index - 1])
    return out


def alignment_rank(cumulative: np.ndarray, rho: float) -> int | None:
    """对齐秩：min{k: E(k) ≥ ρ}；谱内达不到则 None。"""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    reached = np.flatnonzero(cumulative >= float(rho))
    return int(reached[0]) + 1 if len(reached) else None


def diff_in_means(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """原始空间 diff-in-means 方向 μ₊ − μ₋（未单位化，float64）。"""
    y = np.asarray(labels)
    positive = np.asarray(features[y == 1], dtype=np.float64)
    negative = np.asarray(features[y == 0], dtype=np.float64)
    if not len(positive) or not len(negative):
        raise ValueError("diff-in-means 需要正负类同时存在")
    return positive.mean(axis=0) - negative.mean(axis=0)


def coordinate_concentration(direction: np.ndarray) -> dict:
    """给定方向在原始神经元基上的坐标集中度（#37）。

    participation_ratio PR = (Σ v_i²)² / Σ v_i⁴ = 有效非零坐标数（1..D）：稠密均匀
    → PR≈D（方向密布于众多神经元）；集中于少数坐标 → PR≈少数。另报 PR/D、前 16
    坐标能量占比、Gini。**PR 算子本身是向量坐标分布的纯函数（不重训、不涉及 Σ）**；
    但注意读出方向 v\*=w/σ≈Σ⁻¹d 本身受类内协方差整形（实证：单神经元信号在相干负
    相关协方差下 PR(v\*)/D 可达 0.965 → 单看 v\* 会误判分布式）。因此判读用 v\* 与
    **信号方向 d=μ₊−μ₀（协方差无关）双门**：二者皆稠密才判分布式，读出因协方差稠密
    但 d 稀疏时判 indeterminate（见 _summarize 与 tests）。d 取原始空间（"哪些神经元
    承载原始均值移位"），其经验估计在近零信号上有 ~1/3 采样噪声底，故 d 门对强信号
    （如 T4 AUC 0.83）才是有效证书。
    """
    v = unit(direction)
    squared = np.square(v)
    dims = len(v)
    pr = float(1.0 / np.sum(np.square(squared)))  # = (Σv²)²/Σv⁴，此处 Σv²=1
    order = np.sort(squared)[::-1]
    top16 = float(order[: min(16, dims)].sum())
    absolute = np.abs(v)
    sorted_abs = np.sort(absolute)
    cumulative = np.cumsum(sorted_abs)
    total = float(cumulative[-1])
    gini = float(
        (dims + 1 - 2.0 * (cumulative.sum() / total)) / dims if total > 0 else 0.0
    )
    return {
        "participation_ratio": pr,
        "participation_fraction": pr / dims,
        "top16_coord_mass": top16,
        "gini": gini,
        "n_dims": dims,
    }


def remove_directions(features: np.ndarray, basis_rows: np.ndarray) -> np.ndarray:
    """从特征中投影剔除给定（正交化后的）方向组；返回 float32 新副本。

    全程 float32 以控制大矩阵内存峰值；剔除残余 ~1e-7 相对量级，对重训无影响。
    """
    matrix = np.asarray(features, dtype=np.float32)
    basis = np.asarray(basis_rows, dtype=np.float64)
    if basis.ndim == 1:
        basis = basis[None, :]
    q, _ = np.linalg.qr(basis.T)
    q32 = q.astype(np.float32)
    coords = matrix @ q32
    return matrix - coords @ q32.T


def remove_top_pcs(
    features: np.ndarray, center: np.ndarray, vt: np.ndarray, k: int
) -> np.ndarray:
    """剔除中心化后前 k 个方差主轴分量；返回 float32 新副本（内存同上）。"""
    if k <= 0 or k > vt.shape[0]:
        raise ValueError("k 必须落在 PCA 谱范围内")
    basis = np.asarray(vt[:k], dtype=np.float32)
    matrix = np.asarray(features, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    matrix -= (matrix @ basis.T) @ basis
    return matrix


def projection_stats(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """训练集一维投影的类条件统计（E2 剂量 α 的 σ 标定）。"""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if not len(pos) or not len(neg):
        raise ValueError("投影统计需要正负类同时存在")
    return {
        "mean_pos": float(pos.mean()),
        "mean_neg": float(neg.mean()),
        "std_pos": float(pos.std()),
        "std_neg": float(neg.std()),
        "pooled_mean": float(scores.mean()),
        "pooled_std": float(scores.std()),
    }


def refit_auc_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    c_grid: list[float],
    trainer: dict,
    device: str,
) -> dict:
    """变换后特征的全 C 档 warm-start 重训；评估集取最大 AUC。

    残余信号判定不得沿用原始探针选出的正则强度（#35 中风险）：取全网格
    评估集最大值给残余信号最大机会，作为"信号仍在吗"的敏感性上界（非正式
    性能声明，正式口径仍是 inner_val 嵌套选择）。
    """
    warm = None
    auc_by_c: dict[str, float] = {}
    nonconverged = 0
    for c_value in c_grid:
        probe = pg.fit_linear_probe(
            x_train,
            y_train,
            2,
            float(c_value),
            device=device,
            max_iter=int(trainer["lbfgs_max_iter"]),
            tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
            init=warm,
        )
        warm = probe
        nonconverged += int(not probe.converged)
        auc_by_c[str(float(c_value))] = float(
            pg.primary_metric(y_eval, probe.predict_proba(x_eval), 2)
        )
    return {
        "auc_by_c": auc_by_c,
        "max_auc": max(auc_by_c.values()),
        "nonconverged": nonconverged,
    }


def feature_split_redundancy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    c_grid: list[float],
    trainer: dict,
    device: str,
    *,
    folds: int,
    seed: int,
) -> dict:
    """原生坐标半切可解码性（**描述量，非 α/β 判据**，#37 降级）。

    把激活维度随机对半分为不相交两折，各折独立重训探针（全 C 档评估集最大），
    retention_f = (min(AUC_A, AUC_B) − 0.5) / (AUC_full − 0.5)。**警告**：本量并非
    协方差无关，不能独立回答"信号是否分布于众多神经元"——#37 实证反例：一个判别
    均值方向严格均匀分布于全部坐标、但类内协方差沿该方向方差极小（高 SNR 依赖跨
    坐标噪声抵消）的稠密信号，半切后中位保留率仅 0.12，会被误判"局部化"。原因：半
    空间破坏了全局低噪声方向的噪声抵消，SNR 塌陷。故保留为描述量（同时受信号载荷、
    类内协方差、跨半区噪声抵消决定），坐标分布性判读改用 coordinate_concentration
    的 participation ratio（协方差无关）。
    """
    n_dim = int(x_train.shape[1])
    full = refit_auc_grid(x_train, y_train, x_eval, y_eval, c_grid, trainer, device)
    base = full["max_auc"] - 0.5
    nonconverged = int(full["nonconverged"])
    rng = np.random.default_rng(int(seed))
    per_fold = []
    for _ in range(int(folds)):
        perm = rng.permutation(n_dim)
        half = n_dim // 2
        cols_a = np.sort(perm[:half])
        cols_b = np.sort(perm[half:])
        fit_a = refit_auc_grid(
            np.ascontiguousarray(x_train[:, cols_a]), y_train,
            np.ascontiguousarray(x_eval[:, cols_a]), y_eval, c_grid, trainer, device,
        )
        fit_b = refit_auc_grid(
            np.ascontiguousarray(x_train[:, cols_b]), y_train,
            np.ascontiguousarray(x_eval[:, cols_b]), y_eval, c_grid, trainer, device,
        )
        nonconverged += int(fit_a["nonconverged"]) + int(fit_b["nonconverged"])
        min_auc = min(fit_a["max_auc"], fit_b["max_auc"])
        per_fold.append(
            {
                "auc_a": fit_a["max_auc"],
                "auc_b": fit_b["max_auc"],
                "min_auc": min_auc,
                "retention": (min_auc - 0.5) / base if base > 0 else None,
            }
        )
    retentions = [f["retention"] for f in per_fold if f["retention"] is not None]
    return {
        "full_auc": full["max_auc"],
        "per_fold": per_fold,
        "median_retention": float(np.median(retentions)) if retentions else None,
        "min_retention": float(min(retentions)) if retentions else None,
        "folds": int(folds),
        "nonconverged": nonconverged,
    }


def mean_projection_check(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    fit_fn,
) -> dict:
    """正交均值投影自检（Mean Projection；非 LEACE——无白化/斜投影，#36 命名更正）。

    剔除训练集 diff-in-means 后两类训练质心精确相等，凸损失 + L2 下重训 train AUC
    必然 ≈0.5（实现自检的硬信号，偏离 > MEAN_PROJECTION_TOL 即实现错误）。评估 AUC
    为样本外均值泄漏 + 非均值结构的描述量。这不是维数/冗余度量（对任意维均值信号
    一律一轮归零），只作 sanity gate。
    """
    direction = unit(diff_in_means(x_train, y_train))
    x_tr = remove_directions(x_train, direction)
    x_ev = remove_directions(x_eval, direction)
    residual_mean_gap = float(np.linalg.norm(diff_in_means(x_tr, y_train)))
    probe = fit_fn(x_tr, y_train)
    return {
        "train_auc": float(pg.primary_metric(y_train, probe.predict_proba(x_tr), 2)),
        "eval_auc": float(pg.primary_metric(y_eval, probe.predict_proba(x_ev), 2)),
        "train_mean_gap_after_projection": residual_mean_gap,
        "converged": bool(probe.converged),
    }


class MimiRidge:
    """Mimi 特征 → 标量目标的闭式岭回归（train 拟合，eval R²）。"""

    def __init__(self, train_features: np.ndarray, lambda_scale: float) -> None:
        matrix = np.asarray(train_features, dtype=np.float64)
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale = scale
        z = (matrix - self.mean) / self.scale
        self.n_train = len(z)
        lam = float(lambda_scale) * float(self.n_train)
        gram = z.T @ z + lam * np.eye(z.shape[1])
        self._z_train = z
        self._solve = np.linalg.solve
        self._gram = gram
        self.lam = lam

    def r2(self, targets_train: np.ndarray, eval_features: np.ndarray, targets_eval: np.ndarray) -> list[float]:
        """逐列目标的评估集 R²；目标按训练均值中心化后回归。"""
        t_train = np.asarray(targets_train, dtype=np.float64)
        t_eval = np.asarray(targets_eval, dtype=np.float64)
        if t_train.ndim == 1:
            t_train = t_train[:, None]
        if t_eval.ndim == 1:
            t_eval = t_eval[:, None]
        t_mean = t_train.mean(axis=0)
        beta = self._solve(self._gram, self._z_train.T @ (t_train - t_mean))
        z_eval = (np.asarray(eval_features, dtype=np.float64) - self.mean) / self.scale
        predictions = z_eval @ beta + t_mean
        residual = np.square(t_eval - predictions).sum(axis=0)
        total = np.square(t_eval - t_eval.mean(axis=0)).sum(axis=0)
        total[total == 0.0] = np.nan
        return [float(v) for v in 1.0 - residual / total]


def extract_event_windows(
    projections: np.ndarray,
    event_rows: np.ndarray,
    half_window: int,
    valid_row_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按事件中心行切 ±half_window 窗；仅保留完整落在 [1, valid_row_max] 的事件。

    返回 (窗口矩阵 [n_kept, 2·half+1, n_dirs], 保留掩码 [n_events])。
    """
    matrix = np.asarray(projections, dtype=np.float64)
    rows = np.asarray(event_rows, dtype=np.int64)
    keep = (rows - half_window >= 1) & (rows + half_window <= valid_row_max)
    kept_rows = rows[keep]
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    windows = matrix[kept_rows[:, None] + offsets[None, :]]
    return windows, keep


def sustained_onset(excludes: np.ndarray, min_consecutive: int) -> int | None:
    """首个"连续 ≥min_consecutive 个偏移均显著"的起始索引；无则 None。"""
    flags = np.asarray(excludes, dtype=bool)
    window = int(min_consecutive)
    if window <= 0:
        raise ValueError("min_consecutive 必须为正")
    for start in range(len(flags) - window + 1):
        if flags[start : start + window].all():
            return start
    return None


def cluster_bootstrap_separation(
    sum_pos: np.ndarray,
    count_pos: np.ndarray,
    sum_neg: np.ndarray,
    count_neg: np.ndarray,
    n_boot: int,
    seed: int,
    *,
    min_consecutive: int = TRAJ_MIN_CONSECUTIVE,
) -> dict:
    """会话级 cluster bootstrap：sep(t) 点估计、逐点 CI 与 sup-t 同时置信带。

    51 个偏移逐点 95% CI 的"首次排除零"在零效应下假阳性率接近 1（#35 高风险）。
    显著性判定因此改为双重校正：(1) studentized sup-t 同时带——对每个 bootstrap
    抽样计算 max_t |sep_b(t)−sep(t)|/se(t)，取其 95 分位数 q，同时带 = sep ±
    q·se(t)，把全曲线族错误率控制在 5%；(2) 起点须连续 ≥min_consecutive 个偏移
    同时带排除零（与 e1x/trajectory.divergence_step 的持续性约束对齐）。
    逐点 CI 仍输出，仅作描述。输入为 [n_sessions, n_offsets]（sum）与
    [n_sessions]（count）。
    """
    sum_pos = np.asarray(sum_pos, dtype=np.float64)
    sum_neg = np.asarray(sum_neg, dtype=np.float64)
    count_pos = np.asarray(count_pos, dtype=np.float64)
    count_neg = np.asarray(count_neg, dtype=np.float64)
    n_sessions = len(count_pos)
    if not (len(sum_pos) == len(sum_neg) == len(count_neg) == n_sessions):
        raise ValueError("bootstrap 输入的会话维不一致")
    if count_pos.sum() == 0 or count_neg.sum() == 0:
        raise ValueError("bootstrap 需要两组事件都非空")
    point = sum_pos.sum(axis=0) / count_pos.sum() - sum_neg.sum(axis=0) / count_neg.sum()
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_boot), sum_pos.shape[1]), dtype=np.float64)
    dropped = 0
    for b in range(int(n_boot)):
        take = rng.integers(0, n_sessions, size=n_sessions)
        pos_n = count_pos[take].sum()
        neg_n = count_neg[take].sum()
        if pos_n == 0 or neg_n == 0:
            draws[b] = np.nan
            dropped += 1
            continue
        draws[b] = sum_pos[take].sum(axis=0) / pos_n - sum_neg[take].sum(axis=0) / neg_n
    pointwise_lower = np.nanpercentile(draws, 2.5, axis=0)
    pointwise_upper = np.nanpercentile(draws, 97.5, axis=0)
    se = np.nanstd(draws, axis=0, ddof=1)
    positive_se = se > 0
    if not positive_se.any():
        raise ValueError("bootstrap 标准误全为零，无法构造同时带")
    standardized = np.abs(draws[:, positive_se] - point[positive_se]) / se[positive_se]
    sup_stats = np.nanmax(standardized, axis=1)
    sup_quantile = float(np.nanpercentile(sup_stats, 95.0))
    simultaneous_lower = point - sup_quantile * se
    simultaneous_upper = point + sup_quantile * se
    excludes = (simultaneous_lower > 0) | (simultaneous_upper < 0)
    onset = sustained_onset(excludes, min_consecutive)
    return {
        "separation": point,
        "pointwise_ci_lower": pointwise_lower,
        "pointwise_ci_upper": pointwise_upper,
        "bootstrap_se": se,
        "sup_t_quantile_95": sup_quantile,
        "simultaneous_lower": simultaneous_lower,
        "simultaneous_upper": simultaneous_upper,
        "min_consecutive": int(min_consecutive),
        "onset_index_sustained": onset,
        "bootstrap_dropped_draws": dropped,
    }


# ---------------------------------------------------------------------------
# 协议与断点
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _load_json_checkpoint(path: Path, protocol_hash: str, identity: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected = {"schema_version": SCHEMA_VERSION, "protocol_hash": protocol_hash, **identity}
    if all(payload.get(key) == value for key, value in expected.items()):
        return payload
    return None


def _geometry_root(roots: dict) -> Path:
    return roots["work"] / "geometry"


def _n_random() -> int:
    """E2-lite 范数匹配随机方向数（与 E1-X 同源配置，#36 claim 4）。"""
    return int(load_config("grids")["e1"]["e1x"]["n_random_directions"])


def _protocol(probe_cfg: dict, summary: dict, roots: dict, train: list[str], evals: list[str]) -> tuple[dict, str]:
    if summary.get("g2", {}).get("verdict", {}).get("verdict") != "fail":
        raise SystemExit("正式汇总不再是 G2=fail，#32 几何解剖前提失效，请先复核")
    protocol = {
        "name": PROTOCOL_NAME,
        "prereg": [32, 35, 36, 37],
        "target": TARGET,
        "binary_specs": list(BINARY_SPECS),
        "n_layers": N_LAYERS,
        "spectrum_layers": list(SPECTRUM_LAYERS),
        "remove_top_pc_ks": list(REMOVE_TOP_PC_KS),
        "energy_ks": list(ENERGY_KS),
        "align_rhos": list(ALIGN_RHOS),
        "mimi_top_pcs": MIMI_TOP_PCS,
        "ridge_lambda_scale": RIDGE_LAMBDA_SCALE,
        # #37：feature_split 受协方差混淆（降为描述量）、whitened 残差塌缩为 tautology
        # 且漏信号也过（撤销）；坐标分布性判读改协方差无关的 participation ratio。
        "distribution_method": "coordinate_participation_ratio_dual_gate_v1",
        "pr_thresholds": {"alpha_min": PR_ALPHA_MIN, "beta_max": PR_BETA_MAX, "random_baseline": "≈1/3"},
        "feature_split_descriptive": {"folds": FEATURE_SPLIT_FOLDS, "seed": FEATURE_SPLIT_SEED},
        "rank_one_binary": "analytical_S_B_rank_1_no_experiment",
        "mean_projection": "sanity_check_only_not_a_criterion",
        "refit_c_policy": "full_c_grid_eval_max_sensitivity_bound",
        "steering": {"schema": "e1x-directions-v1", "n_random": _n_random(), "proj_std_scale": "train_pooled_deduped"},
        "trajectory": {
            "half_window": TRAJ_HALF_WINDOW,
            "n_boot": TRAJ_N_BOOT,
            "seed": TRAJ_SEED,
            "min_consecutive": TRAJ_MIN_CONSECUTIVE,
            "band": "studentized_sup_t_95_simultaneous",
        },
        "identity_auc_tolerance": IDENTITY_AUC_TOLERANCE,
        "full_refit_tolerance": FULL_REFIT_TOLERANCE,
        "mean_projection_tol": MEAN_PROJECTION_TOL,
        "c_grid": [float(c) for c in probe_cfg["c_grid"]],
        "trainer": probe_cfg["trainer"],
        "pca_solver": "numpy.linalg.svd/full_matrices=False/float64",
        "formal_summary_sha256": _sha256(Path(REPO_ROOT) / "reports" / "wp_e1_probe_summary.json"),
        # 源码与行域绑定（#35/#36）：代码或冻结行域变化必须令旧断点失效。
        "script_sha256": _sha256(Path(__file__)),
        "probe_gpu_sha256": _sha256(Path(pg.__file__)),
        "grid_sha256": _sha256(Path(g.__file__)),
        "engine_sha256": _sha256(Path(engine.__file__)),
        "alignment_sha256": _sha256(Path(_alignment.__file__)),
        "row_plan_signature": engine._row_plan_signature(probe_cfg, roots, train, evals),
        # 全部二分类 fit 文件内容进入协议哈希（#37 claim 3）：任一 fit 内容变化即翻转
        # protocol_hash，令 directions/spectrum/steering/trajectory 全部旧断点失效，
        # 从根上杜绝"fit 更新 → 部分阶段重算 → 其余阶段复用旧结果"的混合来源。
        "fits_aggregate_sha256": _fits_aggregate_sha(roots, [int(s) for s in probe_cfg["seeds"]]),
    }
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()
    return protocol, hashlib.sha256(encoded).hexdigest()


def _load_summary() -> dict:
    path = Path(REPO_ROOT) / "reports" / "wp_e1_probe_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cell_direction(path: Path) -> np.ndarray | None:
    """只取格子里的 __weight__（标准化空间单位方向），不解压逐会话分数。"""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "__weight__" not in payload.files:
            return None
        return np.asarray(payload["__weight__"], dtype=np.float64)


def _load_fit_directions(roots: dict, spec_name: str, layer: int, seed: int) -> dict:
    """从正式 fit 断点取双空间方向、chosen_c 与文件摘要；缺失时抛出带路径的错误。"""
    path = engine._fit_path(roots, spec_name, layer, seed)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    probe, meta = engine._load_fit(path)
    if int(meta["n_classes"]) != 2:
        raise ValueError(f"{path} 不是二分类拟合")
    weight = np.asarray(probe.weight[0], dtype=np.float64)
    return {
        "std": unit(weight),
        "orig": orig_space_direction(weight, probe.scale),
        "chosen_c": float(meta["chosen_c"]),
        "scale": np.asarray(probe.scale, dtype=np.float64),
        "sha256": _sha256(path),
    }


def _fit_sha(roots: dict, spec_name: str, layer: int, seed: int) -> str | None:
    path = engine._fit_path(roots, spec_name, layer, seed)
    return _sha256(path) if path.is_file() else None


def _fits_aggregate_sha(roots: dict, seeds: list[int]) -> str:
    """全部二分类 fit 文件内容的聚合摘要（#36 claim 5：cache 命中也重算比对）。"""
    return hashlib.sha256(
        "\n".join(
            f"{spec_name}|{layer}|{seed}|{_fit_sha(roots, spec_name, layer, seed)}"
            for spec_name in BINARY_SPECS
            for layer in range(N_LAYERS)
            for seed in seeds
        ).encode()
    ).hexdigest()


def _npz_meta_valid(
    path: Path,
    protocol_hash: str,
    *,
    schema: str | None = None,
    layer: int | None = None,
    seed: int | None = None,
) -> bool:
    """向量/转向 NPZ 的协议校验：protocol_hash 必匹配，可选校验 schema/layer/seed。"""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(bytes(payload["__meta__"]).decode())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if meta.get("protocol_hash") != protocol_hash:
        return False
    if schema is not None and meta.get("schema") != schema:
        return False
    if layer is not None and meta.get("layer") != layer:
        return False
    return not (seed is not None and meta.get("seed") != seed)


def _spectrum_fresh(roots: dict, root: Path, protocol_hash: str, layer: int, seed: int) -> bool:
    """spectrum (层,种子) 断点是否可复用：协议匹配 + fit SHA 匹配 + 向量 NPZ 有效。"""
    record = _load_json_checkpoint(
        _spectrum_checkpoint_path(root, layer, seed), protocol_hash, {"layer": layer, "seed": seed}
    )
    if record is None:
        return False
    if record.get("fit_sha256") != _fit_sha(roots, TARGET, layer, seed):
        return False
    return _npz_meta_valid(
        _vectors_path(root, layer, seed), protocol_hash,
        schema="geometry-vectors-v1", layer=layer, seed=seed,
    )


# ---------------------------------------------------------------------------
# stage directions：方向族（零激活载入）
# ---------------------------------------------------------------------------


def stage_directions(args, protocol: dict, protocol_hash: str) -> dict:
    probe_cfg, _cache_cfg = engine._cfg()
    roots = engine._roots()
    summary = _load_summary()
    seeds = [int(s) for s in probe_cfg["seeds"]]
    root = _geometry_root(roots)
    checkpoint = root / "directions.json"
    if not args.force:
        cached = _load_json_checkpoint(checkpoint, protocol_hash, {"stage": "directions"})
        # cache 命中也重算 768-fit 聚合摘要并比对（#36 claim 5：内容变化令旧断点失效）。
        if cached is not None and cached.get("fits_aggregate_sha256") == _fits_aggregate_sha(roots, seeds):
            print("directions 断点已存在且 fit 聚合摘要一致，跳过（--force 重算）")
            return cached
        if cached is not None:
            print("directions 断点存在但 fit 聚合摘要不一致，重算")

    started = perf_counter()
    missing: list[str] = []
    bank: dict[tuple[str, int, int], dict] = {}
    for spec_name in BINARY_SPECS:
        for layer in range(N_LAYERS):
            for seed in seeds:
                try:
                    bank[(spec_name, layer, seed)] = _load_fit_directions(roots, spec_name, layer, seed)
                except FileNotFoundError as exc:
                    missing.append(str(exc))
    if missing:
        raise SystemExit(
            f"缺少 {len(missing)} 个正式 fit 断点（work/fits 可能被清理），前 5 个：\n"
            + "\n".join(missing[:5])
        )

    # 与格子 __weight__ 交叉核验（标准化空间方向应逐位一致，fp32 容差）。
    cell_cosines: list[float] = []
    for (spec_name, layer, seed), entry in bank.items():
        cell_dir = _load_cell_direction(engine._cell_path(roots, spec_name, "acts", layer, seed))
        if cell_dir is not None:
            cell_cosines.append(abs_cosine(entry["std"], cell_dir))
    if not cell_cosines:
        raise SystemExit("没有任何格子方向可供交叉核验，拒绝继续")
    min_cell_cos = float(min(cell_cosines))
    if min_cell_cos < 0.9999:
        raise SystemExit(f"fit 与格子方向最小 |cos|={min_cell_cos:.8f} < 0.9999，来源不一致")

    per_spec: dict[str, dict] = {}
    mean_dirs: dict[tuple[str, int], np.ndarray] = {}
    for spec_name in BINARY_SPECS:
        auc_table = summary["per_spec"][spec_name]["auc_by_seed_layer"]
        layers_out = {}
        for layer in range(N_LAYERS):
            orig = [bank[(spec_name, layer, seed)]["orig"] for seed in seeds]
            std = [bank[(spec_name, layer, seed)]["std"] for seed in seeds]
            # 探针方向 w/σ 天然指向 label=1（正 logit → 类 1），跨种子同向：
            # 直接平均，不做 sign_aligned_mean 的符号翻转（#36 secondary：避免掩盖符号错误）。
            signed_pairs = [float(np.dot(unit(orig[i]), unit(orig[j]))) for i in range(3) for j in range(i + 1, 3)]
            if min(signed_pairs) <= 0:
                raise RuntimeError(
                    f"{spec_name}@L{layer} 三种子探针方向未一致指向 label=1"
                    f"（最小带符号 cos={min(signed_pairs):.4f}）"
                )
            mean_dirs[(spec_name, layer)] = unit(np.mean(np.stack([unit(v) for v in orig]), axis=0))
            layers_out[str(layer)] = {
                "min_pairwise_abs_cos_orig": float(min(abs(v) for v in signed_pairs)),
                "min_pairwise_abs_cos_std": float(
                    min(abs_cosine(std[i], std[j]) for i in range(3) for j in range(i + 1, 3))
                ),
                "auc_seed_mean": float(np.mean([auc_table[str(s)][str(layer)] for s in seeds])),
            }
        per_spec[spec_name] = {"layers": layers_out}

    t4_reference = mean_dirs[(TARGET, 29)]
    t4_propagation = {}
    for layer in range(N_LAYERS):
        entry = {
            "to_L29_abs_cos": abs_cosine(mean_dirs[(TARGET, layer)], t4_reference),
        }
        if layer + 1 < N_LAYERS:
            entry["adjacent_abs_cos"] = abs_cosine(mean_dirs[(TARGET, layer)], mean_dirs[(TARGET, layer + 1)])
        t4_propagation[str(layer)] = entry

    cross_spec = {}
    cross_spec_signed = {}
    for layer in range(N_LAYERS):
        cross_spec[str(layer)] = {
            a: {b: abs_cosine(mean_dirs[(a, layer)], mean_dirs[(b, layer)]) for b in BINARY_SPECS}
            for a in BINARY_SPECS
        }
        # 带符号余弦：各规格方向均指向各自 label=1，符号本身携带"同向/反向编码"语义。
        cross_spec_signed[str(layer)] = {
            a: {
                b: float(np.dot(unit(mean_dirs[(a, layer)]), unit(mean_dirs[(b, layer)])))
                for b in BINARY_SPECS
            }
            for a in BINARY_SPECS
        }
    t1_vs_t4 = {
        spec_name: {str(layer): cross_spec[str(layer)][spec_name][TARGET] for layer in SPECTRUM_LAYERS}
        for spec_name in BINARY_SPECS
        if spec_name.startswith("T1_")
    }
    fits_aggregate = _fits_aggregate_sha(roots, seeds)

    arrays: dict[str, np.ndarray] = {
        "__meta__": np.frombuffer(
            json.dumps(
                {"schema": "geometry-directions-bank-v1", "protocol_hash": protocol_hash, "seeds": seeds},
                ensure_ascii=False,
            ).encode(),
            dtype=np.uint8,
        )
    }
    for spec_name in BINARY_SPECS:
        for layer in range(N_LAYERS):
            stacked = np.stack([bank[(spec_name, layer, seed)]["orig"] for seed in seeds])
            arrays[f"orig__{spec_name}__L{layer}"] = stacked.astype(np.float32)
    _atomic_write_npz(root / "directions_bank.npz", arrays)

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_hash": protocol_hash,
        "stage": "directions",
        "n_fit_files": len(bank),
        "fits_aggregate_sha256": fits_aggregate,
        "min_fit_vs_cell_abs_cos": min_cell_cos,
        "per_spec": per_spec,
        "t4_layer_propagation": t4_propagation,
        "cross_spec_abs_cos_by_layer": cross_spec,
        "cross_spec_signed_cos_by_layer": cross_spec_signed,
        "t1_vs_t4_abs_cos": t1_vs_t4,
        "bank_path": str(root / "directions_bank.npz"),
        "wall_seconds": perf_counter() - started,
    }
    _atomic_write_json(checkpoint, result)
    print(
        f"directions 完成：{len(bank)} 个方向；fit↔cell 最小 |cos|={min_cell_cos:.6f}；"
        f"耗时 {result['wall_seconds']:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# stage spectrum：能量谱、剔除重训、Mimi 鉴定、转向分片
# ---------------------------------------------------------------------------


def _spectrum_checkpoint_path(root: Path, layer: int, seed: int) -> Path:
    return root / f"spectrum__L{layer}__s{seed}.json"


def _vectors_path(root: Path, layer: int, seed: int) -> Path:
    return root / "vectors" / f"L{layer}__s{seed}.npz"


def _steering_path(root: Path, layer: int) -> Path:
    return root / f"steering_L{layer}.npz"


def _union_train_roles(train_rows: dict, seeds: list[int]) -> list:
    """跨种子按 (session, channel, step) 去重的训练行并集（与 E1-X x_all 同口径，#37）。

    E1-X `stage_geometry` 用 seen 集去重 (session_id, agent_channel, step) 后组成
    x_all；本函数复现同一去重口径，避免几何支线按各自出现计数（重叠行最多计 3 次）
    造成 proj_std 与 E1-X 不一致、进而 E2 注入剂量随方向来源改变。
    """
    merged: dict[tuple[str, int], dict[int, int]] = {}
    for seed in seeds:
        for role in train_rows[(TARGET, seed)]:
            bucket = merged.setdefault((role.session_id, role.agent_channel), {})
            for step, label in zip(role.steps.tolist(), role.labels.tolist(), strict=True):
                bucket.setdefault(int(step), int(label))
    roles = []
    for (sid, channel), step_label in sorted(merged.items()):
        steps = np.asarray(sorted(step_label), dtype=np.int64)
        labels = np.asarray([step_label[int(s)] for s in steps], dtype=np.int64)
        roles.append(g.RoleRows(sid, channel, steps, labels))
    return roles


def _steering_union_features(store: dict, train_rows: dict, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """去重训练并集的激活矩阵与标签（供 diffmeans 与 proj_std 同口径计算）。"""
    union = _union_train_roles(train_rows, seeds)
    features, labels, _ = g.assemble(union, "acts", store, dtype=np.float32)
    if len(labels) < 2:
        raise RuntimeError("去重训练并集不足 2 行")
    return features, labels


def _projection_std_on(features: np.ndarray, directions: dict[str, np.ndarray]) -> dict[str, float]:
    """各方向在给定特征矩阵原始空间的投影标准差（E1-X projection_std 同口径）。"""
    names = list(directions)
    basis = np.stack([directions[name] for name in names], axis=1).astype(np.float32)
    proj = np.asarray(features, dtype=np.float32) @ basis
    return {name: float(np.asarray(proj[:, i], dtype=np.float64).std()) for i, name in enumerate(names)}


def _write_steering(
    root: Path,
    layer: int,
    seeds: list[int],
    store: dict,
    train_rows: dict,
    protocol_hash: str,
) -> None:
    """合成文档/04 §2.1-G 承诺的转向包（schema 兼容 e1x-directions-v1）。

    键名与 wp_e2_lite_plan.py 的消费口径一致：probe_s{seed}/probe_meanseed/
    diffmeans_s{seed}/diffmeans/random_r{i}；random 方向数取自配置 n_random_directions
    （与 E1-X 同源，#36 claim 4）。**与 E1-X 完全同口径（#37 claim 5）**：聚合 diffmeans
    在去重训练并集上重算（非 per-seed 均值），所有方向 proj_std 在同一去重并集上算
    （非按种子出现计数）。符号约定：+v 指向 T4 label=1，由类条件投影统计显式核验，
    不做静默翻转；per-seed 键仅作诊断。
    """
    n_random = _n_random()
    directions: dict[str, np.ndarray] = {}
    probe_dirs: list[np.ndarray] = []
    probe_stats: list[dict] = []
    for seed in seeds:
        if not _npz_meta_valid(
            _vectors_path(root, layer, seed), protocol_hash,
            schema="geometry-vectors-v1", layer=layer, seed=seed,
        ):
            raise RuntimeError(f"steering 合成缺少或过期的 L{layer}/s{seed} 向量 NPZ")
        with np.load(_vectors_path(root, layer, seed), allow_pickle=False) as payload:
            v_star = np.asarray(payload["v_star"], dtype=np.float64)
            d_vec = np.asarray(payload["d"], dtype=np.float64)
        record = _load_json_checkpoint(
            _spectrum_checkpoint_path(root, layer, seed), protocol_hash, {"layer": layer, "seed": seed}
        )
        if record is None:
            raise RuntimeError(f"steering 合成缺少 L{layer}/s{seed} 的 spectrum 断点")
        directions[f"probe_s{seed}"] = unit(v_star)
        directions[f"diffmeans_s{seed}"] = unit(d_vec)
        probe_dirs.append(v_star)
        probe_stats.append(record["projection_stats"]["v_star"])
    directions["probe_meanseed"] = mean_direction_label1(probe_dirs, probe_stats)
    # 聚合 diffmeans 在去重训练并集上重算，与 E1-X diff_means_direction(x_all) 一致。
    union_features, union_labels = _steering_union_features(store, train_rows, seeds)
    diffmeans = unit(diff_in_means(union_features, union_labels))
    proj_union = union_features @ diffmeans.astype(np.float32)  # union_features 已是 float32
    if not proj_union[union_labels == 1].mean() > proj_union[union_labels == 0].mean():
        raise RuntimeError(f"L{layer} 聚合 diffmeans 未指向 label=1，符号约定被破坏")
    directions["diffmeans"] = diffmeans
    rng = np.random.default_rng(TRAJ_SEED + layer)
    for index in range(n_random):
        directions[f"random_r{index}"] = unit(rng.standard_normal(len(probe_dirs[0])))
    proj_std = _projection_std_on(union_features, directions)
    del union_features
    gc.collect()
    npz_payload = {name: vec.astype(np.float32) for name, vec in directions.items()}
    npz_payload["__meta__"] = np.frombuffer(
        json.dumps(
            {
                "schema": "e1x-directions-v1",
                "source": "wp_e1_geometry_autopsy.spectrum",
                "protocol_hash": protocol_hash,
                "layer": layer,
                "n_random": n_random,
                "sign": "+v 指向 T4 label=1（对方话轮 complete，可接话感知）",
                "scale": "steer = alpha * proj_std[name] * unit(v)",
                "proj_std": proj_std,
                "proj_std_source": "train_dedup_union_session_channel_step",
            },
            ensure_ascii=False,
        ).encode(),
        dtype=np.uint8,
    )
    _atomic_write_npz(_steering_path(root, layer), npz_payload)
    print(
        f"L{layer}：转向包已写入 {_steering_path(root, layer)}"
        f"（{len(directions)} 方向，{n_random} 随机，train 尺度）",
        flush=True,
    )


def _run_spectrum_task(
    *,
    layer: int,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    mimi_train: np.ndarray,
    mimi_eval: np.ndarray,
    fit_entry: dict,
    fit_sha256: str,
    formal_auc: float,
    c_grid: list[float],
    trainer: dict,
    device: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = perf_counter()
    x64 = np.asarray(x_train, dtype=np.float64)
    center = x64.mean(axis=0)
    centered = x64 - center
    del x64
    print(f"L{layer}/s{seed}：float64 PCA {centered.shape[0]}×{centered.shape[1]}", flush=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    del centered
    gc.collect()
    variance = np.square(singular_values, dtype=np.float64)
    variance_cum = np.cumsum(variance) / variance.sum()

    v_star = fit_entry["orig"]
    chosen_c = float(fit_entry["chosen_c"])
    eval64 = np.asarray(x_eval, dtype=np.float64)
    identity_auc = projection_auc(y_eval, eval64 @ v_star)
    identity_gap = abs(identity_auc - float(formal_auc))
    if identity_gap > IDENTITY_AUC_TOLERANCE:
        raise RuntimeError(
            f"L{layer}/s{seed} 一维投影恒等自检失败：|{identity_auc:.6f}−{formal_auc:.6f}|"
            f"={identity_gap:.6f} > {IDENTITY_AUC_TOLERANCE}"
        )
    if identity_auc <= 0.5:
        raise RuntimeError(f"L{layer}/s{seed} 探针方向未指向 label=1（AUC={identity_auc:.4f}）")

    d_raw = diff_in_means(x_train, y_train)
    d_unit = unit(d_raw)
    d_auc = projection_auc(y_eval, eval64 @ d_unit)
    cos_d_vstar = abs_cosine(d_unit, v_star)

    profile_v = energy_profile(vt, v_star)
    profile_d = energy_profile(vt, d_unit)
    cum_v = profile_v["cumulative"]
    cum_d = profile_d["cumulative"]

    # 变换后重训一律全 C 档取评估集最大（敏感性上界，#35 中风险 1）。
    removals: dict[str, dict] = {
        "full_refit": refit_auc_grid(x_train, y_train, x_eval, y_eval, c_grid, trainer, device)
    }
    formal_c_refit = float(removals["full_refit"]["auc_by_c"][str(chosen_c)])
    full_refit_gap = abs(formal_c_refit - float(formal_auc))
    if full_refit_gap > FULL_REFIT_TOLERANCE:
        raise RuntimeError(
            f"L{layer}/s{seed} 正式 C 全维重训复现失败：|{formal_c_refit:.6f}−{formal_auc:.6f}|"
            f"={full_refit_gap:.6f} > {FULL_REFIT_TOLERANCE}"
        )
    for name, direction in (("v_star", v_star), ("d", d_unit)):
        x_tr_removed = remove_directions(x_train, direction)
        x_ev_removed = remove_directions(x_eval, direction)
        removals[f"remove_dir_{name}"] = refit_auc_grid(
            x_tr_removed, y_train, x_ev_removed, y_eval, c_grid, trainer, device
        )
        del x_tr_removed, x_ev_removed
        gc.collect()
    for k in REMOVE_TOP_PC_KS:
        if k >= int(vt.shape[0]):  # 谱不足时跳过（真实 4096 维不触发）
            continue
        x_tr_removed = remove_top_pcs(x_train, center, vt, k)
        x_ev_removed = remove_top_pcs(x_eval, center, vt, k)
        removals[f"remove_top_pc_{k}"] = refit_auc_grid(
            x_tr_removed, y_train, x_ev_removed, y_eval, c_grid, trainer, device
        )
        del x_tr_removed, x_ev_removed
        gc.collect()

    def _fit(x_block: np.ndarray, y_block: np.ndarray):
        return pg.fit_linear_probe(
            x_block,
            y_block,
            2,
            chosen_c,
            device=device,
            max_iter=int(trainer["lbfgs_max_iter"]),
            tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
        )

    # 坐标集中度（协方差无关，#37 判读用）：读出/信号方向权重多少个神经元。
    concentration = {
        "v_star": coordinate_concentration(v_star),
        "d": coordinate_concentration(d_unit),
    }
    # 原生坐标半切可解码性（#37 降级为描述量，非 α/β；受协方差混淆）。
    feature_split = feature_split_redundancy(
        x_train, y_train, x_eval, y_eval, c_grid, trainer, device,
        folds=FEATURE_SPLIT_FOLDS, seed=FEATURE_SPLIT_SEED + layer,
    )
    gc.collect()
    # 正交均值投影自检（非判据，#36）：train AUC 必≈0.5，硬门。均值擦除后 train
    # 质心相等 ⇒ 任意 C 下零权重最优、train AUC≈0.5 与 C 无关，故沿用 chosen_c。
    mean_projection = mean_projection_check(x_train, y_train, x_eval, y_eval, fit_fn=_fit)
    if abs(mean_projection["train_auc"] - 0.5) > MEAN_PROJECTION_TOL:
        raise RuntimeError(
            f"L{layer}/s{seed} 均值投影自检失败：擦除后 train AUC="
            f"{mean_projection['train_auc']:.4f} 偏离 0.5 > {MEAN_PROJECTION_TOL}，实现有误"
        )
    gc.collect()

    # 目标投影用 f32 输入（train 侧避免再造 float64 大副本；R²/统计在 float64 汇总）。
    ridge = MimiRidge(mimi_train, RIDGE_LAMBDA_SCALE)
    n_pcs = min(MIMI_TOP_PCS, vt.shape[0])
    train32 = np.asarray(x_train, dtype=np.float32)
    center32 = center.astype(np.float32)
    basis32 = vt[:n_pcs].astype(np.float32)
    pc_train = (train32 - center32) @ basis32.T
    pc_eval = (eval64 - center) @ vt[:n_pcs].T
    train_proj_v = train32 @ v_star.astype(np.float32)
    train_proj_d = train32 @ d_unit.astype(np.float32)
    targets_train = np.column_stack([pc_train, train_proj_v, train_proj_d]).astype(np.float64)
    targets_eval = np.column_stack([pc_eval, eval64 @ v_star, eval64 @ d_unit])
    del pc_train, pc_eval, train32
    r2 = ridge.r2(targets_train, mimi_eval, targets_eval)
    mimi_r2 = {
        "per_pc": r2[:n_pcs],
        "v_star": r2[n_pcs],
        "d": r2[n_pcs + 1],
        "lambda": ridge.lam,
        "n_train": ridge.n_train,
    }

    stats_v = projection_stats(train_proj_v, y_train)
    stats_d = projection_stats(train_proj_d, y_train)
    for name, stat in (("v_star", stats_v), ("d", stats_d)):
        if not stat["mean_pos"] > stat["mean_neg"]:
            raise RuntimeError(f"L{layer}/s{seed} 方向 {name} 未指向 label=1，符号约定被破坏")
    del eval64, targets_train, targets_eval
    gc.collect()

    nonconverged_total = (
        sum(entry["nonconverged"] for entry in removals.values())
        + int(feature_split["nonconverged"])
        + int(not mean_projection["converged"])
    )
    if nonconverged_total > 0:
        raise RuntimeError(
            f"L{layer}/s{seed} 存在 {nonconverged_total} 个未收敛拟合（grad_norm≥1e-3），"
            "变换后重训不稳定，拒绝写入断点（#36 硬门）"
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": "spectrum",
        "layer": layer,
        "seed": seed,
        "n_train_rows": len(y_train),
        "n_eval_rows": len(y_eval),
        "formal_auc": float(formal_auc),
        "chosen_c": chosen_c,
        "fit_sha256": fit_sha256,
        "identity_projection_auc": identity_auc,
        "identity_gap": identity_gap,
        "full_refit_gap_at_formal_c": full_refit_gap,
        "diff_means": {"auc": d_auc, "abs_cos_to_v_star": cos_d_vstar},
        "energy_v_star": {
            "at_k": energy_at_ks(cum_v, ENERGY_KS),
            "alignment_rank": {str(rho): alignment_rank(cum_v, rho) for rho in ALIGN_RHOS},
            "span_fraction": profile_v["span_fraction"],
        },
        "energy_d": {
            "at_k": energy_at_ks(cum_d, ENERGY_KS),
            "alignment_rank": {str(rho): alignment_rank(cum_d, rho) for rho in ALIGN_RHOS},
            "span_fraction": profile_d["span_fraction"],
        },
        "pca_variance_cum_at_k": energy_at_ks(variance_cum, ENERGY_KS),
        "removal_auc": removals,
        "coordinate_concentration": concentration,
        "feature_split_descriptive": feature_split,
        "mean_projection": mean_projection,
        "mimi_r2": mimi_r2,
        "projection_stats": {"v_star": stats_v, "d": stats_d},
        "nonconverged_fits": nonconverged_total,
        "wall_seconds": perf_counter() - started,
    }
    vectors = {
        "v_star": v_star.astype(np.float32),
        "d": d_unit.astype(np.float32),
        "pc1": np.asarray(vt[0], dtype=np.float32),
        "energy_cum_v_star": cum_v.astype(np.float32),
        "energy_cum_d": cum_d.astype(np.float32),
        "pca_variance_cum": variance_cum.astype(np.float32),
    }
    return record, vectors


def stage_spectrum(args, protocol: dict, protocol_hash: str) -> None:
    engine._validate_devices([str(args.device)])
    probe_cfg, _cache_cfg = engine._cfg()
    roots = engine._roots()
    summary = _load_summary()
    train, evals = engine._sessions()
    _specs, seeds, _inner, _pools, train_rows, eval_rows = engine._prepare_rows(probe_cfg, roots, train, evals)
    seeds = [int(s) for s in seeds]
    layers = tuple(int(v) for v in str(args.layers).split(",") if v.strip())
    unknown = [layer for layer in layers if layer not in SPECTRUM_LAYERS]
    if unknown:
        raise SystemExit(f"--layers 只支持 {SPECTRUM_LAYERS}，收到 {unknown}")
    root = _geometry_root(roots)
    auc_table = summary["per_spec"][TARGET]["auc_by_seed_layer"]

    pending = [
        (layer, seed)
        for layer in layers
        for seed in seeds
        if args.force or not _spectrum_fresh(roots, root, protocol_hash, layer, seed)
    ]
    steering_pending = [
        layer
        for layer in layers
        if args.force
        or any((layer, seed) in pending for seed in seeds)
        or not _npz_meta_valid(
            _steering_path(root, layer), protocol_hash, schema="e1x-directions-v1", layer=layer
        )
    ]
    print(
        f"spectrum：待计算 {len(pending)}/{len(layers) * len(seeds)} 个层×种子任务，"
        f"待合成转向包 {len(steering_pending)} 层",
        flush=True,
    )
    if not pending and not steering_pending:
        return

    n_steps = engine._load_run_specs(roots)
    unique_steps = set(n_steps.values())
    if len(unique_steps) != 1:
        raise RuntimeError(f"正式角色 n_steps 不一致：{sorted(unique_steps)}")
    n_layer_rows = unique_steps.pop()
    role_groups = [eval_rows[TARGET]] + [train_rows[(TARGET, seed)] for seed in seeds]
    compact_rows = g.required_layer_rows(role_groups, n_rows=n_layer_rows)
    run_keys = sorted(compact_rows)

    mimi_keys = sorted({(sid, channel) for sid, _ in run_keys for channel in (0, 1)})
    print(f"载入 Mimi 潜表征 {len(mimi_keys)} 路（层无关，装配后立即释放）", flush=True)
    mimi_store = g.preload_mimi(roots["runs"], mimi_keys)
    mimi_train_by_seed = {}
    for seed in seeds:
        features, labels_m, _ = g.assemble(train_rows[(TARGET, seed)], "mimi", mimi_store, dtype=np.float32)
        mimi_train_by_seed[seed] = (features, labels_m)
    mimi_eval, _, _ = g.assemble(eval_rows[TARGET], "mimi", mimi_store, dtype=np.float32)
    mimi_store.clear()
    del mimi_store
    gc.collect()

    for layer in layers:
        layer_pending = [seed for seed in seeds if (layer, seed) in pending]
        if not layer_pending and layer not in steering_pending:
            continue
        load_started = perf_counter()
        print(f"L{layer}：载入 {len(run_keys)} 路压紧层缓存", flush=True)
        store = g.preload_layer(roots["runs"], run_keys, layer, row_indices=compact_rows)
        x_eval, y_eval, _ = g.assemble(eval_rows[TARGET], "acts", store, dtype=np.float32)
        print(f"L{layer}：层缓存与评估矩阵就绪，耗时 {perf_counter() - load_started:.1f}s", flush=True)
        for seed in layer_pending:
            x_train, y_train, _ = g.assemble(train_rows[(TARGET, seed)], "acts", store, dtype=np.float32)
            mimi_train, mimi_labels = mimi_train_by_seed[seed]
            if not np.array_equal(mimi_labels, y_train):
                raise RuntimeError(f"L{layer}/s{seed}：Mimi 与 acts 行域标签不一致")
            fit_entry = _load_fit_directions(roots, TARGET, layer, seed)
            record, vectors = _run_spectrum_task(
                layer=layer,
                seed=seed,
                x_train=x_train,
                y_train=y_train,
                x_eval=x_eval,
                y_eval=y_eval,
                mimi_train=mimi_train,
                mimi_eval=mimi_eval,
                fit_entry=fit_entry,
                fit_sha256=fit_entry["sha256"],
                formal_auc=float(auc_table[str(seed)][str(layer)]),
                c_grid=[float(c) for c in probe_cfg["c_grid"]],
                trainer=probe_cfg["trainer"],
                device=str(args.device),
            )
            record["protocol_hash"] = protocol_hash
            vectors["__meta__"] = np.frombuffer(
                json.dumps(
                    {
                        "schema": "geometry-vectors-v1",
                        "protocol_hash": protocol_hash,
                        "layer": layer,
                        "seed": seed,
                    },
                    ensure_ascii=False,
                ).encode(),
                dtype=np.uint8,
            )
            _atomic_write_npz(_vectors_path(root, layer, seed), vectors)
            _atomic_write_json(_spectrum_checkpoint_path(root, layer, seed), record)
            print(
                f"L{layer}/s{seed} 完成：E(16)={record['energy_v_star']['at_k']['16']:.4f}，"
                f"PR(v*)={record['coordinate_concentration']['v_star']['participation_ratio']:.1f}"
                f"/{record['coordinate_concentration']['v_star']['n_dims']}，"
                f"cos(d,v*)={record['diff_means']['abs_cos_to_v_star']:.4f}，"
                f"耗时 {record['wall_seconds']:.1f}s",
                flush=True,
            )
            del x_train, y_train
            gc.collect()
            engine._empty_cuda_cache(str(args.device))
        if layer in steering_pending:
            _write_steering(root, layer, seeds, store, train_rows, protocol_hash)
        store.clear()
        del store, x_eval, y_eval
        gc.collect()


# ---------------------------------------------------------------------------
# stage trajectory：事件锁定方向轨迹（评估集）
# ---------------------------------------------------------------------------


def stage_trajectory(args, protocol: dict, protocol_hash: str) -> dict:
    engine._validate_devices([str(args.device)])
    probe_cfg, _cache_cfg = engine._cfg()
    roots = engine._roots()
    train, evals = engine._sessions()
    _specs, seeds, _inner, _pools, _train_rows, eval_rows = engine._prepare_rows(probe_cfg, roots, train, evals)
    seeds = [int(s) for s in seeds]
    layer = int(args.traj_layer)
    root = _geometry_root(roots)
    checkpoint = root / f"trajectory__L{layer}.json"
    steering_file = _steering_path(root, layer)
    steering_valid = _npz_meta_valid(steering_file, protocol_hash, schema="e1x-directions-v1", layer=layer)
    steering_sha = _sha256(steering_file) if steering_valid else None
    if not args.force:
        cached = _load_json_checkpoint(checkpoint, protocol_hash, {"stage": "trajectory", "layer": layer})
        # 缓存复用另须转向包 SHA 一致（#37 claim 3 防依赖链混用；fits 已入协议哈希，
        # 此为额外的显式依赖绑定）。
        if cached is not None and cached.get("steering_sha256") == steering_sha and steering_sha is not None:
            print("trajectory 断点与转向包 SHA 一致，跳过（--force 重算）")
            return cached
        if cached is not None:
            print("trajectory 断点存在但转向包 SHA 不一致或缺失，重算")

    if not steering_valid:
        raise SystemExit(f"缺少或过期的转向包 {steering_file}——trajectory 依赖 spectrum 完成该层")
    n_random = _n_random()
    with np.load(steering_file, allow_pickle=False) as payload:
        steering_meta = json.loads(bytes(payload["__meta__"]).decode())
        directions = {
            "probe_meanseed": np.asarray(payload["probe_meanseed"], dtype=np.float64),
            "diffmeans": np.asarray(payload["diffmeans"], dtype=np.float64),
        }
        # 载入全部随机方向作经验零假设（#37：原仅读 random_r0）。
        for index in range(n_random):
            directions[f"random_r{index}"] = np.asarray(payload[f"random_r{index}"], dtype=np.float64)
    pc1_members = []
    for seed in seeds:
        path = _vectors_path(root, layer, seed)
        if not _npz_meta_valid(path, protocol_hash, schema="geometry-vectors-v1", layer=layer, seed=seed):
            raise SystemExit(f"缺少或过期的 {path}——请先以当前协议重跑 spectrum")
        with np.load(path, allow_pickle=False) as payload:
            pc1_members.append(np.asarray(payload["pc1"], dtype=np.float64))
    directions["pc1"] = sign_aligned_mean(pc1_members)
    # σ 标准化用转向包里方向自身的训练行投影标准差（train pooled，#36 claim 4）。
    projection_norms = {
        "probe_meanseed": float(steering_meta["proj_std"]["probe_meanseed"]),
        "diffmeans": float(steering_meta["proj_std"]["diffmeans"]),
    }

    roles = eval_rows[TARGET]
    eval_keys = sorted({(role.session_id, role.agent_channel) for role in roles})
    print(f"trajectory：载入评估集 {len(eval_keys)} 路完整 L{layer}", flush=True)
    store = g.preload_layer(roots["runs"], eval_keys, layer)

    names = list(directions)
    basis = np.stack([directions[name] for name in names], axis=1).astype(np.float32)
    n_offsets = 2 * TRAJ_HALF_WINDOW + 1
    session_ids = sorted({role.session_id for role in roles})
    session_index = {sid: i for i, sid in enumerate(session_ids)}
    sums = {
        name: {group: np.zeros((len(session_ids), n_offsets)) for group in (0, 1)} for name in names
    }
    counts = {name: {group: np.zeros(len(session_ids)) for group in (0, 1)} for name in names}
    dropped_events = 0
    total_events = 0
    for role in roles:
        array = store[(role.session_id, role.agent_channel)]
        projections = np.asarray(array, dtype=np.float32) @ basis
        valid_row_max = min(len(array) - 1, ANALYSIS_MAX_LABEL_STEP + 1)
        event_rows = role.steps + 1  # acts 行 = 标签步 + 1（mve/alignment #8）
        windows, keep_mask = extract_event_windows(projections, event_rows, TRAJ_HALF_WINDOW, valid_row_max)
        labels = role.labels[keep_mask]
        dropped_events += int((~keep_mask).sum())
        total_events += len(role.steps)
        row = session_index[role.session_id]
        for group in (0, 1):
            block = windows[labels == group]
            if not len(block):
                continue
            for column, name in enumerate(names):
                sums[name][group][row] += block[:, :, column].sum(axis=0)
                counts[name][group][row] += len(block)
    store.clear()
    del store
    gc.collect()

    offsets_ms = [int(offset * 80) for offset in range(-TRAJ_HALF_WINDOW, TRAJ_HALF_WINDOW + 1)]
    curves = {}
    for name in names:
        boot = cluster_bootstrap_separation(
            sums[name][1],
            counts[name][1],
            sums[name][0],
            counts[name][0],
            TRAJ_N_BOOT,
            TRAJ_SEED,
            min_consecutive=TRAJ_MIN_CONSECUTIVE,
        )
        norm = projection_norms.get(name)
        onset_index = boot["onset_index_sustained"]
        curves[name] = {
            "mean_label1": (sums[name][1].sum(axis=0) / counts[name][1].sum()).tolist(),
            "mean_label0": (sums[name][0].sum(axis=0) / counts[name][0].sum()).tolist(),
            "separation": boot["separation"].tolist(),
            "separation_in_sigma": (boot["separation"] / norm).tolist() if norm else None,
            "pointwise_ci_lower": boot["pointwise_ci_lower"].tolist(),
            "pointwise_ci_upper": boot["pointwise_ci_upper"].tolist(),
            "simultaneous_lower": boot["simultaneous_lower"].tolist(),
            "simultaneous_upper": boot["simultaneous_upper"].tolist(),
            "sup_t_quantile_95": boot["sup_t_quantile_95"],
            "onset_ms_sustained": (None if onset_index is None else offsets_ms[onset_index]),
            "min_consecutive": boot["min_consecutive"],
            "bootstrap_dropped_draws": boot["bootstrap_dropped_draws"],
            "n_events_label1": float(counts[name][1].sum()),
            "n_events_label0": float(counts[name][0].sum()),
        }

    random_names = [f"random_r{index}" for index in range(n_random)]
    random_onsets = [curves[name]["onset_ms_sustained"] for name in random_names]
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_hash": protocol_hash,
        "stage": "trajectory",
        "layer": layer,
        "steering_sha256": steering_sha,
        "offsets_ms": offsets_ms,
        "half_window_steps": TRAJ_HALF_WINDOW,
        "total_events": total_events,
        "dropped_events_incomplete_window": dropped_events,
        "n_sessions": len(session_ids),
        "random_direction_names": random_names,
        # 经验零假设：任一随机方向出现持续显著即须查实现/带宽（#37）。
        "random_any_sustained": any(onset is not None for onset in random_onsets),
        "direction_pairwise_abs_cos": {
            "probe_meanseed__diffmeans": abs_cosine(directions["probe_meanseed"], directions["diffmeans"]),
            "probe_meanseed__pc1": abs_cosine(directions["probe_meanseed"], directions["pc1"]),
        },
        "curves": curves,
    }
    arrays = {
        "__meta__": np.frombuffer(
            json.dumps({"protocol_hash": protocol_hash, "layer": layer}, ensure_ascii=False).encode(),
            dtype=np.uint8,
        )
    }
    for name in names:
        for group in (0, 1):
            arrays[f"sum__{name}__g{group}"] = sums[name][group].astype(np.float64)
            arrays[f"count__{name}__g{group}"] = counts[name][group].astype(np.float64)
    _atomic_write_npz(root / f"trajectory__L{layer}.npz", arrays)
    _atomic_write_json(checkpoint, result)
    print(
        f"trajectory 完成：事件 {total_events}（弃 {dropped_events}），"
        f"probe_meanseed 持续显著起点 = {curves['probe_meanseed']['onset_ms_sustained']} ms"
        f"（sup-t 同时带 + 连续 {TRAJ_MIN_CONSECUTIVE} 步）",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# stage finalize：判读矩阵与报告
# ---------------------------------------------------------------------------


def _verdict_interval(value: float, low: float, high: float) -> str:
    if value < low:
        return "alpha"
    if value >= high:
        return "beta"
    return "indeterminate"


def _summarize(protocol: dict, protocol_hash: str, roots: dict, args) -> dict:
    root = _geometry_root(roots)
    directions = _load_json_checkpoint(root / "directions.json", protocol_hash, {"stage": "directions"})
    if directions is None:
        raise SystemExit("缺少 directions 断点，请先运行 --stage directions")
    spectrum: dict[str, dict] = {}
    for layer in SPECTRUM_LAYERS:
        for seed in (0, 1, 2):
            record = _load_json_checkpoint(
                _spectrum_checkpoint_path(root, layer, seed), protocol_hash, {"layer": layer, "seed": seed}
            )
            if record is None:
                raise SystemExit(f"缺少 spectrum 断点 L{layer}/s{seed}，请先运行 --stage spectrum")
            spectrum[f"L{layer}/s{seed}"] = record
    trajectory = _load_json_checkpoint(
        root / f"trajectory__L{int(args.traj_layer)}.json",
        protocol_hash,
        {"stage": "trajectory", "layer": int(args.traj_layer)},
    )

    def across(metric) -> list[float]:
        return [metric(record) for record in spectrum.values()]

    e16 = across(lambda r: r["energy_v_star"]["at_k"]["16"])
    # 坐标集中度（#37，协方差无关）：读出方向的 participation ratio 占比。
    pr_frac_vstar = across(lambda r: float(r["coordinate_concentration"]["v_star"]["participation_fraction"]))
    pr_frac_d = across(lambda r: float(r["coordinate_concentration"]["d"]["participation_fraction"]))
    # 描述量（非判据）：原生坐标半切可解码性（受协方差混淆）。
    split_median = across(lambda r: float(r["feature_split_descriptive"]["median_retention"]))
    mp_train = across(lambda r: r["mean_projection"]["train_auc"])
    # top16 基线与变换值同口径（全 C 档评估集最大，#36 claim 6：消除选择偏差错配）。
    remove_top16_drop = across(
        lambda r: r["removal_auc"]["full_refit"]["max_auc"] - r["removal_auc"]["remove_top_pc_16"]["max_auc"]
    )
    cos_d = across(lambda r: r["diff_means"]["abs_cos_to_v_star"])
    mimi_gap = across(
        lambda r: float(np.mean(r["mimi_r2"]["per_pc"][:8])) - r["mimi_r2"]["v_star"]
    )
    identity_gap = across(lambda r: r["identity_gap"])
    full_refit_gap = across(lambda r: r["full_refit_gap_at_formal_c"])
    nonconverged_total = int(sum(across(lambda r: r["nonconverged_fits"])))

    # 坐标分布性判读（#37）：读出方向 v* 与原始信号方向 d 的 participation ratio 占比。
    # 参照系：随机高斯方向 PR/D≈1/3（实测），故阈值 0.10 处于"局部化 ~0.03"与"稠密 ~0.33"
    # 之间。要求 v* 与 d **同时** 满足才裁决——若读出仅因协方差而稠密、原始信号却稀疏，
    # 二者不一致 → indeterminate（对冲 PR(v*)=Σ⁻¹d 的协方差整形残余风险）。
    per_task_min = [min(a, b) for a, b in zip(pr_frac_vstar, pr_frac_d, strict=True)]
    per_task_max = [max(a, b) for a, b in zip(pr_frac_vstar, pr_frac_d, strict=True)]
    worst_pr = min(per_task_min)  # 最集中任务里 {v*,d} 更稀疏的一个
    best_pr = max(per_task_max)
    if worst_pr >= PR_ALPHA_MIN:
        distributed_verdict = "alpha"  # 每任务 v* 与 d 皆稠密
    elif best_pr < PR_BETA_MAX:
        distributed_verdict = "beta"  # 每任务 v* 与 d 皆稀疏
    else:
        distributed_verdict = "indeterminate"

    # 均值投影自检（非判读行）：训练 AUC 偏离 0.5 超过容差说明实现有误。
    mp_sanity_ok = all(abs(value - 0.5) <= MEAN_PROJECTION_TOL for value in mp_train)
    if not mp_sanity_ok:
        raise SystemExit("均值投影自检异常（train AUC 偏离 0.5），实现有误，拒绝出判读矩阵")
    if nonconverged_total > 0:
        raise SystemExit(f"存在 {nonconverged_total} 个未收敛拟合，拒绝出判读矩阵（#36/#37 硬门）")

    # T1×T4 方向关系（@L29，种子均值方向）。
    t1_cos_l29 = {
        spec: float(values["29"]) for spec, values in directions["t1_vs_t4_abs_cos"].items()
    }
    t1_min = min(t1_cos_l29.values())
    t1_d80 = t1_cos_l29.get("T1_d80", float("nan"))
    t1_d800 = t1_cos_l29.get("T1_d800", float("nan"))
    if t1_min >= 0.7 and (t1_d80 - t1_d800) <= 0.15:
        t1_verdict = "alpha"
    elif t1_d800 < 0.4:
        t1_verdict = "beta"
    else:
        t1_verdict = "indeterminate"

    # 层间凝聚：末端相邻方向余弦 + 对齐秩（ρ=0.95）沿层单调不升。
    t4_prop = directions["t4_layer_propagation"]
    adjacent_tail = [
        float(t4_prop[str(layer)]["adjacent_abs_cos"]) for layer in (28, 29, 30)
    ]
    rank_by_layer = {
        layer: max(
            (
                spectrum[f"L{layer}/s{seed}"]["energy_v_star"]["alignment_rank"]["0.95"] or 10**9
            )
            for seed in (0, 1, 2)
        )
        for layer in SPECTRUM_LAYERS
    }
    if min(adjacent_tail) >= 0.9 and rank_by_layer[31] <= rank_by_layer[29]:
        condensation_verdict = "alpha"
    elif min(adjacent_tail) < 0.7:
        condensation_verdict = "beta"
    else:
        condensation_verdict = "indeterminate"

    # 轨迹起点（sup-t 同时带 + 持续显著；trajectory 未运行时 not_run）。
    if trajectory is None:
        onset_values: list = []
        onset_verdict = "not_run"
        onset_worst = None
    else:
        probe_onset = trajectory["curves"]["probe_meanseed"]["onset_ms_sustained"]
        random_any = bool(trajectory.get("random_any_sustained"))
        onset_values = [probe_onset, "random_any_sig" if random_any else "random_clean"]
        onset_worst = probe_onset
        if random_any:
            onset_verdict = "indeterminate"  # 任一随机方向持续显著 → 查实现/带宽
        elif probe_onset is not None and probe_onset <= -240:
            onset_verdict = "alpha"
        elif probe_onset is None or probe_onset > 0:
            onset_verdict = "beta"
        else:
            onset_verdict = "indeterminate"

    verdicts = {
        "E16_misalignment": {
            "values": e16,
            "worst": max(e16),
            "verdict": _verdict_interval(max(e16), 0.5, 0.8),
            "reading": "alpha=错向主张成立（E(16)<0.5）；beta=与 rank-k 曲线矛盾须查实现",
        },
        "coordinate_distribution": {
            "values": [f"v*/d={a:.3f}/{b:.3f}" for a, b in zip(pr_frac_vstar, pr_frac_d, strict=True)],
            "worst": worst_pr,
            "verdict": distributed_verdict,
            "reading": (
                "读出方向 v* 与原始信号方向 d 的 participation ratio 占比 PR/D（参照随机方向"
                "≈1/3）：alpha=分布式（每任务 v* 与 d 皆≥0.10，即≥~410/4096）；beta=局部化"
                "（皆<0.01）；v*/d 不一致（读出因协方差稠密但信号稀疏）→ indeterminate。取代 #36"
                "受协方差混淆的特征折（实证把稠密全局信号误判局部化，保留为描述量）。阈值为暂定值"
            ),
        },
        "top16_contribution": {
            "values": remove_top16_drop,
            "worst": max(remove_top16_drop),
            "verdict": "alpha" if max(remove_top16_drop) < 0.01 else "beta",
            "reading": "alpha=方差前 16 主轴对读出无实质贡献（全 C 档上界掉幅<0.01）",
        },
        "diff_means_alignment": {
            "values": cos_d,
            "worst": min(cos_d),
            "verdict": "alpha" if min(cos_d) >= 0.8 else ("beta" if min(cos_d) < 0.5 else "indeterminate"),
            "reading": "alpha=diff-means 可作 E2 稳健转向向量（cos≥0.8）",
        },
        "mimi_contrast": {
            "values": mimi_gap,
            "worst": min(mimi_gap),
            "verdict": "alpha" if min(mimi_gap) > 0.2 else "indeterminate",
            "reading": "alpha=主轴被 Mimi 高预测而 v* 低（前 8 方差主轴均值 R² − v* R² > 0.2）",
        },
        "t1_t4_relation": {
            "values": [t1_cos_l29[spec] for spec in sorted(t1_cos_l29)],
            "worst": t1_min,
            "verdict": t1_verdict,
            "reading": (
                "L29 种子均值方向 |cos(T1_dδ, T4)|：alpha=前瞻≈状态外推（全部 ≥0.7 且 "
                "δ80→δ800 降幅 ≤0.15）；beta=独立前瞻通道（δ800 <0.4）"
            ),
        },
        "layer_condensation": {
            "values": adjacent_tail,
            "worst": min(adjacent_tail),
            "verdict": condensation_verdict,
            "reading": (
                "alpha=末端凝聚（L28→29→30→31 相邻 |cos| 均 ≥0.9 且对齐秩@0.95 "
                "L31≤L29）；beta=方向持续旋转（任一相邻 <0.7）"
            ),
        },
        "trajectory_onset": {
            "values": onset_values,
            "worst": onset_worst,
            "verdict": onset_verdict,
            "reading": (
                "alpha=预判性听觉（probe_meanseed 持续显著起点 ≤−240 ms 且全部随机方向无显著）；"
                "beta=仅事件后分离或从未持续显著；任一随机方向显著 → indeterminate 并查实现"
            ),
        },
    }
    return {
        "kind": "exploratory_non_decisive",
        "prereg": [32, 35, 36, 37],
        "formal_g2_unchanged": True,
        "rank_one_binary": "二类间散度 S_B 秩恒为 1（解析事实），线性可分性只有一个判别方向；"
        "不做实验'确认'（白化残差塌缩为 mean-projection tautology，且漏信号也过，已撤销 #37）",
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "identity_gap_max": max(identity_gap),
        "fit_audit": {
            "full_refit_gap_max": max(full_refit_gap),
            "nonconverged_fits_total": nonconverged_total,
            "mean_projection_sanity_ok": mp_sanity_ok,
            "mean_projection_train_auc": mp_train,
        },
        "coordinate_concentration_descriptive": {
            "pr_fraction_vstar": pr_frac_vstar,
            "pr_fraction_d": pr_frac_d,
            "native_half_split_median_retention_covariance_dependent": split_median,
        },
        "alignment_rank_095_by_layer": {str(k): v for k, v in rank_by_layer.items()},
        "verdict_matrix": verdicts,
        "directions": directions,
        "spectrum": spectrum,
        "trajectory": trajectory,
        "runtime": {"git_head": _git_head()},
    }


def _format_metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _write_markdown(result: dict) -> Path:
    verdicts = result["verdict_matrix"]
    t4_prop = result["directions"]["t4_layer_propagation"]
    audit = result["fit_audit"]
    concentration = result["coordinate_concentration_descriptive"]
    lines = [
        "# E1 几何解剖报告（第一梯队）",
        "",
        "> 探索性、严格非裁决（PREREG #32/#35/#36/#37）。G2 正式判定保持 `fail`，本报告不回写正式汇总。",
        "> 实验定义与判读矩阵见 `文档/04_e1_几何解剖与探索路线.md` §2。",
        "",
        "## 判读矩阵结果",
        "",
        "| 观测量 | 逐任务值 | 保守值 | 判定 |",
        "| --- | --- | ---: | --- |",
    ]
    for name, entry in verdicts.items():
        values = "、".join(_format_metric(value) for value in entry["values"])
        lines.append(f"| {name} | {values} | {_format_metric(entry['worst'])} | **{entry['verdict']}** |")
    lines.extend(["", "各判定的 α/β 含义：", ""])
    lines.extend(f"- `{name}`：{entry['reading']}" for name, entry in verdicts.items())
    lines.extend(
        [
            "",
            "## 关键锚点与拟合审计",
            "",
            f"- 一维投影恒等自检最大偏差：{result['identity_gap_max']:.2e}（门限 {IDENTITY_AUC_TOLERANCE}）。",
            f"- 正式 C 全维重训复现最大偏差：{audit['full_refit_gap_max']:.2e}"
            f"（硬门 {FULL_REFIT_TOLERANCE}）。",
            f"- 未收敛拟合总数：{audit['nonconverged_fits_total']}（硬门：>0 拒绝出矩阵）。",
            "- 秩-1 为解析事实（二类间散度秩=1），不做实验确认（白化残差塌缩为 mean-projection "
            "tautology 且漏信号也过，已撤销）。",
            "- 均值投影自检（train AUC 应≈0.5，Mean Projection 非 LEACE）："
            + "、".join(f"{value:.4f}" for value in audit["mean_projection_train_auc"])
            + ("（通过）" if audit["mean_projection_sanity_ok"] else "（**异常，查实现**）")
            + "。",
            "- 坐标集中度描述量（协方差无关）：PR(v*)/D="
            + "、".join(f"{value:.3f}" for value in concentration["pr_fraction_vstar"])
            + "；PR(d)/D="
            + "、".join(f"{value:.3f}" for value in concentration["pr_fraction_d"])
            + "；原生半切中位保留率（受协方差混淆，仅描述）="
            + "、".join(
                f"{value:.3f}"
                for value in concentration["native_half_split_median_retention_covariance_dependent"]
            )
            + "。",
            f"- fit↔cell 方向最小 |cos|：{result['directions']['min_fit_vs_cell_abs_cos']:.6f}。",
            "- T4 方向层间传播（种子均值方向，相邻层 |cos|）：L28→L29 "
            f"{t4_prop['28'].get('adjacent_abs_cos', float('nan')):.4f}，L29→L30 "
            f"{t4_prop['29'].get('adjacent_abs_cos', float('nan')):.4f}，L30→L31 "
            f"{t4_prop['30'].get('adjacent_abs_cos', float('nan')):.4f}；"
            "对齐秩@0.95（跨种子保守）："
            + "、".join(
                f"L{layer}={result['alignment_rank_095_by_layer'][str(layer)]}"
                for layer in SPECTRUM_LAYERS
            )
            + "。",
        ]
    )
    trajectory = result.get("trajectory")
    if trajectory is not None:
        probe_curve = trajectory["curves"]["probe_meanseed"]
        random_onsets = [
            trajectory["curves"][name]["onset_ms_sustained"]
            for name in trajectory["random_direction_names"]
        ]
        lines.extend(
            [
                f"- 事件锁定轨迹（L{trajectory['layer']}）：事件 {trajectory['total_events']}，"
                f"probe_meanseed 持续显著起点 = {probe_curve['onset_ms_sustained']} ms"
                f"（sup-t 同时带 + 连续 {probe_curve['min_consecutive']} 步）；"
                f"{len(random_onsets)} 个随机方向对照持续显著起点 = {random_onsets}"
                f"（任一非 None 即触发经验零假设警报："
                f"{'是' if trajectory['random_any_sustained'] else '否'}）。",
            ]
        )
    else:
        lines.append("- 事件锁定轨迹：未运行（trajectory 阶段可选）。")
    lines.extend(
        [
            "",
            "完整逐任务数值见 `wp_e1_geometry_autopsy.json`；转向向量包（e1x-directions-v1 "
            "schema，可经 `wp_e2_lite_plan.py --directions-npz` 直接消费）见 "
            "`<data_root>/e1_probe/geometry/steering_L{29,30,31}.npz`。",
        ]
    )
    path = Path(REPO_ROOT) / "reports" / "e1_几何解剖报告.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {path}")
    return path


def stage_finalize(args, protocol: dict, protocol_hash: str) -> dict:
    roots = engine._roots()
    result = _summarize(protocol, protocol_hash, roots, args)
    write_report_json("wp_e1_geometry_autopsy.json", result)
    _write_markdown(result)
    verdict_text = "，".join(
        f"{name}={entry['verdict']}" for name, entry in result["verdict_matrix"].items()
    )
    print(f"finalize 完成：{verdict_text}")
    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run(args) -> None:
    probe_cfg, _cache_cfg = engine._cfg()
    summary = _load_summary()
    roots = engine._roots()
    train, evals = engine._sessions()
    protocol, protocol_hash = _protocol(probe_cfg, summary, roots, train, evals)
    print(f"#32/#35 几何解剖协议 {protocol_hash[:12]}，stage={args.stage}", flush=True)
    if args.stage in ("directions", "all"):
        stage_directions(args, protocol, protocol_hash)
    if args.stage in ("spectrum", "all"):
        stage_spectrum(args, protocol, protocol_hash)
    if args.stage in ("trajectory", "all"):
        stage_trajectory(args, protocol, protocol_hash)
    if args.stage in ("finalize", "all"):
        stage_finalize(args, protocol, protocol_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["directions", "spectrum", "trajectory", "finalize", "all"],
    )
    parser.add_argument("--device", default="cuda:0", help="spectrum/trajectory 的重训与投影设备")
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in SPECTRUM_LAYERS),
        help="spectrum 层列表（仅支持 29,30,31 的子集）",
    )
    parser.add_argument("--traj-layer", type=int, default=29, help="trajectory 使用的层")
    parser.add_argument("--force", action="store_true", help="覆盖当前协议下已有断点")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
