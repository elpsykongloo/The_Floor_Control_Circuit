"""E1 几何解剖第一梯队（PREREG #32，探索性、严格非裁决）。

对 G2 有效秩失败做机制解剖（文档/04 §2）：T4 的线性读出在数学上是一维方向
v* = (w/σ)/‖w/σ‖（AUC 对单调变换不变），本脚本量化该方向与激活方差主轴的
错向程度，并给出 E2 方向注入所需的转向向量包。四个 stage：

  directions  零激活载入：8 个二分类规格 × 32 层 × 3 种子的原始空间方向族
              （来自 work/fits 断点），层间传播、跨种子稳定性、T1×T4 方向关系；
  spectrum    L29/L30/L31 × 3 种子：PCA 能量谱与对齐秩、方向剔除/补空间重训、
              diff-in-means、Mimi 主轴内容鉴定、转向向量分片；
  trajectory  评估集事件锁定轨迹：对方 IPU 末步 ±25 步窗内 v*/d 投影按
              complete/incomplete 分组的会话级 bootstrap 均值带；
  finalize    汇总判读矩阵（文档/04 §2.2）并写 reports/。

一切结果不回写正式汇总，不改变 G2=fail；断点按协议哈希隔离，重复启动只算缺失点。
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

from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg
from floor_circuit.mve.alignment import ANALYSIS_MAX_LABEL_STEP

SCHEMA_VERSION = 1
PROTOCOL_NAME = "geometry-autopsy-v1"
TARGET = "T4"
SPECTRUM_LAYERS = (29, 30, 31)
BINARY_SPECS = ("T1_d0", "T1_d80", "T1_d160", "T1_d240", "T1_d400", "T1_d800", "T2", "T4")
N_LAYERS = 32
REMOVE_TOP_PC_KS = (1, 8, 16, 32, 64, 128)
ENERGY_KS = (1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512)
ALIGN_RHOS = (0.5, 0.8, 0.95)
MIMI_TOP_PCS = 32
RIDGE_LAMBDA_SCALE = 1e-3
NULLING_MAX_DIRECTIONS = 6
NULLING_STOP_AUC = 0.55
TRAJ_HALF_WINDOW = 25
TRAJ_N_BOOT = 1000
TRAJ_SEED = 20260723
IDENTITY_AUC_TOLERANCE = 1e-3


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
    """以第一个方向为符号参考，翻转反向者后平均并单位化。"""
    if not directions:
        raise ValueError("至少需要一个方向")
    reference = unit(directions[0])
    stacked = [reference]
    for vector in directions[1:]:
        v = unit(vector)
        stacked.append(-v if float(np.dot(reference, v)) < 0 else v)
    return unit(np.mean(np.stack(stacked), axis=0))


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


def iterative_nulling(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    fit_fn,
    max_directions: int,
    stop_auc: float,
) -> dict:
    """迭代均值差擦除：每轮重训探针记录 AUC，再剔除当前训练集 diff-in-means 方向。

    擦除方向必须用 diff-in-means 而非探针方向：剔除样本均值差方向使训练集两类
    均值差精确归零（一阶信号被完整擦除），而剔除"探针方向"存在固定点病理——
    估计方向与真方向的角误差每轮留下等比例残余、列内信噪比不变，AUC 永不崩塌
    （合成数据实测卡在固定值；与概念擦除文献 INLP→LEACE 的结论一致）。
    auc_sequence[i] 是已擦除 i 个方向后的重训 AUC；rounds_to_collapse 为首个
    AUC ≤ stop_auc 的擦除数——它度量"承载均值可分性的方向数"。
    fit_fn(x_tr, y_tr) 返回训练好的二分类 LinearProbe（注入以便测试与设备解耦）。
    """
    x_tr = np.array(x_train, dtype=np.float32, copy=True)
    x_ev = np.array(x_eval, dtype=np.float32, copy=True)
    auc_sequence: list[float] = []
    rounds_to_collapse: int | None = None
    for removed in range(int(max_directions) + 1):
        probe = fit_fn(x_tr, y_train)
        auc = float(pg.primary_metric(y_eval, probe.predict_proba(x_ev), 2))
        auc_sequence.append(auc)
        if auc <= float(stop_auc):
            rounds_to_collapse = removed
            break
        if removed == int(max_directions):
            break
        direction = unit(diff_in_means(x_tr, y_train))
        x_tr = remove_directions(x_tr, direction)
        x_ev = remove_directions(x_ev, direction)
    return {
        "auc_sequence": auc_sequence,
        "rounds_to_collapse": rounds_to_collapse,
        "collapsed": rounds_to_collapse is not None,
        "stop_auc": float(stop_auc),
        "max_directions": int(max_directions),
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


def cluster_bootstrap_separation(
    sum_pos: np.ndarray,
    count_pos: np.ndarray,
    sum_neg: np.ndarray,
    count_neg: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """会话级 cluster bootstrap：分组均值差 sep(t) 的点估计与 95% CI。

    输入均为 [n_sessions, n_offsets]（sum）或 [n_sessions]（count）。
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
    lower = np.nanpercentile(draws, 2.5, axis=0)
    upper = np.nanpercentile(draws, 97.5, axis=0)
    excludes = (lower > 0) | (upper < 0)
    first = np.flatnonzero(excludes)
    return {
        "separation": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "first_offset_index_ci_excludes_zero": int(first[0]) if len(first) else None,
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


def _protocol(probe_cfg: dict, summary: dict) -> tuple[dict, str]:
    if summary.get("g2", {}).get("verdict", {}).get("verdict") != "fail":
        raise SystemExit("正式汇总不再是 G2=fail，#32 几何解剖前提失效，请先复核")
    protocol = {
        "name": PROTOCOL_NAME,
        "prereg": 32,
        "target": TARGET,
        "binary_specs": list(BINARY_SPECS),
        "n_layers": N_LAYERS,
        "spectrum_layers": list(SPECTRUM_LAYERS),
        "remove_top_pc_ks": list(REMOVE_TOP_PC_KS),
        "energy_ks": list(ENERGY_KS),
        "align_rhos": list(ALIGN_RHOS),
        "mimi_top_pcs": MIMI_TOP_PCS,
        "ridge_lambda_scale": RIDGE_LAMBDA_SCALE,
        "nulling": {"max_directions": NULLING_MAX_DIRECTIONS, "stop_auc": NULLING_STOP_AUC},
        "trajectory": {
            "half_window": TRAJ_HALF_WINDOW,
            "n_boot": TRAJ_N_BOOT,
            "seed": TRAJ_SEED,
        },
        "identity_auc_tolerance": IDENTITY_AUC_TOLERANCE,
        "c_source": "fit_meta_chosen_c",
        "trainer": probe_cfg["trainer"],
        "pca_solver": "numpy.linalg.svd/full_matrices=False/float64",
        "formal_summary_sha256": _sha256(Path(REPO_ROOT) / "reports" / "wp_e1_probe_summary.json"),
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
    """从正式 fit 断点取双空间方向与 chosen_c；缺失时抛出带路径的错误。"""
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
    }


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
        if cached is not None:
            print("directions 断点已存在，跳过（--force 重算）")
            return cached

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
            pairs_orig = [abs_cosine(orig[i], orig[j]) for i in range(3) for j in range(i + 1, 3)]
            pairs_std = [abs_cosine(std[i], std[j]) for i in range(3) for j in range(i + 1, 3)]
            mean_dirs[(spec_name, layer)] = sign_aligned_mean(orig)
            layers_out[str(layer)] = {
                "min_pairwise_abs_cos_orig": float(min(pairs_orig)),
                "min_pairwise_abs_cos_std": float(min(pairs_std)),
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
    for layer in range(N_LAYERS):
        matrix = {
            a: {b: abs_cosine(mean_dirs[(a, layer)], mean_dirs[(b, layer)]) for b in BINARY_SPECS}
            for a in BINARY_SPECS
        }
        cross_spec[str(layer)] = matrix
    t1_vs_t4 = {
        spec_name: {str(layer): cross_spec[str(layer)][spec_name][TARGET] for layer in SPECTRUM_LAYERS}
        for spec_name in BINARY_SPECS
        if spec_name.startswith("T1_")
    }

    arrays: dict[str, np.ndarray] = {
        "__meta__": np.frombuffer(
            json.dumps({"protocol_hash": protocol_hash, "seeds": seeds}, ensure_ascii=False).encode(),
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
        "min_fit_vs_cell_abs_cos": min_cell_cos,
        "per_spec": per_spec,
        "t4_layer_propagation": t4_propagation,
        "cross_spec_abs_cos_by_layer": cross_spec,
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


def _refit_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    chosen_c: float,
    trainer: dict,
    device: str,
) -> float:
    probe = pg.fit_linear_probe(
        x_train,
        y_train,
        2,
        chosen_c,
        device=device,
        max_iter=int(trainer["lbfgs_max_iter"]),
        tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
    )
    return float(pg.primary_metric(y_eval, probe.predict_proba(x_eval), 2))


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
    formal_auc: float,
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

    d_raw = diff_in_means(x_train, y_train)
    d_unit = unit(d_raw)
    d_auc = projection_auc(y_eval, eval64 @ d_unit)
    cos_d_vstar = abs_cosine(d_unit, v_star)

    profile_v = energy_profile(vt, v_star)
    profile_d = energy_profile(vt, d_unit)
    cum_v = profile_v["cumulative"]
    cum_d = profile_d["cumulative"]

    removals: dict[str, float] = {
        "full_refit": _refit_auc(x_train, y_train, x_eval, y_eval, chosen_c, trainer, device)
    }
    for name, direction in (("v_star", v_star), ("d", d_unit)):
        x_tr_removed = remove_directions(x_train, direction)
        x_ev_removed = remove_directions(x_eval, direction)
        removals[f"remove_dir_{name}"] = _refit_auc(
            x_tr_removed, y_train, x_ev_removed, y_eval, chosen_c, trainer, device
        )
        del x_tr_removed, x_ev_removed
        gc.collect()
    for k in REMOVE_TOP_PC_KS:
        x_tr_removed = remove_top_pcs(x_train, center, vt, k)
        x_ev_removed = remove_top_pcs(x_eval, center, vt, k)
        removals[f"remove_top_pc_{k}"] = _refit_auc(
            x_tr_removed, y_train, x_ev_removed, y_eval, chosen_c, trainer, device
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

    nulling = iterative_nulling(
        x_train,
        y_train,
        x_eval,
        y_eval,
        fit_fn=_fit,
        max_directions=NULLING_MAX_DIRECTIONS,
        stop_auc=NULLING_STOP_AUC,
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
    del eval64, targets_train, targets_eval
    gc.collect()

    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": "spectrum",
        "layer": layer,
        "seed": seed,
        "n_train_rows": len(y_train),
        "n_eval_rows": len(y_eval),
        "formal_auc": float(formal_auc),
        "chosen_c": chosen_c,
        "identity_projection_auc": identity_auc,
        "identity_gap": identity_gap,
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
        "nulling": nulling,
        "mimi_r2": mimi_r2,
        "projection_stats": {"v_star": stats_v, "d": stats_d},
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
        if args.force
        or _load_json_checkpoint(
            _spectrum_checkpoint_path(root, layer, seed), protocol_hash, {"layer": layer, "seed": seed}
        )
        is None
        or not _vectors_path(root, layer, seed).is_file()
    ]
    print(f"spectrum：待计算 {len(pending)}/{len(layers) * len(seeds)} 个层×种子任务", flush=True)
    if not pending:
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
        if not layer_pending:
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
                formal_auc=float(auc_table[str(seed)][str(layer)]),
                trainer=probe_cfg["trainer"],
                device=str(args.device),
            )
            record["protocol_hash"] = protocol_hash
            vectors["__meta__"] = np.frombuffer(
                json.dumps(
                    {"protocol_hash": protocol_hash, "layer": layer, "seed": seed},
                    ensure_ascii=False,
                ).encode(),
                dtype=np.uint8,
            )
            _atomic_write_npz(_vectors_path(root, layer, seed), vectors)
            _atomic_write_json(_spectrum_checkpoint_path(root, layer, seed), record)
            print(
                f"L{layer}/s{seed} 完成：E(16)={record['energy_v_star']['at_k']['16']:.4f}，"
                f"剔除 v* 后 AUC={record['removal_auc']['remove_dir_v_star']:.4f}，"
                f"迭代剔除崩塌轮数={record['nulling']['rounds_to_collapse']}，"
                f"cos(d,v*)={record['diff_means']['abs_cos_to_v_star']:.4f}，"
                f"耗时 {record['wall_seconds']:.1f}s",
                flush=True,
            )
            del x_train, y_train
            gc.collect()
            engine._empty_cuda_cache(str(args.device))
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
    if not args.force:
        cached = _load_json_checkpoint(checkpoint, protocol_hash, {"stage": "trajectory", "layer": layer})
        if cached is not None:
            print("trajectory 断点已存在，跳过（--force 重算）")
            return cached

    fragments = []
    for seed in seeds:
        path = _vectors_path(root, layer, seed)
        if not path.is_file():
            raise SystemExit(f"缺少 {path}——trajectory 依赖 spectrum 阶段先完成该层三种子")
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(bytes(payload["__meta__"]).decode())
            if meta.get("protocol_hash") != protocol_hash:
                raise SystemExit(f"{path} 协议哈希不匹配，请先以当前协议重跑 spectrum")
            fragments.append(
                {key: np.asarray(payload[key], dtype=np.float64) for key in ("v_star", "d", "pc1")}
            )
    stats = {}
    for seed in seeds:
        record = _load_json_checkpoint(
            _spectrum_checkpoint_path(root, layer, seed), protocol_hash, {"layer": layer, "seed": seed}
        )
        if record is None:
            raise SystemExit(f"缺少 L{layer}/s{seed} 的 spectrum 断点")
        stats[seed] = record["projection_stats"]

    directions = {
        "v_star": sign_aligned_mean([fragment["v_star"] for fragment in fragments]),
        "d": sign_aligned_mean([fragment["d"] for fragment in fragments]),
        "pc1": sign_aligned_mean([fragment["pc1"] for fragment in fragments]),
        "random": unit(np.random.default_rng(TRAJ_SEED).standard_normal(len(fragments[0]["v_star"]))),
    }
    projection_norms = {
        "v_star": float(np.mean([stats[seed]["v_star"]["pooled_std"] for seed in seeds])),
        "d": float(np.mean([stats[seed]["d"]["pooled_std"] for seed in seeds])),
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
            sums[name][1], counts[name][1], sums[name][0], counts[name][0], TRAJ_N_BOOT, TRAJ_SEED
        )
        norm = projection_norms.get(name, 1.0)
        first_index = boot["first_offset_index_ci_excludes_zero"]
        curves[name] = {
            "mean_label1": (sums[name][1].sum(axis=0) / counts[name][1].sum()).tolist(),
            "mean_label0": (sums[name][0].sum(axis=0) / counts[name][0].sum()).tolist(),
            "separation": boot["separation"].tolist(),
            "separation_in_train_sigma": (boot["separation"] / norm).tolist() if name in projection_norms else None,
            "ci_lower": boot["ci_lower"].tolist(),
            "ci_upper": boot["ci_upper"].tolist(),
            "first_offset_ms_ci_excludes_zero": (None if first_index is None else offsets_ms[first_index]),
            "bootstrap_dropped_draws": boot["bootstrap_dropped_draws"],
            "n_events_label1": float(counts[name][1].sum()),
            "n_events_label0": float(counts[name][0].sum()),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_hash": protocol_hash,
        "stage": "trajectory",
        "layer": layer,
        "offsets_ms": offsets_ms,
        "half_window_steps": TRAJ_HALF_WINDOW,
        "total_events": total_events,
        "dropped_events_incomplete_window": dropped_events,
        "n_sessions": len(session_ids),
        "direction_pairwise_abs_cos": {
            "v_star__d": abs_cosine(directions["v_star"], directions["d"]),
            "v_star__pc1": abs_cosine(directions["v_star"], directions["pc1"]),
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
    sep_first = curves["v_star"]["first_offset_ms_ci_excludes_zero"]
    print(
        f"trajectory 完成：事件 {total_events}（弃 {dropped_events}），"
        f"v* 分离首个显著偏移 = {sep_first} ms",
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
    nulling_rounds = across(
        lambda r: (
            r["nulling"]["rounds_to_collapse"]
            if r["nulling"]["rounds_to_collapse"] is not None
            else float("inf")
        )
    )
    nulling_last_auc = across(lambda r: r["nulling"]["auc_sequence"][-1])
    remove_top16_drop = across(lambda r: r["formal_auc"] - r["removal_auc"]["remove_top_pc_16"])
    cos_d = across(lambda r: r["diff_means"]["abs_cos_to_v_star"])
    mimi_gap = across(
        lambda r: float(np.mean(r["mimi_r2"]["per_pc"][:8])) - r["mimi_r2"]["v_star"]
    )
    identity_gap = across(lambda r: r["identity_gap"])

    worst_rounds = max(nulling_rounds)
    if worst_rounds <= 2:
        necessity_verdict = "alpha"
    elif worst_rounds == float("inf") and max(nulling_last_auc) >= 0.70:
        necessity_verdict = "beta"
    else:
        necessity_verdict = "indeterminate"
    verdicts = {
        "E16_misalignment": {
            "values": e16,
            "worst": max(e16),
            "verdict": _verdict_interval(max(e16), 0.5, 0.8),
            "reading": "alpha=错向主张成立（E(16)<0.5）；beta=与 rank-k 曲线矛盾须查实现",
        },
        "one_dim_necessity": {
            "values": [None if value == float("inf") else int(value) for value in nulling_rounds],
            "worst": None if worst_rounds == float("inf") else int(worst_rounds),
            "verdict": necessity_verdict,
            "reading": (
                "alpha=一维/近一维码（迭代剔除 ≤2 轮崩塌至 AUC≤0.55）；"
                "beta=厚方向束（6 轮未崩塌且末端 AUC≥0.70）；单次剔除对方向估计误差不稳健，"
                "崩塌轮数才是必要方向数的判据"
            ),
        },
        "top16_contribution": {
            "values": remove_top16_drop,
            "worst": max(remove_top16_drop),
            "verdict": "alpha" if max(remove_top16_drop) < 0.01 else "beta",
            "reading": "alpha=方差前 16 主轴对读出无实质贡献（掉幅<0.01）",
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
            "reading": "alpha=主轴被 Mimi 高预测而 v* 低（top-8 PC 均值 R² − v* R² > 0.2）",
        },
    }
    return {
        "kind": "exploratory_non_decisive",
        "prereg": 32,
        "formal_g2_unchanged": True,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "identity_gap_max": max(identity_gap),
        "verdict_matrix": verdicts,
        "directions": directions,
        "spectrum": spectrum,
        "trajectory": trajectory,
        "runtime": {"git_head": _git_head()},
    }


def _format_metric(value) -> str:
    if value is None:
        return "未崩塌"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _write_markdown(result: dict) -> Path:
    verdicts = result["verdict_matrix"]
    t4_prop = result["directions"]["t4_layer_propagation"]
    lines = [
        "# E1 几何解剖报告（第一梯队）",
        "",
        "> 探索性、严格非裁决（PREREG #32）。G2 正式判定保持 `fail`，本报告不回写正式汇总。",
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
            "## 关键锚点",
            "",
            f"- 一维投影恒等自检最大偏差：{result['identity_gap_max']:.2e}（门限 {IDENTITY_AUC_TOLERANCE}）。",
            f"- fit↔cell 方向最小 |cos|：{result['directions']['min_fit_vs_cell_abs_cos']:.6f}。",
            "- T4 方向层间传播（种子均值方向，相邻层 |cos|）：L28→L29 "
            f"{t4_prop['28'].get('adjacent_abs_cos', float('nan')):.4f}，L29→L30 "
            f"{t4_prop['29'].get('adjacent_abs_cos', float('nan')):.4f}，L30→L31 "
            f"{t4_prop['30'].get('adjacent_abs_cos', float('nan')):.4f}。",
        ]
    )
    trajectory = result.get("trajectory")
    if trajectory is not None:
        v_curve = trajectory["curves"]["v_star"]
        lines.extend(
            [
                f"- 事件锁定轨迹（L{trajectory['layer']}）：事件 {trajectory['total_events']}，"
                f"v* 分离首个显著偏移 = {v_curve['first_offset_ms_ci_excludes_zero']} ms；"
                f"随机方向对照首个显著偏移 = "
                f"{trajectory['curves']['random']['first_offset_ms_ci_excludes_zero']} ms。",
            ]
        )
    else:
        lines.append("- 事件锁定轨迹：未运行（trajectory 阶段可选）。")
    lines.extend(
        [
            "",
            "完整逐任务数值见 `wp_e1_geometry_autopsy.json`；"
            "转向向量包见 `<data_root>/e1_probe/geometry/vectors/`。",
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
    protocol, protocol_hash = _protocol(probe_cfg, summary)
    print(f"#32 几何解剖协议 {protocol_hash[:12]}，stage={args.stage}", flush=True)
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
