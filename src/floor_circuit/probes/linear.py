"""线性探针协议（冻结）：L2-logistic，C 网格在验证段选择，特征 z-score 只在训练段拟合，
负类下采样 5:1，3 种子。输出按会话组织的评估分数，供 cluster bootstrap 使用。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SessionData = dict[str, tuple[np.ndarray, np.ndarray]]  # sid -> (X [n,d], y [n])
SessionProvider = Callable[[str], tuple[np.ndarray, np.ndarray]]


def downsample_negatives(
    X: np.ndarray, y: np.ndarray, ratio: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """保留全部正类，负类下采样至 ratio:1（负类不足时全保留）。"""
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    n_keep = min(len(neg), ratio * max(len(pos), 1))
    keep_neg = rng.choice(neg, size=n_keep, replace=False) if len(neg) > n_keep else neg
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]


def _stack(data: SessionData, sids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sid in sids:
        X, y = data[sid]
        xs.append(np.asarray(X, dtype=np.float32))
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


@dataclass
class ProbeFit:
    best_c: float
    seed: int
    scaler: StandardScaler
    model: LogisticRegression
    val_auc_by_c: dict[float, float]


def fit_probe(
    data: SessionData,
    train_sids: list[str],
    val_sids: list[str],
    c_grid: list[float],
    seed: int,
    neg_ratio: int = 5,
) -> ProbeFit:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    X_tr, y_tr = _stack(data, train_sids)
    X_tr, y_tr = downsample_negatives(X_tr, y_tr, neg_ratio, rng)
    X_va, y_va = _stack(data, val_sids)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)
    best: tuple[float, LogisticRegression] | None = None
    val_aucs: dict[float, float] = {}
    for c in c_grid:
        clf = LogisticRegression(C=float(c), max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(X_tr_s, y_tr)
        auc = float(roc_auc_score(y_va, clf.predict_proba(X_va_s)[:, 1]))
        val_aucs[float(c)] = auc
        if best is None or auc > val_aucs[best[0]]:
            best = (float(c), clf)
    assert best is not None
    return ProbeFit(best_c=best[0], seed=seed, scaler=scaler, model=best[1], val_auc_by_c=val_aucs)


def fit_probe_streaming(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_sids: list[str],
    val_provider: SessionProvider,
    c_grid: list[float],
    seed: int,
) -> tuple[ProbeFit, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """在已抽样训练矩阵上拟合，并逐会话读取验证特征。

    所有 C 先完成拟合，随后验证集只扫描一次；内存中长期保留的验证材料仅有一维标签和分数。
    ``X_train`` 会被原地标准化，调用方不得依赖调用后的原始特征值。
    """

    from sklearn.metrics import roc_auc_score

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    if X_train.ndim != 2 or len(X_train) != len(y_train):
        raise ValueError("训练特征与标签形状不一致")
    if len(np.unique(y_train)) < 2:
        raise ValueError("训练标签必须同时包含正负类")
    if not c_grid:
        raise ValueError("C 网格不能为空")

    scaler = StandardScaler(copy=False).fit(X_train)
    X_train_scaled = scaler.transform(X_train, copy=False)
    models: dict[float, LogisticRegression] = {}
    for c_value in c_grid:
        c = float(c_value)
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed)
        model.fit(X_train_scaled, y_train)
        models[c] = model

    scores_by_c: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        c: {} for c in models
    }
    for sid in val_sids:
        X_val, y_val = val_provider(sid)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.int64)
        if X_val.ndim != 2 or len(X_val) != len(y_val):
            raise ValueError(f"{sid} 验证特征与标签形状不一致")
        X_val_scaled = scaler.transform(X_val, copy=False)
        for c, model in models.items():
            score = model.predict_proba(X_val_scaled)[:, 1].astype(np.float64)
            scores_by_c[c][sid] = (y_val, score)

    val_aucs: dict[float, float] = {}
    for c, per_session in scores_by_c.items():
        y_all = np.concatenate([per_session[sid][0] for sid in val_sids])
        score_all = np.concatenate([per_session[sid][1] for sid in val_sids])
        val_aucs[c] = float(roc_auc_score(y_all, score_all))
    best_c = max(models, key=lambda c: val_aucs[c])
    fit = ProbeFit(
        best_c=best_c,
        seed=seed,
        scaler=scaler,
        model=models[best_c],
        val_auc_by_c=val_aucs,
    )
    return fit, scores_by_c[best_c]


def score_sessions(fit: ProbeFit, data: SessionData, sids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """返回 sid -> (y_true, y_score)，供 stats.cluster_bootstrap。"""
    out = {}
    for sid in sids:
        X, y = data[sid]
        scores = fit.model.predict_proba(fit.scaler.transform(np.asarray(X, dtype=np.float32)))[:, 1]
        out[sid] = (np.asarray(y, dtype=np.int64), scores.astype(np.float64))
    return out


def probe_direction(fit: ProbeFit) -> np.ndarray:
    """标准化空间中的探针方向（单位化），供后续几何/steering 分析复用。"""
    w = fit.model.coef_.ravel()
    return w / (np.linalg.norm(w) + 1e-12)
