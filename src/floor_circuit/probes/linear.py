"""线性探针协议（冻结）：L2-logistic，C 网格在验证段选择，特征 z-score 只在训练段拟合，
负类下采样 5:1，3 种子。输出按会话组织的评估分数，供 cluster bootstrap 使用。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SessionData = dict[str, tuple[np.ndarray, np.ndarray]]  # sid -> (X [n,d], y [n])


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
