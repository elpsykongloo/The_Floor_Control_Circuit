"""统计规范（文档/00 §8）：会话级 cluster bootstrap（1,000 次）、成对优势 CI、G1 裁决。
帧级重采样禁止——一切重采样单位都是会话。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

PerSession = dict[str, tuple[np.ndarray, np.ndarray]]  # sid -> (y_true, y_score)


def _pooled(per_session: PerSession, sids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ys = np.concatenate([per_session[s][0] for s in sids])
    ps = np.concatenate([per_session[s][1] for s in sids])
    return ys, ps


def pooled_metrics(per_session: PerSession) -> dict:
    sids = sorted(per_session)
    y, p = _pooled(per_session, sids)
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_acc": float(balanced_accuracy_score(y, p >= 0.5)),
        "n": len(y),
        "pos_rate": float(np.mean(y)),
        "n_sessions": len(sids),
    }


def _resample_sids(sids: list[str], rng: np.random.Generator) -> list[str]:
    idx = rng.integers(0, len(sids), size=len(sids))
    return [sids[i] for i in idx]


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def cluster_bootstrap_auc(
    per_session: PerSession, n_boot: int = 1000, seed: int = 0
) -> dict:
    """会话级有放回重采样的 AUC 95% CI。"""
    sids = sorted(per_session)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        take = _resample_sids(sids, rng)
        y, p = _pooled(per_session, take)
        auc = _safe_auc(y, p)
        if auc is not None:
            samples.append(auc)
    arr = np.asarray(samples)
    y, p = _pooled(per_session, sids)
    return {
        "point": _safe_auc(y, p),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_boot_effective": len(arr),
    }


def paired_advantage_bootstrap(
    probe: PerSession,
    baselines: dict[str, PerSession],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """优势统计量 = 探针 AUC − max(各基线 AUC)，同一会话重采样下成对计算（文档/00 §7-G1）。"""
    sids = sorted(probe)
    for name, b in baselines.items():
        if sorted(b) != sids:
            raise ValueError(f"基线 {name} 的会话集合与探针不一致")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        take = _resample_sids(sids, rng)
        y_p, p_p = _pooled(probe, take)
        auc_p = _safe_auc(y_p, p_p)
        if auc_p is None:
            continue
        base_aucs = []
        for b in baselines.values():
            y_b, p_b = _pooled(b, take)
            auc_b = _safe_auc(y_b, p_b)
            if auc_b is not None:
                base_aucs.append(auc_b)
        if base_aucs:
            samples.append(auc_p - max(base_aucs))
    arr = np.asarray(samples)
    y, p = _pooled(probe, sids)
    point_probe = _safe_auc(y, p)
    point_bases = {}
    for name, b in baselines.items():
        yb, pb = _pooled(b, sids)
        point_bases[name] = _safe_auc(yb, pb)
    best_base = max(v for v in point_bases.values() if v is not None)
    return {
        "advantage_point": float(point_probe - best_base),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "probe_auc": point_probe,
        "baseline_aucs": point_bases,
        "n_boot_effective": len(arr),
    }


def g1_verdict(adv_point: float, ci_lo: float, full_thr: float, backup_thr: float) -> str:
    """G1 三分支：full_e1 / backup_mve / n1（文档/00 §7）。full 额外要求 CI 下界 > 0。"""
    if adv_point >= full_thr and ci_lo > 0:
        return "full_e1"
    if adv_point >= backup_thr:
        return "backup_mve"
    return "n1"


def shuffle_labels_within_session(
    per_session: PerSession, seed: int = 0
) -> PerSession:
    """shuffled-labels sanity：会话内打乱标签，期望 AUC ≈ 0.5。"""
    rng = np.random.default_rng(seed)
    out = {}
    for sid, (y, p) in per_session.items():
        y2 = np.array(y, copy=True)
        rng.shuffle(y2)
        out[sid] = (y2, p)
    return out
