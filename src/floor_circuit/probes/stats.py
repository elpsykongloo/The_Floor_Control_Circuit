"""统计规范（文档/00 §8）：会话级 cluster bootstrap（1,000 次）、成对优势 CI、G1 裁决。
帧级重采样禁止——一切重采样单位都是会话。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

PerSession = dict[str, tuple[np.ndarray, np.ndarray]]  # sid -> (y_true, y_score)
SeededPerSession = dict[int, PerSession]  # seed -> sid -> (y_true, y_score)
ScoreCollection = PerSession | SeededPerSession


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


def as_seeded_scores(scores: ScoreCollection) -> SeededPerSession:
    """把单种子或多种子分数统一为 ``seed -> PerSession``。"""

    if not scores:
        raise ValueError("会话分数为空")
    first = next(iter(scores.values()))
    if isinstance(first, dict):
        seeded = {int(seed): per_session for seed, per_session in scores.items()}
    else:
        seeded = {0: scores}
    if not seeded or any(not per_session for per_session in seeded.values()):
        raise ValueError("至少一个种子的会话分数为空")
    return seeded


def seed_mean_metrics(scores: ScoreCollection) -> dict:
    """先逐种子计算池化指标，再报告种子均值、样本标准差与逐种子明细。"""

    metrics_by_seed = {
        seed: pooled_metrics(per_session)
        for seed, per_session in sorted(as_seeded_scores(scores).items())
    }

    def summarize(name: str) -> tuple[float, float]:
        values = np.asarray([metrics[name] for metrics in metrics_by_seed.values()])
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return mean, sd

    auc_mean, auc_sd = summarize("auc")
    auprc_mean, auprc_sd = summarize("auprc")
    balanced_acc_mean, balanced_acc_sd = summarize("balanced_acc")
    first = next(iter(metrics_by_seed.values()))
    return {
        "n_seeds": len(metrics_by_seed),
        "auc": auc_mean,
        "auc_mean": auc_mean,
        "auc_sd": auc_sd,
        "auprc": auprc_mean,
        "auprc_mean": auprc_mean,
        "auprc_sd": auprc_sd,
        "balanced_acc": balanced_acc_mean,
        "balanced_acc_mean": balanced_acc_mean,
        "balanced_acc_sd": balanced_acc_sd,
        "n": first["n"],
        "pos_rate": first["pos_rate"],
        "n_sessions": first["n_sessions"],
        "by_seed": metrics_by_seed,
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


def _seed_mean_auc(scores: SeededPerSession, sids: list[str]) -> float | None:
    """在给定会话样本内逐种子计算 AUC，再取种子均值。"""

    aucs: list[float] = []
    for per_session in scores.values():
        y, p = _pooled(per_session, sids)
        auc = _safe_auc(y, p)
        if auc is None:
            return None
        aucs.append(auc)
    return float(np.mean(aucs))


def cluster_bootstrap_seed_mean_auc(
    scores: ScoreCollection,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """会话级重采样；每次先求各探针种子的 AUC，再取种子均值。"""

    seeded = as_seeded_scores(scores)
    sids = sorted(next(iter(seeded.values())))
    for score_seed, per_session in seeded.items():
        if sorted(per_session) != sids:
            raise ValueError(f"种子 {score_seed} 的会话集合不一致")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        auc = _seed_mean_auc(seeded, _resample_sids(sids, rng))
        if auc is not None:
            samples.append(auc)
    arr = np.asarray(samples)
    if not len(arr):
        raise ValueError("会话级 bootstrap 没有产生可用的双类样本")
    return {
        "point": _seed_mean_auc(seeded, sids),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_boot_effective": len(arr),
        "n_seeds": len(seeded),
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


def paired_seed_mean_advantage_bootstrap(
    probe: ScoreCollection,
    baselines: dict[str, ScoreCollection],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """会话重采样内比较种子均值探针与最大种子均值基线。"""

    probe_seeded = as_seeded_scores(probe)
    baseline_seeded = {
        name: as_seeded_scores(scores)
        for name, scores in baselines.items()
    }
    sids = sorted(next(iter(probe_seeded.values())))
    for score_seed, per_session in probe_seeded.items():
        if sorted(per_session) != sids:
            raise ValueError(f"探针种子 {score_seed} 的会话集合不一致")
    for name, seeded in baseline_seeded.items():
        for score_seed, per_session in seeded.items():
            if sorted(per_session) != sids:
                raise ValueError(f"基线 {name} 种子 {score_seed} 的会话集合与探针不一致")

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        take = _resample_sids(sids, rng)
        probe_auc = _seed_mean_auc(probe_seeded, take)
        baseline_aucs = [
            _seed_mean_auc(scores, take)
            for scores in baseline_seeded.values()
        ]
        if probe_auc is None or any(auc is None for auc in baseline_aucs):
            continue
        samples.append(probe_auc - max(baseline_aucs))
    arr = np.asarray(samples)
    if not len(arr):
        raise ValueError("成对优势 bootstrap 没有产生可用的双类样本")

    point_probe = _seed_mean_auc(probe_seeded, sids)
    point_bases = {
        name: _seed_mean_auc(scores, sids)
        for name, scores in baseline_seeded.items()
    }
    if point_probe is None or any(value is None for value in point_bases.values()):
        raise ValueError("完整评估集无法计算双类 AUC")
    best_base = max(point_bases.values())
    return {
        "advantage_point": float(point_probe - best_base),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "probe_auc": point_probe,
        "baseline_aucs": point_bases,
        "n_boot_effective": len(arr),
        "probe_n_seeds": len(probe_seeded),
        "baseline_n_seeds": {
            name: len(scores)
            for name, scores in baseline_seeded.items()
        },
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
