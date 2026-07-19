"""跨模型上下文应力结果的校验、统计与工程性结论。

本模块处理模型专用环境写出的 ``manifest.json + trace.npz``。它不加载模型权重，
也不改动 MVE/E1 的冻结判据。所有阈值均作为描述性工程诊断随报告明示。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA_VERSION = 1

# 这些阈值用于复现 Moshi 已观察到的异常形态，不构成 G1/G2 判据。
DIAGNOSTIC_THRESHOLDS = {
    "norm_robust_z": 8.0,
    "norm_ratio_high": 1.5,
    "norm_ratio_low": 2.0 / 3.0,
    "direction_cosine_max": 0.8,
    "cross_run_cosine_min": 0.9,
    "cross_run_cosine_increase": 0.15,
    "uncentered_pc1_share_min": 0.8,
    "decision_agreement_min": 0.9,
    "decision_agreement_increase": 0.25,
    "decision_entropy_drop": 0.2,
    "dynamic_key_cosine_min": 0.99,
    "task_auc_drop": 0.05,
    "task_auc_pre_min": 0.55,
}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ContextStressRun:
    """一个完整上下文应力运行。"""

    root: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def model(self) -> str:
        return str(self.manifest["model"])

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.manifest["layers"])


def load_context_stress_run(root: Path, *, require_complete: bool = True) -> ContextStressRun:
    """读取并严格校验一个运行目录。"""

    root = Path(root)
    manifest_path = root / "manifest.json"
    trace_path = root / "trace.npz"
    if not manifest_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"{root} 缺少 manifest.json 或 trace.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"{root}: schema_version={manifest.get('schema_version')}，"
            f"期望 {TRACE_SCHEMA_VERSION}"
        )
    if manifest.get("protocol") != "context_stress_v1":
        raise ValueError(f"{root}: protocol 必须为 context_stress_v1")
    if require_complete and not bool(manifest.get("complete", False)):
        raise ValueError(f"{root}: 运行未完整结束；如需故障诊断请显式允许 partial")
    declared_sha = manifest.get("trace_sha256")
    if declared_sha and declared_sha != _sha256(trace_path):
        raise ValueError(f"{root}: trace.npz SHA-256 与 manifest 不一致")

    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {
        "logical_positions",
        "cache_lengths",
        "position_offsets",
        "hidden",
        "decision_probs",
        "decision_ids",
        "dynamic_key_cosines",
        "all_finite",
        "sliding_events",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{root}: trace 缺少字段 {missing}")

    positions = arrays["logical_positions"]
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError(f"{root}: logical_positions 必须为非空一维数组")
    if np.any(np.diff(positions) <= 0):
        raise ValueError(f"{root}: logical_positions 必须严格递增")
    n_samples = len(positions)
    for name in (
        "cache_lengths",
        "position_offsets",
        "decision_probs",
        "decision_ids",
        "dynamic_key_cosines",
        "all_finite",
        "sliding_events",
    ):
        if arrays[name].shape[0] != n_samples:
            raise ValueError(f"{root}: {name} 首维与 logical_positions 不一致")
    hidden = arrays["hidden"]
    if hidden.ndim != 3 or hidden.shape[0] != n_samples:
        raise ValueError(f"{root}: hidden 必须为 [sample, layer, hidden_dim]")
    if hidden.shape[1] != len(manifest.get("layers", [])):
        raise ValueError(f"{root}: hidden 层数与 manifest.layers 不一致")
    if arrays["dynamic_key_cosines"].shape != hidden.shape[:2]:
        raise ValueError(f"{root}: dynamic_key_cosines 必须为 [sample, layer]")
    if arrays["decision_probs"].ndim != 2:
        raise ValueError(f"{root}: decision_probs 必须为二维概率数组")
    probabilities = arrays["decision_probs"].astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{root}: decision_probs 含非有限值")
    if np.any(probabilities < -1e-6):
        raise ValueError(f"{root}: decision_probs 含负值")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=2e-3):
        raise ValueError(f"{root}: decision_probs 每行之和必须约等于 1")
    task_fields = {"task_labels", "task_scores"}
    present_task_fields = task_fields & set(arrays)
    if present_task_fields and present_task_fields != task_fields:
        raise ValueError(f"{root}: task_labels 与 task_scores 必须同时出现")
    if present_task_fields:
        if arrays["task_labels"].shape != (n_samples,):
            raise ValueError(f"{root}: task_labels 必须为 [sample]")
        if arrays["task_scores"].shape != (n_samples,):
            raise ValueError(f"{root}: task_scores 必须为 [sample]")
        valid_labels = arrays["task_labels"] >= 0
        if np.any(~np.isin(arrays["task_labels"][valid_labels], [0, 1])):
            raise ValueError(f"{root}: 有效 task_labels 只能取 0/1，缺失值用 -1")
        if not np.isfinite(arrays["task_scores"]).all():
            raise ValueError(f"{root}: task_scores 含非有限值")
    return ContextStressRun(root=root, manifest=manifest, arrays=arrays)


def _common_positions(runs: list[ContextStressRun]) -> np.ndarray:
    common = runs[0].arrays["logical_positions"].astype(np.int64)
    for run in runs[1:]:
        common = np.intersect1d(
            common,
            run.arrays["logical_positions"].astype(np.int64),
            assume_unique=True,
        )
    if len(common) < 3:
        raise ValueError("各运行的共同采样位置少于 3 个，无法分析")
    return common


def _aligned(run: ContextStressRun, name: str, positions: np.ndarray) -> np.ndarray:
    own = run.arrays["logical_positions"].astype(np.int64)
    indices = np.searchsorted(own, positions)
    if np.any(indices >= len(own)) or not np.array_equal(own[indices], positions):
        raise ValueError(f"{run.run_id}: 无法按共同位置对齐 {name}")
    return run.arrays[name][indices]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _median_pairwise_cosine(unit_hidden: np.ndarray) -> np.ndarray:
    """输入 [run, sample, layer, hidden]，输出 [sample, layer]。"""

    n_runs = unit_hidden.shape[0]
    if n_runs < 2:
        return np.full(unit_hidden.shape[1:3], np.nan, dtype=np.float64)
    summed = unit_hidden.sum(axis=0)
    numerator = np.square(summed).sum(axis=-1) - n_runs
    denominator = n_runs * (n_runs - 1)
    return numerator / denominator


def _uncentered_pc1_share(unit_hidden: np.ndarray) -> np.ndarray:
    """输入 [run, sample, layer, hidden]，输出 [sample, layer]。"""

    n_runs, n_samples, n_layers, _ = unit_hidden.shape
    if n_runs < 2:
        return np.full((n_samples, n_layers), np.nan, dtype=np.float64)
    out = np.empty((n_samples, n_layers), dtype=np.float64)
    for sample in range(n_samples):
        for layer in range(n_layers):
            values = unit_hidden[:, sample, layer, :].astype(np.float64)
            gram = values @ values.T
            eigenvalues = np.linalg.eigvalsh(gram)
            total = float(np.maximum(eigenvalues, 0.0).sum())
            out[sample, layer] = float(eigenvalues[-1] / total) if total > 0 else np.nan
    return out


def _decision_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1)


def _decision_agreement(decision_ids: np.ndarray) -> np.ndarray:
    """输入 [run, sample]，输出每个 sample 的跨运行众数占比。"""

    n_runs, n_samples = decision_ids.shape
    if n_runs < 2:
        return np.full(n_samples, np.nan, dtype=np.float64)
    out = np.empty(n_samples, dtype=np.float64)
    for sample in range(n_samples):
        _, counts = np.unique(decision_ids[:, sample], return_counts=True)
        out[sample] = float(counts.max() / n_runs)
    return out


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """带并列秩的二分类 AUC；任一类别缺失时返回 NaN。"""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = (labels == 0) | (labels == 1)
    labels = labels[valid]
    scores = scores[valid]
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # 秩从 1 开始；并列取平均秩。
        mean_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = mean_rank
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    median = np.median(finite)
    return float(max(1.4826 * np.median(np.abs(finite - median)), 1e-8))


def _safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _direction_cosine(pre: np.ndarray, post: np.ndarray) -> float:
    pre_direction = _normalize_rows(pre.reshape(-1, pre.shape[-1])).mean(axis=0)
    post_direction = _normalize_rows(post.reshape(-1, post.shape[-1])).mean(axis=0)
    denominator = np.linalg.norm(pre_direction) * np.linalg.norm(post_direction)
    if denominator == 0:
        return float("nan")
    return float(np.dot(pre_direction, post_direction) / denominator)


def _boundary_masks(
    positions: np.ndarray,
    boundary: int,
    window_positions: int,
) -> tuple[np.ndarray, np.ndarray]:
    pre = (positions >= boundary - window_positions) & (positions < boundary)
    post = (positions >= boundary) & (positions < boundary + window_positions)
    return pre, post


def _layer_boundary_metrics(
    *,
    layer_index: int,
    layer_number: int,
    boundary: int,
    positions: np.ndarray,
    hidden: np.ndarray,
    finite: np.ndarray,
    pairwise_cosine: np.ndarray,
    pc1_share: np.ndarray,
    key_cosines: np.ndarray,
    cache_lengths: np.ndarray,
    decision_entropy: np.ndarray,
    decision_agreement: np.ndarray,
    task_labels: np.ndarray | None,
    task_scores: np.ndarray | None,
    window_positions: int,
) -> dict[str, Any] | None:
    pre, post = _boundary_masks(positions, boundary, window_positions)
    if int(pre.sum()) < 2 or int(post.sum()) < 2:
        return None
    norms = np.linalg.norm(hidden[:, :, layer_index, :].astype(np.float64), axis=-1)
    pre_norms = norms[:, pre].reshape(-1)
    post_norms = norms[:, post].reshape(-1)
    pre_median = _safe_median(pre_norms)
    post_median = _safe_median(post_norms)
    norm_ratio = float(post_median / pre_median) if pre_median > 0 else float("nan")
    norm_robust_z = float(abs(post_median - pre_median) / _robust_scale(pre_norms))
    direction_cosine = _direction_cosine(
        hidden[:, pre, layer_index, :].astype(np.float64),
        hidden[:, post, layer_index, :].astype(np.float64),
    )
    pair_pre = _safe_median(pairwise_cosine[pre, layer_index])
    pair_post = _safe_median(pairwise_cosine[post, layer_index])
    pc1_pre = _safe_median(pc1_share[pre, layer_index])
    pc1_post = _safe_median(pc1_share[post, layer_index])
    key_post_min = float(np.nanmin(key_cosines[:, post, layer_index]))
    finite_pre = float(np.mean(finite[:, pre, layer_index]))
    finite_post = float(np.mean(finite[:, post, layer_index]))

    # 缓存长度沿每个运行分别检查，避免把不同运行拼接处误判为淘汰。
    cache_drop_count = 0
    for run_cache in cache_lengths:
        indices = np.flatnonzero(pre | post)
        if len(indices) >= 2:
            cache_drop_count += int(np.sum(np.diff(run_cache[indices]) < 0))

    entropy_pre = _safe_median(decision_entropy[:, pre])
    entropy_post = _safe_median(decision_entropy[:, post])
    agreement_pre = _safe_median(decision_agreement[pre])
    agreement_post = _safe_median(decision_agreement[post])
    task_auc_pre = float("nan")
    task_auc_post = float("nan")
    task_rows_pre = 0
    task_rows_post = 0
    if task_labels is not None and task_scores is not None:
        pre_labels = task_labels[:, pre].reshape(-1)
        post_labels = task_labels[:, post].reshape(-1)
        pre_scores = task_scores[:, pre].reshape(-1)
        post_scores = task_scores[:, post].reshape(-1)
        task_rows_pre = int(np.sum(pre_labels >= 0))
        task_rows_post = int(np.sum(post_labels >= 0))
        task_auc_pre = _binary_auc(pre_labels, pre_scores)
        task_auc_post = _binary_auc(post_labels, post_scores)

    thresholds = DIAGNOSTIC_THRESHOLDS
    flags = {
        "nonfinite": bool(finite_post < 1.0),
        "norm_spike": bool(
            norm_robust_z >= thresholds["norm_robust_z"]
            and (
                norm_ratio >= thresholds["norm_ratio_high"]
                or norm_ratio <= thresholds["norm_ratio_low"]
            )
        ),
        "direction_jump": bool(
            np.isfinite(direction_cosine)
            and direction_cosine <= thresholds["direction_cosine_max"]
        ),
        "cross_run_collapse": bool(
            np.isfinite(pair_post)
            and pair_post >= thresholds["cross_run_cosine_min"]
            and pair_post - pair_pre >= thresholds["cross_run_cosine_increase"]
            and pc1_post >= thresholds["uncentered_pc1_share_min"]
        ),
        "decision_collapse": bool(
            np.isfinite(agreement_post)
            and agreement_post >= thresholds["decision_agreement_min"]
            and agreement_post - agreement_pre >= thresholds["decision_agreement_increase"]
            and entropy_pre - entropy_post >= thresholds["decision_entropy_drop"]
        ),
        "structural_eviction": bool(
            cache_drop_count > 0
            or (
                np.isfinite(key_post_min)
                and key_post_min < thresholds["dynamic_key_cosine_min"]
            )
        ),
        "task_performance_drop": bool(
            np.isfinite(task_auc_pre)
            and np.isfinite(task_auc_post)
            and task_auc_pre >= thresholds["task_auc_pre_min"]
            and task_auc_pre - task_auc_post >= thresholds["task_auc_drop"]
        ),
    }
    representation_count = sum(
        int(flags[name])
        for name in (
            "norm_spike",
            "direction_jump",
            "cross_run_collapse",
            "decision_collapse",
            "task_performance_drop",
        )
    )
    pathology_confirmed = bool(flags["nonfinite"] or representation_count >= 2)
    return {
        "layer": layer_number,
        "pre_samples": int(pre.sum()),
        "post_samples": int(post.sum()),
        "norm_pre_median": pre_median,
        "norm_post_median": post_median,
        "norm_ratio": norm_ratio,
        "norm_robust_z": norm_robust_z,
        "pre_post_direction_cosine": direction_cosine,
        "cross_run_pairwise_cosine_pre": pair_pre,
        "cross_run_pairwise_cosine_post": pair_post,
        "cross_run_pairwise_cosine_increase": float(pair_post - pair_pre),
        "uncentered_pc1_share_pre": pc1_pre,
        "uncentered_pc1_share_post": pc1_post,
        "decision_entropy_pre": entropy_pre,
        "decision_entropy_post": entropy_post,
        "decision_agreement_pre": agreement_pre,
        "decision_agreement_post": agreement_post,
        "task_auc_pre": task_auc_pre,
        "task_auc_post": task_auc_post,
        "task_auc_drop": float(task_auc_pre - task_auc_post),
        "task_rows_pre": task_rows_pre,
        "task_rows_post": task_rows_post,
        "dynamic_key_cosine_post_min": key_post_min,
        "finite_fraction_pre": finite_pre,
        "finite_fraction_post": finite_post,
        "cache_drop_count": cache_drop_count,
        "flags": flags,
        "pathology_confirmed": pathology_confirmed,
    }


def _validate_run_group(runs: list[ContextStressRun]) -> None:
    if not runs:
        raise ValueError("至少需要一个运行目录")
    first = runs[0]
    for run in runs[1:]:
        if run.model != first.model:
            raise ValueError("一次分析不能混合不同模型")
        if run.layers != first.layers:
            raise ValueError("所有运行必须使用相同层集合")
        if run.arrays["hidden"].shape[-1] != first.arrays["hidden"].shape[-1]:
            raise ValueError("所有运行的隐藏维度必须一致")
        if run.arrays["decision_probs"].shape[-1] != first.arrays["decision_probs"].shape[-1]:
            raise ValueError("所有运行的决策类别数必须一致")
        if run.manifest.get("context_spec") != first.manifest.get("context_spec"):
            raise ValueError("所有运行的 context_spec 必须逐字段一致")
        first_has_task = {"task_labels", "task_scores"} <= set(first.arrays)
        run_has_task = {"task_labels", "task_scores"} <= set(run.arrays)
        if run_has_task != first_has_task:
            raise ValueError("所有运行必须一致地提供或省略任务标签与分数")


def _derive_recommendation(
    *,
    manifest: dict[str, Any],
    max_tested_position: int,
    tested_positions: np.ndarray,
    boundary_results: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = manifest["context_spec"]
    start_position = int(spec["start_position"])
    official = int(spec["official_max_positions"])
    target_seconds = float(spec["analysis_target_seconds"])
    required = int(spec["analysis_target_required_positions"])

    confirmed_boundaries = [
        int(item["boundary"])
        for item in boundary_results
        if item["pathology_confirmed"]
    ]
    earliest_pathology = min(confirmed_boundaries) if confirmed_boundaries else None
    structural_boundaries = [
        int(item["boundary"])
        for item in boundary_results
        if item["structural_eviction_observed"]
    ]
    earliest_structural = min(structural_boundaries) if structural_boundaries else None
    target_covered = max_tested_position >= required
    target_clean = bool(
        target_covered
        and (earliest_pathology is None or earliest_pathology > required)
    )

    if earliest_pathology is not None:
        safe_candidates = tested_positions[tested_positions < earliest_pathology]
        safe_position = (
            int(safe_candidates[-1]) if len(safe_candidates) else start_position
        )
        status = "在候选边界观察到至少两类表征异常"
    elif max_tested_position >= official:
        safe_candidates = tested_positions[tested_positions <= official]
        safe_position = (
            int(safe_candidates[-1]) if len(safe_candidates) else start_position
        )
        status = "已覆盖官方上限，未在上限前确认表征骤变；仍按官方规格截断"
    else:
        safe_candidates = tested_positions[
            tested_positions <= min(max_tested_position, official)
        ]
        safe_position = (
            int(safe_candidates[-1]) if len(safe_candidates) else start_position
        )
        status = "只得到实测下界，尚未跨过官方上限"

    min_rate = float(spec["positions_per_second_min"])
    max_rate = float(spec["positions_per_second_max"])
    usable_positions = max(safe_position - start_position, 0)
    seconds_range = {
        "at_max_position_rate": float(usable_positions / max_rate),
        "at_min_position_rate": float(usable_positions / min_rate),
    }
    return {
        "status": status,
        "max_tested_position": max_tested_position,
        "official_max_positions": official,
        "official_boundary_covered": max_tested_position >= official,
        "earliest_confirmed_pathology_position": earliest_pathology,
        "earliest_structural_eviction_position": earliest_structural,
        "safe_position_for_this_report": safe_position,
        "safe_position_rule": "共同实测采样点中不越过病理边界或官方上限的最大位置",
        "safe_seconds_range": seconds_range,
        "analysis_target_seconds": target_seconds,
        "analysis_target_required_positions": required,
        "analysis_target_covered": target_covered,
        "analysis_target_clean": target_clean,
        "formal_note": (
            "该结论是上下文工程诊断，不自动改写 PREREG、grids.yaml 或 E1 判据；"
            "若要冻结为正式分析窗，仍需按变更流程登记。"
        ),
    }


def analyze_context_stress_runs(
    runs: list[ContextStressRun],
    *,
    boundaries: list[int] | None = None,
    window_positions: int | None = None,
) -> dict[str, Any]:
    """联合分析多个独立长流运行。"""

    _validate_run_group(runs)
    first = runs[0]
    positions = _common_positions(runs)
    layers = first.layers
    hidden = np.stack([_aligned(run, "hidden", positions) for run in runs]).astype(np.float64)
    decision_probs = np.stack(
        [_aligned(run, "decision_probs", positions) for run in runs]
    ).astype(np.float64)
    decision_ids = np.stack([_aligned(run, "decision_ids", positions) for run in runs])
    key_cosines = np.stack(
        [_aligned(run, "dynamic_key_cosines", positions) for run in runs]
    ).astype(np.float64)
    finite = np.stack([_aligned(run, "all_finite", positions) for run in runs]).astype(bool)
    cache_lengths = np.stack([_aligned(run, "cache_lengths", positions) for run in runs])

    unit_hidden = _normalize_rows(hidden)
    pairwise_cosine = _median_pairwise_cosine(unit_hidden)
    pc1_share = _uncentered_pc1_share(unit_hidden)
    entropy = _decision_entropy(decision_probs)
    agreement = _decision_agreement(decision_ids)
    task_labels = None
    task_scores = None
    if {"task_labels", "task_scores"} <= set(first.arrays):
        task_labels = np.stack(
            [_aligned(run, "task_labels", positions) for run in runs]
        ).astype(np.int8)
        task_scores = np.stack(
            [_aligned(run, "task_scores", positions) for run in runs]
        ).astype(np.float64)
        if not np.any(task_labels >= 0):
            task_labels = None
            task_scores = None

    planned = boundaries
    if planned is None:
        planned = sorted(
            {
                int(value)
                for run in runs
                for value in run.manifest.get("planned_boundaries", [])
            }
        )
    if not planned:
        raise ValueError("没有候选边界；请在 manifest 或参数中提供")
    if window_positions is None:
        window_positions = int(first.manifest["sampling"]["boundary_window_positions"])
    if window_positions <= 0:
        raise ValueError("window_positions 必须为正数")

    boundary_results: list[dict[str, Any]] = []
    for boundary in planned:
        layer_results = []
        for layer_index, layer_number in enumerate(layers):
            metrics = _layer_boundary_metrics(
                layer_index=layer_index,
                layer_number=layer_number,
                boundary=int(boundary),
                positions=positions,
                hidden=hidden,
                finite=finite,
                pairwise_cosine=pairwise_cosine,
                pc1_share=pc1_share,
                key_cosines=key_cosines,
                cache_lengths=cache_lengths,
                decision_entropy=entropy,
                decision_agreement=agreement,
                task_labels=task_labels,
                task_scores=task_scores,
                window_positions=window_positions,
            )
            if metrics is not None:
                layer_results.append(metrics)
        if not layer_results:
            boundary_results.append(
                {
                    "boundary": int(boundary),
                    "covered": False,
                    "layers": [],
                    "pathology_confirmed": False,
                    "structural_eviction_observed": False,
                }
            )
            continue
        pathology_layers = [
            int(item["layer"]) for item in layer_results if item["pathology_confirmed"]
        ]
        structural_layers = [
            int(item["layer"])
            for item in layer_results
            if item["flags"]["structural_eviction"]
        ]
        boundary_results.append(
            {
                "boundary": int(boundary),
                "covered": True,
                "window_positions": int(window_positions),
                "pathology_confirmed": bool(pathology_layers),
                "pathology_layers": pathology_layers,
                "structural_eviction_observed": bool(structural_layers),
                "structural_eviction_layers": structural_layers,
                "layers": layer_results,
            }
        )

    max_tested = int(min(run.arrays["logical_positions"][-1] for run in runs))
    recommendation = _derive_recommendation(
        manifest=first.manifest,
        max_tested_position=max_tested,
        tested_positions=positions,
        boundary_results=boundary_results,
    )
    return {
        "schema_version": 1,
        "analysis": "context_stress_v1",
        "model": first.model,
        "run_ids": [run.run_id for run in runs],
        "n_runs": len(runs),
        "cross_run_evidence_available": len(runs) >= 3,
        "task_performance_evidence_available": task_labels is not None,
        "layers": list(layers),
        "common_sample_count": len(positions),
        "common_position_range": [int(positions[0]), int(positions[-1])],
        "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
        "context_spec": first.manifest["context_spec"],
        "sampling": {
            "boundary_window_positions": int(window_positions),
            "candidate_boundaries": [int(value) for value in planned],
        },
        "boundaries": boundary_results,
        "recommendation": recommendation,
        "limitations": [
            (
                "少于 3 个独立输入运行时，跨运行公共方向与决策收敛指标只返回缺失值，"
                "不能独立确认公共吸引态。"
            ),
            (
                "隐藏状态骤变诊断只说明该运行制式下的数值或表征异常；"
                "若 trace 未携带循环对齐的任务标签，正式任务性能仍需在冻结标签与探针协议上分段复算。"
            ),
            (
                "模型在官方位置上限外继续返回有限值，只能视为外推运行成功，"
                "不能据此扩张正式上下文规格。"
            ),
        ],
    }


def render_context_stress_markdown(report: dict[str, Any]) -> str:
    """把分析报告渲染为简洁 Markdown。"""

    recommendation = report["recommendation"]
    lines = [
        f"# {report['model']} 上下文应力诊断",
        "",
        f"- 独立运行：{report['n_runs']}；共同采样点：{report['common_sample_count']}；"
        f"位置范围：{report['common_position_range'][0]}..{report['common_position_range'][1]}",
        f"- 结论：{recommendation['status']}",
        f"- 计划分析窗 {recommendation['analysis_target_seconds']:.1f} 秒："
        f"覆盖={'是' if recommendation['analysis_target_covered'] else '否'}，"
        f"干净={'是' if recommendation['analysis_target_clean'] else '否'}",
        f"- 本报告安全位置：{recommendation['safe_position_for_this_report']}；"
        f"按最大/最小位置速率折算 "
        f"{recommendation['safe_seconds_range']['at_max_position_rate']:.2f}.."
        f"{recommendation['safe_seconds_range']['at_min_position_rate']:.2f} 秒",
        "",
        "## 候选边界",
        "",
        "| 位置 | 已覆盖 | 表征骤变 | 结构淘汰 | 异常层 |",
        "| ---: | :---: | :---: | :---: | --- |",
    ]
    for item in report["boundaries"]:
        lines.append(
            f"| {item['boundary']} "
            f"| {'是' if item['covered'] else '否'} "
            f"| {'是' if item['pathology_confirmed'] else '否'} "
            f"| {'是' if item['structural_eviction_observed'] else '否'} "
            f"| {','.join(str(value) for value in item.get('pathology_layers', [])) or '—'} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
    ]
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines += ["", f"> {recommendation['formal_note']}", ""]
    return "\n".join(lines)
