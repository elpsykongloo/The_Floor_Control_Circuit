"""MiniCPM-o 密集全双工三角对照上下文实验的校验与统计。

长流侧持续使用 32 位置/秒的密集全双工单元，并在预定时点插入自然决策探针。
两个复位侧只重放探针前最近若干个完整单元，其中一个与长流使用相同绝对位置，
另一个使用低位置。三路收到完全相同的近期音频与生成 token，可分别估计远端
历史内容效应和绝对位置效应。

本模块只做描述性工程诊断，不修改项目冻结的 E1/G2 判据。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DENSE_TRACE_SCHEMA_VERSION = 2

# 阈值同时要求相对早期基线出现实质恶化，避免把稳定存在的长短上下文差异误判为退化。
DENSE_DIAGNOSTIC_THRESHOLDS = {
    "hidden_cosine_max_bad": 0.95,
    "hidden_cosine_drop_min": 0.03,
    "logit_cosine_max_bad": 0.95,
    "logit_cosine_drop_min": 0.03,
    "js_divergence_min": 0.05,
    "js_divergence_increase_min": 0.03,
    "special_tv_min": 0.10,
    "special_tv_increase_min": 0.05,
    "first_token_agreement_max_bad": 0.80,
    "first_token_agreement_drop_min": 0.20,
    "sequence_similarity_max_bad": 0.60,
    "sequence_similarity_drop_min": 0.20,
    "hidden_norm_log2_deviation_min": 1.0,
    "hidden_norm_log2_deviation_increase_min": 0.585,
    "cross_run_cosine_delta_min": 0.20,
    "cross_run_cosine_delta_increase_min": 0.15,
    "common_direction_delta_min": 0.20,
    "common_direction_delta_increase_min": 0.15,
    "required_metric_families": 2,
    "required_consecutive_checkpoints": 2,
}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DenseContextRun:
    """一个完整的 MiniCPM-o 密集全双工运行。"""

    root: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.manifest["layers"])


def load_dense_context_run(root: Path, *, require_complete: bool = True) -> DenseContextRun:
    """读取并严格校验一个密集全双工运行目录。"""

    root = Path(root)
    manifest_path = root / "manifest.json"
    trace_path = root / "trace.npz"
    if not manifest_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"{root} 缺少 manifest.json 或 trace.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != DENSE_TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"{root}: schema_version={manifest.get('schema_version')}，"
            f"期望 {DENSE_TRACE_SCHEMA_VERSION}"
        )
    if manifest.get("protocol") != "minicpm_dense_context_v1":
        raise ValueError(f"{root}: protocol 必须为 minicpm_dense_context_v1")
    if require_complete and not bool(manifest.get("complete", False)):
        raise ValueError(f"{root}: 运行未完整结束")
    declared_sha = manifest.get("trace_sha256")
    if declared_sha and declared_sha != _sha256(trace_path):
        raise ValueError(f"{root}: trace.npz SHA-256 与 manifest 不一致")

    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {
        "target_seconds",
        "long_input_seconds",
        "long_positions",
        "control_positions",
        "position_control_positions",
        "position_control_absolute_positions",
        "position_control_shifts",
        "probe_indices",
        "long_hidden",
        "control_hidden",
        "position_control_hidden",
        "long_logits",
        "control_logits",
        "position_control_logits",
        "long_generated_ids",
        "control_generated_ids",
        "position_control_generated_ids",
        "long_generated_lengths",
        "control_generated_lengths",
        "position_control_generated_lengths",
        "long_cache_lengths",
        "control_cache_lengths",
        "position_control_cache_lengths",
        "long_all_finite",
        "control_all_finite",
        "position_control_all_finite",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{root}: trace 缺少字段 {missing}")

    target_seconds = arrays["target_seconds"]
    if target_seconds.ndim != 1 or len(target_seconds) == 0:
        raise ValueError(f"{root}: target_seconds 必须为非空一维数组")
    n_rows = len(target_seconds)
    one_dimensional = (
        "long_input_seconds",
        "long_positions",
        "control_positions",
        "position_control_positions",
        "position_control_absolute_positions",
        "position_control_shifts",
        "probe_indices",
        "long_generated_lengths",
        "control_generated_lengths",
        "position_control_generated_lengths",
        "long_cache_lengths",
        "control_cache_lengths",
        "position_control_cache_lengths",
    )
    for name in one_dimensional:
        if arrays[name].shape != (n_rows,):
            raise ValueError(f"{root}: {name} 必须为 [{n_rows}]")

    long_hidden = arrays["long_hidden"]
    control_hidden = arrays["control_hidden"]
    position_control_hidden = arrays["position_control_hidden"]
    if (
        long_hidden.ndim != 3
        or long_hidden.shape != control_hidden.shape
        or long_hidden.shape != position_control_hidden.shape
    ):
        raise ValueError(
            f"{root}: 三路 hidden 必须同为 [观测, 层, 隐藏维]"
        )
    if long_hidden.shape[0] != n_rows:
        raise ValueError(f"{root}: hidden 首维与 target_seconds 不一致")
    if long_hidden.shape[1] != len(manifest.get("layers", [])):
        raise ValueError(f"{root}: hidden 层数与 manifest.layers 不一致")

    long_logits = arrays["long_logits"]
    control_logits = arrays["control_logits"]
    position_control_logits = arrays["position_control_logits"]
    if (
        long_logits.ndim != 2
        or long_logits.shape != control_logits.shape
        or long_logits.shape != position_control_logits.shape
    ):
        raise ValueError(f"{root}: 三路 logits 必须同为 [观测, 词表]")
    if long_logits.shape[0] != n_rows:
        raise ValueError(f"{root}: logits 首维与 target_seconds 不一致")

    long_ids = arrays["long_generated_ids"]
    control_ids = arrays["control_generated_ids"]
    position_control_ids = arrays["position_control_generated_ids"]
    if (
        long_ids.ndim != 2
        or long_ids.shape != control_ids.shape
        or long_ids.shape != position_control_ids.shape
    ):
        raise ValueError(f"{root}: 三路 generated_ids 必须同为二维数组")
    if long_ids.shape[0] != n_rows:
        raise ValueError(f"{root}: generated_ids 首维与 target_seconds 不一致")
    max_tokens = long_ids.shape[1]
    if np.any(arrays["long_generated_lengths"] < 0) or np.any(
        arrays["long_generated_lengths"] > max_tokens
    ):
        raise ValueError(f"{root}: long_generated_lengths 越界")
    if np.any(arrays["control_generated_lengths"] < 0) or np.any(
        arrays["control_generated_lengths"] > max_tokens
    ):
        raise ValueError(f"{root}: control_generated_lengths 越界")
    if np.any(arrays["position_control_generated_lengths"] < 0) or np.any(
        arrays["position_control_generated_lengths"] > max_tokens
    ):
        raise ValueError(f"{root}: position_control_generated_lengths 越界")

    expected_finite_shape = long_hidden.shape[:2]
    if arrays["long_all_finite"].shape != expected_finite_shape:
        raise ValueError(f"{root}: long_all_finite 必须为 [观测, 层]")
    if arrays["control_all_finite"].shape != expected_finite_shape:
        raise ValueError(f"{root}: control_all_finite 必须为 [观测, 层]")
    if arrays["position_control_all_finite"].shape != expected_finite_shape:
        raise ValueError(
            f"{root}: position_control_all_finite 必须为 [观测, 层]"
        )
    if not np.isfinite(long_logits.astype(np.float32)).all():
        raise ValueError(f"{root}: long_logits 含非有限值")
    if not np.isfinite(control_logits.astype(np.float32)).all():
        raise ValueError(f"{root}: control_logits 含非有限值")
    if not np.isfinite(position_control_logits.astype(np.float32)).all():
        raise ValueError(f"{root}: position_control_logits 含非有限值")
    if not np.array_equal(
        arrays["position_control_absolute_positions"],
        arrays["long_positions"],
    ):
        raise ValueError(f"{root}: 绝对位置对照未与长流位置逐项对齐")
    if np.any(arrays["position_control_shifts"] < 0):
        raise ValueError(f"{root}: position_control_shifts 不能为负")
    if not np.array_equal(
        arrays["position_control_positions"]
        + arrays["position_control_shifts"],
        arrays["position_control_absolute_positions"],
    ):
        raise ValueError(f"{root}: 绝对位置、低位位置和平移量不自洽")
    return DenseContextRun(root=root, manifest=manifest, arrays=arrays)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(_normalize_rows(left) * _normalize_rows(right), axis=-1)


def _centered_logit_cosine(long_logits: np.ndarray, control_logits: np.ndarray) -> np.ndarray:
    left = long_logits.astype(np.float64)
    right = control_logits.astype(np.float64)
    left -= left.mean(axis=-1, keepdims=True)
    right -= right.mean(axis=-1, keepdims=True)
    return _row_cosine(left, right)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _js_divergence(long_logits: np.ndarray, control_logits: np.ndarray) -> np.ndarray:
    """逐行 Jensen-Shannon 散度，单位为 nat。"""

    log_left = _log_softmax(long_logits)
    log_right = _log_softmax(control_logits)
    left = np.exp(log_left)
    right = np.exp(log_right)
    mixture = 0.5 * (left + right)
    log_mixture = np.log(np.maximum(mixture, 1e-300))
    left_kl = np.sum(left * (log_left - log_mixture), axis=-1)
    right_kl = np.sum(right * (log_right - log_mixture), axis=-1)
    return 0.5 * (left_kl + right_kl)


def _selected_probabilities(logits: np.ndarray, token_ids: list[int]) -> np.ndarray:
    log_probs = _log_softmax(logits)
    return np.exp(log_probs[:, token_ids])


def _special_categorical_probabilities(
    logits: np.ndarray,
    token_ids: list[int],
) -> np.ndarray:
    """把四个关键 token 与其余词表合成完备的五分类分布。"""

    selected = _selected_probabilities(logits, token_ids)
    other = np.maximum(1.0 - selected.sum(axis=-1, keepdims=True), 0.0)
    probabilities = np.concatenate([selected, other], axis=-1)
    return probabilities / np.maximum(
        probabilities.sum(axis=-1, keepdims=True),
        1e-300,
    )


def _uncentered_first_direction_share(values: np.ndarray) -> float:
    """返回未中心化第一奇异方向解释的平方范数占比。"""

    if values.ndim != 2 or len(values) < 2:
        return float("nan")
    singular_values = np.linalg.svd(values.astype(np.float64), compute_uv=False)
    energy = np.square(singular_values)
    denominator = float(energy.sum())
    return float(energy[0] / denominator) if denominator > 0 else float("nan")


def _cross_run_pairwise_cosine(
    values: np.ndarray,
    probe_indices: np.ndarray,
    run_indices: np.ndarray,
) -> float:
    """只比较相同探针、不同独立运行的隐藏方向。"""

    normalized = _normalize_rows(values.astype(np.float64))
    pairwise: list[float] = []
    for probe_index in np.unique(probe_indices):
        selected = np.flatnonzero(probe_indices == probe_index)
        for left_offset, left_index in enumerate(selected):
            for right_index in selected[left_offset + 1 :]:
                if run_indices[left_index] == run_indices[right_index]:
                    continue
                pairwise.append(
                    float(np.dot(normalized[left_index], normalized[right_index]))
                )
    return float(np.median(pairwise)) if pairwise else float("nan")


def _sequence_similarity(
    long_ids: np.ndarray,
    control_ids: np.ndarray,
    long_length: int,
    control_length: int,
) -> float:
    """以归一化 Levenshtein 距离计算生成 token 序列相似度。"""

    left = long_ids[:long_length].tolist()
    right = control_ids[:control_length].tolist()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            substitution = previous[right_index - 1] + int(left_token != right_token)
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    distance = previous[-1]
    return float(1.0 - distance / max(len(left), len(right)))


def _run_observation_metrics(run: DenseContextRun) -> dict[str, np.ndarray]:
    arrays = run.arrays
    long_hidden = arrays["long_hidden"].astype(np.float64)
    low_hidden = arrays["control_hidden"].astype(np.float64)
    position_hidden = arrays["position_control_hidden"].astype(np.float64)
    long_logits = arrays["long_logits"]
    low_logits = arrays["control_logits"]
    position_logits = arrays["position_control_logits"]

    token_mapping = run.manifest["special_token_ids"]
    selected_ids = [
        int(token_mapping[name])
        for name in ("listen", "speak", "chunk_eos", "turn_eos")
    ]
    long_special = _special_categorical_probabilities(long_logits, selected_ids)
    low_special = _special_categorical_probabilities(low_logits, selected_ids)
    position_special = _special_categorical_probabilities(
        position_logits,
        selected_ids,
    )
    special_tv = 0.5 * np.abs(position_special - low_special).sum(axis=-1)
    listen_probability_delta = np.abs(
        position_special[:, 0] - low_special[:, 0]
    )
    history_special_tv = 0.5 * np.abs(
        long_special - position_special
    ).sum(axis=-1)

    long_lengths = arrays["long_generated_lengths"].astype(np.int64)
    low_lengths = arrays["control_generated_lengths"].astype(np.int64)
    position_lengths = arrays[
        "position_control_generated_lengths"
    ].astype(np.int64)
    sequence_similarity = np.asarray(
        [
            _sequence_similarity(
                arrays["position_control_generated_ids"][index],
                arrays["control_generated_ids"][index],
                int(position_lengths[index]),
                int(low_lengths[index]),
            )
            for index in range(len(low_lengths))
        ],
        dtype=np.float64,
    )
    history_sequence_similarity = np.asarray(
        [
            _sequence_similarity(
                arrays["long_generated_ids"][index],
                arrays["position_control_generated_ids"][index],
                int(long_lengths[index]),
                int(position_lengths[index]),
            )
            for index in range(len(long_lengths))
        ],
        dtype=np.float64,
    )
    long_first = np.argmax(long_logits.astype(np.float32), axis=-1)
    low_first = np.argmax(low_logits.astype(np.float32), axis=-1)
    position_first = np.argmax(position_logits.astype(np.float32), axis=-1)
    all_finite = np.logical_and.reduce(
        (
            arrays["long_all_finite"].all(axis=1),
            arrays["control_all_finite"].all(axis=1),
            arrays["position_control_all_finite"].all(axis=1),
        )
    )
    return {
        # 主判据比较“高绝对位置近期后缀”与“低绝对位置近期后缀”。
        "hidden_cosine": _row_cosine(position_hidden, low_hidden),
        "logit_cosine": _centered_logit_cosine(position_logits, low_logits),
        "js_divergence": _js_divergence(position_logits, low_logits),
        "special_tv": special_tv,
        "listen_probability_delta": listen_probability_delta,
        "first_token_equal": position_first == low_first,
        "sequence_similarity": sequence_similarity,
        # 历史效应比较完整长流与绝对位置已匹配的近期后缀。
        "history_hidden_cosine": _row_cosine(long_hidden, position_hidden),
        "history_logit_cosine": _centered_logit_cosine(
            long_logits,
            position_logits,
        ),
        "history_js_divergence": _js_divergence(long_logits, position_logits),
        "history_special_tv": history_special_tv,
        "history_first_token_equal": long_first == position_first,
        "history_sequence_similarity": history_sequence_similarity,
        "long_special_probs": long_special,
        "position_special_probs": position_special,
        "control_special_probs": low_special,
        "hidden_norm_ratio": np.linalg.norm(long_hidden, axis=-1)
        / np.maximum(np.linalg.norm(position_hidden, axis=-1), 1e-12),
        "all_finite": all_finite,
    }


def _validate_run_group(runs: list[DenseContextRun]) -> None:
    if not runs:
        raise ValueError("至少需要一个运行目录")
    first = runs[0]
    first_design = first.manifest.get("design")
    first_context = first.manifest.get("context_spec")
    first_special = first.manifest.get("special_token_ids")
    for run in runs[1:]:
        if run.layers != first.layers:
            raise ValueError("所有运行必须使用相同层集合")
        if run.manifest.get("design") != first_design:
            raise ValueError("所有运行的 design 必须逐字段一致")
        if run.manifest.get("context_spec") != first_context:
            raise ValueError("所有运行的 context_spec 必须逐字段一致")
        if run.manifest.get("special_token_ids") != first_special:
            raise ValueError("所有运行的 special_token_ids 必须一致")
        if run.arrays["long_hidden"].shape[1:] != first.arrays["long_hidden"].shape[1:]:
            raise ValueError("所有运行的隐藏层与维度必须一致")
        if run.arrays["long_logits"].shape[1] != first.arrays["long_logits"].shape[1]:
            raise ValueError("所有运行的词表维度必须一致")


def _zero_shift_validation(runs: list[DenseContextRun]) -> dict[str, Any]:
    """核对位置平移为零时，两条复位路径逐数组完全一致。"""

    hidden_max_abs: list[float] = []
    logits_max_abs: list[float] = []
    generated_equal: list[bool] = []
    for run in runs:
        mask = run.arrays["position_control_shifts"] == 0
        if not np.any(mask):
            continue
        hidden_max_abs.append(
            float(
                np.max(
                    np.abs(
                        run.arrays["position_control_hidden"][mask].astype(
                            np.float32
                        )
                        - run.arrays["control_hidden"][mask].astype(np.float32)
                    )
                )
            )
        )
        logits_max_abs.append(
            float(
                np.max(
                    np.abs(
                        run.arrays["position_control_logits"][mask].astype(
                            np.float32
                        )
                        - run.arrays["control_logits"][mask].astype(np.float32)
                    )
                )
            )
        )
        selected = np.flatnonzero(mask)
        for index in selected:
            position_length = int(
                run.arrays["position_control_generated_lengths"][index]
            )
            control_length = int(
                run.arrays["control_generated_lengths"][index]
            )
            generated_equal.append(
                position_length == control_length
                and np.array_equal(
                    run.arrays["position_control_generated_ids"][
                        index,
                        :position_length,
                    ],
                    run.arrays["control_generated_ids"][
                        index,
                        :control_length,
                    ],
                )
            )
    count = len(generated_equal)
    return {
        "n_zero_shift_pairs": count,
        "hidden_max_abs": max(hidden_max_abs, default=float("nan")),
        "logits_max_abs": max(logits_max_abs, default=float("nan")),
        "generated_ids_all_equal": bool(count and all(generated_equal)),
        "exact_parity": bool(
            count
            and max(hidden_max_abs, default=float("inf")) == 0.0
            and max(logits_max_abs, default=float("inf")) == 0.0
            and all(generated_equal)
        ),
    }


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if len(finite) else float("nan")


def _checkpoint_rows(
    runs: list[DenseContextRun],
    observation_metrics: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    distinct_audio_hashes = {
        str(run.manifest.get("source_audio", {}).get("sha256", ""))
        for run in runs
        if run.manifest.get("source_audio", {}).get("sha256")
    }
    cross_input_group_available = len(distinct_audio_hashes) >= 3
    targets = sorted(
        {
            int(value)
            for run in runs
            for value in run.arrays["target_seconds"].tolist()
        }
    )
    rows: list[dict[str, Any]] = []
    layers = runs[0].layers
    for target in targets:
        hidden_parts = []
        logit_parts = []
        js_parts = []
        tv_parts = []
        listen_delta_parts = []
        first_parts = []
        sequence_parts = []
        history_hidden_parts = []
        history_logit_parts = []
        history_js_parts = []
        history_tv_parts = []
        history_first_parts = []
        history_sequence_parts = []
        norm_ratio_parts = []
        finite_parts = []
        raw_long_hidden_parts = []
        raw_control_hidden_parts = []
        probe_index_parts = []
        run_index_parts = []
        long_positions = []
        long_seconds = []
        control_positions = []
        position_shifts = []
        for run, metrics in zip(runs, observation_metrics, strict=True):
            mask = run.arrays["target_seconds"] == target
            if not np.any(mask):
                continue
            hidden_parts.append(metrics["hidden_cosine"][mask])
            logit_parts.append(metrics["logit_cosine"][mask])
            js_parts.append(metrics["js_divergence"][mask])
            tv_parts.append(metrics["special_tv"][mask])
            listen_delta_parts.append(metrics["listen_probability_delta"][mask])
            first_parts.append(metrics["first_token_equal"][mask])
            sequence_parts.append(metrics["sequence_similarity"][mask])
            history_hidden_parts.append(metrics["history_hidden_cosine"][mask])
            history_logit_parts.append(metrics["history_logit_cosine"][mask])
            history_js_parts.append(metrics["history_js_divergence"][mask])
            history_tv_parts.append(metrics["history_special_tv"][mask])
            history_first_parts.append(metrics["history_first_token_equal"][mask])
            history_sequence_parts.append(
                metrics["history_sequence_similarity"][mask]
            )
            norm_ratio_parts.append(metrics["hidden_norm_ratio"][mask])
            finite_parts.append(metrics["all_finite"][mask])
            raw_long_hidden_parts.append(
                run.arrays["long_hidden"][mask].astype(np.float64)
            )
            raw_control_hidden_parts.append(
                run.arrays["position_control_hidden"][mask].astype(np.float64)
            )
            probe_index_parts.append(run.arrays["probe_indices"][mask])
            run_index_parts.append(
                np.full(int(np.sum(mask)), len(run_index_parts), dtype=np.int16)
            )
            long_positions.append(run.arrays["long_positions"][mask])
            long_seconds.append(run.arrays["long_input_seconds"][mask])
            control_positions.append(run.arrays["control_positions"][mask])
            position_shifts.append(run.arrays["position_control_shifts"][mask])
        hidden = np.concatenate(hidden_parts, axis=0)
        logit = np.concatenate(logit_parts)
        js = np.concatenate(js_parts)
        tv = np.concatenate(tv_parts)
        listen_delta = np.concatenate(listen_delta_parts)
        first = np.concatenate(first_parts)
        sequence = np.concatenate(sequence_parts)
        history_hidden = np.concatenate(history_hidden_parts, axis=0)
        history_logit = np.concatenate(history_logit_parts)
        history_js = np.concatenate(history_js_parts)
        history_tv = np.concatenate(history_tv_parts)
        history_first = np.concatenate(history_first_parts)
        history_sequence = np.concatenate(history_sequence_parts)
        norm_ratio = np.concatenate(norm_ratio_parts, axis=0)
        finite = np.concatenate(finite_parts)
        raw_long_hidden = np.concatenate(raw_long_hidden_parts, axis=0)
        raw_control_hidden = np.concatenate(raw_control_hidden_parts, axis=0)
        probe_indices = np.concatenate(probe_index_parts)
        run_indices = np.concatenate(run_index_parts)
        cross_input_available = (
            len(np.unique(run_indices)) >= 3 and cross_input_group_available
        )
        positions = np.concatenate(long_positions)
        seconds = np.concatenate(long_seconds)
        controls = np.concatenate(control_positions)
        shifts = np.concatenate(position_shifts)
        layer_rows = []
        for layer_index, layer in enumerate(layers):
            layer_values = hidden[:, layer_index]
            history_layer_values = history_hidden[:, layer_index]
            layer_norm_ratio = norm_ratio[:, layer_index]
            long_cross_run = (
                _cross_run_pairwise_cosine(
                    raw_long_hidden[:, layer_index, :],
                    probe_indices,
                    run_indices,
                )
                if cross_input_available
                else float("nan")
            )
            control_cross_run = (
                _cross_run_pairwise_cosine(
                    raw_control_hidden[:, layer_index, :],
                    probe_indices,
                    run_indices,
                )
                if cross_input_available
                else float("nan")
            )
            long_common_share = (
                _uncentered_first_direction_share(
                    raw_long_hidden[:, layer_index, :]
                )
                if cross_input_available
                else float("nan")
            )
            control_common_share = (
                _uncentered_first_direction_share(
                    raw_control_hidden[:, layer_index, :]
                )
                if cross_input_available
                else float("nan")
            )
            layer_rows.append(
                {
                    "layer": int(layer),
                    "hidden_cosine_median": _median(layer_values),
                    "hidden_cosine_p10": _quantile(layer_values, 0.10),
                    "hidden_cosine_min": float(np.min(layer_values)),
                    "history_hidden_cosine_median": _median(
                        history_layer_values
                    ),
                    "hidden_norm_ratio_median": _median(layer_norm_ratio),
                    "hidden_norm_log2_deviation": float(
                        abs(np.log2(max(_median(layer_norm_ratio), 1e-12)))
                    ),
                    "long_cross_run_cosine": long_cross_run,
                    "control_cross_run_cosine": control_cross_run,
                    "cross_run_cosine_delta": float(
                        long_cross_run - control_cross_run
                    ),
                    "long_common_direction_share": long_common_share,
                    "control_common_direction_share": control_common_share,
                    "common_direction_share_delta": float(
                        long_common_share - control_common_share
                    ),
                }
            )
        worst_norm_layer = max(
            layer_rows,
            key=lambda item: item["hidden_norm_log2_deviation"],
        )
        rows.append(
            {
                "target_seconds": target,
                "n_pairs": len(logit),
                "long_input_seconds_median": _median(seconds.astype(np.float64)),
                "long_position_median": _median(positions.astype(np.float64)),
                "long_position_min": int(np.min(positions)),
                "long_position_max": int(np.max(positions)),
                "control_position_median": _median(controls.astype(np.float64)),
                "position_shift_median": _median(shifts.astype(np.float64)),
                "observed_positions_per_second": _median(
                    (positions - int(runs[0].manifest["context_spec"]["start_position"]))
                    / np.maximum(seconds, 1)
                ),
                "layers": layer_rows,
                "worst_layer_hidden_cosine_median": min(
                    item["hidden_cosine_median"] for item in layer_rows
                ),
                "history_worst_layer_hidden_cosine_median": min(
                    item["history_hidden_cosine_median"]
                    for item in layer_rows
                ),
                "worst_hidden_norm_ratio_median": worst_norm_layer[
                    "hidden_norm_ratio_median"
                ],
                "max_hidden_norm_log2_deviation": worst_norm_layer[
                    "hidden_norm_log2_deviation"
                ],
                "max_cross_run_cosine_delta": max(
                    item["cross_run_cosine_delta"] for item in layer_rows
                ),
                "max_common_direction_share_delta": max(
                    item["common_direction_share_delta"] for item in layer_rows
                ),
                "logit_cosine_median": _median(logit),
                "logit_cosine_p10": _quantile(logit, 0.10),
                "js_divergence_median": _median(js),
                "js_divergence_p90": _quantile(js, 0.90),
                "special_tv_median": _median(tv),
                "special_tv_p90": _quantile(tv, 0.90),
                "listen_probability_delta_median": _median(listen_delta),
                "listen_probability_delta_p90": _quantile(listen_delta, 0.90),
                "first_token_agreement": float(np.mean(first)),
                "sequence_similarity_median": _median(sequence),
                "sequence_similarity_p10": _quantile(sequence, 0.10),
                "history_logit_cosine_median": _median(history_logit),
                "history_js_divergence_median": _median(history_js),
                "history_special_tv_median": _median(history_tv),
                "history_first_token_agreement": float(
                    np.mean(history_first)
                ),
                "history_sequence_similarity_median": _median(
                    history_sequence
                ),
                "finite_fraction": float(np.mean(finite)),
            }
        )
    return rows


def _add_degradation_flags(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    baseline = rows[0]
    thresholds = DENSE_DIAGNOSTIC_THRESHOLDS
    for row in rows:
        hidden_drop = (
            baseline["worst_layer_hidden_cosine_median"]
            - row["worst_layer_hidden_cosine_median"]
        )
        logit_drop = baseline["logit_cosine_median"] - row["logit_cosine_median"]
        js_increase = row["js_divergence_median"] - baseline["js_divergence_median"]
        tv_increase = row["special_tv_median"] - baseline["special_tv_median"]
        first_drop = baseline["first_token_agreement"] - row["first_token_agreement"]
        sequence_drop = (
            baseline["sequence_similarity_median"]
            - row["sequence_similarity_median"]
        )
        norm_deviation_increase = (
            row["max_hidden_norm_log2_deviation"]
            - baseline["max_hidden_norm_log2_deviation"]
        )
        cross_run_increase = (
            row["max_cross_run_cosine_delta"]
            - baseline["max_cross_run_cosine_delta"]
        )
        common_direction_increase = (
            row["max_common_direction_share_delta"]
            - baseline["max_common_direction_share_delta"]
        )
        representation = bool(
            (
                row["worst_layer_hidden_cosine_median"]
                <= thresholds["hidden_cosine_max_bad"]
                and hidden_drop >= thresholds["hidden_cosine_drop_min"]
            )
            or (
                row["logit_cosine_median"] <= thresholds["logit_cosine_max_bad"]
                and logit_drop >= thresholds["logit_cosine_drop_min"]
            )
        )
        distribution = bool(
            row["js_divergence_median"] >= thresholds["js_divergence_min"]
            and js_increase >= thresholds["js_divergence_increase_min"]
            and row["special_tv_median"] >= thresholds["special_tv_min"]
            and tv_increase >= thresholds["special_tv_increase_min"]
        )
        behavior = bool(
            (
                row["first_token_agreement"]
                <= thresholds["first_token_agreement_max_bad"]
                and first_drop >= thresholds["first_token_agreement_drop_min"]
            )
            or (
                row["sequence_similarity_median"]
                <= thresholds["sequence_similarity_max_bad"]
                and sequence_drop >= thresholds["sequence_similarity_drop_min"]
            )
        )
        structural = bool(
            (
                row["max_hidden_norm_log2_deviation"]
                >= thresholds["hidden_norm_log2_deviation_min"]
                and norm_deviation_increase
                >= thresholds["hidden_norm_log2_deviation_increase_min"]
            )
            or (
                np.isfinite(row["max_cross_run_cosine_delta"])
                and row["max_cross_run_cosine_delta"]
                >= thresholds["cross_run_cosine_delta_min"]
                and cross_run_increase
                >= thresholds["cross_run_cosine_delta_increase_min"]
            )
            or (
                np.isfinite(row["max_common_direction_share_delta"])
                and row["max_common_direction_share_delta"]
                >= thresholds["common_direction_delta_min"]
                and common_direction_increase
                >= thresholds["common_direction_delta_increase_min"]
            )
        )
        numeric = bool(row["finite_fraction"] < 1.0)
        family_count = sum((representation, distribution, behavior))
        row["change_from_early_baseline"] = {
            "hidden_cosine_drop": float(hidden_drop),
            "logit_cosine_drop": float(logit_drop),
            "js_divergence_increase": float(js_increase),
            "special_tv_increase": float(tv_increase),
            "first_token_agreement_drop": float(first_drop),
            "sequence_similarity_drop": float(sequence_drop),
            "hidden_norm_log2_deviation_increase": float(
                norm_deviation_increase
            ),
            "cross_run_cosine_delta_increase": float(cross_run_increase),
            "common_direction_share_delta_increase": float(
                common_direction_increase
            ),
        }
        row["flags"] = {
            "representation_degradation": representation,
            "distribution_degradation": distribution,
            "behavior_degradation": behavior,
            "structural_degradation": structural,
            "nonfinite": numeric,
        }
        row["metric_family_count"] = family_count
        row["candidate_degradation"] = bool(
            numeric or family_count >= thresholds["required_metric_families"]
        )

        history_hidden_drop = (
            baseline["history_worst_layer_hidden_cosine_median"]
            - row["history_worst_layer_hidden_cosine_median"]
        )
        history_logit_drop = (
            baseline["history_logit_cosine_median"]
            - row["history_logit_cosine_median"]
        )
        history_js_increase = (
            row["history_js_divergence_median"]
            - baseline["history_js_divergence_median"]
        )
        history_tv_increase = (
            row["history_special_tv_median"]
            - baseline["history_special_tv_median"]
        )
        history_first_drop = (
            baseline["history_first_token_agreement"]
            - row["history_first_token_agreement"]
        )
        history_sequence_drop = (
            baseline["history_sequence_similarity_median"]
            - row["history_sequence_similarity_median"]
        )
        history_representation = bool(
            (
                row["history_worst_layer_hidden_cosine_median"]
                <= thresholds["hidden_cosine_max_bad"]
                and history_hidden_drop >= thresholds["hidden_cosine_drop_min"]
            )
            or (
                row["history_logit_cosine_median"]
                <= thresholds["logit_cosine_max_bad"]
                and history_logit_drop >= thresholds["logit_cosine_drop_min"]
            )
        )
        history_distribution = bool(
            row["history_js_divergence_median"]
            >= thresholds["js_divergence_min"]
            and history_js_increase
            >= thresholds["js_divergence_increase_min"]
            and row["history_special_tv_median"]
            >= thresholds["special_tv_min"]
            and history_tv_increase
            >= thresholds["special_tv_increase_min"]
        )
        history_behavior = bool(
            (
                row["history_first_token_agreement"]
                <= thresholds["first_token_agreement_max_bad"]
                and history_first_drop
                >= thresholds["first_token_agreement_drop_min"]
            )
            or (
                row["history_sequence_similarity_median"]
                <= thresholds["sequence_similarity_max_bad"]
                and history_sequence_drop
                >= thresholds["sequence_similarity_drop_min"]
            )
        )
        history_family_count = sum(
            (
                history_representation,
                history_distribution,
                history_behavior,
                structural,
            )
        )
        row["history_change_from_early_baseline"] = {
            "hidden_cosine_drop": float(history_hidden_drop),
            "logit_cosine_drop": float(history_logit_drop),
            "js_divergence_increase": float(history_js_increase),
            "special_tv_increase": float(history_tv_increase),
            "first_token_agreement_drop": float(history_first_drop),
            "sequence_similarity_drop": float(history_sequence_drop),
        }
        row["history_flags"] = {
            "representation_divergence": history_representation,
            "distribution_divergence": history_distribution,
            "behavior_divergence": history_behavior,
            "structural_divergence": structural,
            "nonfinite": numeric,
        }
        row["history_metric_family_count"] = history_family_count
        row["candidate_history_divergence"] = bool(
            numeric
            or history_family_count
            >= thresholds["required_metric_families"]
        )

    required = int(thresholds["required_consecutive_checkpoints"])
    if required != 2:
        raise ValueError("当前持续性标记实现要求连续检查点数为 2")
    for index, row in enumerate(rows):
        previous_position_candidate = bool(
            index > 0 and rows[index - 1]["candidate_degradation"]
        )
        next_position_candidate = bool(
            index + 1 < len(rows)
            and rows[index + 1]["candidate_degradation"]
        )
        persistent = bool(
            row["candidate_degradation"]
            and (previous_position_candidate or next_position_candidate)
        )
        row["confirmed_persistent_degradation"] = bool(
            row["flags"]["nonfinite"] or persistent
        )
        previous_history_candidate = bool(
            index > 0 and rows[index - 1]["candidate_history_divergence"]
        )
        next_history_candidate = bool(
            index + 1 < len(rows)
            and rows[index + 1]["candidate_history_divergence"]
        )
        history_persistent = bool(
            row["candidate_history_divergence"]
            and (previous_history_candidate or next_history_candidate)
        )
        row["confirmed_persistent_history_divergence"] = bool(
            row["history_flags"]["nonfinite"] or history_persistent
        )


def _derive_recommendation(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    n_runs: int,
    n_distinct_audio_inputs: int,
) -> dict[str, Any]:
    context_spec = manifest["context_spec"]
    official = int(context_spec["official_max_positions"])
    start = int(context_spec["start_position"])
    dense_rate = float(context_spec["dense_positions_per_second"])
    confirmed = [
        row for row in rows if row["confirmed_persistent_degradation"]
    ]
    first_confirmed = confirmed[0] if confirmed else None
    if first_confirmed is None:
        empirical_safe = rows[-1]
        status = "实测范围内未确认持续的绝对位置多指标退化"
    else:
        first_index = rows.index(first_confirmed)
        empirical_safe = rows[max(first_index - 1, 0)]
        status = "观察到持续的绝对位置多指标退化，安全下界取前一检查点"

    history_confirmed = [
        row
        for row in rows
        if row["confirmed_persistent_history_divergence"]
    ]
    first_history_confirmed = (
        history_confirmed[0] if history_confirmed else None
    )
    if first_history_confirmed is None:
        history_clean = rows[-1]
        history_status = "实测范围内未确认持续的密集历史行为分叉"
    else:
        history_index = rows.index(first_history_confirmed)
        history_clean = rows[max(history_index - 1, 0)]
        history_status = "观察到持续的密集历史行为分叉"

    probe_rates = np.asarray(
        [row["observed_positions_per_second"] for row in rows],
        dtype=np.float64,
    )
    observed_probe_rate = _median(probe_rates)
    formal_dense_seconds = float(context_spec["formal_dense_seconds"])
    dense_budget_2000 = start + 2000 * dense_rate
    rows_through_2000 = [
        row for row in rows if row["target_seconds"] <= 2000
    ]
    covered_2000 = bool(
        any(
            row["target_seconds"] >= 2000
            and row["long_position_min"] >= dense_budget_2000
            for row in rows
        )
    )
    clean_2000 = bool(
        covered_2000
        and all(
            not row["confirmed_persistent_degradation"]
            for row in rows_through_2000
        )
    )
    nearest_2000 = (
        min(rows, key=lambda row: abs(row["target_seconds"] - 2000))
        if rows
        else None
    )
    return {
        "status": status,
        "n_independent_runs": n_runs,
        "n_distinct_audio_inputs": n_distinct_audio_inputs,
        "cross_input_evidence_available": n_distinct_audio_inputs >= 3,
        "official_max_positions": official,
        "dense_positions_per_second": dense_rate,
        "observed_probe_positions_per_input_second": observed_probe_rate,
        "formal_dense_context_seconds": formal_dense_seconds,
        "formal_dense_complete_one_second_units": int(
            (official - start) // dense_rate
        ),
        "first_confirmed_degradation_seconds": (
            first_confirmed["target_seconds"] if first_confirmed else None
        ),
        "first_confirmed_degradation_position": (
            first_confirmed["long_position_median"] if first_confirmed else None
        ),
        "empirical_clean_lower_bound_seconds": empirical_safe["target_seconds"],
        "empirical_clean_lower_bound_position": empirical_safe["long_position_median"],
        "history_status": history_status,
        "first_confirmed_history_divergence_seconds": (
            first_history_confirmed["target_seconds"]
            if first_history_confirmed
            else None
        ),
        "first_confirmed_history_divergence_position": (
            first_history_confirmed["long_position_median"]
            if first_history_confirmed
            else None
        ),
        "history_clean_lower_bound_seconds": history_clean[
            "target_seconds"
        ],
        "history_clean_lower_bound_position": history_clean[
            "long_position_median"
        ],
        "seconds_2000_covered": covered_2000,
        "seconds_2000_clean": clean_2000,
        "nearest_2000_checkpoint": nearest_2000,
        "formal_note": (
            "官方规格仍由 max_position_embeddings 决定；超过官方位置后的干净结果"
            "只构成密集全双工外推证据。"
        ),
    }


def analyze_dense_context_runs(runs: list[DenseContextRun]) -> dict[str, Any]:
    """联合分析多个独立密集全双工运行。"""

    _validate_run_group(runs)
    distinct_audio_hashes = {
        str(run.manifest.get("source_audio", {}).get("sha256", ""))
        for run in runs
        if run.manifest.get("source_audio", {}).get("sha256")
    }
    observation_metrics = [_run_observation_metrics(run) for run in runs]
    rows = _checkpoint_rows(runs, observation_metrics)
    _add_degradation_flags(rows)
    recommendation = _derive_recommendation(
        rows,
        manifest=runs[0].manifest,
        n_runs=len(runs),
        n_distinct_audio_inputs=len(distinct_audio_hashes),
    )
    return {
        "schema_version": 2,
        "analysis": "minicpm_dense_context_v2",
        "model": "minicpm_o_4_5",
        "run_ids": [run.run_id for run in runs],
        "n_runs": len(runs),
        "layers": list(runs[0].layers),
        "diagnostic_thresholds": DENSE_DIAGNOSTIC_THRESHOLDS,
        "context_spec": runs[0].manifest["context_spec"],
        "design": runs[0].manifest["design"],
        "zero_shift_validation": _zero_shift_validation(runs),
        "checkpoints": rows,
        "recommendation": recommendation,
        "limitations": [
            (
                "三角对照把绝对位置效应与远端历史内容效应分开；生成内容仍来自有限"
                "的真实音频与模型 token 库，不能覆盖所有现实对话分布。"
            ),
            (
                "探针比较原生决策分布、隐藏状态和贪心生成序列；正式 floor 标签 AUC"
                "仍需在 E1 冻结标签协议中复算。"
            ),
            (
                "高位近期后缀检验局部 RoPE 位置外推，不能单独证明模型仍能从远端"
                "前缀准确检索语义；完整长流保留远端 KV，但其内容效应可能是合法的。"
            ),
            (
                "超过官方位置上限的成功运行不能自动扩张模型正式规格。"
            ),
            (
                "跨输入公共吸引态指标至少需要三条独立运行；单条冒烟中的对应字段"
                "仅作缺失值处理。"
            ),
        ],
    }


def render_dense_context_markdown(report: dict[str, Any]) -> str:
    """渲染密集全双工上下文分析摘要。"""

    recommendation = report["recommendation"]
    zero_shift = report["zero_shift_validation"]
    zero_shift_status = (
        "精确一致"
        if zero_shift["exact_parity"]
        else ("未通过" if zero_shift["n_zero_shift_pairs"] else "未覆盖")
    )
    lines = [
        "# MiniCPM-o 密集全双工上下文测量",
        "",
        f"- 独立运行：{report['n_runs']}",
        f"- 不同音频输入：{recommendation['n_distinct_audio_inputs']}",
        f"- 零平移对拍：{zero_shift['n_zero_shift_pairs']} 组，"
        f"{zero_shift_status}",
        f"- 结论：{recommendation['status']}",
        f"- 密集历史：{recommendation['history_status']}",
        f"- 严格密集位置速率：{recommendation['dense_positions_per_second']:.3f} 位置/秒",
        "- 自然探针即时平均速率："
        f"{recommendation['observed_probe_positions_per_input_second']:.3f} 位置/输入秒",
        f"- 官方规格折算：{recommendation['formal_dense_context_seconds']:.2f} 秒",
        "- 官方规格内完整密集单元："
        f"{recommendation['formal_dense_complete_one_second_units']} 个一秒单元",
        "- 绝对位置实证干净下界："
        f"{recommendation['empirical_clean_lower_bound_seconds']} 秒，"
        f"位置 {recommendation['empirical_clean_lower_bound_position']:.0f}",
        "- 历史分叉前下界："
        f"{recommendation['history_clean_lower_bound_seconds']} 秒，"
        f"位置 {recommendation['history_clean_lower_bound_position']:.0f}",
        f"- 2000 秒：覆盖={'是' if recommendation['seconds_2000_covered'] else '否'}，"
        "未确认绝对位置退化="
        f"{'是' if recommendation['seconds_2000_clean'] else '否'}",
        "",
        "## 检查点",
        "",
        "| 目标秒 | 中位位置 | 位置层余弦 | 位置 logits | 位置 JS "
        "| 位置首词一致率 | 历史 logits | 历史范数比 | 跨输入余弦增量 "
        "| 位置序列相似度 "
        "| 位置退化 | 历史分叉 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for row in report["checkpoints"]:
        cross_run_delta = row["max_cross_run_cosine_delta"]
        cross_run_text = (
            f"{cross_run_delta:.4f}" if np.isfinite(cross_run_delta) else "—"
        )
        lines.append(
            f"| {row['target_seconds']} "
            f"| {row['long_position_median']:.0f} "
            f"| {row['worst_layer_hidden_cosine_median']:.4f} "
            f"| {row['logit_cosine_median']:.4f} "
            f"| {row['js_divergence_median']:.4f} "
            f"| {row['first_token_agreement']:.3f} "
            f"| {row['history_logit_cosine_median']:.4f} "
            f"| {row['worst_hidden_norm_ratio_median']:.4f} "
            f"| {cross_run_text} "
            f"| {row['sequence_similarity_median']:.3f} "
            f"| {'是' if row['confirmed_persistent_degradation'] else '否'} "
            f"| {'是' if row['confirmed_persistent_history_divergence'] else '否'} |"
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
