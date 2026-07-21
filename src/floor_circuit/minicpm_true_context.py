"""MiniCPM-o 因果双工与远端记忆上下文实验的校验、计分和联合分析。

本模块服务于探索性协议，不修改项目冻结的 E1/G2 判据。实验把三类失效拆开：

1. 近期内容放到高绝对位置后的听说决策失效；
2. 完整长历史相对同位置近期历史的听说决策失效；
3. 高位置近期口令仍可回答时，远端口令已经无法检索。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRUE_CONTEXT_SCHEMA_VERSION = 1
TRUE_CONTEXT_PROTOCOL = "minicpm_true_context_v1"

PROBE_KIND_FLOOR = 0
PROBE_KIND_MEMORY = 1

TRUE_CONTEXT_THRESHOLDS = {
    "floor_oracle_min_balanced_accuracy": 0.80,
    "floor_failure_max_balanced_accuracy": 0.65,
    "floor_gap_min": 0.15,
    "memory_oracle_min_accuracy": 0.80,
    "memory_failure_max_accuracy": 0.65,
    "memory_gap_min": 0.20,
    "negative_memory_max_accuracy": 0.20,
    "required_consecutive_checkpoints": 2,
    "minimum_independent_runs": 3,
    "minimum_distinct_audio_inputs": 3,
    "floor_minimum_per_class_per_checkpoint": 6,
    "memory_minimum_per_checkpoint": 6,
}

_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(text: str) -> tuple[str, ...]:
    """把生成文本规范化为可审计的词元序列。

    数字统一写成英文单词，使 ``amber 7`` 与 ``amber seven`` 可以匹配。
    中文字符逐字保留，便于后续扩展中文口令。
    """

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = re.sub(
        r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"(?<!\d)([0-9])(?!\d)",
        lambda match: f" {_NUMBER_WORDS[match.group(1)]} ",
        normalized,
    )
    return tuple(re.findall(r"[a-z]+|[\u3400-\u9fff]+", normalized))


def contains_answer(text: str, aliases: Iterable[str]) -> bool:
    """判断任一答案别名是否作为连续词元片段出现在生成文本中。"""

    output = normalize_answer(text)
    for alias in aliases:
        expected = normalize_answer(alias)
        if not expected or len(expected) > len(output):
            continue
        width = len(expected)
        if any(output[index : index + width] == expected for index in range(len(output) - width + 1)):
            return True
    return False


def answer_token_f1(text: str, aliases: Iterable[str]) -> float:
    """返回生成文本相对最佳答案别名的词元集合 F1。"""

    output = normalize_answer(text)
    if not output:
        return 0.0
    best = 0.0
    output_counts: dict[str, int] = {}
    for token in output:
        output_counts[token] = output_counts.get(token, 0) + 1
    for alias in aliases:
        expected = normalize_answer(alias)
        if not expected:
            continue
        expected_counts: dict[str, int] = {}
        for token in expected:
            expected_counts[token] = expected_counts.get(token, 0) + 1
        overlap = sum(
            min(count, output_counts.get(token, 0))
            for token, count in expected_counts.items()
        )
        if overlap == 0:
            continue
        precision = overlap / len(output)
        recall = overlap / len(expected)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return float(best)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。"""

    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return float(center - margin), float(center + margin)


@dataclass(frozen=True)
class TrueContextRun:
    """一个完整的 MiniCPM-o 真实上下文运行。"""

    root: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])


def _require_shape(
    root: Path,
    arrays: dict[str, np.ndarray],
    names: Iterable[str],
    expected: tuple[int, ...],
) -> None:
    for name in names:
        if arrays[name].shape != expected:
            raise ValueError(f"{root}: {name} 形状为 {arrays[name].shape}，期望 {expected}")


def load_true_context_run(root: Path, *, require_complete: bool = True) -> TrueContextRun:
    """读取并严格校验一个因果双工上下文运行。"""

    root = Path(root)
    manifest_path = root / "manifest.json"
    trace_path = root / "trace.npz"
    if not manifest_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"{root} 缺少 manifest.json 或 trace.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != TRUE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(f"{root}: schema_version 不匹配")
    if manifest.get("protocol") != TRUE_CONTEXT_PROTOCOL:
        raise ValueError(f"{root}: protocol 必须为 {TRUE_CONTEXT_PROTOCOL}")
    if require_complete and not bool(manifest.get("complete", False)):
        raise ValueError(f"{root}: 运行未完整结束")
    declared_sha = manifest.get("trace_sha256")
    if declared_sha and sha256_file(trace_path) != declared_sha:
        raise ValueError(f"{root}: trace.npz SHA-256 与 manifest 不一致")

    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    observation_required = {
        "target_seconds",
        "history_input_seconds",
        "lane_indices",
        "probe_kinds",
        "probe_ids",
        "expected_is_listen",
        "long_positions",
        "low_positions",
        "high_positions",
        "high_absolute_positions",
        "high_position_shifts",
        "long_hidden",
        "low_hidden",
        "high_hidden",
        "long_logits",
        "low_logits",
        "high_logits",
        "long_generated_ids",
        "low_generated_ids",
        "high_generated_ids",
        "long_generated_lengths",
        "low_generated_lengths",
        "high_generated_lengths",
        "long_generated_text",
        "low_generated_text",
        "high_generated_text",
        "long_is_listen",
        "low_is_listen",
        "high_is_listen",
        "long_end_of_turn",
        "low_end_of_turn",
        "high_end_of_turn",
        "long_all_finite",
        "low_all_finite",
        "high_all_finite",
        "memory_available",
        "memory_age_seconds",
        "oracle_low_positions",
        "oracle_high_positions",
        "oracle_high_absolute_positions",
        "oracle_high_position_shifts",
        "oracle_low_generated_text",
        "oracle_high_generated_text",
        "oracle_low_is_listen",
        "oracle_high_is_listen",
        "oracle_low_end_of_turn",
        "oracle_high_end_of_turn",
        "memory_long_correct",
        "memory_low_correct",
        "memory_high_correct",
        "memory_oracle_low_correct",
        "memory_oracle_high_correct",
        "memory_long_token_f1",
        "memory_oracle_high_token_f1",
    }
    interaction_required = {
        "interaction_lane_indices",
        "interaction_history_unit_indices",
        "interaction_bank_indices",
        "interaction_modes",
        "interaction_forced_actions",
        "interaction_position_before",
        "interaction_decision_positions",
        "interaction_cache_after",
        "interaction_is_listen",
        "interaction_end_of_turn",
        "interaction_generated_lengths",
        "interaction_generated_text",
    }
    required = observation_required | interaction_required
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{root}: trace 缺少字段 {missing}")

    n_rows = len(arrays["target_seconds"])
    if n_rows == 0:
        raise ValueError(f"{root}: trace 没有观测")
    one_dimensional = observation_required - {
        "long_hidden",
        "low_hidden",
        "high_hidden",
        "long_logits",
        "low_logits",
        "high_logits",
        "long_generated_ids",
        "low_generated_ids",
        "high_generated_ids",
        "long_all_finite",
        "low_all_finite",
        "high_all_finite",
    }
    _require_shape(root, arrays, one_dimensional, (n_rows,))
    n_interactions = len(arrays["interaction_lane_indices"])
    if n_interactions == 0:
        raise ValueError(f"{root}: trace 没有完整长流交互事件")
    _require_shape(root, arrays, interaction_required, (n_interactions,))
    if np.any(arrays["interaction_lane_indices"] < 0):
        raise ValueError(f"{root}: 交互事件出现负 lane")
    if np.any(
        arrays["interaction_decision_positions"]
        < arrays["interaction_position_before"]
    ):
        raise ValueError(f"{root}: 交互决策位置早于单元起始位置")
    if np.any(
        arrays["interaction_cache_after"]
        < arrays["interaction_decision_positions"]
    ):
        raise ValueError(f"{root}: 交互单元结束位置早于决策位置")
    allowed_actions = {"natural", "forced_listen", "forced_speak"}
    if not set(arrays["interaction_forced_actions"].tolist()) <= allowed_actions:
        raise ValueError(f"{root}: 交互事件含未知 forced_action")
    for lane in np.unique(arrays["interaction_lane_indices"]):
        selected_lane = arrays["interaction_lane_indices"] == lane
        expected_indices = np.arange(int(selected_lane.sum()), dtype=np.int32)
        if not np.array_equal(
            arrays["interaction_history_unit_indices"][selected_lane],
            expected_indices,
        ):
            raise ValueError(f"{root}: lane={lane} 的完整历史单元索引不连续")

    hidden_shape = arrays["long_hidden"].shape
    if len(hidden_shape) != 3 or hidden_shape != arrays["low_hidden"].shape:
        raise ValueError(f"{root}: long/low hidden 形状不一致")
    if hidden_shape != arrays["high_hidden"].shape or hidden_shape[0] != n_rows:
        raise ValueError(f"{root}: 三臂 hidden 必须同为 [观测, 层, 隐藏维]")
    logits_shape = arrays["long_logits"].shape
    if len(logits_shape) != 2 or logits_shape[0] != n_rows:
        raise ValueError(f"{root}: logits 必须为 [观测, 词表]")
    if logits_shape != arrays["low_logits"].shape or logits_shape != arrays["high_logits"].shape:
        raise ValueError(f"{root}: 三臂 logits 形状不一致")
    ids_shape = arrays["long_generated_ids"].shape
    if len(ids_shape) != 2 or ids_shape[0] != n_rows:
        raise ValueError(f"{root}: generated_ids 必须为二维数组")
    if ids_shape != arrays["low_generated_ids"].shape or ids_shape != arrays["high_generated_ids"].shape:
        raise ValueError(f"{root}: 三臂 generated_ids 形状不一致")
    finite_shape = hidden_shape[:2]
    _require_shape(
        root,
        arrays,
        ("long_all_finite", "low_all_finite", "high_all_finite"),
        finite_shape,
    )
    for name in ("long_logits", "low_logits", "high_logits"):
        if not np.isfinite(arrays[name].astype(np.float32)).all():
            raise ValueError(f"{root}: {name} 含非有限值")
    if not np.array_equal(arrays["high_absolute_positions"], arrays["long_positions"]):
        raise ValueError(f"{root}: 高位近期臂未与长流位置逐项对齐")
    if not np.array_equal(
        arrays["high_positions"] + arrays["high_position_shifts"],
        arrays["high_absolute_positions"],
    ):
        raise ValueError(f"{root}: 高位近期臂位置平移不自洽")
    if np.any(arrays["high_position_shifts"] < 0):
        raise ValueError(f"{root}: 高位近期臂出现负平移")

    memory_mask = arrays["memory_available"].astype(bool)
    if np.any(
        arrays["oracle_high_absolute_positions"][memory_mask]
        != arrays["long_positions"][memory_mask]
    ):
        raise ValueError(f"{root}: 高位近期记忆臂未与长流位置对齐")
    if np.any(
        arrays["oracle_low_positions"][memory_mask]
        + arrays["oracle_high_position_shifts"][memory_mask]
        != arrays["oracle_high_absolute_positions"][memory_mask]
    ):
        raise ValueError(f"{root}: 高位近期记忆臂位置平移不自洽")
    if np.any(arrays["oracle_high_position_shifts"][memory_mask] < 0):
        raise ValueError(f"{root}: 高位近期记忆臂出现负平移")
    return TrueContextRun(root=root, manifest=manifest, arrays=arrays)


def _balanced_accuracy(expected: np.ndarray, observed: np.ndarray) -> float:
    scores = []
    for label in (False, True):
        selected = expected == label
        if np.any(selected):
            scores.append(float(np.mean(observed[selected] == expected[selected])))
    return float(np.mean(scores)) if len(scores) == 2 else float("nan")


def _centered_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left -= left.mean(axis=-1, keepdims=True)
    right -= right.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.sum(left * right, axis=-1) / np.maximum(denominator, 1e-12)


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.sum(left * right, axis=-1) / np.maximum(denominator, 1e-12)


def _mark_persistent(candidates: list[bool]) -> list[bool]:
    confirmed = [False] * len(candidates)
    for index in range(1, len(candidates)):
        if candidates[index - 1] and candidates[index]:
            confirmed[index - 1] = True
            confirmed[index] = True
    return confirmed


def _rate_summary(values: np.ndarray) -> dict[str, Any]:
    values = values.astype(bool)
    successes = int(values.sum())
    total = len(values)
    low, high = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": float(successes / total) if total else float("nan"),
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def analyze_true_context_runs(runs: list[TrueContextRun]) -> dict[str, Any]:
    """联合分析多条因果双工上下文运行。"""

    if not runs:
        raise ValueError("至少需要一条运行")
    task_set = {str(run.manifest["task"]) for run in runs}
    if len(task_set) != 1:
        raise ValueError("联合分析中的运行必须属于同一 task")
    task = next(iter(task_set))
    probe_catalogs = [run.manifest.get("probe_catalog", []) for run in runs]
    if any(catalog != probe_catalogs[0] for catalog in probe_catalogs[1:]):
        raise ValueError("联合分析中的 probe_catalog 不一致")
    design_keys = (
        "recent_units",
        "max_new_speak_tokens",
        "system_prompt",
        "floor_family",
        "memory_query_readout",
    )
    design_signatures = [
        {
            key: run.manifest.get("design", {}).get(key)
            for key in design_keys
        }
        for run in runs
    ]
    if any(
        signature != design_signatures[0]
        for signature in design_signatures[1:]
    ):
        raise ValueError("联合分析中的关键实验设计不一致")

    merged: dict[str, np.ndarray] = {}
    keys = runs[0].arrays
    for name in keys:
        merged[name] = np.concatenate([run.arrays[name] for run in runs], axis=0)
    distinct_audio = {
        str(run.manifest.get("source_audio", {}).get("sha256", ""))
        for run in runs
    } - {""}
    full_history_bidirectional_runs = 0
    spontaneous_bidirectional_runs = 0
    for run in runs:
        actions = run.arrays["interaction_forced_actions"].astype(str)
        all_states = run.arrays["interaction_is_listen"].astype(bool)
        if all_states.size and bool(all_states.any()) and bool((~all_states).any()):
            full_history_bidirectional_runs += 1
        natural = actions == "natural"
        states = run.arrays["interaction_is_listen"][natural].astype(bool)
        if states.size and bool(states.any()) and bool((~states).any()):
            spontaneous_bidirectional_runs += 1
    global_confirmation_ready = bool(
        len(runs) >= TRUE_CONTEXT_THRESHOLDS["minimum_independent_runs"]
        and len(distinct_audio)
        >= TRUE_CONTEXT_THRESHOLDS["minimum_distinct_audio_inputs"]
        and full_history_bidirectional_runs == len(runs)
    )
    checkpoints: list[dict[str, Any]] = []
    targets = sorted(int(value) for value in np.unique(merged["target_seconds"]))

    floor_candidates: list[bool] = []
    memory_candidates: list[bool] = []
    diagnostic_candidates: list[bool] = []
    for target in targets:
        selected = merged["target_seconds"] == target
        floor_mask = selected & (merged["probe_kinds"] == PROBE_KIND_FLOOR)
        memory_mask = selected & (merged["probe_kinds"] == PROBE_KIND_MEMORY)
        row: dict[str, Any] = {
            "target_seconds": target,
            "n_observations": int(selected.sum()),
            "history_input_seconds_median": float(
                np.median(merged["history_input_seconds"][selected])
            ),
            "long_position_median": float(np.median(merged["long_positions"][selected])),
        }
        if np.any(floor_mask):
            expected = merged["expected_is_listen"][floor_mask].astype(bool)
            long_state = merged["long_is_listen"][floor_mask].astype(bool)
            low_state = merged["low_is_listen"][floor_mask].astype(bool)
            high_state = merged["high_is_listen"][floor_mask].astype(bool)
            long_ba = _balanced_accuracy(expected, long_state)
            low_ba = _balanced_accuracy(expected, low_state)
            high_ba = _balanced_accuracy(expected, high_state)
            history_gap = high_ba - long_ba
            position_gap = low_ba - high_ba
            high_hidden = merged["high_hidden"][floor_mask].astype(np.float64)
            low_hidden = merged["low_hidden"][floor_mask].astype(np.float64)
            layer_cosines = _row_cosine(high_hidden, low_hidden)
            logits_cosine = _centered_cosine(
                merged["high_logits"][floor_mask],
                merged["low_logits"][floor_mask],
            )
            expected_listen_n = int(expected.sum())
            expected_speak_n = int((~expected).sum())
            sample_ready = bool(
                expected_listen_n
                >= TRUE_CONTEXT_THRESHOLDS[
                    "floor_minimum_per_class_per_checkpoint"
                ]
                and expected_speak_n
                >= TRUE_CONTEXT_THRESHOLDS[
                    "floor_minimum_per_class_per_checkpoint"
                ]
            )
            recent_control_pass = bool(
                low_ba
                >= TRUE_CONTEXT_THRESHOLDS[
                    "floor_oracle_min_balanced_accuracy"
                ]
            )
            if not recent_control_pass:
                floor_failure_mode = "recent_control_failure"
            elif high_ba < TRUE_CONTEXT_THRESHOLDS[
                "floor_oracle_min_balanced_accuracy"
            ]:
                floor_failure_mode = "high_position_control_failure"
            elif (
                long_ba
                <= TRUE_CONTEXT_THRESHOLDS[
                    "floor_failure_max_balanced_accuracy"
                ]
                and history_gap >= TRUE_CONTEXT_THRESHOLDS["floor_gap_min"]
            ):
                floor_failure_mode = "remote_history_failure"
            else:
                floor_failure_mode = "none"
            floor_candidate = bool(
                global_confirmation_ready
                and sample_ready
                and floor_failure_mode
                in {"high_position_control_failure", "remote_history_failure"}
            )
            floor_diagnostic = bool(
                global_confirmation_ready
                and sample_ready
                and floor_failure_mode != "none"
            )
            row["floor"] = {
                "n": int(floor_mask.sum()),
                "expected_listen_n": expected_listen_n,
                "expected_speak_n": expected_speak_n,
                "long_balanced_accuracy": long_ba,
                "low_recent_balanced_accuracy": low_ba,
                "high_recent_balanced_accuracy": high_ba,
                "history_accuracy_gap": history_gap,
                "position_accuracy_gap": position_gap,
                "long_expected_accuracy": _rate_summary(long_state == expected),
                "high_expected_accuracy": _rate_summary(high_state == expected),
                "long_is_listen_rate": float(np.mean(long_state)),
                "long_end_of_turn_rate": float(
                    np.mean(merged["long_end_of_turn"][floor_mask])
                ),
                "position_worst_layer_hidden_cosine": float(
                    np.median(np.min(layer_cosines, axis=1))
                ),
                "position_logit_cosine": float(np.median(logits_cosine)),
                "sample_ready": sample_ready,
                "recent_control_pass": recent_control_pass,
                "failure_mode": floor_failure_mode,
                "candidate_failure": floor_candidate,
                "diagnostic_failure": floor_diagnostic,
            }
        else:
            floor_candidate = False
            floor_diagnostic = False
        floor_candidates.append(floor_candidate)

        if np.any(memory_mask):
            long_correct = merged["memory_long_correct"][memory_mask].astype(bool)
            low_correct = merged["memory_low_correct"][memory_mask].astype(bool)
            high_correct = merged["memory_high_correct"][memory_mask].astype(bool)
            oracle_low = merged["memory_oracle_low_correct"][memory_mask].astype(bool)
            oracle_high = merged["memory_oracle_high_correct"][memory_mask].astype(bool)
            long_accuracy = float(np.mean(long_correct))
            oracle_high_accuracy = float(np.mean(oracle_high))
            gap = oracle_high_accuracy - long_accuracy
            negative_accuracy = float(np.mean(high_correct))
            oracle_low_accuracy = float(np.mean(oracle_low))
            negative_low_accuracy = float(np.mean(low_correct))
            sample_ready = bool(
                int(memory_mask.sum())
                >= TRUE_CONTEXT_THRESHOLDS["memory_minimum_per_checkpoint"]
            )
            negative_control_pass = bool(
                max(negative_low_accuracy, negative_accuracy)
                <= TRUE_CONTEXT_THRESHOLDS["negative_memory_max_accuracy"]
            )
            recent_fact_control_pass = bool(
                oracle_low_accuracy
                >= TRUE_CONTEXT_THRESHOLDS["memory_oracle_min_accuracy"]
            )
            if not negative_control_pass:
                memory_failure_mode = "negative_control_leakage"
            elif not recent_fact_control_pass:
                memory_failure_mode = "recent_fact_control_failure"
            elif oracle_high_accuracy < TRUE_CONTEXT_THRESHOLDS[
                "memory_oracle_min_accuracy"
            ]:
                memory_failure_mode = "high_position_control_failure"
            elif (
                long_accuracy
                <= TRUE_CONTEXT_THRESHOLDS["memory_failure_max_accuracy"]
                and gap >= TRUE_CONTEXT_THRESHOLDS["memory_gap_min"]
            ):
                memory_failure_mode = "remote_retrieval_failure"
            else:
                memory_failure_mode = "none"
            memory_candidate = bool(
                global_confirmation_ready
                and sample_ready
                and memory_failure_mode
                in {"high_position_control_failure", "remote_retrieval_failure"}
            )
            memory_diagnostic = bool(
                global_confirmation_ready
                and sample_ready
                and memory_failure_mode != "none"
            )
            row["memory"] = {
                "n": int(memory_mask.sum()),
                "age_seconds_median": float(
                    np.median(merged["memory_age_seconds"][memory_mask])
                ),
                "long_accuracy": _rate_summary(long_correct),
                "low_negative_accuracy": _rate_summary(low_correct),
                "high_negative_accuracy": _rate_summary(high_correct),
                "oracle_low_accuracy": _rate_summary(oracle_low),
                "oracle_high_accuracy": _rate_summary(oracle_high),
                "oracle_gap": gap,
                "negative_high_accuracy": negative_accuracy,
                "negative_control_pass": negative_control_pass,
                "recent_fact_control_pass": recent_fact_control_pass,
                "long_token_f1_median": float(
                    np.median(merged["memory_long_token_f1"][memory_mask])
                ),
                "oracle_high_token_f1_median": float(
                    np.median(merged["memory_oracle_high_token_f1"][memory_mask])
                ),
                "sample_ready": sample_ready,
                "failure_mode": memory_failure_mode,
                "candidate_failure": memory_candidate,
                "diagnostic_failure": memory_diagnostic,
            }
        else:
            memory_candidate = False
            memory_diagnostic = False
        memory_candidates.append(memory_candidate)
        diagnostic_candidates.append(floor_diagnostic or memory_diagnostic)
        checkpoints.append(row)

    floor_confirmed = _mark_persistent(floor_candidates)
    memory_confirmed = _mark_persistent(memory_candidates)
    diagnostic_confirmed = _mark_persistent(diagnostic_candidates)
    for index, row in enumerate(checkpoints):
        if "floor" in row:
            row["floor"]["confirmed_persistent_failure"] = floor_confirmed[index]
        if "memory" in row:
            row["memory"]["confirmed_persistent_failure"] = memory_confirmed[index]
        row["confirmed_persistent_diagnostic_failure"] = diagnostic_confirmed[index]

    first_failure_index = next(
        (
            index
            for index in range(len(checkpoints))
            if floor_confirmed[index] or memory_confirmed[index]
        ),
        None,
    )
    first_candidate_index = next(
        (
            index
            for index in range(len(checkpoints))
            if floor_candidates[index] or memory_candidates[index]
        ),
        None,
    )
    first_diagnostic_anomaly_index = next(
        (
            index
            for index, diagnostic in enumerate(diagnostic_candidates)
            if diagnostic
        ),
        None,
    )
    first_persistent_diagnostic_index = next(
        (
            index
            for index, diagnostic in enumerate(diagnostic_confirmed)
            if diagnostic
        ),
        None,
    )
    if not global_confirmation_ready:
        status = "证据不足，尚不能确认真实上下文边界"
        safe_seconds = None
        first_failure_seconds = None
        first_candidate_seconds = None
        first_diagnostic_anomaly_seconds = None
        first_persistent_diagnostic_seconds = None
        boundary_basis = "evidence_not_ready"
    else:
        first_failure_seconds = (
            targets[first_failure_index]
            if first_failure_index is not None
            else None
        )
        first_candidate_seconds = (
            targets[first_candidate_index]
            if first_candidate_index is not None
            else None
        )
        first_diagnostic_anomaly_seconds = (
            targets[first_diagnostic_anomaly_index]
            if first_diagnostic_anomaly_index is not None
            else None
        )
        first_persistent_diagnostic_seconds = (
            targets[first_persistent_diagnostic_index]
            if first_persistent_diagnostic_index is not None
            else None
        )
        boundary_indices = [
            index
            for index in (
                first_failure_index,
                first_persistent_diagnostic_index,
            )
            if index is not None
        ]
        if boundary_indices:
            boundary_index = min(boundary_indices)
            safe_seconds = targets[boundary_index - 1] if boundary_index > 0 else None
            if (
                first_failure_index is not None
                and first_failure_index == boundary_index
            ):
                status = "观察到持续的任务能力失效"
                boundary_basis = "confirmed_task_failure"
            else:
                status = "控制臂或近期任务持续失效，边界辨识在此中止"
                boundary_basis = "persistent_diagnostic_failure"
        elif first_candidate_index is None:
            status = "实测范围内未观察到任务能力失效候选"
            safe_seconds = targets[-1]
            boundary_basis = "maximum_tested_lower_bound"
        else:
            status = "观察到单点任务能力失效候选，仍需相邻检查点确认"
            safe_seconds = (
                targets[first_candidate_index - 1]
                if first_candidate_index > 0
                else None
            )
            boundary_basis = "isolated_task_failure_candidate"

    interaction_modes = merged["interaction_modes"].astype(str)
    interaction_actions = merged["interaction_forced_actions"].astype(str)
    filler_mask = (
        (interaction_modes == "filler")
        & (interaction_actions == "natural")
    )
    natural_mask = interaction_actions == "natural"
    filler_states = merged["interaction_is_listen"][filler_mask].astype(bool)
    filler_end = merged["interaction_end_of_turn"][filler_mask].astype(bool)
    natural_states = merged["interaction_is_listen"][natural_mask].astype(bool)
    natural_end = merged["interaction_end_of_turn"][natural_mask].astype(bool)
    full_states = merged["interaction_is_listen"].astype(bool)
    full_end = merged["interaction_end_of_turn"].astype(bool)
    return {
        "schema_version": 1,
        "analysis": "minicpm_true_context_joint_v1",
        "task": task,
        "run_ids": [run.run_id for run in runs],
        "n_runs": len(runs),
        "n_distinct_audio_inputs": len(distinct_audio),
        "evidence_ready": global_confirmation_ready,
        "interaction_audit": {
            "full_history_events": int(full_states.size),
            "full_history_listen_events": int(full_states.sum()),
            "full_history_speak_events": int((~full_states).sum()),
            "full_history_listen_rate": (
                float(np.mean(full_states)) if full_states.size else float("nan")
            ),
            "full_history_end_of_turn_rate": (
                float(np.mean(full_end)) if full_end.size else float("nan")
            ),
            "filler_events": int(filler_mask.sum()),
            "filler_listen_events": int(filler_states.sum()),
            "filler_speak_events": int((~filler_states).sum()),
            "filler_listen_rate": (
                float(np.mean(filler_states)) if filler_states.size else float("nan")
            ),
            "filler_end_of_turn_rate": (
                float(np.mean(filler_end)) if filler_end.size else float("nan")
            ),
            "natural_events": int(natural_mask.sum()),
            "natural_listen_events": int(natural_states.sum()),
            "natural_speak_events": int((~natural_states).sum()),
            "natural_listen_rate": (
                float(np.mean(natural_states)) if natural_states.size else float("nan")
            ),
            "natural_end_of_turn_rate": (
                float(np.mean(natural_end)) if natural_end.size else float("nan")
            ),
            "full_history_bidirectional_runs": full_history_bidirectional_runs,
            "spontaneous_bidirectional_runs": spontaneous_bidirectional_runs,
            "bidirectional_runs": full_history_bidirectional_runs,
            "all_runs_bidirectional": full_history_bidirectional_runs == len(runs),
        },
        "thresholds": TRUE_CONTEXT_THRESHOLDS,
        "checkpoints": checkpoints,
        "recommendation": {
            "status": status,
            "empirical_safe_checkpoint_seconds": safe_seconds,
            "first_candidate_failure_seconds": first_candidate_seconds,
            "first_confirmed_failure_seconds": first_failure_seconds,
            "first_diagnostic_anomaly_seconds": first_diagnostic_anomaly_seconds,
            "first_persistent_diagnostic_failure_seconds": (
                first_persistent_diagnostic_seconds
            ),
            "boundary_basis": boundary_basis,
            "first_tested_seconds": targets[0],
            "max_tested_seconds": targets[-1],
            "formal_max_positions": int(
                min(run.manifest["context_spec"]["official_max_positions"] for run in runs)
            ),
            "scope": (
                "该边界只对本协议的因果双工听说与口令检索任务成立；"
                "正式规格仍由模型配置决定。"
            ),
        },
    }


def render_true_context_markdown(report: dict[str, Any]) -> str:
    """渲染联合分析 Markdown。"""

    recommendation = report["recommendation"]
    safe_checkpoint = recommendation["empirical_safe_checkpoint_seconds"]
    safe_text = (
        f"{safe_checkpoint} 秒"
        if safe_checkpoint is not None
        else f"低于 {recommendation['first_tested_seconds']} 秒，尚未定量"
    )
    first_failure = recommendation["first_confirmed_failure_seconds"]
    first_candidate = recommendation["first_candidate_failure_seconds"]
    first_diagnostic = recommendation[
        "first_persistent_diagnostic_failure_seconds"
    ]
    audit = report["interaction_audit"]
    lines = [
        "# MiniCPM-o 因果双工真实上下文测量",
        "",
        f"- 任务：{report['task']}",
        f"- 独立运行：{report['n_runs']}",
        f"- 不同音频输入：{report['n_distinct_audio_inputs']}",
        f"- 判定证据完备：{'是' if report['evidence_ready'] else '否'}",
        f"- 结论：{recommendation['status']}",
        f"- 实证安全检查点：{safe_text}",
        f"- 首个候选失效：{first_candidate if first_candidate is not None else '未观察到'}",
        f"- 首个确认失效：{first_failure if first_failure is not None else '未观察到'}",
        f"- 首个持续诊断失效：{first_diagnostic if first_diagnostic is not None else '未观察到'}",
        f"- 安全点依据：{recommendation['boundary_basis']}",
        f"- 最大测试距离：{recommendation['max_tested_seconds']} 秒",
        (
            f"- 完整历史全部交互：倾听 {audit['full_history_listen_events']} 次，"
            f"发言 {audit['full_history_speak_events']} 次，"
            f"双向运行 {audit['full_history_bidirectional_runs']}/{report['n_runs']}"
        ),
        (
            f"- 其中自然决策：倾听 {audit['natural_listen_events']} 次，"
            f"发言 {audit['natural_speak_events']} 次，"
            f"自然双向运行 {audit['spontaneous_bidirectional_runs']}/{report['n_runs']}"
        ),
        (
            f"- 其中自然填充段：倾听 {audit['filler_listen_events']} 次，"
            f"发言 {audit['filler_speak_events']} 次"
        ),
        "",
        "## 检查点",
        "",
        (
            "| 目标秒 | 观测数 | 长流位置 | 长流听说 BA | 高位近期 BA | "
            "远端记忆 | 高位近期记忆 | 模式 | 任务失效 | 诊断失效 |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :---: | :---: |",
    ]
    for row in report["checkpoints"]:
        floor = row.get("floor", {})
        memory = row.get("memory", {})
        failure = bool(
            floor.get("confirmed_persistent_failure", False)
            or memory.get("confirmed_persistent_failure", False)
        )
        failure_mode = floor.get(
            "failure_mode",
            memory.get("failure_mode", "none"),
        )
        diagnostic_failure = bool(
            row.get("confirmed_persistent_diagnostic_failure", False)
        )
        lines.append(
            "| "
            f"{row['target_seconds']} | "
            f"{row['n_observations']} | "
            f"{row['long_position_median']:.0f} | "
            f"{floor.get('long_balanced_accuracy', float('nan')):.3f} | "
            f"{floor.get('high_recent_balanced_accuracy', float('nan')):.3f} | "
            f"{memory.get('long_accuracy', {}).get('rate', float('nan')):.3f} | "
            f"{memory.get('oracle_high_accuracy', {}).get('rate', float('nan')):.3f} | "
            f"{failure_mode} | "
            f"{'是' if failure else '否'} | "
            f"{'是' if diagnostic_failure else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"- {recommendation['scope']}",
            "- 听说平衡准确率使用预先校准且类别均衡的探针。",
            "- 远端记忆只有在同高位置近期事实仍可回答时，才把失败归因于远端检索。",
        ]
    )
    return "\n".join(lines) + "\n"
