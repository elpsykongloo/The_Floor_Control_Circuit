"""WP7：独立复算 G1 分数包并审计正式裁决。

本脚本只把 ``floor_circuit.mve.artifacts`` 当作磁盘契约读取器使用。所有指标、
会话级 bootstrap、shuffled-labels sanity 与 G1 分支均在本文件内重新实现，
不调用正式分析链的统计或裁决函数。

用法：
    uv run python scripts/wp7_audit_mve.py
    uv run python scripts/wp7_audit_mve.py --summary reports/mve_summary.json
    uv run python scripts/wp7_audit_mve.py --manifest D:\\...\\manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from _bootstrap import REPO_ROOT, REPORTS_DIR
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from floor_circuit.config import load_config
from floor_circuit.mve.artifacts import (
    ANALYSIS_SOURCE_PATHS,
    FROZEN_G1_BOOTSTRAP_N,
    FROZEN_G1_BOOTSTRAP_SEED,
    FROZEN_G1_ITEM_COUNT,
    FROZEN_G1_LAYERS,
    FROZEN_G1_SEEDS,
    FROZEN_G1_TARGETS,
    read_per_session_npz,
    sha256_file,
    validate_score_bundle,
)

PerSession = dict[str, tuple[np.ndarray, np.ndarray]]
SeededPerSession = dict[int, PerSession]
ItemKey = tuple[str, str, int | None, int]

AUDIT_SCHEMA = "floor_circuit.mve.independent_audit.v1"
FLOAT_ATOL = 1e-12
CI_SCOPE = "给定已在 probe_val 上选择的目标、层与各种子 C 后的会话级条件 CI"
DEFAULT_SUMMARY = REPORTS_DIR / "mve_summary.json"
DEFAULT_OUTPUT = REPORTS_DIR / "mve_independent_audit.json"


class IndependentAuditError(RuntimeError):
    """独立审计遇到不可继续或违反冻结协议的输入。"""


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IndependentAuditError(f"无法读取{description}：{path}") from exc
    if not isinstance(payload, dict):
        raise IndependentAuditError(f"{description}顶层必须为对象：{path}")
    return payload


def _item_key(item: dict[str, Any]) -> ItemKey:
    layer = item["layer"]
    return (
        str(item["target"]),
        str(item["kind"]),
        None if layer is None else int(layer),
        int(item["seed"]),
    )


def _analysis_content_sha256(sources: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sources:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise IndependentAuditError(f"分析源码不存在：{relative}")
        encoded_path = relative.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_commit(relative: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%H", "--", relative],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IndependentAuditError(f"无法读取分析源码提交：{relative}") from exc
    return value or None


def _verify_metadata(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """独立核对冻结配置、分析代码指纹与预检快照。"""

    grids = load_config("grids")
    mve = grids["mve"]
    clocks = grids["clocks"]
    expected_contract = {
        "targets": list(FROZEN_G1_TARGETS),
        "layers": list(FROZEN_G1_LAYERS),
        "seeds": list(FROZEN_G1_SEEDS),
        "expected_items": FROZEN_G1_ITEM_COUNT,
    }
    if manifest["contract"] != expected_contract:
        raise IndependentAuditError("manifest contract 与冻结 G1 配置不一致")
    config_checks = {
        "targets": list(mve["targets"]) == expected_contract["targets"],
        "layers": [int(value) for value in mve["layers"]] == expected_contract["layers"],
        "seeds": [int(value) for value in mve["seeds"]] == expected_contract["seeds"],
        "bootstrap_n": int(mve["bootstrap_n"]) == FROZEN_G1_BOOTSTRAP_N,
        "full_threshold": float(mve["g1_full_threshold"]) == 0.05,
        "backup_threshold": float(mve["g1_backup_threshold"]) == 0.02,
        "n_train_sessions": int(mve["n_sessions_train"]) == 160,
        "n_eval_sessions": int(mve["n_sessions_eval"]) == len(manifest["eval_session_order"]),
    }
    failed_config = [name for name, passed in config_checks.items() if not passed]
    if failed_config:
        raise IndependentAuditError(f"当前冻结配置校验失败：{failed_config}")

    analysis = manifest["analysis_protocol"]
    if (
        int(analysis["bootstrap_n"]) != FROZEN_G1_BOOTSTRAP_N
        or int(analysis["bootstrap_seed"]) != FROZEN_G1_BOOTSTRAP_SEED
    ):
        raise IndependentAuditError("manifest bootstrap 元数据与冻结协议不一致")
    code = analysis["code"]
    sources = [str(value) for value in code["sources"]]
    if tuple(sources) != ANALYSIS_SOURCE_PATHS:
        raise IndependentAuditError("manifest 分析源码列表与冻结集合不一致")
    content_sha256 = _analysis_content_sha256(sources)
    if code["content_sha256"] != content_sha256:
        raise IndependentAuditError("manifest 分析代码内容指纹与当前源码不一致")
    current_commits = {relative: _source_commit(relative) for relative in sources}
    if code["source_commits"] != current_commits:
        raise IndependentAuditError("manifest 分析源码逐文件提交记录与仓库不一致")

    snapshot_path = manifest_path.parent / manifest["preflight_report_path"]
    snapshot = _load_json_object(snapshot_path, "预检快照")
    if snapshot.get("status") != "passed":
        raise IndependentAuditError("预检快照 status 未通过")
    expected_n_sessions = int(mve["n_sessions_train"]) + int(mve["n_sessions_eval"])
    expected_clock_hz = float(clocks["moshi"]["hz"])
    preflight_checks = {
        "n_sessions": snapshot.get("n_sessions") == expected_n_sessions,
        "n_runs": snapshot.get("n_runs") == expected_n_sessions * 2,
        "layers": snapshot.get("layers") == expected_contract["layers"],
        "n_labels": snapshot.get("n_labels") == expected_n_sessions,
        "expected_n_steps": snapshot.get("expected_n_steps")
        == round(float(mve["max_minutes_per_session"]) * 60.0 * expected_clock_hz),
        "expected_clock_hz": snapshot.get("expected_clock_hz") == expected_clock_hz,
        "expected_max_seconds": snapshot.get("expected_max_seconds")
        == float(mve["max_minutes_per_session"]) * 60.0,
        "expected_mimi_chunk_seconds": snapshot.get("expected_mimi_chunk_seconds")
        == float(mve["mimi_chunk_seconds"]),
        "expected_forward_chunk_steps": snapshot.get("expected_forward_chunk_steps")
        == int(mve["forward_chunk_steps"]),
        "runner_code_version": snapshot.get("expected_code_version")
        == manifest["runner_code_version"],
        "label_sha256": snapshot.get("label_sha256") == manifest["label_sha256"],
    }
    failed_preflight = [name for name, passed in preflight_checks.items() if not passed]
    if failed_preflight:
        raise IndependentAuditError(f"预检快照元数据校验失败：{failed_preflight}")
    eval_sessions = set(manifest["eval_session_order"])
    if not eval_sessions.issubset(manifest["label_sha256"]):
        raise IndependentAuditError("评估会话未被 manifest.label_sha256 完整覆盖")

    return {
        "config": config_checks,
        "analysis_code_content_sha256": content_sha256,
        "analysis_source_commits": current_commits,
        "preflight": preflight_checks,
        "preflight_status": snapshot["status"],
    }


def _load_and_align_items(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[dict[ItemKey, PerSession], dict[str, Any]]:
    """读取 34 个 NPZ，并核对同一目标的会话顺序与标签逐项全等。"""

    expected_sessions = sorted(str(value) for value in manifest["eval_session_order"])
    loaded: dict[ItemKey, PerSession] = {}
    for item in manifest["items"]:
        key = _item_key(item)
        per_session = read_per_session_npz(manifest_path.parent / item["path"])
        if list(per_session) != expected_sessions:
            raise IndependentAuditError(f"{key}: NPZ 会话顺序与排序后的评估会话不一致")
        loaded[key] = per_session
    if len(loaded) != FROZEN_G1_ITEM_COUNT:
        raise IndependentAuditError(f"独立读取到 {len(loaded)} 项，期望 {FROZEN_G1_ITEM_COUNT}")

    comparisons = 0
    rows_by_target: dict[str, int] = {}
    for target in FROZEN_G1_TARGETS:
        target_items = sorted(
            ((key, value) for key, value in loaded.items() if key[0] == target),
            key=lambda pair: repr(pair[0]),
        )
        if not target_items:
            raise IndependentAuditError(f"{target}: 没有分数条目")
        reference_key, reference = target_items[0]
        rows_by_target[target] = sum(len(labels) for labels, _scores in reference.values())
        for key, per_session in target_items[1:]:
            if list(per_session) != list(reference):
                raise IndependentAuditError(f"{key}: 会话序列与 {reference_key} 不一致")
            for session_id in expected_sessions:
                expected_labels = reference[session_id][0]
                actual_labels = per_session[session_id][0]
                if (
                    expected_labels.dtype != actual_labels.dtype
                    or expected_labels.shape != actual_labels.shape
                    or not np.array_equal(expected_labels, actual_labels)
                ):
                    raise IndependentAuditError(
                        f"{key}: 会话 {session_id} 的标签与 {reference_key} 不完全一致"
                    )
                comparisons += 1
    return loaded, {
        "n_items": len(loaded),
        "n_eval_sessions": len(expected_sessions),
        "label_array_comparisons": comparisons,
        "rows_by_target": rows_by_target,
    }


def _pooled(per_session: PerSession, session_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([per_session[session_id][0] for session_id in session_ids])
    scores = np.concatenate([per_session[session_id][1] for session_id in session_ids])
    return labels, scores


def _metrics(per_session: PerSession) -> dict[str, Any]:
    session_ids = sorted(per_session)
    labels, scores = _pooled(per_session, session_ids)
    if len(np.unique(labels)) < 2:
        raise IndependentAuditError("完整评估集缺少正类或负类，无法计算指标")
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "balanced_acc": float(balanced_accuracy_score(labels, scores >= 0.5)),
        "n": len(labels),
        "pos_rate": float(np.mean(labels)),
        "n_sessions": len(session_ids),
    }


def _mean_and_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def _seed_mean_metrics(scores: SeededPerSession) -> dict[str, Any]:
    by_seed = {seed: _metrics(scores[seed]) for seed in sorted(scores)}
    auc_mean, auc_sd = _mean_and_sd([value["auc"] for value in by_seed.values()])
    auprc_mean, auprc_sd = _mean_and_sd([value["auprc"] for value in by_seed.values()])
    balanced_mean, balanced_sd = _mean_and_sd(
        [value["balanced_acc"] for value in by_seed.values()]
    )
    first = next(iter(by_seed.values()))
    return {
        "n_seeds": len(by_seed),
        "auc": auc_mean,
        "auc_mean": auc_mean,
        "auc_sd": auc_sd,
        "auprc": auprc_mean,
        "auprc_mean": auprc_mean,
        "auprc_sd": auprc_sd,
        "balanced_acc": balanced_mean,
        "balanced_acc_mean": balanced_mean,
        "balanced_acc_sd": balanced_sd,
        "n": first["n"],
        "pos_rate": first["pos_rate"],
        "n_sessions": first["n_sessions"],
        "by_seed": {str(seed): value for seed, value in by_seed.items()},
    }


def _safe_auc(per_session: PerSession, session_ids: list[str]) -> float | None:
    labels, scores = _pooled(per_session, session_ids)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _seed_mean_auc(
    scores: SeededPerSession,
    session_ids: list[str],
) -> float | None:
    values: list[float] = []
    for seed in sorted(scores):
        auc = _safe_auc(scores[seed], session_ids)
        if auc is None:
            return None
        values.append(auc)
    return float(np.mean(values))


def _resample_sessions(
    sorted_sessions: list[str],
    rng: np.random.Generator,
) -> list[str]:
    indices = rng.integers(0, len(sorted_sessions), size=len(sorted_sessions))
    return [sorted_sessions[index] for index in indices]


def _bootstrap_seed_mean_auc(
    probe: SeededPerSession,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """使用独立新 RNG 复算探针种子均值的会话级 CI。"""

    session_ids = sorted(next(iter(probe.values())))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        value = _seed_mean_auc(probe, _resample_sessions(session_ids, rng))
        if value is not None:
            samples.append(value)
    if not samples:
        raise IndependentAuditError("探针会话级 bootstrap 没有有效双类样本")
    array = np.asarray(samples, dtype=np.float64)
    return {
        "point": _seed_mean_auc(probe, session_ids),
        "ci_lo": float(np.percentile(array, 2.5)),
        "ci_hi": float(np.percentile(array, 97.5)),
        "n_boot_effective": len(array),
        "n_seeds": len(probe),
    }


def _bootstrap_advantage(
    probe: SeededPerSession,
    baselines: dict[str, SeededPerSession],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """在同一会话重采样中复算探针减最大基线的成对优势。"""

    session_ids = sorted(next(iter(probe.values())))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        take = _resample_sessions(session_ids, rng)
        probe_auc = _seed_mean_auc(probe, take)
        baseline_aucs = {
            name: _seed_mean_auc(scores, take)
            for name, scores in baselines.items()
        }
        if probe_auc is None or any(value is None for value in baseline_aucs.values()):
            continue
        samples.append(probe_auc - max(baseline_aucs.values()))
    if not samples:
        raise IndependentAuditError("成对优势 bootstrap 没有有效双类样本")
    array = np.asarray(samples, dtype=np.float64)
    point_probe = _seed_mean_auc(probe, session_ids)
    point_baselines = {
        name: _seed_mean_auc(scores, session_ids)
        for name, scores in baselines.items()
    }
    if point_probe is None or any(value is None for value in point_baselines.values()):
        raise IndependentAuditError("完整评估集无法计算探针或基线 AUC")
    return {
        "advantage_point": float(point_probe - max(point_baselines.values())),
        "ci_lo": float(np.percentile(array, 2.5)),
        "ci_hi": float(np.percentile(array, 97.5)),
        "probe_auc": point_probe,
        "baseline_aucs": point_baselines,
        "n_boot_effective": len(array),
        "probe_n_seeds": len(probe),
        "baseline_n_seeds": {
            name: len(scores)
            for name, scores in baselines.items()
        },
    }


def _shuffle_labels(
    scores: PerSession,
    seed: int,
    session_order: list[str],
) -> PerSession:
    """按正式评估写入顺序逐会话洗牌；分数包 NPZ 自身按标识排序。"""

    if sorted(session_order) != sorted(scores):
        raise IndependentAuditError("shuffled sanity 的会话顺序与分数项不一致")
    rng = np.random.default_rng(seed)
    shuffled: PerSession = {}
    for session_id in session_order:
        labels, values = scores[session_id]
        shuffled_labels = np.array(labels, copy=True)
        rng.shuffle(shuffled_labels)
        shuffled[session_id] = (shuffled_labels, values)
    return shuffled


def _verdict(
    advantage_point: float,
    ci_lo: float,
    full_threshold: float,
    backup_threshold: float,
) -> str:
    if advantage_point >= full_threshold and ci_lo > 0:
        return "full_e1"
    if advantage_point >= backup_threshold:
        return "backup_mve"
    return "n1"


def recompute_summary(
    manifest: dict[str, Any],
    loaded: dict[ItemKey, PerSession],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """完全独立复算 summary 的 ``overall`` 与 ``per_target``。"""

    mve = load_config("grids")["mve"]
    full_threshold = float(mve["g1_full_threshold"])
    backup_threshold = float(mve["g1_backup_threshold"])
    n_boot = int(manifest["analysis_protocol"]["bootstrap_n"])
    boot_seed = int(manifest["analysis_protocol"]["bootstrap_seed"])
    item_metadata = {_item_key(item): item for item in manifest["items"]}
    per_target: dict[str, Any] = {}
    bootstrap_samples: dict[str, Any] = {}

    for target in FROZEN_G1_TARGETS:
        layer_summary: dict[str, Any] = {}
        probes_by_layer: dict[int, SeededPerSession] = {}
        for layer in FROZEN_G1_LAYERS:
            probes = {
                seed: loaded[(target, "probe", layer, seed)]
                for seed in FROZEN_G1_SEEDS
            }
            probes_by_layer[layer] = probes
            per_seed = {seed: _metrics(probes[seed]) for seed in FROZEN_G1_SEEDS}
            auc_mean, auc_sd = _mean_and_sd(
                [per_seed[seed]["auc"] for seed in FROZEN_G1_SEEDS]
            )
            auprc_mean, auprc_sd = _mean_and_sd(
                [per_seed[seed]["auprc"] for seed in FROZEN_G1_SEEDS]
            )
            balanced_mean, balanced_sd = _mean_and_sd(
                [per_seed[seed]["balanced_acc"] for seed in FROZEN_G1_SEEDS]
            )
            layer_summary[str(layer)] = {
                "n_seeds": len(FROZEN_G1_SEEDS),
                "auc_mean": auc_mean,
                "auc_sd": auc_sd,
                "auprc_mean": auprc_mean,
                "auprc_sd": auprc_sd,
                "balanced_acc_mean": balanced_mean,
                "balanced_acc_sd": balanced_sd,
                "auc_by_seed": {
                    str(seed): per_seed[seed]["auc"]
                    for seed in FROZEN_G1_SEEDS
                },
                "best_c_by_seed": {
                    str(seed): float(
                        item_metadata[(target, "probe", layer, seed)]["best_c"]
                    )
                    for seed in FROZEN_G1_SEEDS
                },
            }

        best_layer = max(
            FROZEN_G1_LAYERS,
            key=lambda layer: layer_summary[str(layer)]["auc_mean"],
        )
        selected_probe = probes_by_layer[best_layer]
        baselines: dict[str, SeededPerSession] = {
            "hazard": {0: loaded[(target, "hazard", None, 0)]},
            "mimi": {
                seed: loaded[(target, "mimi", None, seed)]
                for seed in FROZEN_G1_SEEDS
            },
            "acoustic_gru": {0: loaded[(target, "acoustic_gru", None, 0)]},
        }
        advantage = _bootstrap_advantage(
            selected_probe,
            baselines,
            n_boot=n_boot,
            seed=boot_seed,
        )
        # 这里有意从相同 seed 新建 RNG，复现正式协议中的单独探针 CI。
        probe_ci = _bootstrap_seed_mean_auc(
            selected_probe,
            n_boot=n_boot,
            seed=boot_seed,
        )
        shuffled = {
            seed: _shuffle_labels(
                selected_probe[seed],
                boot_seed,
                [str(value) for value in manifest["eval_session_order"]],
            )
            for seed in FROZEN_G1_SEEDS
        }
        shuffled_metrics = _seed_mean_metrics(shuffled)
        target_result = {
            "best_layer": best_layer,
            "layer_summary": layer_summary,
            "advantage": advantage,
            "baseline_metrics": {
                name: _seed_mean_metrics(scores)
                for name, scores in baselines.items()
            },
            "probe_ci": probe_ci,
            "shuffled_auc": shuffled_metrics["auc_mean"],
            "shuffled_auc_sd": shuffled_metrics["auc_sd"],
            "shuffled_n_seeds": shuffled_metrics["n_seeds"],
            "ci_scope": CI_SCOPE,
            "covers_model_selection_uncertainty": False,
            "verdict": _verdict(
                advantage["advantage_point"],
                advantage["ci_lo"],
                full_threshold,
                backup_threshold,
            ),
        }
        per_target[target] = target_result
        bootstrap_samples[target] = {
            "advantage_n_effective": advantage["n_boot_effective"],
            "probe_n_effective": probe_ci["n_boot_effective"],
        }

    decisive_target = max(
        FROZEN_G1_TARGETS,
        key=lambda target: per_target[target]["advantage"]["advantage_point"],
    )
    decisive = per_target[decisive_target]["advantage"]
    overall = {
        "decisive_target": decisive_target,
        "advantage_point": decisive["advantage_point"],
        "ci_lo": decisive["ci_lo"],
        "verdict": _verdict(
            decisive["advantage_point"],
            decisive["ci_lo"],
            full_threshold,
            backup_threshold,
        ),
    }
    return {"overall": overall, "per_target": per_target}, bootstrap_samples


def _difference(
    path: str,
    reason: str,
    expected: Any,
    actual: Any,
) -> dict[str, Any]:
    return {
        "path": path,
        "reason": reason,
        "expected": expected,
        "actual": actual,
    }


def _compare_values(
    expected: Any,
    actual: Any,
    path: str,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            differences.append(_difference(path, "类型不一致", "object", type(actual).__name__))
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            differences.append(_difference(f"{path}.{key}", "字段缺失", expected[key], None))
        for key in sorted(actual_keys - expected_keys):
            differences.append(_difference(f"{path}.{key}", "出现额外字段", None, actual[key]))
        for key in sorted(expected_keys & actual_keys):
            _compare_values(expected[key], actual[key], f"{path}.{key}", differences)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            differences.append(_difference(path, "类型不一致", "array", type(actual).__name__))
            return
        if len(expected) != len(actual):
            differences.append(_difference(path, "数组长度不一致", len(expected), len(actual)))
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_values(expected_item, actual_item, f"{path}[{index}]", differences)
        return
    if isinstance(expected, float) and not isinstance(expected, bool):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not np.isfinite(actual)
            or not np.isclose(expected, float(actual), rtol=0.0, atol=FLOAT_ATOL)
        ):
            differences.append(_difference(path, "浮点值不一致", expected, actual))
        return
    if type(expected) is not type(actual) or expected != actual:
        differences.append(_difference(path, "值或类型不一致", expected, actual))


def _validate_generated_at(summary: dict[str, Any]) -> None:
    value = summary.get("generated_at")
    if not isinstance(value, str):
        raise IndependentAuditError("summary.generated_at 缺失或类型错误")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise IndependentAuditError("summary.generated_at 不是 ISO-8601 时间") from exc
    if parsed.tzinfo is None:
        raise IndependentAuditError("summary.generated_at 缺少时区")


def _resolve_manifest(
    summary: dict[str, Any],
    manifest_override: str | Path | None,
) -> tuple[Path, dict[str, Any], str, dict[str, Any]]:
    score_bundle = summary.get("score_bundle")
    if not isinstance(score_bundle, dict):
        raise IndependentAuditError("summary 缺少 score_bundle 引用")
    required = {
        "absolute_path",
        "relative_path",
        "manifest_path",
        "manifest_sha256",
        "n_items",
    }
    if set(score_bundle) != required:
        raise IndependentAuditError("summary.score_bundle 字段集合不完整")
    manifest_path = (
        Path(manifest_override)
        if manifest_override is not None
        else Path(str(score_bundle["manifest_path"]))
    ).resolve()
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != score_bundle["manifest_sha256"]:
        raise IndependentAuditError("summary 引用的 manifest SHA-256 与文件不一致")
    manifest = validate_score_bundle(manifest_path)
    expected_bundle = {
        "absolute_path": str(manifest_path.parent.resolve()),
        "relative_path": manifest["bundle_path"]["relative"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_sha256,
        "n_items": len(manifest["items"]),
    }
    return manifest_path, manifest, actual_sha256, expected_bundle


def audit_summary(
    summary_path: str | Path = DEFAULT_SUMMARY,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """执行独立审计；校验失败时抛出异常，summary 数值差异写入返回值。"""

    summary_file = Path(summary_path).resolve()
    summary = _load_json_object(summary_file, "G1 summary")
    _validate_generated_at(summary)
    expected_top_keys = {"generated_at", "overall", "per_target", "score_bundle"}
    if set(summary) != expected_top_keys:
        raise IndependentAuditError(
            f"summary 顶层字段集合不一致：{sorted(summary)}"
        )
    resolved_manifest, manifest, manifest_sha256, expected_bundle = _resolve_manifest(
        summary,
        manifest_path,
    )
    metadata_checks = _verify_metadata(manifest, resolved_manifest)
    loaded, alignment_checks = _load_and_align_items(manifest, resolved_manifest)
    recomputed, bootstrap_checks = recompute_summary(manifest, loaded)
    expected_summary = {
        **recomputed,
        "score_bundle": expected_bundle,
    }
    actual_summary = {
        "overall": summary["overall"],
        "per_target": summary["per_target"],
        "score_bundle": summary["score_bundle"],
    }
    differences: list[dict[str, Any]] = []
    _compare_values(expected_summary, actual_summary, "$", differences)
    return {
        "summary_path": str(summary_file),
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": manifest_sha256,
        "n_items": len(manifest["items"]),
        "checks": {
            "manifest_validate_score_bundle": "passed",
            "metadata": metadata_checks,
            "item_alignment": alignment_checks,
            "bootstrap": bootstrap_checks,
            "summary_fields_compared": True,
            "float_atol": FLOAT_ATOL,
        },
        "recomputed": recomputed,
        "differences": differences,
    }


def run_audit(
    summary_path: str | Path = DEFAULT_SUMMARY,
    manifest_path: str | Path | None = None,
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """成功或失败均原子发布独立审计报告。"""

    output = Path(output_path)
    generated_at = datetime.now(UTC).isoformat()
    try:
        details = audit_summary(summary_path, manifest_path)
        differences = details["differences"]
        payload = {
            "schema": AUDIT_SCHEMA,
            "generated_at": generated_at,
            "status": "passed" if not differences else "failed",
            **details,
        }
        if differences:
            payload["error"] = f"mve_summary 与独立复算结果存在 {len(differences)} 项差异"
    except Exception as exc:
        payload = {
            "schema": AUDIT_SCHEMA,
            "generated_at": generated_at,
            "status": "failed",
            "summary_path": str(Path(summary_path).resolve()),
            "manifest_path": (
                None if manifest_path is None else str(Path(manifest_path).resolve())
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "differences": [],
        }
    _write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="独立复算 G1 并审计 mve_summary")
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
        help="G1 summary JSON，默认 reports/mve_summary.json",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="可选的 manifest 路径；仍须与 summary 中记录的 SHA-256 一致",
    )
    args = parser.parse_args()
    result = run_audit(args.summary, args.manifest)
    if result["status"] == "passed":
        overall = result["recomputed"]["overall"]
        print(
            "独立复算通过："
            f"{overall['verdict']}，{overall['decisive_target']} 优势 "
            f"{overall['advantage_point']:+.6f}"
        )
        return
    print(f"独立复算失败：{result.get('error', '未知错误')}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
