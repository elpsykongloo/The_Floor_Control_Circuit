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
import pandas as pd
from _bootstrap import REPO_ROOT, REPORTS_DIR
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from floor_circuit.config import data_root, load_config
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
AuthoritativeLabels = dict[str, dict[str, np.ndarray]]

AUDIT_SCHEMA = "floor_circuit.mve.independent_audit.v1"
FLOAT_ATOL = 1e-12
CI_SCOPE = (
    "层与各种子 C 在 probe_train 内层划分上选择，probe_val 仅用于最终报告；"
    "会话级 bootstrap CI 对选定模型无报告集选择泄漏（目标间取优仍为冻结决策规则）"
)
MIN_ELIGIBLE_STEP = 0  # 与 mve/alignment.py 一致（本脚本独立复算，不 import 正式链）
# PREREG #8 锚定：标签步 s 观测截止 (s+1)·τ；acts 读行 s+1、基线读行 s；末标签步丢弃
EXPECTED_TIME_ALIGNMENT = {
    "initial_token_position": 0,
    "acts_observed_through_offset_steps": 0,
    "label_step_observed_through_offset_steps": 1,
    "acts_row_for_step": "s+1",
    "baseline_row_for_step": "s",
    "min_eligible_step": MIN_ELIGIBLE_STEP,
    "last_label_step_dropped": True,
}
DEFAULT_SUMMARY = REPORTS_DIR / "mve_summary.json"
DEFAULT_OUTPUT = REPORTS_DIR / "mve_independent_audit.json"
DEFAULT_SPLIT = REPO_ROOT / "configs" / "splits" / "candor.json"
REQUIRED_SCORE_SOURCE_PATHS = (
    "configs/grids.yaml",
    "configs/splits/candor.json",
    "src/floor_circuit/config.py",
    "src/floor_circuit/schemas.py",
    "src/floor_circuit/cachelib/manifest.py",
    "src/floor_circuit/cachelib/zarr_io.py",
    "src/floor_circuit/mve/preflight.py",
    "src/floor_circuit/probes/linear.py",
    "src/floor_circuit/probes/baselines.py",
    "src/floor_circuit/probes/gru.py",
)


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


def _sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    """独立计算文件摘要，不复用正式分数包的摘要函数。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(chunk_size):
                digest.update(block)
    except OSError as exc:
        raise IndependentAuditError(f"无法读取待核验文件：{path}") from exc
    return digest.hexdigest()


def _git_commit_file(repository_head: str, relative: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", f"{repository_head}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IndependentAuditError(f"提交 {repository_head} 不含要求的分数来源：{relative}") from exc


def _canonical_text_bytes(payload: bytes) -> bytes:
    """按 Git 文本规范统一换行，避免 Windows 检出层的 CRLF 造成假差异。"""

    return payload.replace(b"\r\n", b"\n")


def _verify_repository_sources(
    analysis: dict[str, Any],
    source_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """确认记录的仓库头真实存在，且分数生成来源等于该提交中的内容。"""

    repository_head = str(analysis["code"]["repository_head"])
    try:
        object_type = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-t", repository_head],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IndependentAuditError(f"analysis repository_head 不是当前仓库可解析对象：{repository_head}") from exc
    if object_type != "commit":
        raise IndependentAuditError(f"analysis repository_head 对象类型为 {object_type!r}，期望 commit")

    required = tuple(dict.fromkeys((*ANALYSIS_SOURCE_PATHS, *REQUIRED_SCORE_SOURCE_PATHS)))
    selected = required if source_paths is None else tuple(source_paths)
    missing = sorted(set(required) - set(selected))
    if missing:
        raise IndependentAuditError(f"分数生成来源核验集合不完整：{missing}")

    verified: dict[str, Any] = {}
    for relative in selected:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise IndependentAuditError(f"当前分数生成来源不存在：{relative}")
        current = _canonical_text_bytes(path.read_bytes())
        committed = _canonical_text_bytes(_git_commit_file(repository_head, relative))
        current_sha256 = hashlib.sha256(current).hexdigest()
        commit_sha256 = hashlib.sha256(committed).hexdigest()
        if current_sha256 != commit_sha256:
            raise IndependentAuditError(f"当前分数生成来源与 repository_head 内容不一致：{relative}")
        verified[relative] = {
            "sha256": current_sha256,
            "repository_head_sha256": commit_sha256,
        }
    return {
        "repository_head": repository_head,
        "object_type": object_type,
        "required_sources": list(required),
        "verified_sources": verified,
    }


def _load_frozen_sessions(
    split_path: Path,
    *,
    n_train: int,
    n_eval: int,
) -> tuple[list[str], list[str]]:
    split = _load_json_object(split_path, "CANDOR 冻结划分")
    groups = split.get("splits")
    if not isinstance(groups, dict):
        raise IndependentAuditError("CANDOR 冻结划分缺少 splits")
    train_all = groups.get("probe_train")
    eval_all = groups.get("probe_val")
    if not isinstance(train_all, list) or not isinstance(eval_all, list):
        raise IndependentAuditError("CANDOR 冻结划分缺少 probe_train/probe_val")
    train = [str(value) for value in train_all[:n_train]]
    evals = [str(value) for value in eval_all[:n_eval]]
    if len(train) != n_train or len(evals) != n_eval:
        raise IndependentAuditError("CANDOR 冻结划分容量不足")
    if (
        any(not value for value in [*train, *evals])
        or len(train) != len(set(train))
        or len(evals) != len(set(evals))
        or set(train) & set(evals)
    ):
        raise IndependentAuditError("CANDOR 冻结训练/验证会话为空、重复或有交叠")
    return train, evals


def _marker_label_record(marker: dict[str, Any], session_id: str) -> dict[str, Any]:
    try:
        record = marker["outputs"]["labels"]
    except (KeyError, TypeError) as exc:
        raise IndependentAuditError(f"{session_id}: WP1 完成标记缺少 outputs.labels") from exc
    if not isinstance(record, dict):
        raise IndependentAuditError(f"{session_id}: WP1 完成标记 labels 字段无效")
    return record


def _assemble_authoritative_labels(
    path: Path,
    session_id: str,
    *,
    expected_n_steps: int,
    t1_delta_ms: int,
) -> dict[str, np.ndarray]:
    """从 WP1 权威 parquet 独立重建评估标签顺序。"""

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise IndependentAuditError(f"{session_id}: 无法读取权威标签 parquet") from exc
    required = {"target", "agent_channel", "step", "delta_ms", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise IndependentAuditError(f"{session_id}: 权威标签缺列 {missing}")

    assembled: dict[str, np.ndarray] = {}
    for target in FROZEN_G1_TARGETS:
        parts: list[np.ndarray] = []
        for channel in (0, 1):
            mask = (frame["target"] == target) & (frame["agent_channel"] == channel)
            if target == "T1":
                mask &= frame["delta_ms"] == t1_delta_ms
            rows = frame.loc[mask, ["step", "label"]].copy()
            if rows.empty:
                parts.append(np.empty(0, dtype=np.int8))
                continue
            steps_numeric = pd.to_numeric(rows["step"], errors="coerce").to_numpy(dtype=np.float64)
            if (
                not np.isfinite(steps_numeric).all()
                or not np.equal(steps_numeric, np.floor(steps_numeric)).all()
                or (steps_numeric < 0).any()
            ):
                raise IndependentAuditError(f"{session_id}/{target}/agent{channel}: step 不是非负整数")
            # 时间对齐（PREREG #7/#8）：与正式链一致——末标签步（无对应 acts 行）丢弃
            rows = rows.loc[
                (steps_numeric < expected_n_steps - 1) & (steps_numeric >= MIN_ELIGIBLE_STEP)
            ]
            rows = rows.sort_values("step", kind="stable")
            steps = rows["step"].to_numpy(dtype=np.int64)
            if len(steps) != len(np.unique(steps)):
                raise IndependentAuditError(f"{session_id}/{target}/agent{channel}: 评估步号重复")
            labels = rows["label"].to_numpy()
            if not np.isin(labels, (0, 1)).all():
                raise IndependentAuditError(f"{session_id}/{target}/agent{channel}: 标签不是二值")
            parts.append(labels.astype(np.int8, copy=False))
        assembled[target] = np.concatenate(parts)
    return assembled


def _verify_authoritative_labels(
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    split_path: Path,
    data_root_path: Path,
    mve: dict[str, Any],
    expected_n_steps: int,
) -> tuple[dict[str, Any], AuthoritativeLabels]:
    """四方核对权威标签、完成标记、平铺副本和清单摘要。"""

    n_train = int(mve["n_sessions_train"])
    n_eval = int(mve["n_sessions_eval"])
    train, evals = _load_frozen_sessions(
        split_path,
        n_train=n_train,
        n_eval=n_eval,
    )
    if manifest["eval_session_order"] != evals:
        raise IndependentAuditError("manifest.eval_session_order 与冻结 probe_val 前 40 会话顺序不全等")
    expected_sessions = [*train, *evals]
    expected_set = set(expected_sessions)
    manifest_hashes = manifest["label_sha256"]
    snapshot_hashes = snapshot.get("label_sha256")
    if set(manifest_hashes) != expected_set or len(manifest_hashes) != n_train + n_eval:
        raise IndependentAuditError("manifest.label_sha256 键集合不等于冻结 160+40 会话")
    if not isinstance(snapshot_hashes, dict) or snapshot_hashes != manifest_hashes:
        raise IndependentAuditError("预检快照与 manifest 的标签哈希不全等")

    source_root = data_root_path / "events" / "candor"
    flat_root = data_root_path / "events" / "candor_labels_flat"
    eval_set = set(evals)
    actual_hashes: dict[str, str] = {}
    authoritative: AuthoritativeLabels = {}
    for session_id in expected_sessions:
        source = source_root / f"{session_id}.labels.parquet"
        marker_path = source_root / f"{session_id}.complete.json"
        flat = flat_root / f"{session_id}.parquet"
        source_sha256 = _sha256_path(source)
        flat_sha256 = _sha256_path(flat)
        try:
            source_size = source.stat().st_size
        except OSError as exc:
            raise IndependentAuditError(f"{session_id}: 无法读取权威标签大小") from exc
        marker = _load_json_object(marker_path, f"{session_id} WP1 完成标记")
        record = _marker_label_record(marker, session_id)
        marker_ok = (
            marker.get("schema_version") == 1
            and marker.get("session") == session_id
            and record.get("name") == source.name
            and record.get("size") == source_size
            and record.get("sha256") == source_sha256
        )
        if not marker_ok:
            raise IndependentAuditError(f"{session_id}: WP1 完成标记与权威标签不一致")
        if (
            flat_sha256 != source_sha256
            or manifest_hashes[session_id] != source_sha256
            or snapshot_hashes[session_id] != source_sha256
        ):
            raise IndependentAuditError(f"{session_id}: 权威标签、平铺副本、manifest、预检快照四方哈希不一致")
        actual_hashes[session_id] = source_sha256
        if session_id in eval_set:
            authoritative[session_id] = _assemble_authoritative_labels(
                source,
                session_id,
                expected_n_steps=expected_n_steps,
                t1_delta_ms=int(mve["t1_delta_ms"]),
            )
    return {
        "split_path": str(split_path.resolve()),
        "source_root": str(source_root.resolve()),
        "flat_root": str(flat_root.resolve()),
        "n_train_sessions": len(train),
        "n_eval_sessions": len(evals),
        "n_label_hashes": len(actual_hashes),
        "eval_order_exact": True,
        "label_hashes_four_way_exact": True,
    }, authoritative


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
        "t1_delta_ms": int(mve["t1_delta_ms"]) == 400,  # PREREG #8 δ 读法裁决
        "layers": [int(value) for value in mve["layers"]] == expected_contract["layers"],
        "seeds": [int(value) for value in mve["seeds"]] == expected_contract["seeds"],
        "bootstrap_n": int(mve["bootstrap_n"]) == FROZEN_G1_BOOTSTRAP_N,
        "full_threshold": float(mve["g1_full_threshold"]) == 0.05,
        "backup_threshold": float(mve["g1_backup_threshold"]) == 0.02,
        "n_train_sessions": int(mve["n_sessions_train"]) == 160,
        "n_eval_sessions": int(mve["n_sessions_eval"]) == len(manifest["eval_session_order"]),
    }
    c_grid = {float(value) for value in mve["probe_c_grid"]}
    config_checks["best_c_grid"] = all(
        (item["best_c"] is None if item["kind"] in {"hazard", "acoustic_gru"} else float(item["best_c"]) in c_grid)
        for item in manifest["items"]
    )
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
    repository_sources = _verify_repository_sources(analysis)

    snapshot_path = manifest_path.parent / manifest["preflight_report_path"]
    snapshot = _load_json_object(snapshot_path, "预检快照")
    if snapshot.get("status") != "passed":
        raise IndependentAuditError("预检快照 status 未通过")
    expected_n_sessions = int(mve["n_sessions_train"]) + int(mve["n_sessions_eval"])
    expected_clock_hz = float(clocks["moshi"]["hz"])
    expected_versions = sorted(
        str(value)
        for value in snapshot.get(
            "expected_code_versions",
            [snapshot.get("expected_code_version")],
        )
        if value
    )
    observed_versions = sorted(
        str(value) for value in snapshot.get("observed_code_versions", [])
    )
    if len(observed_versions) == 1:
        version_set_id = observed_versions[0]
    else:
        version_set_id = "runner-set." + hashlib.sha256(
            json.dumps(observed_versions, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
    declared_snapshot_set_id = snapshot.get(
        "runner_code_version_set_id",
        snapshot.get("expected_code_version"),
    )
    declared_analysis_versions = analysis.get("runner_code_versions")
    analysis_versions_ok = (
        declared_analysis_versions == observed_versions
        and analysis.get("runner_code_version_set_id") == version_set_id
        if declared_analysis_versions is not None
        else len(observed_versions) == 1
        and manifest["runner_code_version"] == observed_versions[0]
    )
    preflight_checks = {
        "n_sessions": snapshot.get("n_sessions") == expected_n_sessions,
        "n_runs": snapshot.get("n_runs") == expected_n_sessions * 2,
        "layers": snapshot.get("layers") == expected_contract["layers"],
        "n_labels": snapshot.get("n_labels") == expected_n_sessions,
        "expected_n_steps": snapshot.get("expected_n_steps")
        == round(float(mve["max_minutes_per_session"]) * 60.0 * expected_clock_hz),
        "expected_clock_hz": snapshot.get("expected_clock_hz") == expected_clock_hz,
        "expected_max_seconds": snapshot.get("expected_max_seconds") == float(mve["max_minutes_per_session"]) * 60.0,
        "expected_mimi_chunk_seconds": snapshot.get("expected_mimi_chunk_seconds") == float(mve["mimi_chunk_seconds"]),
        "expected_forward_chunk_steps": snapshot.get("expected_forward_chunk_steps") == int(mve["forward_chunk_steps"]),
        "expected_text_mode": snapshot.get("expected_text_mode") == str(mve["text_mode"]),
        "enforce_code_version": snapshot.get("enforce_code_version") is True,
        "require_time_alignment": snapshot.get("require_time_alignment") is True,
        "runner_code_versions_allowed": set(observed_versions).issubset(
            set(expected_versions)
        ),
        "runner_code_version_set_id": (
            declared_snapshot_set_id
            == manifest["runner_code_version"]
            == version_set_id
        ),
        "analysis_runner_versions": analysis_versions_ok,
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
        "repository_sources": repository_sources,
        "preflight": preflight_checks,
        "preflight_status": snapshot["status"],
    }


def _load_and_align_items(
    manifest: dict[str, Any],
    manifest_path: Path,
    authoritative: AuthoritativeLabels | None = None,
) -> tuple[dict[ItemKey, PerSession], dict[str, Any]]:
    """读取 34 个 NPZ，并核对权威标签与同目标各项逐项全等。"""

    expected_sessions = sorted(str(value) for value in manifest["eval_session_order"])
    loaded: dict[ItemKey, PerSession] = {}
    authoritative_comparisons = 0
    for item in manifest["items"]:
        key = _item_key(item)
        per_session = read_per_session_npz(manifest_path.parent / item["path"])
        if list(per_session) != expected_sessions:
            raise IndependentAuditError(f"{key}: NPZ 会话顺序与排序后的评估会话不一致")
        if authoritative is not None:
            if set(authoritative) != set(expected_sessions):
                raise IndependentAuditError("权威评估标签的会话集合与 manifest 不一致")
            for session_id in expected_sessions:
                expected_labels = authoritative[session_id][key[0]]
                actual_labels = per_session[session_id][0]
                if expected_labels.shape != actual_labels.shape or not np.array_equal(expected_labels, actual_labels):
                    raise IndependentAuditError(f"{key}: 会话 {session_id} 的 NPZ 标签与 WP1 权威标签不全等")
                authoritative_comparisons += 1
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
                    raise IndependentAuditError(f"{key}: 会话 {session_id} 的标签与 {reference_key} 不完全一致")
                comparisons += 1
    return loaded, {
        "n_items": len(loaded),
        "n_eval_sessions": len(expected_sessions),
        "label_array_comparisons": comparisons,
        "authoritative_label_comparisons": authoritative_comparisons,
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
    balanced_mean, balanced_sd = _mean_and_sd([value["balanced_acc"] for value in by_seed.values()])
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
        baseline_aucs = {name: _seed_mean_auc(scores, take) for name, scores in baselines.items()}
        if probe_auc is None or any(value is None for value in baseline_aucs.values()):
            continue
        samples.append(probe_auc - max(baseline_aucs.values()))
    if not samples:
        raise IndependentAuditError("成对优势 bootstrap 没有有效双类样本")
    array = np.asarray(samples, dtype=np.float64)
    point_probe = _seed_mean_auc(probe, session_ids)
    point_baselines = {name: _seed_mean_auc(scores, session_ids) for name, scores in baselines.items()}
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
        "baseline_n_seeds": {name: len(scores) for name, scores in baselines.items()},
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


def _verify_runner_manifests(manifest: dict[str, Any]) -> dict[str, Any]:
    """独立打开 runs_root 下全部 400 个缓存 manifest，核验冻结文本流与时间对齐声明。

    这是 PREREG #7 "四层核验"中独立审计层的实体检查——不信任预检快照的转述，
    直接读缓存产物本身。runs_root 缺失即硬失败（正式审计必须在数据所在机器运行）。
    """

    runs_root = Path(str(manifest["runs_root"]))
    if not runs_root.is_dir():
        raise IndependentAuditError(f"runs_root 不存在，无法独立核验缓存 manifest：{runs_root}")
    sessions = sorted(str(value) for value in manifest["label_sha256"])
    expected_alignment = {
        "initial_token_position": 0,
        "acts_observed_through_offset_steps": 0,
    }
    allowed_versions = {
        str(value)
        for value in manifest.get("analysis_protocol", {}).get(
            "runner_code_versions",
            [],
        )
    }
    if not allowed_versions:
        allowed_versions = {str(manifest["runner_code_version"])}
    n_verified = 0
    version_counts: dict[str, int] = {}
    for session_id in sessions:
        for channel in (0, 1):
            path = runs_root / f"{session_id}_agent{channel}" / "manifest.json"
            payload = _load_json_object(path, "缓存 manifest")
            if payload.get("text_mode") != "greedy":
                raise IndependentAuditError(
                    f"{session_id}/agent{channel}: 缓存 text_mode={payload.get('text_mode')!r}，"
                    "冻结协议要求 greedy"
                )
            code_version = str(payload.get("code_version", ""))
            if code_version not in allowed_versions:
                raise IndependentAuditError(
                    f"{session_id}/agent{channel}: 缓存 code_version={code_version!r}，"
                    f"未登记在 {sorted(allowed_versions)!r}"
                )
            declared = payload.get("extra", {}).get("execution", {}).get("time_alignment")
            if declared != expected_alignment:
                raise IndependentAuditError(
                    f"{session_id}/agent{channel}: 缓存 time_alignment={declared!r}，"
                    f"期望 {expected_alignment!r}"
                )
            version_counts[code_version] = version_counts.get(code_version, 0) + 1
            n_verified += 1
    return {
        "n_manifests": n_verified,
        "text_mode": "greedy",
        "time_alignment_verified": True,
        "runner_code_version_counts": dict(sorted(version_counts.items())),
    }


def _verify_descriptive(summary: dict[str, Any]) -> dict[str, Any]:
    """核验描述性附表：仅作形状与非裁决性检查（其输入不进入分数包，不做数值复算）。"""

    descriptive = summary.get("descriptive")
    if not isinstance(descriptive, dict):
        raise IndependentAuditError("summary.descriptive 缺失")
    note = descriptive.get("note")
    if not isinstance(note, str) or "非 G1 判据" not in note:
        raise IndependentAuditError("descriptive.note 必须声明其非判据地位")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if "verdict" in value:
                raise IndependentAuditError(f"描述性附表不得含裁决字段：{path}.verdict")
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(descriptive, "descriptive")
    entries = descriptive.get("T1")
    if not isinstance(entries, dict):
        raise IndependentAuditError("descriptive.T1 必须为对象")
    for delta_key, entry in entries.items():
        if not isinstance(entry, dict) or int(entry.get("delta_ms", -1)) != int(delta_key):
            raise IndependentAuditError(f"descriptive.T1[{delta_key}] 的 delta_ms 不一致")
    return {"n_entries": len(entries), "non_decisional": True}


def _declared_selection(summary: dict[str, Any]) -> dict[str, int]:
    """核验 per_target.selection 声明的内部一致性并返回声明的最优层。

    嵌套选择在 inner_val（probe_train 内层划分）上进行，其原始分数不进入 34 项
    分数包；本审计验证 (a) 声明的 best_layer 是声明的逐层选择 AUC 表的 argmax
    （并列取较小层号，与正式链一致），(b) 层集合与冻结层一致，(c) best_layer 字段
    与 selection 声明互相一致。选择过程本身由确定性种子与分析代码指纹约束，可整链
    复跑复现。
    """
    per_target = summary.get("per_target")
    if not isinstance(per_target, dict):
        raise IndependentAuditError("summary.per_target 缺失")
    declared: dict[str, int] = {}
    for target in FROZEN_G1_TARGETS:
        result = per_target.get(target)
        if not isinstance(result, dict) or not isinstance(result.get("selection"), dict):
            raise IndependentAuditError(f"{target}: summary 缺少 selection 声明")
        selection = result["selection"]
        try:
            best_layer = int(selection["best_layer"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IndependentAuditError(f"{target}: selection.best_layer 无效") from exc
        table = selection.get("selection_auc_mean_by_layer")
        if not isinstance(table, dict):
            raise IndependentAuditError(f"{target}: selection 缺少逐层选择 AUC 表")
        try:
            by_layer = {int(layer): value for layer, value in table.items()}
        except (TypeError, ValueError) as exc:
            raise IndependentAuditError(f"{target}: 选择 AUC 表层键无效") from exc
        if sorted(by_layer) != sorted(FROZEN_G1_LAYERS):
            raise IndependentAuditError(
                f"{target}: 选择 AUC 表层集合 {sorted(by_layer)} 与冻结层不一致"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in by_layer.values()
        ):
            raise IndependentAuditError(f"{target}: 选择 AUC 表含非法值")
        argmax_layer = max(sorted(by_layer), key=lambda ell: by_layer[ell])
        if argmax_layer != best_layer:
            raise IndependentAuditError(
                f"{target}: 声明 best_layer=L{best_layer} 与选择表 argmax=L{argmax_layer} 不一致"
            )
        if result.get("best_layer") != best_layer:
            raise IndependentAuditError(f"{target}: best_layer 字段与 selection 声明不一致")
        layer_summary = result.get("layer_summary")
        if isinstance(layer_summary, dict):
            for layer, declared_mean in by_layer.items():
                entry = layer_summary.get(str(layer), {})
                mean_value = entry.get("selection_auc_mean")
                by_seed = entry.get("selection_auc_by_seed")
                if mean_value is None or not isinstance(by_seed, dict) or not by_seed:
                    raise IndependentAuditError(f"{target}/L{layer}: layer_summary 缺少选择 AUC 证据")
                seed_values = [float(value) for value in by_seed.values()]
                if not np.isclose(float(mean_value), float(np.mean(seed_values)), rtol=0.0, atol=FLOAT_ATOL):
                    raise IndependentAuditError(
                        f"{target}/L{layer}: selection_auc_mean 与逐种子均值不一致"
                    )
                if not np.isclose(float(mean_value), float(declared_mean), rtol=0.0, atol=FLOAT_ATOL):
                    raise IndependentAuditError(
                        f"{target}/L{layer}: 选择表与 layer_summary 的选择 AUC 不一致"
                    )
        declared[target] = best_layer
    return declared


def _verify_protocol(
    summary: dict[str, Any],
    mve: dict[str, Any],
    *,
    split_path: Path,
) -> dict[str, Any]:
    """核对 summary.protocol 的文本流模式、时间对齐声明与嵌套选择划分。"""

    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise IndependentAuditError("summary.protocol 缺失")
    if protocol.get("ablation") is not None:
        raise IndependentAuditError("正式 G1 summary 不得为消融变体")
    checks: dict[str, bool] = {
        "text_mode": protocol.get("text_mode") == str(mve["text_mode"]) == "greedy",
        "time_alignment": protocol.get("time_alignment") == EXPECTED_TIME_ALIGNMENT,
    }
    split = _load_json_object(split_path, "冻结划分")
    train = [str(value) for value in split["splits"]["probe_train"][: int(mve["n_sessions_train"])]]
    expected_inner_val = train[: int(mve["inner_val_sessions"])]
    nested = protocol.get("nested_selection")
    checks["nested_selection_inner_val"] = (
        isinstance(nested, dict)
        and nested.get("inner_val_sessions") == expected_inner_val
        and nested.get("n_inner_val") == len(expected_inner_val)
        and nested.get("n_inner_train") == len(train) - len(expected_inner_val)
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise IndependentAuditError(f"summary.protocol 校验失败：{failed}")
    return checks


def recompute_summary(
    manifest: dict[str, Any],
    loaded: dict[ItemKey, PerSession],
    declared_selection: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """完全独立复算 summary 的 ``overall`` 与 ``per_target``。

    最优层取声明值（其与选择表的 argmax 一致性已由 ``_declared_selection`` 验证；
    选择输入在 inner_val 上，按协议不进入 probe_val 分数包）。
    """

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
            probes = {seed: loaded[(target, "probe", layer, seed)] for seed in FROZEN_G1_SEEDS}
            probes_by_layer[layer] = probes
            per_seed = {seed: _metrics(probes[seed]) for seed in FROZEN_G1_SEEDS}
            auc_mean, auc_sd = _mean_and_sd([per_seed[seed]["auc"] for seed in FROZEN_G1_SEEDS])
            auprc_mean, auprc_sd = _mean_and_sd([per_seed[seed]["auprc"] for seed in FROZEN_G1_SEEDS])
            balanced_mean, balanced_sd = _mean_and_sd([per_seed[seed]["balanced_acc"] for seed in FROZEN_G1_SEEDS])
            layer_summary[str(layer)] = {
                "n_seeds": len(FROZEN_G1_SEEDS),
                "auc_mean": auc_mean,
                "auc_sd": auc_sd,
                "auprc_mean": auprc_mean,
                "auprc_sd": auprc_sd,
                "balanced_acc_mean": balanced_mean,
                "balanced_acc_sd": balanced_sd,
                "auc_by_seed": {str(seed): per_seed[seed]["auc"] for seed in FROZEN_G1_SEEDS},
                "best_c_by_seed": {
                    str(seed): float(item_metadata[(target, "probe", layer, seed)]["best_c"])
                    for seed in FROZEN_G1_SEEDS
                },
            }

        best_layer = declared_selection[target]
        selected_probe = probes_by_layer[best_layer]
        baselines: dict[str, SeededPerSession] = {
            "hazard": {0: loaded[(target, "hazard", None, 0)]},
            "mimi": {seed: loaded[(target, "mimi", None, seed)] for seed in FROZEN_G1_SEEDS},
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
            "baseline_metrics": {name: _seed_mean_metrics(scores) for name, scores in baselines.items()},
            "probe_ci": probe_ci,
            "shuffled_auc": shuffled_metrics["auc_mean"],
            "shuffled_auc_sd": shuffled_metrics["auc_sd"],
            "shuffled_n_seeds": shuffled_metrics["n_seeds"],
            "ci_scope": CI_SCOPE,
            "selection_disjoint_from_report": True,
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
        Path(manifest_override) if manifest_override is not None else Path(str(score_bundle["manifest_path"]))
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
    *,
    data_root_path: str | Path | None = None,
    split_path: str | Path = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """执行独立审计；校验失败时抛出异常，summary 数值差异写入返回值。"""

    summary_file = Path(summary_path).resolve()
    summary = _load_json_object(summary_file, "G1 summary")
    _validate_generated_at(summary)
    expected_top_keys = {
        "generated_at",
        "overall",
        "per_target",
        "score_bundle",
        "protocol",
        "descriptive",
    }
    if set(summary) != expected_top_keys:
        raise IndependentAuditError(f"summary 顶层字段集合不一致：{sorted(summary)}")
    resolved_manifest, manifest, manifest_sha256, expected_bundle = _resolve_manifest(
        summary,
        manifest_path,
    )
    metadata_checks = _verify_metadata(manifest, resolved_manifest)
    authoritative_root = Path(data_root_path).resolve() if data_root_path is not None else data_root().resolve()
    if Path(manifest["bundle_path"]["relative_base"]).resolve() != authoritative_root:
        raise IndependentAuditError("manifest.bundle_path.relative_base 与权威 data_root 不一致")
    snapshot = _load_json_object(
        resolved_manifest.parent / manifest["preflight_report_path"],
        "预检快照",
    )
    mve = load_config("grids")["mve"]
    expected_n_steps = round(
        float(mve["max_minutes_per_session"]) * 60.0 * float(load_config("grids")["clocks"]["moshi"]["hz"])
    )
    frozen_checks, authoritative = _verify_authoritative_labels(
        manifest,
        snapshot,
        split_path=Path(split_path),
        data_root_path=authoritative_root,
        mve=mve,
        expected_n_steps=expected_n_steps,
    )
    metadata_checks["frozen_inputs"] = frozen_checks
    loaded, alignment_checks = _load_and_align_items(
        manifest,
        resolved_manifest,
        authoritative,
    )
    protocol_checks = _verify_protocol(summary, mve, split_path=Path(split_path))
    metadata_checks["protocol"] = protocol_checks
    metadata_checks["descriptive"] = _verify_descriptive(summary)
    metadata_checks["runner_manifests"] = _verify_runner_manifests(manifest)
    declared_selection = _declared_selection(summary)
    recomputed, bootstrap_checks = recompute_summary(manifest, loaded, declared_selection)
    expected_summary = {
        **recomputed,
        "score_bundle": expected_bundle,
    }
    # selection 证据（inner_val 上产生）已由 _declared_selection/_verify_protocol 单独
    # 核验，其原始输入按协议不进入 probe_val 分数包，因此从数值比对中剥离。
    selection_evidence_keys = {"selection_auc_mean", "selection_auc_by_seed"}
    actual_summary = {
        "overall": summary["overall"],
        "per_target": {
            target: {
                key: (
                    {
                        layer: {
                            field: field_value
                            for field, field_value in entry.items()
                            if field not in selection_evidence_keys
                        }
                        for layer, entry in value.items()
                    }
                    if key == "layer_summary"
                    else value
                )
                for key, value in result.items()
                if key != "selection"
            }
            for target, result in summary["per_target"].items()
        },
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
    data_root_path: str | Path | None = None,
    split_path: str | Path = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """成功或失败均原子发布独立审计报告。"""

    output = Path(output_path)
    generated_at = datetime.now(UTC).isoformat()
    try:
        details = audit_summary(
            summary_path,
            manifest_path,
            data_root_path=data_root_path,
            split_path=split_path,
        )
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
            "manifest_path": (None if manifest_path is None else str(Path(manifest_path).resolve())),
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
            f"独立复算通过：{overall['verdict']}，{overall['decisive_target']} 优势 {overall['advantage_point']:+.6f}"
        )
        return
    print(f"独立复算失败：{result.get('error', '未知错误')}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
