"""WP1：严格审计 CANDOR 事件与标签产物，不修改任何输入数据。

默认审计冻结划分的 ``probe_train`` 前 160 个会话与 ``probe_val`` 前 40 个会话。
新事件目录中位于该集合之外的完整会话按额外样例审计，但不参与旧标签基线比较。

用法：
  uv run python scripts/wp1_audit_events.py
  uv run python scripts/wp1_audit_events.py --events-root D:\\...\\events\\candor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT, REPORTS_DIR
from pandas.testing import assert_frame_equal
from wp1_run_events import _event_pipeline_code_sha256

from floor_circuit.config import data_root
from floor_circuit.data.splits import load_split
from floor_circuit.events.labels import LABEL_KEY_COLUMNS

TARGETS = ("T1", "T2", "T3", "T4", "T5")
KEY_COLUMNS = list(LABEL_KEY_COLUMNS)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(splits: dict[str, list[str]]) -> str:
    payload = json.dumps(splits, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_ids(root: Path, suffix: str) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.name.removesuffix(suffix)
        for path in root.glob(f"*{suffix}")
        if path.is_file() and path.name.endswith(suffix)
    }


def _temporary_files(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".tmp" in path.name.lower()
    )


def _duplicate_stats(frame: pd.DataFrame) -> dict[str, int]:
    duplicated_all = frame.duplicated(KEY_COLUMNS, keep=False)
    duplicated_after_first = frame.duplicated(KEY_COLUMNS, keep="first")
    return {
        "duplicate_rows": int(duplicated_all.sum()),
        "duplicate_redundancy": int(duplicated_after_first.sum()),
        "duplicate_keys": int(frame.loc[duplicated_all, KEY_COLUMNS].drop_duplicates().shape[0]),
    }


def _conflicting_duplicate_keys(frame: pd.DataFrame) -> dict:
    """统计同一唯一键下完整行不全等的旧标签冲突，空值按 pandas 语义等价。"""
    duplicate_rows = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False)]
    by_target = {target: 0 for target in TARGETS}
    samples: list[str] = []
    total = 0
    for key, group in duplicate_rows.groupby(
        KEY_COLUMNS,
        dropna=False,
        sort=False,
    ):
        if len(group.drop_duplicates(keep="first")) <= 1:
            continue
        total += 1
        key_tuple = key if isinstance(key, tuple) else (key,)
        target = str(key_tuple[0])
        if target in by_target:
            by_target[target] += 1
        if len(samples) < 10:
            samples.append(repr(key_tuple))
    return {
        "old_conflicting_keys": total,
        "old_conflicting_keys_by_target": by_target,
        "old_conflicting_key_samples": samples,
    }


def _target_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "target" not in frame.columns:
        return {target: 0 for target in TARGETS}
    counts = frame["target"].value_counts(dropna=False)
    return {target: int(counts.get(target, 0)) for target in TARGETS}


def _sort_labels(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        KEY_COLUMNS,
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)


def _empty_target_totals() -> dict[str, dict[str, int]]:
    return {
        target: {
            "old_rows": 0,
            "old_folded_rows": 0,
            "old_duplicate_redundancy": 0,
            "old_conflicting_keys": 0,
            "new_rows": 0,
        }
        for target in TARGETS
    }


def _add_issue(container: dict, message: str) -> None:
    container["issues"].append(message)


def _read_parquet(path: Path, sid: str, kind: str, session_report: dict) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        _add_issue(session_report, f"{sid}: {kind} parquet 无法读取（{exc!r}）")
        return None


def _check_output(
    sid: str,
    kind: str,
    path: Path,
    marker: dict | None,
    session_report: dict,
) -> tuple[dict, pd.DataFrame | None]:
    result: dict = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        _add_issue(session_report, f"{sid}: 缺少 {kind} 输出 {path.name}")
        return result, None

    stat = path.stat()
    digest = _sha256_file(path)
    result.update({"size": int(stat.st_size), "sha256": digest})
    declared = None
    try:
        if marker is not None:
            declared = marker["outputs"][kind]
    except (KeyError, TypeError):
        _add_issue(session_report, f"{sid}: 完成标记缺少 outputs.{kind}")
    if declared is not None and not isinstance(declared, dict):
        _add_issue(session_report, f"{sid}: 完成标记 outputs.{kind} 不是对象")
        declared = None
    if declared is not None:
        result["declared"] = declared
        if declared.get("name") != path.name:
            _add_issue(session_report, f"{sid}: {kind} 输出文件名与完成标记不一致")
        if declared.get("size") != stat.st_size:
            _add_issue(session_report, f"{sid}: {kind} 输出大小与完成标记不一致")
        if declared.get("sha256") != digest:
            _add_issue(session_report, f"{sid}: {kind} 输出哈希与完成标记不一致")

    frame = _read_parquet(path, sid, kind, session_report)
    if frame is not None:
        result["rows"] = len(frame)
        result["columns"] = list(frame.columns)
    return result, frame


def _check_audio(
    sid: str,
    audio_root: Path,
    marker: dict | None,
    session_report: dict,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    try:
        declared_audio = marker["input"]["source_audio"] if marker is not None else {}
    except (KeyError, TypeError):
        declared_audio = {}
        _add_issue(session_report, f"{sid}: 完成标记缺少 input.source_audio")
    if not isinstance(declared_audio, dict):
        declared_audio = {}
        _add_issue(session_report, f"{sid}: 完成标记 input.source_audio 不是对象")

    for channel in (0, 1):
        name = f"audio_ch{channel}.wav"
        path = audio_root / sid / name
        item: dict = {"path": str(path), "exists": path.is_file()}
        result[name] = item
        if not path.is_file():
            _add_issue(session_report, f"{sid}: 缺少源音频 {name}")
            continue
        stat = path.stat()
        item.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        declared = declared_audio.get(name)
        item["declared"] = declared
        if not isinstance(declared, dict):
            _add_issue(session_report, f"{sid}: 完成标记缺少源音频元数据 {name}")
            continue
        if declared.get("size") != stat.st_size:
            _add_issue(session_report, f"{sid}: 源音频 {name} 大小与完成标记不一致")
        if declared.get("mtime_ns") != stat.st_mtime_ns:
            _add_issue(session_report, f"{sid}: 源音频 {name} 修改时间与完成标记不一致")
    extra_names = sorted(set(declared_audio) - {"audio_ch0.wav", "audio_ch1.wav"})
    if extra_names:
        _add_issue(session_report, f"{sid}: 完成标记含额外源音频项 {extra_names}")
    return result


def _check_label_keys(
    sid: str,
    frame: pd.DataFrame,
    session_report: dict,
) -> dict | None:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        _add_issue(session_report, f"{sid}: 新标签缺少唯一键列 {missing}")
        return None
    stats = _duplicate_stats(frame)
    if stats["duplicate_redundancy"] != 0:
        _add_issue(
            session_report,
            f"{sid}: 新标签唯一键重复 {stats['duplicate_keys']} 个，冗余 {stats['duplicate_redundancy']} 行",
        )
    unexpected = sorted(
        str(value)
        for value in frame["target"].dropna().unique()
        if str(value) not in TARGETS
    )
    if unexpected:
        _add_issue(session_report, f"{sid}: 新标签含未知目标 {unexpected}")
    stats["target_counts"] = _target_counts(frame)
    return stats


def _compare_old_labels(
    sid: str,
    new_frame: pd.DataFrame,
    old_path: Path,
    session_report: dict,
) -> tuple[dict, pd.DataFrame | None]:
    result: dict = {"path": str(old_path), "exists": old_path.is_file(), "strict_equal": False}
    if not old_path.is_file():
        _add_issue(session_report, f"{sid}: 缺少旧标签基线 {old_path.name}")
        return result, None
    old_frame = _read_parquet(old_path, sid, "旧标签", session_report)
    if old_frame is None:
        return result, None
    result["rows"] = len(old_frame)
    result["columns"] = list(old_frame.columns)

    old_missing = [column for column in KEY_COLUMNS if column not in old_frame.columns]
    new_missing = [column for column in KEY_COLUMNS if column not in new_frame.columns]
    if old_missing or new_missing:
        if old_missing:
            _add_issue(session_report, f"{sid}: 旧标签缺少唯一键列 {old_missing}")
        return result, old_frame

    old_stats = _duplicate_stats(old_frame)
    old_conflicts = _conflicting_duplicate_keys(old_frame)
    old_folded = old_frame.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    result.update(old_stats)
    result.update(old_conflicts)
    result["folded_rows"] = len(old_folded)
    result["target_counts"] = _target_counts(old_frame)
    result["folded_target_counts"] = _target_counts(old_folded)
    if old_conflicts["old_conflicting_keys"]:
        _add_issue(
            session_report,
            f"{sid}: 旧标签同一唯一键存在 {old_conflicts['old_conflicting_keys']} 个完整行冲突",
        )

    if list(new_frame.columns) != list(old_folded.columns):
        _add_issue(
            session_report,
            f"{sid}: 新旧折叠标签列或列序不同，新={list(new_frame.columns)}，旧={list(old_folded.columns)}",
        )
        return result, old_folded
    try:
        assert_frame_equal(
            _sort_labels(new_frame),
            _sort_labels(old_folded),
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_names=True,
        )
    except AssertionError as exc:
        detail = " ".join(str(exc).splitlines())
        if len(detail) > 500:
            detail = detail[:500] + "……"
        _add_issue(session_report, f"{sid}: 新标签与旧标签按唯一键 keep=first 折叠后不全等（{detail}）")
    else:
        result["strict_equal"] = True
    return result, old_folded


def _audit_session(
    sid: str,
    scope: str,
    events_root: Path,
    old_labels_root: Path,
    audio_root: Path,
    pipeline_sha256: str,
) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    session_report: dict = {
        "session": sid,
        "scope": scope,
        "status": "passed",
        "issues": [],
    }
    marker_path = events_root / f"{sid}.complete.json"
    marker: dict | None = None
    if not marker_path.is_file():
        _add_issue(session_report, f"{sid}: 缺少完成标记")
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _add_issue(session_report, f"{sid}: 完成标记无法读取（{exc!r}）")
        if marker is not None:
            if not isinstance(marker, dict):
                _add_issue(session_report, f"{sid}: 完成标记顶层不是对象")
                marker = None
            elif marker.get("schema_version") != 1:
                _add_issue(session_report, f"{sid}: 完成标记 schema_version 不是 1")
        if marker is not None:
            if marker.get("session") != sid:
                _add_issue(session_report, f"{sid}: 完成标记会话标识不一致")
            marker_input = marker.get("input")
            marker_code = (
                marker_input.get("event_pipeline_code_sha256")
                if isinstance(marker_input, dict)
                else None
            )
            session_report["marker_pipeline_sha256"] = marker_code
            if not isinstance(marker_input, dict):
                _add_issue(session_report, f"{sid}: 完成标记 input 不是对象")
            if marker_code != pipeline_sha256:
                _add_issue(session_report, f"{sid}: 完成标记事件管线代码指纹不是当前值")

    session_report["audio"] = _check_audio(sid, audio_root, marker, session_report)
    events_info, _events_frame = _check_output(
        sid,
        "events",
        events_root / f"{sid}.events.parquet",
        marker,
        session_report,
    )
    labels_info, labels_frame = _check_output(
        sid,
        "labels",
        events_root / f"{sid}.labels.parquet",
        marker,
        session_report,
    )
    session_report["events"] = events_info
    session_report["labels"] = labels_info

    if labels_frame is not None:
        key_stats = _check_label_keys(sid, labels_frame, session_report)
        if key_stats is not None:
            session_report["labels"].update(key_stats)

    old_folded: pd.DataFrame | None = None
    if scope == "frozen" and labels_frame is not None:
        old_info, old_folded = _compare_old_labels(
            sid,
            labels_frame,
            old_labels_root / f"{sid}.parquet",
            session_report,
        )
        session_report["old_labels"] = old_info

    session_report["status"] = "passed" if not session_report["issues"] else "failed"
    return session_report, labels_frame, old_folded


def audit_events(
    *,
    split_path: str | Path,
    events_root: str | Path,
    old_labels_root: str | Path,
    audio_root: str | Path,
    train_limit: int = 160,
    val_limit: int = 40,
    expected_pipeline_sha256: str | None = None,
) -> dict:
    """执行只读审计并返回可直接序列化的报告。"""
    split_path = Path(split_path)
    events_root = Path(events_root)
    old_labels_root = Path(old_labels_root)
    audio_root = Path(audio_root)
    report: dict = {
        "status": "failed",
        "event_pipeline_code_sha256": None,
        "roots": {
            "events": str(events_root),
            "old_labels": str(old_labels_root),
            "audio": str(audio_root),
        },
        "split": {
            "path": str(split_path),
            "probe_train_limit": train_limit,
            "probe_val_limit": val_limit,
        },
        "artifact_sets": {},
        "temporary_files": {},
        "totals": {
            "events_rows": 0,
            "frozen_new_label_rows": 0,
            "frozen_old_label_rows": 0,
            "frozen_old_folded_label_rows": 0,
            "extra_new_label_rows": 0,
            "new_duplicate_keys": 0,
            "new_duplicate_redundancy": 0,
            "old_duplicate_keys": 0,
            "old_duplicate_redundancy": 0,
            "old_conflicting_keys": 0,
        },
        "target_counts": _empty_target_totals(),
        "extra_target_counts": {target: 0 for target in TARGETS},
        "sessions": [],
        "issues": [],
    }

    for name, root in (
        ("events", events_root),
        ("old_labels", old_labels_root),
        ("audio", audio_root),
    ):
        if not root.is_dir():
            _add_issue(report, f"{name} 根目录不存在：{root}")

    try:
        pipeline_sha256 = expected_pipeline_sha256 or _event_pipeline_code_sha256()
    except Exception as exc:
        pipeline_sha256 = ""
        _add_issue(report, f"当前事件管线代码指纹计算失败（{exc!r}）")
    report["event_pipeline_code_sha256"] = pipeline_sha256

    expected_ids: list[str] = []
    try:
        split_bytes = split_path.read_bytes()
        split_payload = load_split(split_path)
        split_groups = split_payload["splits"]
        train_all = list(split_groups["probe_train"])
        val_all = list(split_groups["probe_val"])
        train_ids = train_all[:train_limit]
        val_ids = val_all[:val_limit]
        expected_ids = train_ids + val_ids
        computed_payload_sha256 = _payload_hash(split_groups)
        report["split"].update(
            {
                "file_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "embedded_sha256": split_payload.get("sha256"),
                "computed_payload_sha256": computed_payload_sha256,
                "dataset": split_payload.get("dataset"),
                "probe_train_ids": train_ids,
                "probe_val_ids": val_ids,
                "expected_session_ids": expected_ids,
            }
        )
        if split_payload.get("dataset") != "candor":
            _add_issue(report, f"划分数据集不是 candor：{split_payload.get('dataset')!r}")
        declared_counts = split_payload.get("counts", {})
        actual_counts = {name: len(ids) for name, ids in split_groups.items()}
        if declared_counts != actual_counts:
            _add_issue(report, f"划分 counts 与实际列表长度不一致：声明={declared_counts}，实际={actual_counts}")
        if len(train_ids) != train_limit or len(val_ids) != val_limit:
            _add_issue(
                report,
                f"划分前缀不足：probe_train={len(train_ids)}/{train_limit}，probe_val={len(val_ids)}/{val_limit}",
            )
        if len(expected_ids) != train_limit + val_limit:
            _add_issue(report, f"冻结前缀会话数为 {len(expected_ids)}，期望 {train_limit + val_limit}")
        duplicate_expected = len(expected_ids) - len(set(expected_ids))
        if duplicate_expected:
            _add_issue(report, f"冻结前缀含 {duplicate_expected} 个重复会话")
    except Exception as exc:
        _add_issue(report, f"冻结划分无法读取或校验（{exc!r}）")

    marker_ids = _artifact_ids(events_root, ".complete.json")
    event_ids = _artifact_ids(events_root, ".events.parquet")
    label_ids = _artifact_ids(events_root, ".labels.parquet")
    old_ids = _artifact_ids(old_labels_root, ".parquet")
    expected_set = set(expected_ids)
    extra_ids = sorted(marker_ids - expected_set)
    report["artifact_sets"] = {
        "expected_frozen_count": len(expected_ids),
        "expected_frozen_ids": expected_ids,
        "complete_marker_count": len(marker_ids),
        "complete_marker_ids": sorted(marker_ids),
        "events_count": len(event_ids),
        "events_ids": sorted(event_ids),
        "labels_count": len(label_ids),
        "labels_ids": sorted(label_ids),
        "old_baseline_count": len(old_ids),
        "old_baseline_ids": sorted(old_ids),
        "extra_sample_count": len(extra_ids),
        "extra_sample_ids": extra_ids,
    }
    if marker_ids != event_ids:
        _add_issue(
            report,
            "完成标记集合与 events 集合不同："
            f"缺失={sorted(marker_ids - event_ids)}，孤立={sorted(event_ids - marker_ids)}",
        )
    if marker_ids != label_ids:
        _add_issue(
            report,
            "完成标记集合与 labels 集合不同："
            f"缺失={sorted(marker_ids - label_ids)}，孤立={sorted(label_ids - marker_ids)}",
        )
    if expected_set - marker_ids:
        _add_issue(report, f"冻结集合缺少完成标记：{sorted(expected_set - marker_ids)}")
    if old_ids != expected_set:
        _add_issue(
            report,
            "旧标签基线集合与划分前缀不一致："
            f"缺失={sorted(expected_set - old_ids)}，额外={sorted(old_ids - expected_set)}",
        )

    event_temps = _temporary_files(events_root)
    old_temps = _temporary_files(old_labels_root)
    report["temporary_files"] = {
        "events_count": len(event_temps),
        "events": event_temps,
        "old_labels_count": len(old_temps),
        "old_labels": old_temps,
        "total": len(event_temps) + len(old_temps),
    }
    if event_temps or old_temps:
        _add_issue(report, f"发现 {len(event_temps) + len(old_temps)} 个临时文件")

    for sid in expected_ids:
        session_report, new_labels, _old_folded = _audit_session(
            sid,
            "frozen",
            events_root,
            old_labels_root,
            audio_root,
            pipeline_sha256,
        )
        report["sessions"].append(session_report)
        report["issues"].extend(session_report["issues"])
        report["totals"]["events_rows"] += int(session_report.get("events", {}).get("rows", 0))
        if new_labels is not None:
            report["totals"]["frozen_new_label_rows"] += len(new_labels)
            if all(column in new_labels.columns for column in KEY_COLUMNS):
                stats = _duplicate_stats(new_labels)
                report["totals"]["new_duplicate_keys"] += stats["duplicate_keys"]
                report["totals"]["new_duplicate_redundancy"] += stats["duplicate_redundancy"]
            new_counts = _target_counts(new_labels)
            for target in TARGETS:
                report["target_counts"][target]["new_rows"] += new_counts[target]
        old_info = session_report.get("old_labels", {})
        report["totals"]["frozen_old_label_rows"] += int(old_info.get("rows", 0))
        report["totals"]["frozen_old_folded_label_rows"] += int(old_info.get("folded_rows", 0))
        report["totals"]["old_duplicate_keys"] += int(old_info.get("duplicate_keys", 0))
        report["totals"]["old_duplicate_redundancy"] += int(old_info.get("duplicate_redundancy", 0))
        report["totals"]["old_conflicting_keys"] += int(old_info.get("old_conflicting_keys", 0))
        old_counts = old_info.get("target_counts", {})
        old_folded_counts = old_info.get("folded_target_counts", {})
        old_conflicts_by_target = old_info.get("old_conflicting_keys_by_target", {})
        for target in TARGETS:
            report["target_counts"][target]["old_rows"] += int(old_counts.get(target, 0))
            report["target_counts"][target]["old_folded_rows"] += int(old_folded_counts.get(target, 0))
            report["target_counts"][target]["old_duplicate_redundancy"] += int(
                old_counts.get(target, 0) - old_folded_counts.get(target, 0)
            )
            report["target_counts"][target]["old_conflicting_keys"] += int(
                old_conflicts_by_target.get(target, 0)
            )

    for sid in extra_ids:
        session_report, new_labels, _old_folded = _audit_session(
            sid,
            "extra_sample",
            events_root,
            old_labels_root,
            audio_root,
            pipeline_sha256,
        )
        report["sessions"].append(session_report)
        report["issues"].extend(session_report["issues"])
        report["totals"]["events_rows"] += int(session_report.get("events", {}).get("rows", 0))
        if new_labels is not None:
            report["totals"]["extra_new_label_rows"] += len(new_labels)
            if all(column in new_labels.columns for column in KEY_COLUMNS):
                stats = _duplicate_stats(new_labels)
                report["totals"]["new_duplicate_keys"] += stats["duplicate_keys"]
                report["totals"]["new_duplicate_redundancy"] += stats["duplicate_redundancy"]
            counts = _target_counts(new_labels)
            for target in TARGETS:
                report["extra_target_counts"][target] += counts[target]

    report["n_sessions_audited"] = len(report["sessions"])
    report["n_frozen_sessions_audited"] = sum(
        item["scope"] == "frozen" for item in report["sessions"]
    )
    report["n_extra_sessions_audited"] = sum(
        item["scope"] == "extra_sample" for item in report["sessions"]
    )
    report["n_session_failures"] = sum(item["status"] == "failed" for item in report["sessions"])
    if (
        report["totals"]["frozen_new_label_rows"]
        != report["totals"]["frozen_old_folded_label_rows"]
    ):
        _add_issue(
            report,
            "冻结新标签总行数与旧标签折叠总行数不相等："
            f"新={report['totals']['frozen_new_label_rows']}，"
            f"旧折叠={report['totals']['frozen_old_folded_label_rows']}",
        )
    for target in TARGETS:
        counts = report["target_counts"][target]
        if counts["new_rows"] != counts["old_folded_rows"]:
            _add_issue(
                report,
                f"{target} 冻结新标签行数与旧标签折叠行数不相等："
                f"新={counts['new_rows']}，旧折叠={counts['old_folded_rows']}",
            )
    report["status"] = "passed" if not report["issues"] else "failed"
    return report


def _write_report(path_arg: str | Path, report: dict) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = REPORTS_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        **report,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def main() -> None:
    root = data_root()
    parser = argparse.ArgumentParser(description="只读严格审计 WP1 CANDOR 事件与标签")
    parser.add_argument(
        "--split-file",
        default=str(REPO_ROOT / "configs" / "splits" / "candor.json"),
    )
    parser.add_argument("--events-root", default=str(root / "events" / "candor"))
    parser.add_argument(
        "--old-labels-root",
        default=str(root / "events" / "candor_labels_flat"),
    )
    parser.add_argument("--audio-root", default=str(root / "candor_extracted"))
    parser.add_argument("--report", default="wp1_events_integrity_audit.json")
    args = parser.parse_args()

    try:
        report = audit_events(
            split_path=args.split_file,
            events_root=args.events_root,
            old_labels_root=args.old_labels_root,
            audio_root=args.audio_root,
        )
    except Exception as exc:
        report = {
            "status": "failed",
            "issues": [f"审计发生未捕获异常（{exc!r}）"],
        }
    report_path = _write_report(args.report, report)
    print(f"[report] {report_path}")
    if report["status"] != "passed":
        for issue in report.get("issues", [])[:40]:
            print(f"- {issue}", file=sys.stderr)
        if len(report.get("issues", [])) > 40:
            print(f"- ……另有 {len(report['issues']) - 40} 项", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"事件标签审计通过：冻结会话 {report['n_frozen_sessions_audited']}，"
        f"额外样例 {report['n_extra_sessions_audited']}，临时文件 0"
    )


if __name__ == "__main__":
    main()
