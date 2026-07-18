"""WP1 事件标签严格审计脚本的临时世界测试。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SHA256 = "a" * 64


def _load_module():
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "wp1_audit_events_test",
        scripts / "wp1_audit_events.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_hash(splits: dict[str, list[str]]) -> str:
    payload = json.dumps(splits, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"target": "T1", "agent_channel": 0, "step": 0, "t": 0.08, "label": 0, "delta_ms": 240.0},
            {"target": "T2", "agent_channel": 0, "step": 1, "t": 0.16, "label": 1, "delta_ms": None},
            {"target": "T3", "agent_channel": 0, "step": 2, "t": 0.24, "label": 2, "delta_ms": None},
            {"target": "T4", "agent_channel": 1, "step": 3, "t": 0.32, "label": 1, "delta_ms": None},
            {"target": "T5", "agent_channel": 1, "step": 4, "t": 0.40, "label": 3, "delta_ms": None},
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_session(
    events_root: Path,
    audio_root: Path,
    sid: str,
    labels: pd.DataFrame,
) -> None:
    session_audio = audio_root / sid
    session_audio.mkdir(parents=True, exist_ok=True)
    for channel in (0, 1):
        (session_audio / f"audio_ch{channel}.wav").write_bytes(
            (f"{sid}-{channel}-音频").encode()
        )

    events_path = events_root / f"{sid}.events.parquet"
    labels_path = events_root / f"{sid}.labels.parquet"
    pd.DataFrame({"kind": ["turnend"], "t": [0.4]}).to_parquet(events_path)
    labels.to_parquet(labels_path)
    source_audio = {}
    for channel in (0, 1):
        path = session_audio / f"audio_ch{channel}.wav"
        stat = path.stat()
        source_audio[path.name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    marker = {
        "schema_version": 1,
        "session": sid,
        "input": {
            "settings_sha256": "b" * 64,
            "event_pipeline_code_sha256": PIPELINE_SHA256,
            "source_audio": source_audio,
        },
        "outputs": {
            "events": {
                "name": events_path.name,
                "size": events_path.stat().st_size,
                "sha256": _sha256(events_path),
            },
            "labels": {
                "name": labels_path.name,
                "size": labels_path.stat().st_size,
                "sha256": _sha256(labels_path),
            },
        },
    }
    (events_root / f"{sid}.complete.json").write_text(
        json.dumps(marker, ensure_ascii=False),
        encoding="utf-8",
    )


def _world(tmp_path: Path):
    events_root = tmp_path / "events"
    old_root = tmp_path / "old"
    audio_root = tmp_path / "audio"
    events_root.mkdir()
    old_root.mkdir()
    audio_root.mkdir()

    splits = {
        "probe_train": ["s1", "train-unused"],
        "probe_val": ["s2", "val-unused"],
        "causal_eval": ["eval"],
    }
    split_payload = {
        "dataset": "candor",
        "seed": 1,
        "counts": {name: len(ids) for name, ids in splits.items()},
        "sha256": _split_hash(splits),
        "splits": splits,
    }
    split_path = tmp_path / "candor.json"
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")

    base = _labels()
    old = pd.concat(
        [base, base.loc[base["target"] == "T2"]],
        ignore_index=True,
    )
    for sid in ("s1", "s2"):
        _write_session(events_root, audio_root, sid, base.copy())
        old.to_parquet(old_root / f"{sid}.parquet")
    _write_session(events_root, audio_root, "extra", base.copy())
    return split_path, events_root, old_root, audio_root


def _audit(module, world):
    split_path, events_root, old_root, audio_root = world
    return module.audit_events(
        split_path=split_path,
        events_root=events_root,
        old_labels_root=old_root,
        audio_root=audio_root,
        train_limit=1,
        val_limit=1,
        expected_pipeline_sha256=PIPELINE_SHA256,
    )


def test_event_integrity_audit_passes_complete_world(tmp_path):
    module = _load_module()
    report = _audit(module, _world(tmp_path))

    assert report["status"] == "passed"
    assert report["artifact_sets"]["expected_frozen_ids"] == ["s1", "s2"]
    assert report["artifact_sets"]["extra_sample_ids"] == ["extra"]
    assert report["totals"]["new_duplicate_redundancy"] == 0
    assert report["target_counts"]["T2"] == {
        "old_rows": 4,
        "old_folded_rows": 2,
        "old_duplicate_redundancy": 2,
        "old_conflicting_keys": 0,
        "new_rows": 2,
    }
    assert all(item["status"] == "passed" for item in report["sessions"])


def test_event_integrity_audit_rejects_output_hash_tampering(tmp_path):
    module = _load_module()
    world = _world(tmp_path)
    labels_path = world[1] / "s1.labels.parquet"
    tampered = pd.read_parquet(labels_path)
    tampered.loc[tampered["target"] == "T4", "label"] = 0
    tampered.to_parquet(labels_path)

    report = _audit(module, world)

    assert report["status"] == "failed"
    assert any("s1: labels 输出哈希与完成标记不一致" in issue for issue in report["issues"])


def test_event_integrity_audit_rejects_duplicate_new_key(tmp_path):
    module = _load_module()
    world = _world(tmp_path)
    labels_path = world[1] / "s1.labels.parquet"
    labels = pd.read_parquet(labels_path)
    labels = pd.concat([labels, labels.loc[labels["target"] == "T2"]], ignore_index=True)
    labels.to_parquet(labels_path)
    _write_session(world[1], world[3], "s1", labels)

    report = _audit(module, world)

    assert report["status"] == "failed"
    assert report["totals"]["new_duplicate_redundancy"] == 1
    assert any("新标签唯一键重复" in issue for issue in report["issues"])


def test_event_integrity_audit_rejects_old_folded_mismatch(tmp_path):
    module = _load_module()
    world = _world(tmp_path)
    old_path = world[2] / "s2.parquet"
    old = pd.read_parquet(old_path)
    old.loc[old["target"] == "T3", "label"] = 1
    old.to_parquet(old_path)

    report = _audit(module, world)

    assert report["status"] == "failed"
    assert any("keep=first 折叠后不全等" in issue for issue in report["issues"])


def test_event_integrity_audit_rejects_conflicting_old_duplicate_key(tmp_path):
    module = _load_module()
    world = _world(tmp_path)
    old_path = world[2] / "s1.parquet"
    old = pd.read_parquet(old_path)
    duplicate_t2 = old.index[old["target"] == "T2"][-1]
    old.loc[duplicate_t2, "label"] = 0
    old.to_parquet(old_path)

    report = _audit(module, world)

    assert report["status"] == "failed"
    assert report["totals"]["old_conflicting_keys"] == 1
    assert report["target_counts"]["T2"]["old_conflicting_keys"] == 1
    s1 = next(item for item in report["sessions"] if item["session"] == "s1")
    assert s1["old_labels"]["strict_equal"] is True
    assert any("旧标签同一唯一键存在 1 个完整行冲突" in issue for issue in report["issues"])


def test_event_integrity_audit_rejects_frozen_set_error(tmp_path):
    module = _load_module()
    world = _world(tmp_path)
    (world[2] / "s2.parquet").replace(world[2] / "wrong.parquet")

    report = _audit(module, world)

    assert report["status"] == "failed"
    assert report["artifact_sets"]["old_baseline_count"] == 2
    assert any("旧标签基线集合与划分前缀不一致" in issue for issue in report["issues"])
