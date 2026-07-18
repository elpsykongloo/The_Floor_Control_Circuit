"""G1 独立复算脚本的 34 项合成世界与篡改护栏。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from floor_circuit.mve.artifacts import (
    ANALYSIS_SOURCE_PATHS,
    ScoreBundleWriter,
    expected_item_keys,
    read_per_session_npz,
    sha256_file,
    validate_score_bundle,
    write_per_session_npz_atomic,
)
from floor_circuit.mve.run import (
    ProbeCell,
    average_over_seeds,
    evaluate_target,
    overall_g1,
)
from floor_circuit.probes.stats import pooled_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    scripts = REPO_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "wp7_audit_mve_test",
        scripts / "wp7_audit_mve.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()
TARGETS = ["T1", "T4"]
LAYERS = [4, 12, 20, 28]
SEEDS = [0, 1, 2]
EVAL_SESSIONS = [f"eval-{index:03d}" for index in range(40)]
TRAIN_SESSIONS = [f"train-{index:03d}" for index in range(160)]


def _analysis_protocol() -> dict[str, Any]:
    content_sha256 = AUDIT._analysis_content_sha256(list(ANALYSIS_SOURCE_PATHS))
    repository_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    source_commits = {relative: AUDIT._source_commit(relative) for relative in ANALYSIS_SOURCE_PATHS}
    return {
        "bootstrap_n": 1000,
        "bootstrap_seed": 0,
        "code": {
            "version": f"{repository_head[:7]}+analysis.{content_sha256}",
            "repository_head": repository_head,
            "content_sha256": content_sha256,
            "sources": list(ANALYSIS_SOURCE_PATHS),
            "source_commits": source_commits,
        },
    }


def _labels(target: str) -> np.ndarray:
    if target == "T1":
        return np.asarray([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    return np.asarray([1, 0, 0, 1, 0, 1], dtype=np.int64)


def _strength(kind: str, layer: int | None, target: str) -> float:
    if kind == "probe":
        values = {4: 0.58, 12: 0.67, 20: 0.79, 28: 0.71}
        return values[int(layer)] - (0.02 if target == "T4" else 0.0)
    return {
        "mimi": 0.64,
        "hazard": 0.60,
        "acoustic_gru": 0.57,
    }[kind]


def _scores_for_item(
    target: str,
    kind: str,
    layer: int | None,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    labels = _labels(target)
    direction = 2.0 * labels.astype(np.float64) - 1.0
    strength = _strength(kind, layer, target)
    result = {}
    for session_index, session_id in enumerate(reversed(EVAL_SESSIONS)):
        row_index = np.arange(len(labels), dtype=np.float64)
        nuisance = 0.23 * np.sin(0.73 * row_index + 0.31 * session_index + 0.19 * seed)
        values = 0.5 + direction * (strength - 0.5) + nuisance
        result[session_id] = (
            labels.copy(),
            np.clip(values, 0.001, 0.999).astype(np.float64),
        )
    return result


def _official_summary() -> dict[str, Any]:
    """用正式链构造合成 summary，专门检验独立实现的逐字段等价性。"""

    per_target = {}
    for target in TARGETS:
        cells = []
        for layer in LAYERS:
            for seed in SEEDS:
                per_session = _scores_for_item(target, "probe", layer, seed)
                metrics = {**pooled_metrics(per_session), "best_c": 0.1}
                cells.append(
                    ProbeCell(
                        layer=layer,
                        target=target,
                        seed=seed,
                        metrics=metrics,
                        per_session=per_session,
                    )
                )
        baselines = {
            "hazard": _scores_for_item(target, "hazard", None, 0),
            "mimi": {seed: _scores_for_item(target, "mimi", None, seed) for seed in SEEDS},
            "acoustic_gru": _scores_for_item(target, "acoustic_gru", None, 0),
        }
        per_target[target] = evaluate_target(
            average_over_seeds(cells),
            baselines,
            n_boot=1000,
            full_thr=0.05,
            backup_thr=0.02,
            boot_seed=0,
        )
    return {
        "overall": overall_g1(per_target, full_thr=0.05, backup_thr=0.02),
        "per_target": per_target,
    }


def _preflight_payload(label_hashes: dict[str, str], runner_version: str) -> dict[str, Any]:
    return {
        "status": "passed",
        "n_sessions": 200,
        "n_runs": 400,
        "layers": LAYERS,
        "required_arrays_per_run": [
            "acts_L4",
            "acts_L12",
            "acts_L20",
            "acts_L28",
            "mimi_latent",
        ],
        "expected_n_steps": 7500,
        "expected_clock_hz": 12.5,
        "expected_code_version": runner_version,
        "expected_max_seconds": 600.0,
        "expected_mimi_chunk_seconds": 0.08,
        "expected_forward_chunk_steps": 128,
        "n_labels": 200,
        "label_sha256": label_hashes,
    }


def _write_frozen_inputs(root: Path) -> tuple[Path, dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    eval_order = list(reversed(EVAL_SESSIONS))
    split_path = root / "candor.json"
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    "probe_train": TRAIN_SESSIONS,
                    "probe_val": eval_order,
                    "causal_eval": [],
                }
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    t1 = _labels("T1")
    t4 = _labels("T4")
    frame = pd.DataFrame(
        {
            "target": ["T1"] * len(t1) + ["T4"] * len(t4),
            "agent_channel": [0] * len(t1) + [1] * len(t4),
            "step": [*range(len(t1)), *range(len(t4))],
            "delta_ms": [240] * len(t1) + [-1] * len(t4),
            "label": np.concatenate([t1, t4]),
        }
    )
    template = root / "label-template.parquet"
    frame.to_parquet(template, index=False)
    payload = template.read_bytes()
    template.unlink()
    digest = hashlib.sha256(payload).hexdigest()

    source_root = root / "data" / "events" / "candor"
    flat_root = root / "data" / "events" / "candor_labels_flat"
    source_root.mkdir(parents=True)
    flat_root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for session_id in [*TRAIN_SESSIONS, *eval_order]:
        source = source_root / f"{session_id}.labels.parquet"
        source.write_bytes(payload)
        (flat_root / f"{session_id}.parquet").write_bytes(payload)
        marker = {
            "schema_version": 1,
            "session": session_id,
            "outputs": {
                "labels": {
                    "name": source.name,
                    "size": len(payload),
                    "sha256": digest,
                }
            },
        }
        (source_root / f"{session_id}.complete.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        hashes[session_id] = digest
    return split_path, hashes


def _audit_paths(summary_path: Path) -> dict[str, Path]:
    return {
        "data_root_path": summary_path.parent / "data",
        "split_path": summary_path.parent / "candor.json",
    }


def _write_package(
    root: Path,
    *,
    recomputed: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    _split_path, label_hashes = _write_frozen_inputs(root)
    writer = ScoreBundleWriter.create(
        root / "data" / "mve" / "g1_scores",
        relative_base=root / "data",
        eval_session_order=list(reversed(EVAL_SESSIONS)),
        targets=TARGETS,
        layers=LAYERS,
        seeds=SEEDS,
        run_id="synthetic-run",
    )
    for target, kind, layer, seed in sorted(
        expected_item_keys(TARGETS, LAYERS, SEEDS),
        key=repr,
    ):
        writer.add(
            target=target,
            kind=kind,
            layer=layer,
            seed=seed,
            best_c=0.1 if kind in {"probe", "mimi"} else None,
            per_session=_scores_for_item(target, kind, layer, seed),
        )
    runner_version = "951e839+runner." + "a" * 64
    preflight = root / "mve_preflight.json"
    preflight.write_text(
        json.dumps(
            _preflight_payload(label_hashes, runner_version),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    reference = writer.finalize(
        runs_root=root / "runs",
        runner_code_version=runner_version,
        label_hashes=label_hashes,
        preflight_report_path=preflight,
        analysis_protocol=_analysis_protocol(),
    )
    manifest_path = Path(reference["manifest_path"])
    if recomputed is None:
        recomputed = _official_summary()
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        **recomputed,
        "score_bundle": reference,
    }
    summary_path = root / "mve_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return summary_path, manifest_path, recomputed


def _rewrite_manifest_reference(
    summary_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["score_bundle"]["manifest_sha256"] = sha256_file(manifest_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def synthetic_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("mve_independent_audit")
    return _write_package(root)


def test_independent_audit_accepts_complete_34_item_world(
    synthetic_world,
    tmp_path: Path,
):
    summary_path, manifest_path, _recomputed = synthetic_world
    output = tmp_path / "audit.json"

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=output,
        **_audit_paths(summary_path),
    )

    assert result["status"] == "passed"
    assert result["n_items"] == 34
    assert result["differences"] == []
    assert result["checks"]["item_alignment"]["n_eval_sessions"] == 40
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert not list(tmp_path.glob(".*.tmp"))


def test_independent_audit_rejects_score_tamper_by_sha(
    synthetic_world,
    tmp_path: Path,
):
    _summary_path, _manifest_path, recomputed = synthetic_world
    summary_path, manifest_path, _ = _write_package(
        tmp_path / "tampered-score",
        recomputed=recomputed,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_path = manifest_path.parent / manifest["items"][0]["path"]
    with score_path.open("ab") as handle:
        handle.write(b"tampered")
    output = tmp_path / "score-audit.json"

    result = AUDIT.run_audit(
        summary_path,
        output_path=output,
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "SHA-256" in result["error"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    assert not list(tmp_path.glob(".*.tmp"))


def test_independent_audit_reports_summary_numeric_tamper(
    synthetic_world,
    tmp_path: Path,
):
    _summary_path, _manifest_path, recomputed = synthetic_world
    summary_path, manifest_path, _ = _write_package(
        tmp_path / "tampered-summary",
        recomputed=recomputed,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["overall"]["advantage_point"] += 0.001
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    output = tmp_path / "summary-audit.json"

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=output,
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert any(difference["path"] == "$.overall.advantage_point" for difference in result["differences"])
    assert "1 项差异" in result["error"]


def test_independent_audit_rejects_seed_label_misalignment(
    synthetic_world,
    tmp_path: Path,
):
    _summary_path, _manifest_path, recomputed = synthetic_world
    summary_path, manifest_path, _ = _write_package(
        tmp_path / "misaligned-labels",
        recomputed=recomputed,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(
        candidate
        for candidate in manifest["items"]
        if (
            candidate["target"],
            candidate["kind"],
            candidate["layer"],
            candidate["seed"],
        )
        == ("T1", "probe", 20, 1)
    )
    score_path = manifest_path.parent / item["path"]
    per_session = read_per_session_npz(score_path)
    first_session = sorted(per_session)[0]
    labels, scores = per_session[first_session]
    changed = labels.copy()
    changed[0] = 1 - changed[0]
    per_session[first_session] = (changed, scores)
    write_per_session_npz_atomic(score_path, per_session)
    item["sha256"] = sha256_file(score_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["score_bundle"]["manifest_sha256"] = sha256_file(manifest_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    output = tmp_path / "labels-audit.json"

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=output,
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "标签" in result["error"]
    assert "不全等" in result["error"]


def test_bootstrap_is_bitwise_reproducible_with_fresh_seed_zero_rng(
    synthetic_world,
):
    _summary_path, manifest_path, recomputed = synthetic_world
    manifest = validate_score_bundle(manifest_path)
    loaded, _checks = AUDIT._load_and_align_items(manifest, manifest_path)
    best_layer = recomputed["per_target"]["T1"]["best_layer"]
    probe = {seed: loaded[("T1", "probe", best_layer, seed)] for seed in SEEDS}
    baselines = {
        "hazard": {0: loaded[("T1", "hazard", None, 0)]},
        "mimi": {seed: loaded[("T1", "mimi", None, seed)] for seed in SEEDS},
        "acoustic_gru": {0: loaded[("T1", "acoustic_gru", None, 0)]},
    }

    advantage_first = AUDIT._bootstrap_advantage(probe, baselines, n_boot=1000, seed=0)
    advantage_second = AUDIT._bootstrap_advantage(probe, baselines, n_boot=1000, seed=0)
    probe_first = AUDIT._bootstrap_seed_mean_auc(probe, n_boot=1000, seed=0)
    probe_second = AUDIT._bootstrap_seed_mean_auc(probe, n_boot=1000, seed=0)

    assert advantage_first == advantage_second
    assert probe_first == probe_second
    assert advantage_first == recomputed["per_target"]["T1"]["advantage"]
    assert probe_first == recomputed["per_target"]["T1"]["probe_ci"]


def test_independent_audit_rejects_wrong_frozen_split(tmp_path: Path):
    summary_path, manifest_path, _ = _write_package(tmp_path / "wrong-split")
    split_path = _audit_paths(summary_path)["split_path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["splits"]["probe_val"] = list(reversed(split["splits"]["probe_val"]))
    split_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=tmp_path / "wrong-split-audit.json",
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "probe_val" in result["error"]
    assert "顺序不全等" in result["error"]


def test_independent_audit_rejects_consistently_wrong_npz_labels(tmp_path: Path):
    summary_path, manifest_path, _ = _write_package(tmp_path / "wrong-labels")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["items"]:
        if item["target"] != "T1":
            continue
        score_path = manifest_path.parent / item["path"]
        per_session = read_per_session_npz(score_path)
        session_id = sorted(per_session)[0]
        labels, scores = per_session[session_id]
        changed = labels.copy()
        changed[0] = 1 - changed[0]
        per_session[session_id] = (changed, scores)
        write_per_session_npz_atomic(score_path, per_session)
        item["sha256"] = sha256_file(score_path)
    _rewrite_manifest_reference(summary_path, manifest_path, manifest)

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=tmp_path / "wrong-labels-audit.json",
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "WP1 权威标签不全等" in result["error"]


def test_independent_audit_rejects_authoritative_label_tamper(tmp_path: Path):
    summary_path, manifest_path, _ = _write_package(tmp_path / "source-tamper")
    session_id = TRAIN_SESSIONS[0]
    source = _audit_paths(summary_path)["data_root_path"] / "events" / "candor" / f"{session_id}.labels.parquet"
    with source.open("ab") as handle:
        handle.write(b"tampered")

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=tmp_path / "source-tamper-audit.json",
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "完成标记与权威标签不一致" in result["error"]


def test_independent_audit_requires_exact_200_label_hash_keys(tmp_path: Path):
    summary_path, manifest_path, _ = _write_package(tmp_path / "wrong-hash-keys")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed_hash = manifest["label_sha256"].pop(TRAIN_SESSIONS[0])
    manifest["label_sha256"]["foreign-session"] = removed_hash
    snapshot_path = manifest_path.parent / manifest["preflight_report_path"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["label_sha256"] = manifest["label_sha256"]
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    manifest["preflight_report_sha256"] = sha256_file(snapshot_path)
    _rewrite_manifest_reference(summary_path, manifest_path, manifest)

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=tmp_path / "wrong-hash-keys-audit.json",
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "160+40" in result["error"]


def test_independent_audit_rejects_invalid_repository_head(tmp_path: Path):
    summary_path, manifest_path, _ = _write_package(tmp_path / "bad-head")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    code = manifest["analysis_protocol"]["code"]
    code["repository_head"] = "0" * 40
    code["version"] = f"{'0' * 7}+analysis.{code['content_sha256']}"
    _rewrite_manifest_reference(summary_path, manifest_path, manifest)

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=tmp_path / "bad-head-audit.json",
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "repository_head" in result["error"]
    assert "可解析对象" in result["error"]


def test_repository_source_guard_rejects_omitted_required_source(synthetic_world):
    _summary_path, manifest_path, _ = synthetic_world
    manifest = validate_score_bundle(manifest_path)

    with pytest.raises(AUDIT.IndependentAuditError, match="核验集合不完整"):
        AUDIT._verify_repository_sources(
            manifest["analysis_protocol"],
            tuple(ANALYSIS_SOURCE_PATHS),
        )


def test_independent_audit_rejects_best_c_outside_frozen_grid(tmp_path: Path):
    summary_path, manifest_path, _ = _write_package(tmp_path / "bad-c")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(candidate for candidate in manifest["items"] if candidate["kind"] == "probe")
    item["best_c"] = 123.0
    _rewrite_manifest_reference(summary_path, manifest_path, manifest)

    result = AUDIT.run_audit(
        summary_path,
        manifest_path,
        output_path=tmp_path / "bad-c-audit.json",
        **_audit_paths(summary_path),
    )

    assert result["status"] == "failed"
    assert "best_c_grid" in result["error"]
