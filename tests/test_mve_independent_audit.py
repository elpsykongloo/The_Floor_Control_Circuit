"""G1 独立复算脚本的 34 项合成世界与篡改护栏。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
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


def _analysis_protocol() -> dict[str, Any]:
    content_sha256 = AUDIT._analysis_content_sha256(list(ANALYSIS_SOURCE_PATHS))
    repository_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    source_commits = {
        relative: AUDIT._source_commit(relative)
        for relative in ANALYSIS_SOURCE_PATHS
    }
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
        nuisance = 0.23 * np.sin(
            0.73 * row_index + 0.31 * session_index + 0.19 * seed
        )
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
            "mimi": {
                seed: _scores_for_item(target, "mimi", None, seed)
                for seed in SEEDS
            },
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


def _write_package(
    root: Path,
    *,
    recomputed: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
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
    label_ids = [*EVAL_SESSIONS, *(f"train-{index:03d}" for index in range(160))]
    label_hashes = {
        session_id: f"{index + 1:064x}"
        for index, session_id in enumerate(label_ids)
    }
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

    result = AUDIT.run_audit(summary_path, output_path=output)

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

    result = AUDIT.run_audit(summary_path, manifest_path, output_path=output)

    assert result["status"] == "failed"
    assert any(
        difference["path"] == "$.overall.advantage_point"
        for difference in result["differences"]
    )
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

    result = AUDIT.run_audit(summary_path, manifest_path, output_path=output)

    assert result["status"] == "failed"
    assert "标签" in result["error"]
    assert "不完全一致" in result["error"]


def test_bootstrap_is_bitwise_reproducible_with_fresh_seed_zero_rng(
    synthetic_world,
):
    _summary_path, manifest_path, recomputed = synthetic_world
    manifest = validate_score_bundle(manifest_path)
    loaded, _checks = AUDIT._load_and_align_items(manifest, manifest_path)
    best_layer = recomputed["per_target"]["T1"]["best_layer"]
    probe = {
        seed: loaded[("T1", "probe", best_layer, seed)]
        for seed in SEEDS
    }
    baselines = {
        "hazard": {0: loaded[("T1", "hazard", None, 0)]},
        "mimi": {
            seed: loaded[("T1", "mimi", None, seed)]
            for seed in SEEDS
        },
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
