"""G1 逐会话分数包的原子性、契约与报告引用测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.mve.artifacts import (
    ANALYSIS_SOURCE_PATHS,
    MANIFEST_NAME,
    ScoreBundleError,
    ScoreBundleWriter,
    expected_item_keys,
    read_per_session_npz,
    validate_score_bundle,
    write_per_session_npz_atomic,
)
from floor_circuit.mve.run import render_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scores() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "session-b": (
            np.asarray([0, 1, 1], dtype=np.int64),
            np.asarray([0.1, 0.8, 0.7], dtype=np.float32),
        ),
        "session-a": (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        ),
    }


def _complete_writer(
    tmp_path: Path,
) -> tuple[ScoreBundleWriter, Path]:
    targets = ["T1", "T4"]
    layers = [4, 12, 20, 28]
    seeds = [0, 1, 2]
    writer = ScoreBundleWriter.create(
        tmp_path / "g1_scores",
        relative_base=tmp_path,
        eval_session_order=["session-b", "session-a"],
        targets=targets,
        layers=layers,
        seeds=seeds,
        run_id="fixed-run",
    )
    for target, kind, layer, seed in sorted(
        expected_item_keys(targets, layers, seeds),
        key=repr,
    ):
        writer.add(
            target=target,
            kind=kind,
            per_session=_scores(),
            layer=layer,
            seed=seed,
            best_c=0.1 if kind in {"probe", "mimi"} else None,
        )
    preflight = tmp_path / "mve_preflight.json"
    preflight.write_text('{"status":"passed"}\n', encoding="utf-8")
    return writer, preflight


def _finalize(
    writer: ScoreBundleWriter,
    preflight: Path,
    *,
    git_oid_length: int = 64,
) -> dict:
    repository_head = "d" * git_oid_length
    return writer.finalize(
        runs_root=preflight.parent / "runs",
        runner_code_version="abc1234+runner." + "1" * 64,
        label_hashes={"session-a": "a" * 64, "session-b": "b" * 64},
        preflight_report_path=preflight,
        analysis_protocol={
            "bootstrap_n": 1000,
            "bootstrap_seed": 0,
            "code": {
                "version": f"{repository_head[:7]}+analysis." + "c" * 64,
                "repository_head": repository_head,
                "content_sha256": "c" * 64,
                "sources": list(ANALYSIS_SOURCE_PATHS),
                "source_commits": {
                    source: "e" * git_oid_length
                    for source in ANALYSIS_SOURCE_PATHS
                },
            },
        },
    )


def test_per_session_npz_atomic_roundtrip_preserves_empty_session(tmp_path: Path):
    path = tmp_path / "scores.npz"
    counts = write_per_session_npz_atomic(path, _scores())

    assert counts == {"rows": 3, "sessions": 2}
    restored = read_per_session_npz(path)
    assert list(restored) == ["session-a", "session-b"]
    assert restored["session-a"][0].shape == (0,)
    np.testing.assert_array_equal(restored["session-b"][0], [0, 1, 1])
    np.testing.assert_allclose(restored["session-b"][1], [0.1, 0.8, 0.7])
    with np.load(path, allow_pickle=False) as archive:
        assert archive["session_ids"].dtype.kind == "U"
        assert archive["offsets"].dtype == np.int64
        assert archive["labels"].dtype == np.int8
        assert archive["scores"].dtype == np.float64
    assert not list(tmp_path.glob(".*.tmp"))


def test_full_g1_bundle_has_exactly_34_unique_items(tmp_path: Path):
    writer, preflight = _complete_writer(tmp_path)
    reference = _finalize(writer, preflight)
    manifest = validate_score_bundle(reference["manifest_path"])

    assert reference["n_items"] == 34
    assert manifest["contract"]["expected_items"] == 34
    assert manifest["analysis_protocol"]["bootstrap_n"] == 1000
    assert manifest["analysis_protocol"]["bootstrap_seed"] == 0
    keys = {
        (item["target"], item["kind"], item["layer"], item["seed"])
        for item in manifest["items"]
    }
    assert keys == expected_item_keys(
        ["T1", "T4"],
        [4, 12, 20, 28],
        [0, 1, 2],
    )
    assert manifest["eval_session_order"] == ["session-b", "session-a"]
    assert all(
        item["best_c"] == 0.1
        for item in manifest["items"]
        if item["kind"] in {"probe", "mimi"}
    )
    assert all(
        item["best_c"] is None
        for item in manifest["items"]
        if item["kind"] in {"hazard", "acoustic_gru"}
    )


def test_bundle_accepts_sha1_git_object_ids(tmp_path: Path):
    """本仓库使用 40 位 SHA-1；清单也兼容 64 位新格式。"""

    writer, preflight = _complete_writer(tmp_path)
    reference = _finalize(writer, preflight, git_oid_length=40)
    manifest = validate_score_bundle(reference["manifest_path"])

    code = manifest["analysis_protocol"]["code"]
    assert len(code["repository_head"]) == 40
    assert {len(value) for value in code["source_commits"].values()} == {40}


def test_bundle_rejects_tampered_npz_and_incomplete_manifest(tmp_path: Path):
    writer, preflight = _complete_writer(tmp_path)
    reference = _finalize(writer, preflight)
    manifest_path = Path(reference["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_path = manifest_path.parent / manifest["items"][0]["path"]
    with score_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ScoreBundleError, match="SHA-256"):
        validate_score_bundle(manifest_path)

    manifest["items"].pop()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ScoreBundleError, match="条目数"):
        validate_score_bundle(manifest_path)


def test_incomplete_contract_never_publishes_final_manifest(tmp_path: Path):
    writer = ScoreBundleWriter.create(
        tmp_path / "g1_scores",
        relative_base=tmp_path,
        eval_session_order=["session-b", "session-a"],
        targets=["T1", "T4"],
        layers=[4, 12, 20, 28],
        seeds=[0, 1, 2],
        run_id="incomplete-run",
    )
    writer.add(
        target="T1",
        kind="probe",
        per_session=_scores(),
        layer=4,
        seed=0,
        best_c=0.1,
    )
    preflight = tmp_path / "mve_preflight.json"
    preflight.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ScoreBundleError, match="条目数"):
        _finalize(writer, preflight)
    assert not (writer.bundle_dir / MANIFEST_NAME).exists()


def test_bundle_rejects_path_that_does_not_resolve_to_manifest_directory(
    tmp_path: Path,
):
    writer, preflight = _complete_writer(tmp_path)
    reference = _finalize(writer, preflight)
    manifest_path = Path(reference["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_path"]["relative"] = "mve/g1_scores/another-run"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ScoreBundleError, match=r"bundle_path\.relative"):
        validate_score_bundle(manifest_path)


def _load_wp7():
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "wp7_run_mve_artifacts",
        scripts / "wp7_run_mve.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_repository_provenance_can_publish_manifest(tmp_path: Path):
    """真实 Git OID 长度必须通过最终清单校验，防止长跑结束后才失败。"""

    module = _load_wp7()
    writer, preflight = _complete_writer(tmp_path)
    reference = writer.finalize(
        runs_root=preflight.parent / "runs",
        runner_code_version="abc1234+runner." + "1" * 64,
        label_hashes={"session-a": "a" * 64, "session-b": "b" * 64},
        preflight_report_path=preflight,
        analysis_protocol={
            "bootstrap_n": 1000,
            "bootstrap_seed": 0,
            "code": module._analysis_code_provenance(),
        },
    )

    manifest = validate_score_bundle(reference["manifest_path"])
    assert manifest["analysis_protocol"]["code"]["repository_head"] == (
        module._analysis_code_provenance()["repository_head"]
    )


def test_json_and_markdown_reports_reference_score_bundle():
    bundle = {
        "absolute_path": r"D:\data\mve\g1_scores\run",
        "relative_path": "mve/g1_scores/run",
        "manifest_path": r"D:\data\mve\g1_scores\run\manifest.json",
        "manifest_sha256": "c" * 64,
        "n_items": 34,
    }
    target = {
        "best_layer": 4,
        "layer_summary": {
            4: {
                "n_seeds": 3,
                "auc_mean": 0.7,
                "auc_sd": 0.01,
                "auprc_mean": 0.6,
                "auprc_sd": 0.01,
                "balanced_acc_mean": 0.65,
                "balanced_acc_sd": 0.01,
            }
        },
        "advantage": {
            "probe_auc": 0.7,
            "advantage_point": 0.1,
            "ci_lo": 0.05,
            "ci_hi": 0.15,
        },
        "probe_ci": {"ci_lo": 0.65, "ci_hi": 0.75},
        "baseline_metrics": {
            "hazard": {
                "n_seeds": 1,
                "auc_mean": 0.6,
                "auc_sd": 0.0,
                "auprc_mean": 0.5,
                "auprc_sd": 0.0,
                "balanced_acc_mean": 0.55,
                "balanced_acc_sd": 0.0,
            }
        },
        "shuffled_auc": 0.5,
        "shuffled_auc_sd": 0.01,
        "shuffled_n_seeds": 3,
        "verdict": "full_e1",
    }
    overall = {
        "decisive_target": "T1",
        "advantage_point": 0.1,
        "ci_lo": 0.05,
        "verdict": "full_e1",
    }
    metadata = {
        "layers": [4],
        "seeds": [0, 1, 2],
        "bootstrap_n": 1000,
        "n_train_sessions": 160,
        "n_eval_sessions": 40,
        "score_bundle": bundle,
    }

    report = render_report({"T1": target}, overall, metadata)
    summary = _load_wp7()._mve_summary_payload(
        overall,
        {"T1": target},
        bundle,
        {"text_mode": "greedy", "ablation": None},
        {"note": "非 G1 判据", "T1": {}},
    )

    assert bundle["absolute_path"] in report
    assert bundle["relative_path"] in report
    assert bundle["manifest_sha256"] in report
    assert summary["score_bundle"] == bundle


def test_publish_failure_removes_manifest_and_both_official_reports(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_wp7()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(module, "render_report", lambda *_args, **_kwargs: "报告\n")

    class FakeWriter:
        def __init__(self):
            self.manifest_path = tmp_path / "bundle" / MANIFEST_NAME

        def finalize(self, **_kwargs):
            self.manifest_path.parent.mkdir()
            self.manifest_path.write_text("{}\n", encoding="utf-8")
            return {
                "absolute_path": str(self.manifest_path.parent),
                "relative_path": "mve/g1_scores/fake",
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": "a" * 64,
                "n_items": 34,
            }

        def remove_manifest(self):
            self.manifest_path.unlink(missing_ok=True)

    writer = FakeWriter()
    (reports_dir / "mve_summary.json").write_text("旧小结\n", encoding="utf-8")

    def fail_summary(*_args, **_kwargs):
        raise OSError("模拟小结发布失败")

    monkeypatch.setattr(module, "_write_report_json_atomic", fail_summary)
    with pytest.raises(OSError, match="模拟小结发布失败"):
        module._publish_g1_outputs(
            score_writer=writer,
            runs_root=tmp_path / "runs",
            runner_code_version="runner",
            label_hashes={"session": "a" * 64},
            preflight_report_path=tmp_path / "preflight.json",
            analysis_protocol={},
            per_target={"T1": {"layer_summary": {}}},
            overall={},
            meta={},
            protocol={"text_mode": "greedy", "ablation": None},
            descriptive={"note": "非 G1 判据", "T1": {}},
        )

    assert not writer.manifest_path.exists()
    assert not (reports_dir / "mve_报告.md").exists()
    assert not (reports_dir / "mve_summary.json").exists()
