"""上下文应力诊断的采样、证据合取与防误判测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.context_stress import (
    analyze_context_stress_runs,
    load_context_stress_run,
    render_context_stress_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "runners" / "_shared"))

from context_stress_runner import build_sample_positions  # noqa: E402


def _write_synthetic_run(
    root: Path,
    *,
    run_index: int,
    collapse: bool,
    complete: bool = True,
) -> None:
    root.mkdir(parents=True)
    positions = np.arange(10, 3010, 10, dtype=np.int64)
    n_samples = len(positions)
    layers = [4, 12]
    hidden_dim = 24
    rng = np.random.default_rng(100 + run_index)
    base = rng.normal(size=(len(layers), hidden_dim))
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    hidden = np.empty((n_samples, len(layers), hidden_dim), dtype=np.float32)
    common = np.zeros((len(layers), hidden_dim), dtype=np.float32)
    common[:, 0] = 1.0
    for index, position in enumerate(positions):
        noise = rng.normal(scale=0.03, size=(len(layers), hidden_dim))
        if collapse and position >= 1000:
            hidden[index] = (common + noise * 0.05) * 10.0
        else:
            hidden[index] = base + noise

    cache_lengths = positions.copy()
    key_cosines = np.ones((n_samples, len(layers)), dtype=np.float32)
    if collapse:
        cache_lengths[positions >= 1000] -= 500
        key_cosines[positions >= 1000] = 0.2
    decision_probs = np.empty((n_samples, 2), dtype=np.float32)
    if run_index % 2:
        decision_probs[:] = [0.25, 0.75]
    else:
        decision_probs[:] = [0.75, 0.25]
    if collapse:
        decision_probs[positions >= 1000] = [0.995, 0.005]
    decision_ids = np.argmax(decision_probs, axis=1).astype(np.int16)

    np.savez_compressed(
        root / "trace.npz",
        logical_positions=positions,
        cache_lengths=cache_lengths,
        position_offsets=np.zeros(n_samples, dtype=np.int64),
        hidden=hidden.astype(np.float16),
        decision_probs=decision_probs,
        decision_ids=decision_ids,
        dynamic_key_cosines=key_cosines,
        all_finite=np.ones((n_samples, len(layers)), dtype=bool),
        sliding_events=(positions >= 1000).astype(np.int64) if collapse else np.zeros(n_samples, dtype=np.int64),
    )
    manifest = {
        "schema_version": 1,
        "protocol": "context_stress_v1",
        "model": "synthetic",
        "run_id": f"run{run_index}",
        "complete": complete,
        "layers": layers,
        "planned_boundaries": [1000, 2000],
        "sampling": {
            "boundary_window_positions": 100,
        },
        "context_spec": {
            "official_max_positions": 1000,
            "start_position": 0,
            "logical_step_positions": 10,
            "positions_per_second_min": 10.0,
            "positions_per_second_max": 10.0,
            "positions_per_second_stress": 10.0,
            "analysis_target_seconds": 20.0,
            "analysis_target_required_positions": 200,
            "cache_policy": "no_sliding",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )


def test_sample_schedule_is_quantized_dense_and_boundary_aware():
    positions = build_sample_positions(
        start_position=7,
        max_position=120,
        quantum=13,
        coarse_stride=40,
        boundaries=[60, 100],
        dense_radius=14,
    )

    assert np.all((positions - 7) % 13 == 0)
    assert positions[0] == 20
    assert positions[-1] == 111
    # 两个边界都必须有一个最近的逻辑步端点。
    assert np.min(np.abs(positions - 60)) <= 7
    assert np.min(np.abs(positions - 100)) <= 7


def test_joint_evidence_detects_moshi_like_collapse(tmp_path):
    roots = []
    for run_index in range(4):
        root = tmp_path / f"run{run_index}"
        _write_synthetic_run(root, run_index=run_index, collapse=True)
        roots.append(root)
    runs = [load_context_stress_run(root) for root in roots]

    report = analyze_context_stress_runs(runs)

    first = report["boundaries"][0]
    assert first["boundary"] == 1000
    assert first["pathology_confirmed"] is True
    assert first["structural_eviction_observed"] is True
    assert first["pathology_layers"] == [4, 12]
    assert report["recommendation"]["earliest_confirmed_pathology_position"] == 1000
    assert report["recommendation"]["safe_position_for_this_report"] == 990
    assert report["cross_run_evidence_available"] is True


def test_single_direction_shift_does_not_trigger_false_positive(tmp_path):
    roots = []
    for run_index in range(4):
        root = tmp_path / f"clean{run_index}"
        _write_synthetic_run(root, run_index=run_index, collapse=False)
        roots.append(root)
    runs = [load_context_stress_run(root) for root in roots]

    report = analyze_context_stress_runs(runs)

    assert all(not item["pathology_confirmed"] for item in report["boundaries"])
    assert report["recommendation"]["analysis_target_covered"] is True
    assert report["recommendation"]["analysis_target_clean"] is True
    assert report["recommendation"]["safe_position_for_this_report"] == 1000
    markdown = render_context_stress_markdown(report)
    assert "计划分析窗 20.0 秒" in markdown
    assert "仍按官方规格截断" in markdown


def test_safe_position_uses_largest_aligned_observation(tmp_path):
    root = tmp_path / "aligned"
    _write_synthetic_run(root, run_index=0, collapse=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["context_spec"]["official_max_positions"] = 1005
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    report = analyze_context_stress_runs([load_context_stress_run(root)])

    assert report["recommendation"]["safe_position_for_this_report"] == 1000
    assert "共同实测采样点" in report["recommendation"]["safe_position_rule"]


def test_partial_run_is_rejected_by_default(tmp_path):
    root = tmp_path / "partial"
    _write_synthetic_run(root, run_index=0, collapse=False, complete=False)

    with pytest.raises(ValueError, match="未完整结束"):
        load_context_stress_run(root)
    loaded = load_context_stress_run(root, require_complete=False)
    assert loaded.run_id == "run0"


def test_probability_rows_must_be_normalized(tmp_path):
    root = tmp_path / "bad"
    _write_synthetic_run(root, run_index=0, collapse=False)
    trace_path = root / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["decision_probs"][:] = 0.8
    np.savez_compressed(trace_path, **arrays)

    with pytest.raises(ValueError, match="每行之和"):
        load_context_stress_run(root)
