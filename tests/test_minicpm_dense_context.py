from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.minicpm_dense_context import (
    analyze_dense_context_runs,
    load_dense_context_run,
    render_dense_context_markdown,
)


def _write_dense_run(
    root: Path,
    *,
    run_id: str,
    collapse_from_seconds: int | None = None,
    history_only_collapse_from_seconds: int | None = None,
    complete: bool = True,
) -> Path:
    root.mkdir(parents=True)
    targets = np.asarray([64, 1000, 2000, 2050], dtype=np.int32)
    probes = 2
    repeated_targets = np.repeat(targets, probes)
    n_rows = len(repeated_targets)
    hidden_dim = 6
    n_layers = 2

    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    control_hidden = rng.normal(size=(n_rows, n_layers, hidden_dim)).astype(np.float32)
    position_control_hidden = control_hidden.copy()
    long_hidden = position_control_hidden.copy()
    control_logits = np.tile(
        np.asarray([5.0, 0.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0]),
        (n_rows, 1),
    ).astype(np.float32)
    position_control_logits = control_logits.copy()
    long_logits = position_control_logits.copy()
    long_ids = np.full((n_rows, 5), -1, dtype=np.int32)
    control_ids = np.full((n_rows, 5), -1, dtype=np.int32)
    long_ids[:, :3] = np.asarray([0, 4, 2])
    control_ids[:, :3] = np.asarray([0, 4, 2])
    position_control_ids = control_ids.copy()

    if collapse_from_seconds is not None:
        collapse = repeated_targets >= collapse_from_seconds
        position_control_hidden[collapse] = -control_hidden[collapse]
        long_hidden[collapse] = position_control_hidden[collapse]
        position_control_logits[collapse] = np.asarray(
            [0.0, 5.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0]
        )
        long_logits[collapse] = position_control_logits[collapse]
        position_control_ids[collapse, :3] = np.asarray([1, 5, 3])
        long_ids[collapse] = position_control_ids[collapse]
    if history_only_collapse_from_seconds is not None:
        history_collapse = repeated_targets >= history_only_collapse_from_seconds
        long_hidden[history_collapse] = -position_control_hidden[
            history_collapse
        ]
        long_logits[history_collapse] = np.asarray(
            [0.0, 5.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0]
        )
        long_ids[history_collapse, :3] = np.asarray([1, 5, 3])

    manifest = {
        "schema_version": 2,
        "protocol": "minicpm_dense_context_v1",
        "model": "minicpm_o_4_5",
        "run_id": run_id,
        "complete": complete,
        "source_audio": {
            "path": f"{run_id}.wav",
            "sha256": f"sha256-{run_id}",
        },
        "layers": [1, 3],
        "special_token_ids": {
            "listen": 0,
            "speak": 1,
            "chunk_eos": 2,
            "turn_eos": 3,
        },
        "context_spec": {
            "official_max_positions": 40960,
            "start_position": 14,
            "dense_unit_positions": 32,
            "dense_positions_per_second": 32.0,
            "formal_dense_seconds": 1279.5625,
            "cache_policy": "off",
        },
        "design": {
            "checkpoint_seconds": targets.tolist(),
            "max_seconds": 2050,
            "checkpoint_clock": "synthetic",
            "probes_per_checkpoint": probes,
            "probe_indices": [0, 1],
            "control_suffix_units": 64,
            "filler_group_units": 8,
            "max_new_speak_tokens": 20,
            "prefill_positions": 11,
            "dense_tail_positions": 21,
            "filler_decode": "synthetic",
            "probe_decode": "synthetic",
            "generate_audio": False,
            "pairing": "synthetic",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    long_positions = 14 + repeated_targets.astype(np.int64) * 32 + 11
    control_positions = np.full(n_rows, 14 + 64 * 32 + 11, dtype=np.int64)
    position_shifts = long_positions - control_positions
    np.savez_compressed(
        root / "trace.npz",
        target_seconds=repeated_targets,
        long_input_seconds=repeated_targets + np.tile([1, 2], len(targets)),
        long_positions=long_positions,
        control_positions=control_positions,
        position_control_positions=control_positions,
        position_control_absolute_positions=long_positions,
        position_control_shifts=position_shifts,
        probe_indices=np.tile([0, 1], len(targets)).astype(np.int16),
        long_hidden=long_hidden.astype(np.float16),
        control_hidden=control_hidden.astype(np.float16),
        position_control_hidden=position_control_hidden.astype(np.float16),
        long_logits=long_logits.astype(np.float16),
        control_logits=control_logits.astype(np.float16),
        position_control_logits=position_control_logits.astype(np.float16),
        long_generated_ids=long_ids,
        control_generated_ids=control_ids,
        position_control_generated_ids=position_control_ids,
        long_generated_lengths=np.full(n_rows, 3, dtype=np.int16),
        control_generated_lengths=np.full(n_rows, 3, dtype=np.int16),
        position_control_generated_lengths=np.full(n_rows, 3, dtype=np.int16),
        long_cache_lengths=long_positions + 3,
        control_cache_lengths=np.full(n_rows, 14 + 64 * 32 + 14, dtype=np.int64),
        position_control_cache_lengths=np.full(
            n_rows,
            14 + 64 * 32 + 14,
            dtype=np.int64,
        ),
        long_all_finite=np.ones((n_rows, n_layers), dtype=bool),
        control_all_finite=np.ones((n_rows, n_layers), dtype=bool),
        position_control_all_finite=np.ones(
            (n_rows, n_layers),
            dtype=bool,
        ),
    )
    return root


def test_clean_dense_runs_support_2000_second_empirical_lower_bound(tmp_path):
    runs = [
        load_dense_context_run(
            _write_dense_run(tmp_path / f"run{index}", run_id=f"run{index}")
        )
        for index in range(3)
    ]
    report = analyze_dense_context_runs(runs)

    assert report["recommendation"]["seconds_2000_covered"] is True
    assert report["recommendation"]["seconds_2000_clean"] is True
    assert report["recommendation"]["first_confirmed_degradation_seconds"] is None
    assert report["recommendation"]["empirical_clean_lower_bound_seconds"] == 2050
    assert report["recommendation"]["cross_input_evidence_available"] is True
    assert report["zero_shift_validation"]["n_zero_shift_pairs"] == 6
    assert report["zero_shift_validation"]["exact_parity"] is True
    assert report["recommendation"]["dense_positions_per_second"] == 32.0
    assert report["recommendation"]["formal_dense_context_seconds"] == 1279.5625
    assert (
        report["recommendation"]["formal_dense_complete_one_second_units"]
        == 1279
    )
    assert (
        "2000 秒：覆盖=是，未确认绝对位置退化=是"
        in render_dense_context_markdown(report)
    )


def test_persistent_multi_metric_collapse_is_detected_at_2000_seconds(tmp_path):
    runs = [
        load_dense_context_run(
            _write_dense_run(
                tmp_path / f"run{index}",
                run_id=f"run{index}",
                collapse_from_seconds=2000,
            )
        )
        for index in range(3)
    ]
    report = analyze_dense_context_runs(runs)

    assert report["recommendation"]["seconds_2000_clean"] is False
    assert report["recommendation"]["first_confirmed_degradation_seconds"] == 2000
    assert report["recommendation"]["empirical_clean_lower_bound_seconds"] == 1000
    row_2000 = next(
        row for row in report["checkpoints"] if row["target_seconds"] == 2000
    )
    assert row_2000["confirmed_persistent_degradation"] is True
    assert row_2000["flags"]["representation_degradation"] is True
    assert row_2000["flags"]["distribution_degradation"] is True
    assert row_2000["flags"]["behavior_degradation"] is True


def test_duplicate_audio_does_not_count_as_cross_input_evidence(tmp_path):
    roots = [
        _write_dense_run(tmp_path / f"run{index}", run_id=f"run{index}")
        for index in range(3)
    ]
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_audio"]["sha256"] = "same-audio"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
    report = analyze_dense_context_runs(
        [load_dense_context_run(root) for root in roots]
    )

    assert report["recommendation"]["n_distinct_audio_inputs"] == 1
    assert report["recommendation"]["cross_input_evidence_available"] is False
    for row in report["checkpoints"]:
        assert np.isnan(row["max_cross_run_cosine_delta"])


def test_history_only_divergence_is_not_called_position_degradation(tmp_path):
    runs = [
        load_dense_context_run(
            _write_dense_run(
                tmp_path / f"run{index}",
                run_id=f"run{index}",
                history_only_collapse_from_seconds=2000,
            )
        )
        for index in range(3)
    ]
    report = analyze_dense_context_runs(runs)

    assert report["recommendation"]["first_confirmed_degradation_seconds"] is None
    assert report["recommendation"]["seconds_2000_clean"] is True
    row_2000 = next(
        row for row in report["checkpoints"] if row["target_seconds"] == 2000
    )
    assert row_2000["history_logit_cosine_median"] < 0.95
    assert row_2000["logit_cosine_median"] == pytest.approx(1.0)
    assert row_2000["candidate_degradation"] is False


def test_partial_dense_run_is_rejected_by_default(tmp_path):
    root = _write_dense_run(tmp_path / "partial", run_id="partial", complete=False)
    with pytest.raises(ValueError, match="未完整结束"):
        load_dense_context_run(root)
    assert load_dense_context_run(root, require_complete=False).run_id == "partial"


def test_dense_loader_rejects_mismatched_control_hidden_shape(tmp_path):
    root = _write_dense_run(tmp_path / "bad-shape", run_id="bad-shape")
    with np.load(root / "trace.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["control_hidden"] = arrays["control_hidden"][:, :1]
    np.savez_compressed(root / "trace.npz", **arrays)

    with pytest.raises(ValueError, match="三路 hidden"):
        load_dense_context_run(root)
