from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.minicpm_true_context import (
    PROBE_KIND_FLOOR,
    PROBE_KIND_MEMORY,
    analyze_true_context_runs,
    answer_token_f1,
    contains_answer,
    load_true_context_run,
    normalize_answer,
)


def test_answer_normalization_accepts_digit_alias() -> None:
    assert normalize_answer("The answer is AMBER 7.") == (
        "the",
        "answer",
        "is",
        "amber",
        "seven",
    )
    assert contains_answer("The password was amber 7.", ["amber seven"])
    assert contains_answer("The password was Amber7.", ["amber seven"])
    assert not contains_answer("The password was amber eight.", ["amber seven"])
    assert answer_token_f1("amber seven", ["amber seven"]) == 1.0


def _write_run(
    root: Path,
    *,
    task: str,
    targets: list[int],
    kinds: list[int],
    expected: list[int],
    long_is_listen: list[bool],
    high_is_listen: list[bool],
    memory_long: list[bool] | None = None,
    memory_oracle_high: list[bool] | None = None,
) -> None:
    n_rows = len(targets)
    layers = 2
    hidden_dim = 4
    vocab = 7
    width = 5
    long_positions = np.arange(n_rows, dtype=np.int64) * 100 + 1000
    low_positions = np.full(n_rows, 100, dtype=np.int64)
    shifts = long_positions - low_positions
    memory_mask = np.asarray(kinds) == PROBE_KIND_MEMORY
    if memory_long is None:
        memory_long = [False] * n_rows
    if memory_oracle_high is None:
        memory_oracle_high = [False] * n_rows

    arrays = {
        "target_seconds": np.asarray(targets, dtype=np.int32),
        "history_input_seconds": np.asarray(targets, dtype=np.int32),
        "lane_indices": np.zeros(n_rows, dtype=np.int16),
        "probe_kinds": np.asarray(kinds, dtype=np.int8),
        "probe_ids": np.arange(n_rows, dtype=np.int16),
        "expected_is_listen": np.asarray(expected, dtype=np.int8),
        "long_positions": long_positions,
        "low_positions": low_positions,
        "high_positions": low_positions.copy(),
        "high_absolute_positions": long_positions.copy(),
        "high_position_shifts": shifts,
        "long_hidden": np.ones((n_rows, layers, hidden_dim), dtype=np.float16),
        "low_hidden": np.ones((n_rows, layers, hidden_dim), dtype=np.float16),
        "high_hidden": np.ones((n_rows, layers, hidden_dim), dtype=np.float16),
        "long_logits": np.ones((n_rows, vocab), dtype=np.float16),
        "low_logits": np.ones((n_rows, vocab), dtype=np.float16),
        "high_logits": np.ones((n_rows, vocab), dtype=np.float16),
        "long_generated_ids": np.zeros((n_rows, width), dtype=np.int32),
        "low_generated_ids": np.zeros((n_rows, width), dtype=np.int32),
        "high_generated_ids": np.zeros((n_rows, width), dtype=np.int32),
        "long_generated_lengths": np.ones(n_rows, dtype=np.int16),
        "low_generated_lengths": np.ones(n_rows, dtype=np.int16),
        "high_generated_lengths": np.ones(n_rows, dtype=np.int16),
        "long_generated_text": np.asarray([""] * n_rows, dtype="U16"),
        "low_generated_text": np.asarray([""] * n_rows, dtype="U16"),
        "high_generated_text": np.asarray([""] * n_rows, dtype="U16"),
        "long_is_listen": np.asarray(long_is_listen, dtype=bool),
        "low_is_listen": np.asarray(high_is_listen, dtype=bool),
        "high_is_listen": np.asarray(high_is_listen, dtype=bool),
        "long_end_of_turn": np.zeros(n_rows, dtype=bool),
        "low_end_of_turn": np.zeros(n_rows, dtype=bool),
        "high_end_of_turn": np.zeros(n_rows, dtype=bool),
        "long_all_finite": np.ones((n_rows, layers), dtype=bool),
        "low_all_finite": np.ones((n_rows, layers), dtype=bool),
        "high_all_finite": np.ones((n_rows, layers), dtype=bool),
        "memory_available": memory_mask,
        "memory_age_seconds": np.asarray(
            [
                target if memory else -1
                for target, memory in zip(targets, memory_mask, strict=True)
            ],
            dtype=np.int32,
        ),
        "oracle_low_positions": np.where(memory_mask, low_positions, -1),
        "oracle_high_positions": np.where(memory_mask, low_positions, -1),
        "oracle_high_absolute_positions": np.where(memory_mask, long_positions, -1),
        "oracle_high_position_shifts": np.where(memory_mask, shifts, -1),
        "oracle_low_generated_text": np.asarray([""] * n_rows, dtype="U16"),
        "oracle_high_generated_text": np.asarray([""] * n_rows, dtype="U16"),
        "oracle_low_is_listen": np.zeros(n_rows, dtype=bool),
        "oracle_high_is_listen": np.zeros(n_rows, dtype=bool),
        "oracle_low_end_of_turn": np.zeros(n_rows, dtype=bool),
        "oracle_high_end_of_turn": np.zeros(n_rows, dtype=bool),
        "memory_long_correct": np.asarray(memory_long, dtype=bool),
        "memory_low_correct": np.zeros(n_rows, dtype=bool),
        "memory_high_correct": np.zeros(n_rows, dtype=bool),
        "memory_oracle_low_correct": np.asarray(memory_oracle_high, dtype=bool),
        "memory_oracle_high_correct": np.asarray(memory_oracle_high, dtype=bool),
        "memory_long_token_f1": np.asarray(memory_long, dtype=np.float32),
        "memory_oracle_high_token_f1": np.asarray(
            memory_oracle_high,
            dtype=np.float32,
        ),
        "interaction_lane_indices": np.zeros(2, dtype=np.int16),
        "interaction_history_unit_indices": np.arange(2, dtype=np.int32),
        "interaction_bank_indices": np.arange(2, dtype=np.int32),
        "interaction_modes": np.asarray(["filler", "filler"], dtype="U32"),
        "interaction_forced_actions": np.asarray(
            ["natural", "natural"],
            dtype="U16",
        ),
        "interaction_position_before": np.asarray([10, 20], dtype=np.int64),
        "interaction_decision_positions": np.asarray([15, 25], dtype=np.int64),
        "interaction_cache_after": np.asarray([20, 30], dtype=np.int64),
        "interaction_is_listen": np.asarray([True, False], dtype=bool),
        "interaction_end_of_turn": np.asarray([False, True], dtype=bool),
        "interaction_generated_lengths": np.asarray([1, 3], dtype=np.int16),
        "interaction_generated_text": np.asarray(["", "okay"], dtype="U256"),
    }
    root.mkdir(parents=True)
    trace_path = root / "trace.npz"
    np.savez_compressed(trace_path, **arrays)
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "protocol": "minicpm_true_context_v1",
        "model": "minicpm_o_4_5",
        "task": task,
        "run_id": root.name,
        "complete": True,
        "source_audio": {"sha256": f"audio-{root.name}"},
        "probe_catalog": [{"probe_id": "shared"}],
        "context_spec": {"official_max_positions": 40960},
        "trace_sha256": trace_sha,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_floor_failure_requires_two_adjacent_checkpoints(tmp_path: Path) -> None:
    targets = [576] * 4 + [600] * 4 + [624] * 4
    expected = [0, 0, 1, 1] * 3
    high = [False, False, True, True] * 3
    long_state = high[:4] + [True, True, False, False] * 2
    roots = [tmp_path / f"floor_{index}" for index in range(3)]
    for root in roots:
        _write_run(
            root,
            task="floor",
            targets=targets,
            kinds=[PROBE_KIND_FLOOR] * len(targets),
            expected=expected,
            long_is_listen=long_state,
            high_is_listen=high,
        )
    report = analyze_true_context_runs(
        [load_true_context_run(root) for root in roots]
    )
    assert report["recommendation"]["first_confirmed_failure_seconds"] == 600
    assert report["recommendation"]["empirical_safe_checkpoint_seconds"] == 576


def test_memory_failure_uses_recent_high_oracle(tmp_path: Path) -> None:
    targets = [576, 576, 600, 600, 624, 624]
    long_correct = [True, True, False, False, False, False]
    oracle = [True] * len(targets)
    roots = [tmp_path / f"memory_{index}" for index in range(3)]
    for root in roots:
        _write_run(
            root,
            task="memory",
            targets=targets,
            kinds=[PROBE_KIND_MEMORY] * len(targets),
            expected=[-1] * len(targets),
            long_is_listen=[False] * len(targets),
            high_is_listen=[False] * len(targets),
            memory_long=long_correct,
            memory_oracle_high=oracle,
        )
    report = analyze_true_context_runs(
        [load_true_context_run(root) for root in roots]
    )
    assert report["recommendation"]["first_confirmed_failure_seconds"] == 600
    assert report["checkpoints"][1]["memory"]["oracle_high_accuracy"]["rate"] == 1.0
    assert report["checkpoints"][1]["memory"]["high_negative_accuracy"]["rate"] == 0.0


def test_terminal_candidate_is_not_reported_safe(tmp_path: Path) -> None:
    targets = [768] * 4 + [896] * 4
    expected = [False, False, True, True] * 2
    high_state = [False, False, True, True] * 2
    long_state = [*high_state[:4], True, True, True, True]
    roots = [tmp_path / f"candidate_{index}" for index in range(3)]
    for root in roots:
        _write_run(
            root,
            task="floor",
            targets=targets,
            kinds=[PROBE_KIND_FLOOR] * len(targets),
            expected=expected,
            long_is_listen=long_state,
            high_is_listen=high_state,
        )
    report = analyze_true_context_runs(
        [load_true_context_run(root) for root in roots]
    )
    recommendation = report["recommendation"]
    assert recommendation["first_candidate_failure_seconds"] == 896
    assert recommendation["first_confirmed_failure_seconds"] is None
    assert recommendation["empirical_safe_checkpoint_seconds"] == 768


def test_persistent_recent_control_failure_stops_safe_boundary(tmp_path: Path) -> None:
    targets = [784] * 4 + [800] * 4 + [816] * 4
    expected = [False, False, True, True] * 3
    clean = [False, False, True, True]
    broken = [False, True, True, False]
    recent_state = clean + broken * 2
    roots = [tmp_path / f"control_{index}" for index in range(3)]
    for root in roots:
        _write_run(
            root,
            task="floor",
            targets=targets,
            kinds=[PROBE_KIND_FLOOR] * len(targets),
            expected=expected,
            long_is_listen=[False, False, True, True] * 3,
            high_is_listen=recent_state,
        )
    report = analyze_true_context_runs(
        [load_true_context_run(root) for root in roots]
    )
    recommendation = report["recommendation"]
    assert recommendation["first_confirmed_failure_seconds"] is None
    assert recommendation["first_persistent_diagnostic_failure_seconds"] == 800
    assert recommendation["empirical_safe_checkpoint_seconds"] == 784
    assert recommendation["boundary_basis"] == "persistent_diagnostic_failure"


def test_loader_rejects_position_misalignment(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    _write_run(
        root,
        task="floor",
        targets=[600, 600],
        kinds=[PROBE_KIND_FLOOR, PROBE_KIND_FLOOR],
        expected=[0, 1],
        long_is_listen=[False, True],
        high_is_listen=[False, True],
    )
    trace_path = root / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["high_absolute_positions"][0] += 1
    np.savez_compressed(trace_path, **arrays)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="高位近期臂"):
        load_true_context_run(root)
