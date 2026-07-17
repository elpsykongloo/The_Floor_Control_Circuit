"""MVE 内存安全数据路径：先抽样、单层读取、逐会话验证。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from floor_circuit.mve.dataset import (
    build_session_data,
    build_training_sample_plan,
    load_session_feature,
    load_training_sample,
    run_dir_for,
)
from floor_circuit.mve.preflight import RunSpec
from floor_circuit.probes.gru import predict_gru_batched
from floor_circuit.probes.linear import (
    downsample_negatives,
    fit_probe,
    fit_probe_streaming,
    score_sessions,
)


def _memory_world(
    tmp_path: Path,
    session_ids: list[str],
    n_steps: int = 18,
    dim: int = 5,
) -> tuple[Path, Path, dict[tuple[str, int], RunSpec]]:
    runs = tmp_path / "runs"
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    specs: dict[tuple[str, int], RunSpec] = {}
    rng = np.random.default_rng(91)
    for session_index, sid in enumerate(session_ids):
        rows = []
        for channel in (0, 1):
            run_dir = run_dir_for(runs, sid, channel)
            run_dir.mkdir(parents=True)
            group = zarr.open_group(str(run_dir), mode="w")
            labels = ((np.arange(n_steps) + session_index + channel) % 7 == 0).astype(np.int64)
            features = rng.normal(size=(n_steps, dim)).astype(np.float32)
            features[:, 0] += labels * 1.5
            acts = group.create_array(
                "acts_L4",
                shape=features.shape,
                dtype="float16",
                chunks=features.shape,
            )
            acts[:] = features.astype(np.float16)
            latent = group.create_array(
                "mimi_latent",
                shape=features.shape,
                dtype="float16",
                chunks=features.shape,
            )
            latent[:] = features.astype(np.float16)
            specs[(sid, channel)] = RunSpec(
                sid,
                channel,
                run_dir,
                n_steps,
                12.5,
                ("a" * 64, "b" * 64),
            )
            for step, label in enumerate(labels):
                rows.append(
                    {
                        "target": "T1",
                        "agent_channel": channel,
                        "step": step,
                        "label": int(label),
                        "delta_ms": 240,
                    }
                )
        pd.DataFrame(rows).to_parquet(labels_root / f"{sid}.parquet")
    return runs, labels_root, specs


def test_training_plan_matches_legacy_global_downsample(tmp_path):
    session_ids = [f"s{i}" for i in range(5)]
    train = session_ids[:3]
    runs, labels, specs = _memory_world(tmp_path, session_ids)
    full = build_session_data(runs, labels, train, 4, "T1", 240)
    X_all = np.concatenate([full[sid][0] for sid in train])
    y_all = np.concatenate([full[sid][1] for sid in train])
    expected_X, expected_y = downsample_negatives(
        X_all,
        y_all,
        ratio=5,
        rng=np.random.default_rng(3),
    )

    plan = build_training_sample_plan(labels, train, specs, "T1", 240, 5, 3)
    actual_X, actual_y = load_training_sample(runs, plan, 4)

    assert plan.n_available == len(y_all)
    assert np.array_equal(actual_y, expected_y)
    assert np.array_equal(actual_X, expected_X)


def test_streaming_probe_matches_legacy_and_reads_each_eval_once(tmp_path):
    session_ids = [f"s{i}" for i in range(8)]
    train, evals = session_ids[:5], session_ids[5:]
    runs, labels, specs = _memory_world(tmp_path, session_ids, n_steps=24)
    full = build_session_data(runs, labels, session_ids, 4, "T1", 240)
    legacy_fit = fit_probe(full, train, evals, [0.1, 1.0], seed=2, neg_ratio=5)
    legacy_scores = score_sessions(legacy_fit, full, evals)

    plan = build_training_sample_plan(labels, train, specs, "T1", 240, 5, 2)
    X_train, y_train = load_training_sample(runs, plan, 4)
    calls: list[str] = []

    def provide(sid: str) -> tuple[np.ndarray, np.ndarray]:
        calls.append(sid)
        return load_session_feature(runs, labels, sid, specs, 4, "T1", 240)

    streaming_fit, streaming_scores = fit_probe_streaming(
        X_train,
        y_train,
        evals,
        provide,
        [0.1, 1.0],
        seed=2,
    )

    assert calls == evals
    assert streaming_fit.best_c == legacy_fit.best_c
    assert streaming_fit.val_auc_by_c == pytest.approx(legacy_fit.val_auc_by_c)
    for sid in evals:
        assert np.array_equal(streaming_scores[sid][0], legacy_scores[sid][0])
        assert streaming_scores[sid][1] == pytest.approx(legacy_scores[sid][1])


def test_feature_loaders_enforce_explicit_memory_limits(tmp_path):
    runs, labels, specs = _memory_world(tmp_path, ["s0"])
    plan = build_training_sample_plan(labels, ["s0"], specs, "T1", 240, 5, 0)

    with pytest.raises(MemoryError, match="训练特征预计峰值"):
        load_training_sample(runs, plan, 4, max_bytes=1)
    with pytest.raises(MemoryError, match="评估特征预计峰值"):
        load_session_feature(runs, labels, "s0", specs, 4, "T1", 240, max_bytes=1)


def test_double_empty_t4_session_is_preserved_as_zero_row_cluster(tmp_path):
    runs, labels, specs = _memory_world(tmp_path, ["s0"])

    features, values = load_session_feature(
        runs,
        labels,
        "s0",
        specs,
        4,
        "T4",
        None,
    )

    assert features.shape == (0, 5)
    assert values.shape == (0,)


def test_streaming_probe_scores_empty_validation_cluster_explicitly():
    rng = np.random.default_rng(8)
    X_train = rng.normal(size=(40, 3)).astype(np.float32)
    y_train = np.tile([0, 1], 20)
    X_full = rng.normal(size=(8, 3)).astype(np.float32)
    y_full = np.tile([0, 1], 4)

    def provide(sid: str) -> tuple[np.ndarray, np.ndarray]:
        if sid == "empty":
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)
        return X_full.copy(), y_full.copy()

    _fit, scores = fit_probe_streaming(
        X_train,
        y_train,
        ["full", "empty"],
        provide,
        [0.1],
        seed=0,
    )

    assert scores["full"][0].shape == (8,)
    assert scores["empty"][0].shape == (0,)
    assert scores["empty"][1].shape == (0,)


def test_gru_validation_and_evaluation_are_batched():
    calls: list[int] = []

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, value):
            calls.append(len(value))
            return value[:, 0, 0]

    windows = np.arange(11 * 3 * 2, dtype=np.float32).reshape(11, 3, 2)
    scores = predict_gru_batched(
        FakeModel(),
        windows,
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        batch_size=4,
    )

    expected = torch.from_numpy(windows[:, 0, 0]).sigmoid().numpy()
    assert calls == [4, 4, 3]
    assert scores == pytest.approx(expected)
    empty = predict_gru_batched(
        FakeModel(),
        np.empty((0, 3, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        batch_size=4,
    )
    assert empty.shape == (0,)
