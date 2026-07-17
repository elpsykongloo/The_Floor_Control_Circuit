"""G1 正式裁决输入预检与时间域对齐护栏。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import zarr

from floor_circuit.cachelib.manifest import RunManifest, save_manifest, sha256_file
from floor_circuit.mve.dataset import eligible_rows, run_dir_for
from floor_circuit.mve.preflight import (
    MvePreflightError,
    RunSpec,
    preflight_mve_inputs,
    sync_labels_atomic,
    validate_baseline_alignment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_wp7():
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("wp7_run_mve_hardening", scripts / "wp7_run_mve.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labels(n_steps: int) -> pd.DataFrame:
    rows = []
    for channel in (0, 1):
        for step in range(n_steps):
            label = step % 2
            rows.extend(
                [
                    {
                        "target": "T1",
                        "agent_channel": channel,
                        "step": step,
                        "label": label,
                        "delta_ms": 240,
                    },
                    {
                        "target": "T4",
                        "agent_channel": channel,
                        "step": step,
                        "label": label,
                        "delta_ms": None,
                    },
                    {
                        "target": "T5",
                        "agent_channel": channel,
                        "step": step,
                        "label": step % 5,
                        "delta_ms": None,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _write_run(
    root: Path,
    audio_root: Path,
    sid: str,
    channel: int,
    n_steps: int,
    layers: list[int],
) -> None:
    run_dir = run_dir_for(root, sid, channel)
    run_dir.mkdir(parents=True)
    group = zarr.open_group(str(run_dir), mode="w")
    for layer in layers:
        array = group.create_array(
            f"acts_L{layer}",
            shape=(n_steps, 8),
            dtype="float16",
            chunks=(n_steps, 8),
        )
        array[:] = np.zeros((n_steps, 8), dtype=np.float16)
    latent = group.create_array(
        "mimi_latent",
        shape=(n_steps, 4),
        dtype="float16",
        chunks=(n_steps, 4),
    )
    latent[:] = np.zeros((n_steps, 4), dtype=np.float16)
    save_manifest(
        run_dir,
        RunManifest(
            model="moshi",
            mode="R1",
            session_id=sid,
            layers=layers,
            hidden_dim=8,
            clock_hz=12.5,
            n_steps=n_steps,
            source_audio={
                str(audio_root / sid / "audio_ch0.wav"): sha256_file(
                    audio_root / sid / "audio_ch0.wav"
                ),
                str(audio_root / sid / "audio_ch1.wav"): sha256_file(
                    audio_root / sid / "audio_ch1.wav"
                ),
            },
            mimi_latent=True,
            code_version="test-runner",
            extra={
                "delay_application": "global_once_before_streaming_forward",
                "execution": {
                    "forward_mode": "streaming_teacher_forced_backbone",
                    "max_seconds": n_steps / 12.5,
                    "mimi_chunk_seconds": 0.08,
                    "mimi_chunk_frames": 1,
                    "forward_chunk_steps": 128,
                    "transformer_state_preserved": True,
                    "mimi_state_preserved": True,
                    "depformer_skipped": True,
                    "latent_kind": "pre_quantization_continuous",
                },
            },
        ),
    )


def _world(tmp_path: Path, sessions: list[str], n_steps: int = 6):
    runs = tmp_path / "runs"
    labels = tmp_path / "labels"
    audio = tmp_path / "audio"
    labels.mkdir()
    layers = [4, 12, 20, 28]
    for sid in sessions:
        session_audio = audio / sid
        session_audio.mkdir(parents=True)
        for channel in (0, 1):
            sf.write(
                session_audio / f"audio_ch{channel}.wav",
                np.zeros(160, dtype=np.float32),
                16_000,
            )
        for channel in (0, 1):
            _write_run(runs, audio, sid, channel, n_steps, layers)
        _labels(n_steps).to_parquet(labels / f"{sid}.parquet")
    return runs, labels, audio, layers


def test_sync_labels_atomic_refreshes_stale_copy(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    fresh = _labels(4)
    source_path = source / "s1.labels.parquet"
    fresh.to_parquet(source_path)
    payload = source_path.read_bytes()
    (source / "s1.complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session": "s1",
                "outputs": {
                    "labels": {
                        "name": source_path.name,
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (dest / "s1.parquet").write_bytes("陈旧内容".encode())

    hashes = sync_labels_atomic(source, dest, ["s1"])

    assert pd.read_parquet(dest / "s1.parquet").equals(fresh)
    assert len(hashes["s1"]) == 64
    assert not list(dest.glob(".s1.parquet.*.tmp"))


def test_sync_labels_rejects_missing_completion_marker(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    _labels(4).to_parquet(source / "s1.labels.parquet")

    with pytest.raises(MvePreflightError, match="标签或完成标记"):
        sync_labels_atomic(source, dest, ["s1"])


def test_preflight_validates_runs_arrays_latent_and_labels(tmp_path):
    sessions = ["s1", "s2"]
    runs, labels, audio, layers = _world(tmp_path, sessions)

    specs, report = preflight_mve_inputs(
        runs,
        labels,
        audio,
        sessions,
        layers,
        expected_n_steps=6,
        expected_clock_hz=12.5,
        t1_delta_ms=240,
        expected_code_version="test-runner",
        expected_max_seconds=6 / 12.5,
        expected_mimi_chunk_seconds=0.08,
        expected_forward_chunk_steps=128,
    )

    assert len(specs) == 4
    assert report["n_runs"] == 4
    assert report["required_arrays_per_run"][-1] == "mimi_latent"


def test_preflight_rejects_quantized_latent_and_missing_layer(tmp_path):
    runs, labels, audio, layers = _world(tmp_path, ["s1"])
    run_dir = run_dir_for(runs, "s1", 1)
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.extra["execution"]["latent_kind"] = "quantized"
    save_manifest(run_dir, manifest)
    group = zarr.open_group(str(run_dir_for(runs, "s1", 0)), mode="a")
    del group["acts_L28"]

    with pytest.raises(MvePreflightError) as caught:
        preflight_mve_inputs(
            runs,
            labels,
            audio,
            ["s1"],
            layers,
            6,
            12.5,
            240,
            "test-runner",
            6 / 12.5,
            0.08,
            128,
        )

    message = str(caught.value)
    assert "缺少数组 acts_L28" in message
    assert "不是量化前连续表征" in message


def test_preflight_rejects_stale_runner_execution_and_source_audio(tmp_path):
    runs, labels, audio, layers = _world(tmp_path, ["s1"])
    run_dir = run_dir_for(runs, "s1", 0)
    manifest = RunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest.code_version = "stale-runner"
    manifest.extra["execution"]["mimi_chunk_seconds"] = 0.16
    manifest.extra["execution"]["mimi_state_preserved"] = False
    save_manifest(run_dir, manifest)
    sf.write(
        audio / "s1" / "audio_ch0.wav",
        np.ones(160, dtype=np.float32),
        16_000,
    )

    with pytest.raises(MvePreflightError) as caught:
        preflight_mve_inputs(
            runs,
            labels,
            audio,
            ["s1"],
            layers,
            6,
            12.5,
            240,
            "test-runner",
            6 / 12.5,
            0.08,
            128,
        )

    message = str(caught.value)
    assert "code_version" in message
    assert "mimi_chunk_seconds" in message
    assert "mimi_state_preserved" in message
    assert "当前 CANDOR WAV 不一致" in message


def test_eligible_rows_caps_and_orders_by_manifest_steps():
    labels = pd.DataFrame(
        {
            "target": ["T1"] * 4,
            "agent_channel": [0] * 4,
            "step": [5, 1, 3, 0],
            "label": [1, 0, 1, 0],
            "delta_ms": [240] * 4,
        }
    )
    rows = eligible_rows(labels, "T1", 240, 0, max_steps=4)
    assert rows["step"].tolist() == [0, 1, 3]


def test_hazard_baseline_uses_each_role_n_steps(tmp_path, monkeypatch):
    module = _load_wp7()
    labels_root = tmp_path / "flat"
    labels_root.mkdir()
    for sid in ("train", "eval"):
        _labels(8).to_parquet(labels_root / f"{sid}.parquet")
    monkeypatch.setattr(module, "_labels_root", lambda: labels_root)
    specs = {
        (sid, channel): RunSpec(
            sid,
            channel,
            tmp_path,
            4,
            12.5,
            ("a" * 64, "b" * 64),
        )
        for sid in ("train", "eval")
        for channel in (0, 1)
    }

    result = module.hazard_baseline(["train"], ["eval"], "T1", 240, specs)

    assert len(result["eval"][0]) == 8
    assert len(result["eval"][1]) == 8


def test_audio_prefix_reader_stops_at_run_time_domain(tmp_path):
    module = _load_wp7()
    path = tmp_path / "long.wav"
    sample_rate = 16_000
    sf.write(path, np.zeros(sample_rate * 3, dtype=np.float32), sample_rate)

    wav, loaded_rate = module._load_wav_prefix(path, 0.8)

    assert loaded_rate == sample_rate
    assert len(wav) == 12_800


def test_baseline_alignment_rejects_different_label_rows():
    reference = {"s1": (np.array([0, 1]), np.array([0.2, 0.8]))}
    baselines = {
        "hazard": {"s1": (np.array([0, 1]), np.array([0.1, 0.7]))},
        "mimi": {"s1": (np.array([1, 0]), np.array([0.6, 0.4]))},
        "acoustic_gru": {"s1": (np.array([0, 1]), np.array([0.3, 0.9]))},
    }

    with pytest.raises(MvePreflightError, match="标签行与探针不一致"):
        validate_baseline_alignment(
            reference,
            baselines,
            {"hazard", "mimi", "acoustic_gru"},
            "T1",
        )
