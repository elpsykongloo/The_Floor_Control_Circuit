"""G1 正式裁决输入预检与时间域对齐护栏。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import zarr

from floor_circuit.cachelib.manifest import RunManifest, save_manifest
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


def _write_run(root: Path, sid: str, channel: int, n_steps: int, layers: list[int]) -> None:
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
            source_audio={"a.wav": "a" * 64, "b.wav": "b" * 64},
            mimi_latent=True,
            extra={"execution": {"latent_kind": "pre_quantization_continuous"}},
        ),
    )


def _world(tmp_path: Path, sessions: list[str], n_steps: int = 6):
    runs = tmp_path / "runs"
    labels = tmp_path / "labels"
    labels.mkdir()
    layers = [4, 12, 20, 28]
    for sid in sessions:
        for channel in (0, 1):
            _write_run(runs, sid, channel, n_steps, layers)
        _labels(n_steps).to_parquet(labels / f"{sid}.parquet")
    return runs, labels, layers


def test_sync_labels_atomic_refreshes_stale_copy(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    fresh = _labels(4)
    fresh.to_parquet(source / "s1.labels.parquet")
    (dest / "s1.parquet").write_bytes("陈旧内容".encode())

    hashes = sync_labels_atomic(source, dest, ["s1"])

    assert pd.read_parquet(dest / "s1.parquet").equals(fresh)
    assert len(hashes["s1"]) == 64
    assert not list(dest.glob(".s1.parquet.*.tmp"))


def test_preflight_validates_runs_arrays_latent_and_labels(tmp_path):
    sessions = ["s1", "s2"]
    runs, labels, layers = _world(tmp_path, sessions)

    specs, report = preflight_mve_inputs(
        runs,
        labels,
        sessions,
        layers,
        expected_n_steps=6,
        expected_clock_hz=12.5,
        t1_delta_ms=240,
    )

    assert len(specs) == 4
    assert report["n_runs"] == 4
    assert report["required_arrays_per_run"][-1] == "mimi_latent"


def test_preflight_rejects_quantized_latent_and_missing_layer(tmp_path):
    runs, labels, layers = _world(tmp_path, ["s1"])
    run_dir = run_dir_for(runs, "s1", 1)
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.extra["execution"]["latent_kind"] = "quantized"
    save_manifest(run_dir, manifest)
    group = zarr.open_group(str(run_dir_for(runs, "s1", 0)), mode="a")
    del group["acts_L28"]

    with pytest.raises(MvePreflightError) as caught:
        preflight_mve_inputs(runs, labels, ["s1"], layers, 6, 12.5, 240)

    message = str(caught.value)
    assert "缺少数组 acts_L28" in message
    assert "不是量化前连续表征" in message


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
