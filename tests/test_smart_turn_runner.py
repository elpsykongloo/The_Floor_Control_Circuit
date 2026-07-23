"""Smart Turn v3.2 Windows 运行器的轻量回归测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO_ROOT / "runners" / "smart_turn" / "run.py"
    spec = importlib.util.spec_from_file_location("smart_turn_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probability_threshold_matches_official_rule():
    runner = _load_runner()

    assert runner.classify_probability(0.5) == (0, "incomplete", "话轮未完成")
    assert runner.classify_probability(0.500001) == (1, "complete", "话轮完成")


def test_prepare_audio_window_left_pads_short_audio():
    runner = _load_runner()
    audio = np.array([0.25, -0.5], dtype=np.float32)

    window, metadata = runner.prepare_audio_window(audio)

    assert window.shape == (runner.WINDOW_SAMPLES,)
    assert np.count_nonzero(window[:-2]) == 0
    np.testing.assert_array_equal(window[-2:], audio)
    assert metadata["padded_samples"] == runner.WINDOW_SAMPLES - 2
    assert metadata["truncated_samples"] == 0


def test_prepare_audio_window_keeps_latest_samples():
    runner = _load_runner()
    audio = np.arange(runner.WINDOW_SAMPLES + 3, dtype=np.float32)

    window, metadata = runner.prepare_audio_window(audio)

    np.testing.assert_array_equal(window, audio[-runner.WINDOW_SAMPLES :])
    assert metadata["padded_samples"] == 0
    assert metadata["truncated_samples"] == 3


def test_resolve_paths_matches_windows_layout():
    runner = _load_runner()

    paths = runner.resolve_paths(Path("C:/models/Smart-Turn"))

    assert paths["cpu_model"].name == "smart-turn-v3.2-cpu.onnx"
    assert paths["gpu_model"].name == "smart-turn-v3.2-gpu.onnx"
    assert paths["source"].name == "_source"


def test_whisper_features_have_official_shape():
    runner = _load_runner()
    time_axis = np.arange(runner.WINDOW_SAMPLES, dtype=np.float32) / runner.SAMPLE_RATE
    audio = np.sin(2 * np.pi * 440.0 * time_axis).astype(np.float32)

    features = runner.compute_input_features(audio)

    assert features.shape == (1, 80, 800)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_load_audio_resamples_with_soxr(tmp_path):
    runner = _load_runner()
    source_path = tmp_path / "sample-8khz.wav"
    time_axis = np.arange(800, dtype=np.float32) / 8_000
    audio = np.sin(2 * np.pi * 220.0 * time_axis).astype(np.float32)
    sf.write(source_path, audio, 8_000, subtype="PCM_16")

    waveform, metadata = runner._load_audio(source_path)

    assert waveform.dtype == np.float32
    assert waveform.shape == (1_600,)
    assert metadata["original_sample_rate"] == 8_000
    assert metadata["sample_rate"] == runner.SAMPLE_RATE
    assert metadata["resampled_samples"] == 1_600


def test_cuda_runtime_manifest_checks_size_and_hash(tmp_path):
    runner = _load_runner()
    runtime_path = tmp_path / "cuda12"
    runtime_path.mkdir()
    dll_path = runtime_path / "cudart64_12.dll"
    dll_path.write_bytes(b"cuda-runtime")
    manifest = {
        "cuda": "12.4",
        "cudnn_major": 9,
        "total_size_bytes": dll_path.stat().st_size,
        "files": [
            {
                "name": dll_path.name,
                "size_bytes": dll_path.stat().st_size,
                "sha256": runner.sha256_file(dll_path),
            }
        ],
    }
    (runtime_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    report, errors = runner._inspect_cuda_runtime(runtime_path, verify_hashes=True)

    assert errors == []
    assert report["files"] == 1
    assert report["hashes_verified"] is True

    dll_path.write_bytes(b"corrupted")
    _, errors = runner._inspect_cuda_runtime(runtime_path, verify_hashes=True)
    assert any("大小不匹配" in error for error in errors)
    assert any("SHA-256 不匹配" in error for error in errors)


def test_windows_paths_register_smart_turn_runner():
    path_config = yaml.safe_load((REPO_ROOT / "configs" / "paths.windows.yaml").read_text(encoding="utf-8"))

    smart_turn = path_config["models"]["smart_turn"]

    assert smart_turn["root"].endswith(r"Full-Duplex-Casecade\Smart-Turn")
    assert smart_turn["venv_python"].endswith(r"Smart-Turn\.venv\Scripts\python.exe")
    assert smart_turn["manifest"].endswith(r"Smart-Turn\model_manifest.json")
    assert smart_turn["cuda_runtime"].endswith(r"Smart-Turn\runtime\cuda12")
    assert smart_turn["runner"].endswith(r"The_Floor_Control_Circuit\runners\smart_turn\run.py")
