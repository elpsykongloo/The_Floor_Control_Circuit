"""MaAI Windows 运行器的轻量回归测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO_ROOT / "runners" / "maai" / "run.py"
    spec = importlib.util.spec_from_file_location("maai_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_paths_matches_windows_layout():
    runner = _load_runner()

    paths = runner.resolve_paths(Path("C:/models/MaAI"))

    assert paths["source"].name == "_source"
    assert paths["vap"].name == "vap_mimi_state_dict_tri_12.5hz_20000msec.pt"
    assert paths["vap_mc"].name == "vap_mc_mimi_state_dict_tri_12.5hz_20000msec.pt"
    assert paths["mimi_int8"].name == "continuous_mimi_int8.onnx"


def test_parse_device_maps_cuda_index():
    runner = _load_runner()

    assert runner.parse_device("cpu") == ("cpu", None)
    assert runner.parse_device("cuda") == ("cuda", 0)
    assert runner.parse_device("cuda:1") == ("cuda", 1)
    with pytest.raises(ValueError, match="设备仅支持"):
        runner.parse_device("gpu")


def test_prepare_mono_stream_adds_silent_channel_and_warmup_frame():
    runner = _load_runner()
    audio = np.array([0.25, -0.5], dtype=np.float32)

    channel1, channel2, metadata = runner.prepare_streams(audio, None, "vap_mono")

    assert channel1.shape == (runner.FRAME_SAMPLES * 2,)
    assert channel2.shape == channel1.shape
    np.testing.assert_array_equal(channel1[:2], audio)
    assert np.count_nonzero(channel1[2:]) == 0
    assert np.count_nonzero(channel2) == 0
    assert metadata["channel1_padded_samples"] == runner.FRAME_SAMPLES * 2 - 2


def test_prepare_dyadic_stream_aligns_and_pads_channels():
    runner = _load_runner()
    channel1 = np.ones(1_500, dtype=np.float32)
    channel2 = np.ones(3_000, dtype=np.float32)

    prepared1, prepared2, metadata = runner.prepare_streams(channel1, channel2, "vap_mc")

    assert prepared1.shape == prepared2.shape == (3_840,)
    assert metadata["channel1_padded_samples"] == 2_340
    assert metadata["channel2_padded_samples"] == 840


def test_prepare_dyadic_stream_requires_second_channel():
    runner = _load_runner()

    with pytest.raises(ValueError, match="必须提供通道 2"):
        runner.prepare_streams(np.ones(10, dtype=np.float32), None, "vap")


def test_summarize_frames_keeps_raw_probabilities():
    runner = _load_runner()
    mono_frames = [
        {"p_now": 0.2, "p_future": 0.3, "vad": 0.4},
        {"p_now": 0.6, "p_future": 0.7, "vad": 0.8},
    ]
    stereo_frames = [
        {"p_now": [0.2, 0.8], "p_future": [0.3, 0.7], "vad": [0.4, 0.6]},
        {"p_now": [0.6, 0.4], "p_future": [0.7, 0.3], "vad": [0.8, 0.2]},
    ]

    mono = runner.summarize_frames(mono_frames, "vap_mono")
    stereo = runner.summarize_frames(stereo_frames, "vap_mc")

    assert mono["mean_p_now"] == pytest.approx(0.4)
    assert stereo["mean_p_now"] == pytest.approx([0.4, 0.6])
    assert stereo["last"] == stereo_frames[-1]


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


def test_encoder_providers_reject_cuda_fallback():
    runner = _load_runner()

    class FakeSession:
        def __init__(self, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

    class FakeEncoder:
        def __init__(self, providers):
            self._onnx_sess = FakeSession(providers)

    class FakeVap:
        encoder1 = FakeEncoder(["CUDAExecutionProvider", "CPUExecutionProvider"])
        encoder2 = FakeEncoder(["CUDAExecutionProvider", "CPUExecutionProvider"])

    providers, fallback = runner._inspect_encoder_providers(FakeVap(), "cuda:0")
    assert fallback is False
    assert providers["encoder1"][0] == "CUDAExecutionProvider"

    FakeVap.encoder2 = FakeEncoder(["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="静默回退"):
        runner._inspect_encoder_providers(FakeVap(), "cuda:0")


def test_windows_paths_register_maai_runner():
    path_config = yaml.safe_load((REPO_ROOT / "configs" / "paths.windows.yaml").read_text(encoding="utf-8"))

    maai = path_config["models"]["maai"]

    assert maai["root"].endswith(r"Full-Duplex-Casecade\MaAI")
    assert maai["venv_python"].endswith(r"MaAI\.venv\Scripts\python.exe")
    assert maai["manifest"].endswith(r"MaAI\model_manifest.json")
    assert maai["runner"].endswith(r"The_Floor_Control_Circuit\runners\maai\run.py")
