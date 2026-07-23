"""Easy-Turn Windows 运行器的轻量回归测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO_ROOT / "runners" / "easy_turn" / "run.py"
    spec = importlib.util.spec_from_file_location("easy_turn_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_prediction_uses_last_valid_state():
    runner = _load_runner()

    state, transcript = runner.extract_prediction(
        "我先想想<incomplete> 后来讲完了。<complete><|endoftext|>",
    )

    assert state == "complete"
    assert transcript == "我先想想 后来讲完了。"


def test_extract_prediction_reports_unknown_without_tag():
    runner = _load_runner()

    state, transcript = runner.extract_prediction("只有转录，没有状态")

    assert state == "unknown"
    assert transcript == "只有转录，没有状态"


def test_resolve_paths_matches_windows_layout():
    runner = _load_runner()

    paths = runner.resolve_paths(Path("C:/models/Easy-Turn"))

    assert paths["checkpoint"].parts[-4:] == ("models", "core", "easy-turn", "checkpoint.pt")
    assert paths["config"].parts[-2:] == ("conf", "train.yaml")
    assert paths["qwen_config"].name == "Qwen2.5-0.5B-Instruct-config"


def test_windows_paths_register_easy_turn_runner():
    path_config = yaml.safe_load((REPO_ROOT / "configs" / "paths.windows.yaml").read_text(encoding="utf-8"))

    easy_turn = path_config["models"]["easy_turn"]

    assert easy_turn["root"].endswith(r"Full-Duplex-Casecade\Easy-Turn")
    assert easy_turn["venv_python"].endswith(r"Easy-Turn\.venv\Scripts\python.exe")
    assert easy_turn["runner"].endswith(r"The_Floor_Control_Circuit\runners\easy_turn\run.py")


def test_log_mel_spectrogram_uses_official_frame_geometry():
    runner = _load_runner()
    waveform = torch.zeros(16_000, dtype=torch.float32)

    features = runner.compute_log_mel_spectrogram(waveform)

    assert features.shape == (100, 80)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_generation_compatibility_normalizes_upstream_arguments():
    runner = _load_runner()
    received = {}

    class DummyLanguageModel:
        def generate(self, *args, **kwargs):
            received.update(kwargs)
            return args

    model = SimpleNamespace(
        llama_model=DummyLanguageModel(),
        tokenizer=SimpleNamespace(pad_token_id=151665),
    )
    runner._patch_generation_compatibility(model)

    model.llama_model.generate(
        do_sample=False,
        pad_token_id=-100,
        top_p=0.0,
        top_k=0,
        temperature=1.0,
    )

    assert received["pad_token_id"] == 151665
    assert received["do_sample"] is False
    assert received["temperature"] == 1.0
    assert "top_p" not in received
    assert "top_k" not in received
