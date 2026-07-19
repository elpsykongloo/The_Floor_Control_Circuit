"""MiniCPM-o 运行器的 Windows 路径与纯读出兼容性测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO_ROOT / "runners" / "minicpm_o" / "run.py"
    spec = importlib.util.spec_from_file_location("minicpm_o_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TextOnlyDuplexModel:
    """复现上游无条件调用 init_tts 的最小模型。"""

    def __init__(self) -> None:
        self.init_tts_calls = 0
        self.kwargs = None

    def init_tts(self, **kwargs) -> None:
        self.init_tts_calls += 1
        raise AttributeError("没有 tts 属性")

    def as_duplex(self, **kwargs):
        self.kwargs = kwargs
        self.init_tts()
        return SimpleNamespace(kind="duplex")


def test_text_only_duplex_skips_and_restores_upstream_tts_initialization():
    runner = _load_runner()
    model = _TextOnlyDuplexModel()

    duplex = runner._as_duplex_without_unused_tts(
        model,
        generate_audio=False,
        chunk_ms=1000,
    )

    assert duplex.kind == "duplex"
    assert model.kwargs == {"generate_audio": False, "chunk_ms": 1000}
    assert model.init_tts_calls == 0
    with pytest.raises(AttributeError, match="没有 tts 属性"):
        model.init_tts()
    assert model.init_tts_calls == 1


def test_audio_duplex_keeps_real_tts_initialization():
    runner = _load_runner()
    model = _TextOnlyDuplexModel()

    with pytest.raises(AttributeError, match="没有 tts 属性"):
        runner._as_duplex_without_unused_tts(
            model,
            generate_audio=True,
            chunk_ms=1000,
        )
    assert model.init_tts_calls == 1


def test_existing_wrong_nodot_alias_is_rejected(tmp_path):
    runner = _load_runner()
    root = tmp_path / "MiniCPM-o-4.5"
    root.mkdir()
    wrong_alias = tmp_path / "MiniCPM-o-4_5_nodot"
    wrong_alias.mkdir()

    with pytest.raises(SystemExit, match="指向错误"):
        runner._sanitize_model_root(str(root))
