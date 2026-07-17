"""事件与标签双文件缓存的事务护栏。"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "wp1_run_events_transaction",
        REPO_ROOT / "scripts" / "wp1_run_events.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(np.zeros(240, dtype="<i2").tobytes())


def test_event_cache_requires_matching_completion_marker_and_hashes(tmp_path):
    module = _load_module()
    session_dir = tmp_path / "audio" / "session-a"
    out_dir = tmp_path / "events"
    session_dir.mkdir(parents=True)
    out_dir.mkdir()
    _write_wav(session_dir / "audio_ch0.wav")
    _write_wav(session_dir / "audio_ch1.wav")
    fingerprint = module._session_fingerprint(session_dir, "settings")
    events = pd.DataFrame({"event_id": ["e1"], "value": [1]})
    labels = pd.DataFrame({"step": [0, 1], "label": [0, 1]})

    module._write_session_outputs(events, labels, session_dir, out_dir, fingerprint)

    cached = module._cached_session_summary(session_dir, out_dir, fingerprint)
    assert cached is not None
    assert cached["n_events"] == 1
    assert cached["n_labels"] == 2

    pd.DataFrame({"step": [0, 1, 2], "label": [0, 1, 1]}).to_parquet(
        out_dir / "session-a.labels.parquet"
    )
    assert module._cached_session_summary(session_dir, out_dir, fingerprint) is None


def test_event_cache_rejects_two_readable_files_without_completion_marker(tmp_path):
    module = _load_module()
    session_dir = tmp_path / "audio" / "session-a"
    out_dir = tmp_path / "events"
    session_dir.mkdir(parents=True)
    out_dir.mkdir()
    _write_wav(session_dir / "audio_ch0.wav")
    _write_wav(session_dir / "audio_ch1.wav")
    pd.DataFrame({"event_id": ["e1"]}).to_parquet(
        out_dir / "session-a.events.parquet"
    )
    pd.DataFrame({"step": [0], "label": [1]}).to_parquet(
        out_dir / "session-a.labels.parquet"
    )
    fingerprint = module._session_fingerprint(session_dir, "settings")

    assert module._cached_session_summary(session_dir, out_dir, fingerprint) is None
