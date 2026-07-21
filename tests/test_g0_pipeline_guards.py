"""G0 准备与解码的划分隔离护栏。"""

from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_reads_only_target_shard_ids(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    module = _load_script("wp1_g0_prepare_guard", REPO_ROOT / "scripts" / "wp1_g0_prepare.py")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pq.write_table(pa.table({"session_id": ["val_a", "val_b"]}), data_dir / "val-00000.parquet")
    pq.write_table(pa.table({"session_id": ["test_a"]}), data_dir / "test-00000.parquet")

    assert module._read_shard_session_ids(tmp_path, "val-*.parquet") == {"val_a", "val_b"}


def test_dualturn_row_can_skip_continuous_mimi_features():
    from floor_circuit.data.dualturn import _row_to_session

    row = {
        "session_id": "demo",
        "dataset": "switchboard",
        "duration_s": 0.08,
        "num_frames": 1,
    }
    for channel in (0, 1):
        row[f"codes_ch{channel}"] = list(range(8))
        row[f"fvad_ch{channel}"] = [0.0] * 4
        for name in ("vad", "eot", "hold", "bot", "bc"):
            row[f"{name}_ch{channel}"] = [0]

    session = _row_to_session(row, include_mimi_feat=False)

    assert session.mimi_feat == {}
    assert session.codes[0].shape == (1, 8)
    assert session.fvad[1].shape == (1, 4)


def test_decode_split_filter_rejects_unreadable_metadata(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "runners" / "_shared"))
    module = _load_script("decode_mimi_guard", REPO_ROOT / "runners" / "moshi" / "decode_mimi.py")
    val_dir = tmp_path / "val_session"
    test_dir = tmp_path / "test_session"
    broken_dir = tmp_path / "broken_session"
    for path in (val_dir, test_dir, broken_dir):
        path.mkdir()
        (path / "codes_ch0.npy").touch()
    (val_dir / "meta.json").write_text(json.dumps({"split": "val"}), encoding="utf-8")
    (test_dir / "meta.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
    (broken_dir / "meta.json").write_text("{", encoding="utf-8")

    with pytest.raises(module.AdapterError, match="不可读"):
        module.select_batch_dirs(tmp_path, "val")

    (broken_dir / "codes_ch0.npy").unlink()
    assert module.select_batch_dirs(tmp_path, "val") == [val_dir]


def _write_pcm_wav(path: Path, n_frames: int, sample_rate: int = 24_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.zeros(n_frames, dtype="<i2").tobytes())


def test_decode_cache_validates_wav_format_and_length(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "runners" / "_shared"))
    module = _load_script(
        "decode_mimi_cache_guard",
        REPO_ROOT / "runners" / "moshi" / "decode_mimi.py",
    )
    codes_path = tmp_path / "codes.npy"
    out_path = tmp_path / "audio.wav"
    np.save(codes_path, np.zeros((10, 8), dtype=np.int64))

    _write_pcm_wav(out_path, n_frames=10 * 1920)
    cached = module.validate_existing_wav(codes_path, out_path, 24_000, 12.5)
    assert cached is not None
    assert cached["skipped"]

    _write_pcm_wav(out_path, n_frames=100)
    assert module.validate_existing_wav(codes_path, out_path, 24_000, 12.5) is None

    out_path.write_bytes("截断".encode())
    assert module.validate_existing_wav(codes_path, out_path, 24_000, 12.5) is None


def test_decode_shards_are_disjoint_and_complete(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "runners" / "_shared"))
    module = _load_script(
        "decode_mimi_shard_guard",
        REPO_ROOT / "runners" / "moshi" / "decode_mimi.py",
    )
    dirs = [tmp_path / f"session_{index:03d}" for index in range(11)]
    shards = [module.shard_batch_dirs(dirs, 3, index) for index in range(3)]

    assert set().union(*(set(shard) for shard in shards)) == set(dirs)
    assert all(set(shards[left]).isdisjoint(shards[right]) for left in range(3) for right in range(left + 1, 3))
    assert [len(shard) for shard in shards] == [4, 4, 3]

    with pytest.raises(module.AdapterError, match="至少为 1"):
        module.shard_batch_dirs(dirs, 0, 0)
    with pytest.raises(module.AdapterError, match="必须位于"):
        module.shard_batch_dirs(dirs, 2, 2)


def test_calibrate_parallelism_bounds_and_merges_integer_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    module = _load_script(
        "wp1_g0_calibrate_parallel_guard",
        REPO_ROOT / "scripts" / "wp1_g0_calibrate.py",
    )

    assert module._resolve_jobs(8, 3) == 3
    assert 1 <= module._resolve_jobs(0, 100) <= 100
    with pytest.raises(ValueError, match="非负整数"):
        module._resolve_jobs(-1, 3)

    first = {
        cls: {"hits": 1, "n_pred": 2, "n_gold": 3}
        for cls in module.G0_CLASSES
    }
    second = {
        cls: {"hits": 4, "n_pred": 5, "n_gold": 6}
        for cls in module.G0_CLASSES
    }
    merged = module._merge_counts(None, first)
    merged = module._merge_counts(merged, second)
    assert all(
        merged[cls] == {"hits": 5, "n_pred": 7, "n_gold": 9}
        for cls in module.G0_CLASSES
    )
