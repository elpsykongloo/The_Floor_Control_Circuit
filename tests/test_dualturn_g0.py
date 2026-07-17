"""第二轮新增：DualTurn 真实 schema 载入器 + G0 二值轨评分单测。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from floor_circuit.data.dualturn import (
    DualturnSession,
    iter_sessions,
    load_frame_labels,
    split_sessions,
)
from floor_circuit.events.g0 import (
    accumulate_counts,
    build_pred_tracks,
    finalize_counts,
    score_binary_tracks,
)
from floor_circuit.schemas import Event, EventKind, Seg

HZ = 12.5


def make_dualturn_dir(tmp_path, sessions=("sw_0001", "sw_0002"), n_frames=50):
    """按 V2 盘点冻结的 schema 构造合成发布物。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for sid in sessions:
        row = {
            "session_id": sid,
            "dataset": "switchboard",
            "duration_s": n_frames / HZ,
            "num_frames": n_frames,
        }
        for ch in (0, 1):
            row[f"codes_ch{ch}"] = rng.integers(0, 2048, n_frames * 8, dtype=np.int16).tolist()
            row[f"mimi_feat_ch{ch}"] = np.asarray(rng.normal(0, 1, n_frames * 512), dtype=np.float16).tolist()
            for name in ("vad", "eot", "hold", "bot", "bc"):
                track = np.zeros(n_frames, dtype=np.int8)
                if name == "vad":
                    track[5:20] = 1
                elif name == "eot":
                    track[19] = 1
                elif name == "bot":
                    track[5] = 1
                row[f"{name}_ch{ch}"] = track.tolist()
            row[f"fvad_ch{ch}"] = np.linspace(0, 1, n_frames * 4, dtype=np.float32).tolist()
        rows.append(row)
    pd.DataFrame(rows).to_parquet(data_dir / "test-00000.parquet")
    (tmp_path / "splits.json").write_text(
        json.dumps(
            {
                "description": "synthetic",
                "total_sessions": len(sessions),
                "split_counts": {"test": len(sessions)},
                "sessions_without_audio": 0,
                "splits": {"test": list(sessions)},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestDualturnLoader:
    def test_iter_and_shapes(self, tmp_path):
        root = make_dualturn_dir(tmp_path)
        sessions = list(iter_sessions(root))
        assert len(sessions) == 2 and isinstance(sessions[0], DualturnSession)
        s = sessions[0]
        assert s.codes[0].shape == (50, 8) and s.codes[0].dtype == np.int16
        assert s.mimi_feat[1].shape == (50, 512) and s.mimi_feat[1].dtype == np.float16
        assert s.tracks[0]["eot"][19] == 1 and s.tracks[0]["bot"][5] == 1
        assert s.fvad[0].shape == (50, 4)

    def test_filter_and_limit(self, tmp_path):
        root = make_dualturn_dir(tmp_path)
        only = list(iter_sessions(root, sessions={"sw_0002"}))
        assert [s.session_id for s in only] == ["sw_0002"]
        assert len(list(iter_sessions(root, limit=1))) == 1

    def test_split_sessions(self, tmp_path):
        root = make_dualturn_dir(tmp_path)
        assert split_sessions(root, "test") == ["sw_0001", "sw_0002"]
        with pytest.raises(KeyError):
            split_sessions(root, "train")

    def test_split_sessions_session_to_split_mapping(self, tmp_path):
        root = make_dualturn_dir(tmp_path)
        (root / "splits.json").write_text(
            json.dumps(
                {
                    "split_counts": {"train": 1, "test": 1},
                    "splits": {"sw_0001": "train", "sw_0002": "test"},
                }
            ),
            encoding="utf-8",
        )
        assert split_sessions(root, "test") == ["sw_0002"]
        with pytest.raises(KeyError, match=r"\['test', 'train'\]"):
            split_sessions(root, "val")

    def test_load_frame_labels(self, tmp_path):
        root = make_dualturn_dir(tmp_path)
        tracks = load_frame_labels(root, "sw_0001")
        assert set(tracks[0]) == {"vad", "eot", "hold", "bot", "bc"}

    def test_bad_length_raises(self, tmp_path):
        root = make_dualturn_dir(tmp_path)
        df = pd.read_parquet(root / "data" / "test-00000.parquet")
        df.loc[0, "num_frames"] = 49  # 与展平长度不再匹配
        df.to_parquet(root / "data" / "test-00000.parquet")
        with pytest.raises(ValueError, match="展平长度"):
            list(iter_sessions(root, limit=1))


class TestG0BinaryTracks:
    def make_pred(self, n=100):
        ipus = [Seg(1.0, 2.0), Seg(2.6, 4.0), Seg(6.0, 6.5)]
        events = [
            Event(EventKind.ONSET, 0, 1.0),
            Event(EventKind.ONSET, 0, 2.6),
            Event(EventKind.TURNEND, 0, 4.0),
            Event(EventKind.BC, 0, 6.0, t_end=6.5),
        ]
        return build_pred_tracks(ipus, events, 0, n, HZ)

    def test_build_semantics(self):
        tracks = self.make_pred()
        assert tracks["bot"][round(1.0 * HZ)] == 1
        assert tracks["bot"][round(2.6 * HZ)] == 1
        assert tracks["hold"][round(2.0 * HZ)] == 1  # IPU 末非 TURNEND → hold
        assert tracks["eot"][round(4.0 * HZ)] == 1
        assert tracks["bc"][int(6.0 * HZ) : int(np.ceil(6.5 * HZ))].all()
        # bc IPU 的起止不产生 bot/eot/hold
        assert tracks["bot"][round(6.0 * HZ)] == 0
        assert tracks["eot"][round(6.5 * HZ)] == 0 and tracks["hold"][round(6.5 * HZ)] == 0

    def test_score_perfect_and_tolerance(self):
        pred = self.make_pred()
        gold = {k: v.copy() for k, v in pred.items()}
        rep = score_binary_tracks(pred, gold, tolerance_frames=2)
        assert rep["macro_f1"] == 1.0
        gold2 = {k: v.copy() for k, v in pred.items()}
        idx = int(np.nonzero(gold2["eot"])[0][0])
        gold2["eot"][idx] = 0
        gold2["eot"][idx + 2] = 1  # 容差内偏移
        assert score_binary_tracks(pred, gold2, 2)["per_class"]["eot"]["f1"] == 1.0
        gold3 = {k: v.copy() for k, v in pred.items()}
        gold3["eot"][idx] = 0
        gold3["eot"][idx + 4] = 1  # 容差外
        assert score_binary_tracks(pred, gold3, 2)["per_class"]["eot"]["f1"] == 0.0

    def test_endpoint_clamp(self):
        # IPU 一直延伸到音频末尾：round(t_end*hz) == n_frames，应钳到末帧而非丢弃
        n = 50
        t_end = n / HZ  # 恰为片段末尾
        ipus = [Seg(1.0, t_end)]
        events = [Event(EventKind.ONSET, 0, 1.0), Event(EventKind.TURNEND, 0, t_end)]
        tracks = build_pred_tracks(ipus, events, 0, n, HZ)
        assert tracks["eot"][n - 1] == 1
        # 非 TURNEND 的末尾 IPU 同理落 hold
        tracks2 = build_pred_tracks(ipus, [Event(EventKind.ONSET, 0, 1.0)], 0, n, HZ)
        assert tracks2["hold"][n - 1] == 1

    def test_micro_accumulate(self):
        pred = self.make_pred()
        gold = {k: v.copy() for k, v in pred.items()}
        totals = accumulate_counts(None, pred, gold, 2)
        totals = accumulate_counts(totals, pred, gold, 2)
        rep = finalize_counts(totals)
        assert rep["macro_f1"] == 1.0
        assert rep["per_class"]["bot"]["n_gold"] == 4  # 两个会话 × 2 个非-bc onset


class TestSelfCheckUpgrade:
    def test_label_value_hits_locates_key(self, tmp_path):
        payload = {
            "segments": [
                {
                    "channelIndex": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "你好",
                    "annotation": {"turnState": "Complete"},
                }
            ]
        }
        (tmp_path / "a.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        from floor_circuit.data.smoothconv import self_check

        rep = self_check(tmp_path)
        assert rep["label_value_hits"] == {"annotation.turnState": 1}
        assert "annotation.turnState" in rep["key_union"]
        assert rep["sample_item"]["text"] == "你好"

    def test_inventory_fallback(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "big.tar").write_bytes(b"0" * 2048)
        from floor_circuit.data.smoothconv import self_check

        rep = self_check(tmp_path)
        assert rep["n_files_found"] == 0
        assert rep["inventory"]["by_suffix"][".jsonl"] == 1
        assert rep["inventory"]["largest"][0]["path"] == "big.tar"
