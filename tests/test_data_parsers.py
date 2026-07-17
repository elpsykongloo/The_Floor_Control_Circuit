"""语料解析器单测：SmoothConv JSON 变体、CANDOR zip 索引/解压、Backbiter CSV、DualTurn 盘点。"""

from __future__ import annotations

import json
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from floor_circuit.data import candor, dualturn, smoothconv


class TestSmoothConv:
    def make(self, tmp_path, payload, name="a.json"):
        p = tmp_path / name
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return p

    def test_standard_schema(self, tmp_path):
        p = self.make(
            tmp_path,
            {
                "segments": [
                    {"channelIndex": 0, "start": 0.5, "end": 2.0, "text": "今天天气[laugh]不错", "turn": "complete"},
                    {"channelIndex": 1, "start": 2.2, "end": 2.5, "text": "嗯嗯", "turn": "backchannel"},
                ]
            },
        )
        df = smoothconv.parse_file(p)
        assert len(df) == 2
        assert df.iloc[0]["text_clean"] == "今天天气 不错"
        assert df.iloc[1]["turn_label"] == "backchannel"

    def test_alias_and_ms(self, tmp_path):
        p = self.make(
            tmp_path,
            [
                {"channel": "A", "startTime": 500, "endTime": 2000, "transcript": "你好", "turnType": "Incomplete"},
                {"channel": "B", "startTime": 2500, "endTime": 4000, "transcript": "好", "turnType": "wait"},
            ],
        )
        df = smoothconv.parse_file(p)
        assert df.iloc[0]["start_s"] == 0.5 and df.iloc[0]["end_s"] == 2.0  # 毫秒→秒
        assert df.iloc[0]["channel"] == 0 and df.iloc[1]["channel"] == 1
        assert df.iloc[0]["turn_label"] == "incomplete"

    def test_gold_turns_and_va(self, tmp_path):
        p = self.make(
            tmp_path,
            {"segments": [{"channelIndex": 0, "start": 1.0, "end": 2.0, "text": "x", "turn": "complete"}]},
        )
        df = smoothconv.parse_file(p)
        turns = smoothconv.to_gold_turns(df, 0)
        assert turns[0].label == "complete" and turns[0].end == 2.0
        assert smoothconv.to_va_segs(df, 0)[0].start == 1.0

    def test_self_check(self, tmp_path):
        self.make(tmp_path, {"segments": [{"channelIndex": 0, "start": 0, "end": 1, "text": "a", "turn": "complete"}]})
        self.make(tmp_path, {"bad": "shape"}, name="b.json")
        rep = smoothconv.self_check(tmp_path)
        assert rep["n_files_found"] == 2 and rep["files_ok"] == 1 and len(rep["errors"]) == 1


class TestCandor:
    def make_zip(self, tmp_path):
        zp = tmp_path / "processed_media_part_001.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("sessA/audio/mix.wav", b"RIFFxxxx")
            zf.writestr("sessA/transcription/transcript_backbiter.csv", "speaker,start,stop,utterance\nA,0,1,hi\n")
            zf.writestr("sessA/video.mp4", b"\x00\x00")
            zf.writestr("sessB/audio/mix.flac", b"fLaC")
        return zp

    def test_index_and_summary(self, tmp_path):
        self.make_zip(tmp_path)
        idx = candor.index_zips(tmp_path)
        assert set(idx["session_id"]) == {"sessA", "sessB"}
        assert (idx["kind"] == "transcript_backbiter").sum() == 1
        summary = candor.summarize_index(idx)
        assert summary["n_sessions"] == 2 and summary["sessions_with_backbiter"] == 1

    def test_extract(self, tmp_path):
        self.make_zip(tmp_path)
        idx = candor.index_zips(tmp_path)
        out = candor.extract_members(idx, ["sessA"], ["audio", "transcript_backbiter"], tmp_path / "out")
        rels = sorted(str(p.relative_to(tmp_path / "out")).replace("\\", "/") for p in out)
        assert rels == ["sessA/audio/mix.wav", "sessA/transcription/transcript_backbiter.csv"]

    def test_backbiter_import(self, tmp_path):
        p = tmp_path / "bb.csv"
        p.write_text("speaker,start,stop,utterance\nA,0.0,1.5,hello there\nB,1.6,2.0,yeah\n", encoding="utf-8")
        df = candor.import_backbiter(p)
        assert list(df.columns) == ["speaker", "start_s", "end_s", "text"]
        assert df.iloc[1]["text"] == "yeah"

    def test_backbiter_alias_error(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("who_knows,a,b\n1,2,3\n", encoding="utf-8")
        with pytest.raises(KeyError):
            candor.import_backbiter(p)


class TestDualturn:
    def test_safetensors_header(self, tmp_path):
        header = {"feat": {"dtype": "I32", "shape": [10, 8], "data_offsets": [0, 320]}}
        blob = json.dumps(header).encode()
        p = tmp_path / "x.safetensors"
        p.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 320)
        info = dualturn.read_safetensors_header(p)
        assert info["feat"]["dtype"] == "I32" and info["feat"]["shape"] == [10, 8]

    def test_inspect_dir(self, tmp_path):
        np.save(tmp_path / "codes.npy", np.zeros((5, 8), dtype=np.int32))
        (tmp_path / "splits.json").write_text(json.dumps({"train": ["a"], "test": ["b"]}), encoding="utf-8")
        rep = dualturn.inspect_dir(tmp_path)
        assert rep["n_files"] == 2
        peek = rep["peeks"][".npy"][0]["info"]
        assert peek["dtype"] == "int32"  # 离散码判据
        assert rep["splits_json"] == {"train": 1, "test": 1}

    def test_loader_not_ready(self, tmp_path):
        with pytest.raises(NotImplementedError):
            dualturn.load_frame_labels(tmp_path, "s1")


class TestPandasParquetRoundtrip:
    def test_events_labels_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.to_parquet(tmp_path / "t.parquet")
        back = pd.read_parquet(tmp_path / "t.parquet")
        assert back.equals(df)
