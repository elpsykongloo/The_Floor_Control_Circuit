"""SmoothConv / DuplexConv 专家 JSON → 统一 schema（两库同 schema，解析器复用）。

统一输出列：channel:int, start_s:float, end_s:float, text_raw:str, text_clean:str,
turn_label ∈ {complete, incomplete, backchannel, wait}。
字段名做了别名兼容；真实文件的字段覆盖率用 self_check() 在本机核对
（scripts/wp2_parse_smoothconv.py --self-check），确认后可收紧别名表。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from floor_circuit.schemas import Seg, Turn, TurnLabel

_CHANNEL_KEYS = ("channelIndex", "channel_index", "channel", "channelId", "spk", "speaker")
_START_KEYS = ("start", "startTime", "start_time", "begin", "beginTime", "from")
_END_KEYS = ("end", "endTime", "end_time", "stop", "stopTime", "to")
_TEXT_KEYS = ("text", "transcript", "transcription", "content", "utterance", "sentence")
_TURN_KEYS = ("turn", "turnType", "turn_type", "turnLabel", "turn_label", "type", "label")
_SEGMENT_KEYS = ("segments", "utterances", "annotations", "items", "data", "sentences")

_INLINE_MARK_RE = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)|【[^】]*】|（[^）]*）")

_LABEL_MAP = {
    "complete": TurnLabel.COMPLETE,
    "incomplete": TurnLabel.INCOMPLETE,
    "backchannel": TurnLabel.BACKCHANNEL,
    "wait": TurnLabel.WAIT,
}


def clean_text(text: str) -> str:
    return " ".join(_INLINE_MARK_RE.sub(" ", text).split())


def _pick(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _iter_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in _SEGMENT_KEYS:
            if isinstance(payload.get(k), list):
                return [x for x in payload[k] if isinstance(x, dict)]
        # 兜底：取第一个 list-of-dict 值
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError("无法在 JSON 中定位分段列表（segments/utterances/...）")


def parse_file(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = _iter_items(payload)
    rows = []
    for it in items:
        start, end = _pick(it, _START_KEYS), _pick(it, _END_KEYS)
        if start is None or end is None:
            continue
        start, end = float(start), float(end)
        rows.append(
            {
                "channel": _norm_channel(_pick(it, _CHANNEL_KEYS)),
                "start_s": start,
                "end_s": end,
                "text_raw": str(_pick(it, _TEXT_KEYS) or ""),
                "turn_label": _norm_label(_pick(it, _TURN_KEYS)),
            }
        )
    df = pd.DataFrame(rows, columns=["channel", "start_s", "end_s", "text_raw", "turn_label"])
    if len(df) and df["end_s"].max() > 1000.0:  # 毫秒判定（均长 144.6 s 的语料不应超 1000 s）
        df["start_s"] = df["start_s"] / 1000.0
        df["end_s"] = df["end_s"] / 1000.0
    df["text_clean"] = df["text_raw"].map(clean_text)
    return df.sort_values(["channel", "start_s"], kind="stable").reset_index(drop=True)


def _norm_channel(v: Any) -> int:
    if v is None:
        return -1
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int | float):
        return int(v)
    s = str(v).strip().upper()
    if s in ("A", "L", "LEFT", "0", "SPK0", "S0"):
        return 0
    if s in ("B", "R", "RIGHT", "1", "SPK1", "S1"):
        return 1
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits else -1


def _norm_label(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    return _LABEL_MAP[s].value if s in _LABEL_MAP else s


def to_gold_turns(df: pd.DataFrame, channel: int) -> list[Turn]:
    sub = df[df["channel"] == channel]
    return [
        Turn(channel=channel, start=float(r.start_s), end=float(r.end_s), label=r.turn_label)
        for r in sub.itertuples(index=False)
    ]


def to_va_segs(df: pd.DataFrame, channel: int) -> list[Seg]:
    sub = df[df["channel"] == channel]
    return [Seg(float(r.start_s), float(r.end_s)) for r in sub.itertuples(index=False)]


def self_check(dir_path: str | Path, n_files: int = 20) -> dict:
    """抽样解析并统计字段覆盖率与标签分布，供本机核对 schema。"""
    files = sorted(Path(dir_path).rglob("*.json"))[:n_files]
    report: dict = {"n_files_found": len(files), "files_ok": 0, "errors": [], "label_counts": {}}
    n_rows, n_label, n_text, n_ch_unknown = 0, 0, 0, 0
    for f in files:
        try:
            df = parse_file(f)
        except Exception as e:
            report["errors"].append({"file": str(f), "error": repr(e)})
            continue
        report["files_ok"] += 1
        n_rows += len(df)
        n_label += int(df["turn_label"].notna().sum())
        n_text += int((df["text_raw"].str.len() > 0).sum())
        n_ch_unknown += int((df["channel"] < 0).sum())
        for k, v in df["turn_label"].value_counts(dropna=True).items():
            report["label_counts"][str(k)] = report["label_counts"].get(str(k), 0) + int(v)
    report.update(
        n_rows=n_rows,
        label_coverage=(n_label / n_rows if n_rows else 0.0),
        text_coverage=(n_text / n_rows if n_rows else 0.0),
        channel_unknown_rows=n_ch_unknown,
    )
    return report
