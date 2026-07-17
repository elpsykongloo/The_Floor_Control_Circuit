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


def _pick_nested(it: dict, keys: tuple[str, ...]) -> Any:
    """先查顶层别名，再查 attributes 子对象（SmoothConv 真实标签路径 = attributes.turn，
    2026-07-17 自检 label_value_hits 定位）。"""
    v = _pick(it, keys)
    if v is not None:
        return v
    attrs = it.get("attributes")
    return _pick(attrs, keys) if isinstance(attrs, dict) else None


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
                "turn_label": _norm_label(_pick_nested(it, _TURN_KEYS)),
                "other_turn_label": _norm_label(_pick_nested(it, ("other_turn", "otherTurn"))),
            }
        )
    df = pd.DataFrame(
        rows, columns=["channel", "start_s", "end_s", "text_raw", "turn_label", "other_turn_label"]
    )
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


_LABEL_VALUES = {"complete", "incomplete", "backchannel", "wait"}


def _flat_items(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flat_items(v, prefix=f"{key}."))
        else:
            out.append((key, v))
    return out


def _dir_inventory(dir_path: Path, sample_n: int = 12) -> dict:
    """目录里没有 *.json 时的清点回退（DuplexConv 情形）：找出标注实际形态。
    发现 .tar/.tar.gz 时额外流式窥探首个包：成员名样例 + 首个 JSON 成员的键结构。"""
    from collections import Counter

    files = [p for p in dir_path.rglob("*") if p.is_file()]
    suffixes = Counter(p.suffix.lower() for p in files)
    samples = [str(p.relative_to(dir_path)) for p in sorted(files, key=str)[:sample_n]]
    big = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:5]
    report = {
        "n_files_total": len(files),
        "by_suffix": dict(suffixes.most_common(20)),
        "sample_paths": samples,
        "largest": [{"path": str(p.relative_to(dir_path)), "mb": round(p.stat().st_size / 1e6, 1)} for p in big],
    }
    tars = sorted(p for p in files if p.suffix.lower() in (".tar", ".gz", ".tgz"))
    if tars:
        report["tar_peek"] = _tar_peek(tars[0])
    return report


def _tar_peek(tar_path: Path, max_members: int = 30) -> dict:
    """流式窥探 tar（不解包落盘）：成员名分布 + 首个 JSON 成员的解析样例。"""
    import tarfile

    peek: dict = {"tar": tar_path.name, "members": [], "json_sample": None}
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            json_done = False
            for i, member in enumerate(tf):
                if i < max_members:
                    peek["members"].append({"name": member.name, "size": member.size})
                if not json_done and member.isfile() and member.name.lower().endswith(".json"):
                    fh = tf.extractfile(member)
                    if fh is not None:
                        try:
                            obj = json.loads(fh.read(1 << 20).decode("utf-8"))
                            items = _iter_items(obj)
                            peek["json_sample"] = {
                                "member": member.name,
                                "n_items": len(items),
                                "first_item_keys": [k for k, _ in _flat_items(items[0])][:30] if items else None,
                            }
                        except Exception as e:
                            peek["json_sample"] = {"member": member.name, "error": repr(e)}
                    json_done = True
                if i >= max_members and json_done:
                    break
    except Exception as e:
        peek["error"] = repr(e)
    return peek


def self_check(dir_path: str | Path, n_files: int = 20) -> dict:
    """抽样解析并统计字段覆盖率与标签分布；额外输出定位信息：
    - key_union：条目字段名全集（含一层嵌套 parent.child）→ 收紧别名表的依据；
    - label_value_hits：值命中 {complete,incomplete,backchannel,wait} 的字段路径 → 标签真身；
    - sample_item：首条原始条目（字符串截断 80 字符）；
    - 目录无 *.json 时回退为 inventory（后缀分布 + 样例路径 + 最大文件）。"""
    from collections import Counter

    dir_path = Path(dir_path)
    files = sorted(dir_path.rglob("*.json"))[:n_files]
    report: dict = {"n_files_found": len(files), "files_ok": 0, "errors": [], "label_counts": {}}
    if not files:
        report["inventory"] = _dir_inventory(dir_path)
        report.update(n_rows=0, label_coverage=0.0, text_coverage=0.0, channel_unknown_rows=0)
        return report
    key_union: Counter = Counter()
    label_hits: Counter = Counter()
    n_rows, n_label, n_text, n_ch_unknown = 0, 0, 0, 0
    for fi, f in enumerate(files):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            items = _iter_items(payload)
            for it in items[:200]:
                for key, val in _flat_items(it):
                    key_union[key] += 1
                    if isinstance(val, str) and val.strip().lower() in _LABEL_VALUES:
                        label_hits[key] += 1
            if fi == 0 and items:
                report["sample_item"] = {
                    k: (v[:80] if isinstance(v, str) else v) for k, v in _flat_items(items[0])
                }
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
    report["key_union"] = dict(key_union.most_common(40))
    report["label_value_hits"] = dict(label_hits.most_common(10))
    report.update(
        n_rows=n_rows,
        label_coverage=(n_label / n_rows if n_rows else 0.0),
        text_coverage=(n_text / n_rows if n_rows else 0.0),
        channel_unknown_rows=n_ch_unknown,
    )
    return report
