"""dualturn-switchboard-turn-taking 载入器（schema 已按 V2 盘点冻结，2026-07-17）。

发布物：data/*.parquet（61 片，每行一会话）+ splits.json + README.md，**不含原始音频**。
每行列（12.5 Hz 帧级）：
  session_id, dataset, duration_s, num_frames,
  codes_ch{0,1}      list<int16>    → reshape (num_frames, 8)   离散 Mimi 码（可解码回音频/teacher-force）
  mimi_feat_ch{0,1}  list<float16>  → reshape (num_frames, 512) 连续 Mimi 潜表征（编码器基线）
  vad/eot/hold/bot/bc_ch{0,1}  list<int8> → (num_frames,)       金标二值轨（G0 靶）
  fvad_ch{0,1}       list<float>    → (num_frames, 4)           四个未来窗口的软 VAD

展平顺序假定为帧主序（frame-major）；解码后音频须人工听 5 秒确认（wp1_g0_prepare 的听感核对项）。
G0 音源 = Mimi 解码音频（runners/moshi/decode_mimi.py，在 Moshi venv 内运行）。
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FRAME_HZ = 12.5
N_CODEBOOKS = 8
FEAT_DIM = 512
FVAD_DIM = 4
TRACK_NAMES = ("vad", "eot", "hold", "bot", "bc")


@dataclass
class DualturnSession:
    session_id: str
    dataset: str
    duration_s: float
    num_frames: int
    codes: dict[int, np.ndarray]  # ch -> (F, 8) int16
    mimi_feat: dict[int, np.ndarray]  # ch -> (F, 512) float16
    tracks: dict[int, dict[str, np.ndarray]]  # ch -> {vad/eot/hold/bot/bc: (F,) int8}
    fvad: dict[int, np.ndarray]  # ch -> (F, 4) float32


def load_splits(dualturn_dir: str | Path) -> dict:
    path = Path(dualturn_dir) / "splits.json"
    return json.loads(path.read_text(encoding="utf-8"))


def split_sessions(dualturn_dir: str | Path, split: str) -> list[str]:
    """读取指定划分的会话。

    兼容两种已见结构：
    - ``{划分名: [会话列表]}``
    - ``{会话名: 划分名}``（DualTurn 本机真实发布物）
    """
    payload = load_splits(dualturn_dir)
    splits = payload.get("splits", payload)
    if not isinstance(splits, dict):
        raise TypeError(f"splits.json 的 splits 应为对象，实际为 {type(splits).__name__}")

    if splits and all(isinstance(value, str) for value in splits.values()):
        available = sorted({str(value) for value in splits.values()})
        if split not in available:
            raise KeyError(f"splits.json 无划分 '{split}'，可选：{available}")
        return [str(session_id) for session_id, name in splits.items() if name == split]

    if split not in splits:
        raise KeyError(f"splits.json 无划分 '{split}'，可选：{list(splits)}")
    items = splits[split]
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            sid = it.get("session_id") or it.get("session") or it.get("id")
            if sid:
                out.append(str(sid))
    return out


def _reshape(values: Any, n_frames: int, width: int, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if width == 1:
        if len(arr) != n_frames:
            raise ValueError(f"轨长 {len(arr)} ≠ num_frames {n_frames}")
        return arr
    if len(arr) != n_frames * width:
        raise ValueError(f"展平长度 {len(arr)} ≠ num_frames×{width}（{n_frames * width}）")
    return arr.reshape(n_frames, width)  # 帧主序假定（解码听感核对）


def _row_to_session(row: dict) -> DualturnSession:
    n = int(row["num_frames"])
    codes, feat, tracks, fvad = {}, {}, {}, {}
    for ch in (0, 1):
        codes[ch] = _reshape(row[f"codes_ch{ch}"], n, N_CODEBOOKS, np.int16)
        feat[ch] = _reshape(row[f"mimi_feat_ch{ch}"], n, FEAT_DIM, np.float16)
        tracks[ch] = {name: _reshape(row[f"{name}_ch{ch}"], n, 1, np.int8) for name in TRACK_NAMES}
        fvad[ch] = _reshape(row[f"fvad_ch{ch}"], n, FVAD_DIM, np.float32)
    return DualturnSession(
        session_id=str(row["session_id"]),
        dataset=str(row.get("dataset", "")),
        duration_s=float(row["duration_s"]),
        num_frames=n,
        codes=codes,
        mimi_feat=feat,
        tracks=tracks,
        fvad=fvad,
    )


def iter_sessions(
    dualturn_dir: str | Path,
    sessions: set[str] | None = None,
    limit: int | None = None,
    shard_glob: str = "*.parquet",
) -> Iterator[DualturnSession]:
    """流式遍历 data/*.parquet 的会话行；sessions 给定时只产出命中的会话。"""
    import pyarrow.parquet as pq

    data_dir = Path(dualturn_dir) / "data"
    if not data_dir.exists():
        data_dir = Path(dualturn_dir)
    n_out = 0
    for shard in sorted(data_dir.glob(shard_glob)):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=4):
            for row in batch.to_pylist():
                if sessions is not None and str(row.get("session_id")) not in sessions:
                    continue
                yield _row_to_session(row)
                n_out += 1
                if limit is not None and n_out >= limit:
                    return


def load_frame_labels(dualturn_dir: str | Path, session: str) -> dict[int, dict[str, np.ndarray]]:
    """单会话金标二值轨 {channel: {vad/eot/hold/bot/bc: (F,)}}（全量扫描，批处理请用 iter_sessions）。"""
    for sess in iter_sessions(dualturn_dir, sessions={session}):
        return sess.tracks
    raise KeyError(f"未在 {dualturn_dir} 找到会话 {session}")


# ---------- 盘点工具（V2 已完成，保留复用） ----------


def read_safetensors_header(path: str | Path) -> dict:
    """不依赖 torch/safetensors 的头部读取：{tensor 名: {dtype, shape}}。"""
    with Path(path).open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return {k: {"dtype": v.get("dtype"), "shape": v.get("shape")} for k, v in header.items() if k != "__metadata__"}


def _peek_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: {"dtype": str(z[k].dtype), "shape": list(z[k].shape)} for k in z.files}


def _peek_json(path: Path, max_bytes: int = 1 << 20) -> Any:
    raw = path.read_bytes()[:max_bytes]
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:  # 盘点容错
        return {"note": "JSON 过大或截断，未完整解析", "head": raw[:200].decode("utf-8", "replace")}
    if isinstance(obj, dict):
        return {"keys": list(obj.keys())[:30]}
    if isinstance(obj, list):
        first_keys = list(obj[0].keys())[:30] if obj and isinstance(obj[0], dict) else None
        return {"list_len": len(obj), "first_item_keys": first_keys}
    return {"type": type(obj).__name__}


def inspect_dir(dualturn_dir: str | Path, max_peek_per_type: int = 3) -> dict:
    """盘点发布物：后缀计数 + 每类抽样看形状/键名（V2 验证项所用）。"""
    root = Path(dualturn_dir)
    files = [p for p in root.rglob("*") if p.is_file()]
    by_suffix = Counter(p.suffix.lower() for p in files)
    report: dict = {
        "root": str(root),
        "n_files": len(files),
        "by_suffix": dict(by_suffix.most_common()),
        "peeks": {},
        "top_level": sorted({p.relative_to(root).parts[0] for p in files})[:40],
    }
    peeked: Counter = Counter()
    for p in sorted(files):
        suffix = p.suffix.lower()
        if peeked[suffix] >= max_peek_per_type:
            continue
        try:
            if suffix == ".npz":
                info: Any = _peek_npz(p)
            elif suffix == ".npy":
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                info = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
            elif suffix == ".safetensors":
                info = read_safetensors_header(p)
            elif suffix == ".json":
                info = _peek_json(p)
            elif suffix in (".parquet", ".pq"):
                import pyarrow.parquet as pq

                info = {"schema": str(pq.read_schema(p))}
            else:
                continue
        except Exception as e:  # 盘点容错
            info = {"error": repr(e)}
        report["peeks"].setdefault(suffix, []).append({"file": str(p.relative_to(root)), "info": info})
        peeked[suffix] += 1
    try:
        splits = load_splits(root)
        report["splits_json"] = {k: (len(v) if isinstance(v, list) else type(v).__name__) for k, v in splits.items()}
    except FileNotFoundError:
        report["splits_json"] = "splits.json 未在根目录找到"
    return report
