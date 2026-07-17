"""dualturn-switchboard-turn-taking：官方划分载入与发布物盘点（V2 验证项）。

帧级标签/Mimi 特征的正式载入器在 V2 盘点结论回传后冻结实现——
inspect_dir() 先把发布物的文件类型、形状、键名盘点清楚（不需要 torch）。
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_splits(dualturn_dir: str | Path) -> dict:
    path = Path(dualturn_dir) / "splits.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors_header(path: str | Path) -> dict:
    """不依赖 torch/safetensors 的头部读取：{tensor 名: {dtype, shape}}。"""
    with Path(path).open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return {
        k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
        for k, v in header.items()
        if k != "__metadata__"
    }


def _peek_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: {"dtype": str(z[k].dtype), "shape": list(z[k].shape)} for k in z.files}


def _peek_json(path: Path, max_bytes: int = 1 << 20) -> Any:
    raw = path.read_bytes()[:max_bytes]
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return {"note": "JSON 过大或截断，未完整解析", "head": raw[:200].decode("utf-8", "replace")}
    if isinstance(obj, dict):
        return {"keys": list(obj.keys())[:30]}
    if isinstance(obj, list):
        first_keys = list(obj[0].keys())[:30] if obj and isinstance(obj[0], dict) else None
        return {"list_len": len(obj), "first_item_keys": first_keys}
    return {"type": type(obj).__name__}


def inspect_dir(dualturn_dir: str | Path, max_peek_per_type: int = 3) -> dict:
    """盘点发布物：后缀计数 + 每类抽样看形状/键名。判定 Mimi 特征是离散码还是连续 embedding
    的直接证据 = 抽样张量的 dtype（int* → 离散码；float*/bf16 → 连续 embedding）。"""
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
        except Exception as e:
            info = {"error": repr(e)}
        report["peeks"].setdefault(suffix, []).append({"file": str(p.relative_to(root)), "info": info})
        peeked[suffix] += 1
    try:
        splits = load_splits(root)
        report["splits_json"] = {
            k: (len(v) if isinstance(v, list) else type(v).__name__) for k, v in splits.items()
        }
    except FileNotFoundError:
        report["splits_json"] = "splits.json 未在根目录找到"
    return report


def load_frame_labels(dualturn_dir: str | Path, session: str) -> np.ndarray:
    """12.5 Hz 帧级 {eot, hold, bot, bc} 标签载入——待 V2 盘点结论回传后实现。"""
    raise NotImplementedError(
        "帧级标签载入器待 V2 盘点结论冻结：请先运行 "
        "`uv run python scripts/wp3_v2_inspect_dualturn.py` 并回传 "
        "reports/v1_v6/V2_dualturn_inventory.json"
    )
