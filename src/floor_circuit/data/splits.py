"""会话级划分冻结（文档/00 §4.3）：生成即冻结，绝不跨用途复用。

CANDOR 60/15/25（probe_train / probe_val / causal_eval）；SmoothConv 70/30；
DuplexConv 仅训练侧；DualTurn 沿官方 splits.json。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

CANDOR_RATIOS = {"probe_train": 0.60, "probe_val": 0.15, "causal_eval": 0.25}
SMOOTHCONV_RATIOS = {"train": 0.70, "eval_sdt": 0.30}


def freeze_split(session_ids: list[str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"划分比例之和必须为 1，得到 {ratios}")
    ids = sorted(set(map(str, session_ids)))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    # 最大余数法分配，保证覆盖全部会话且无重叠
    quotas = {k: v * n for k, v in ratios.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    rem = n - sum(counts.values())
    for k in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[:rem]:
        counts[k] += 1
    out: dict[str, list[str]] = {}
    i = 0
    for k in ratios:
        out[k] = sorted(ids[i : i + counts[k]])
        i += counts[k]
    return out


def _payload_hash(splits: dict[str, list[str]]) -> str:
    blob = json.dumps(splits, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_split(
    path: str | Path, dataset: str, splits: dict[str, list[str]], seed: int, ratios: dict[str, float]
) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} 已存在：划分文件生成即冻结，禁止覆盖。确需重生成，先在 PREREG.md 变更记录登记并手动删除。"
        )
    payload = {
        "dataset": dataset,
        "seed": seed,
        "ratios": ratios,
        "counts": {k: len(v) for k, v in splits.items()},
        "sha256": _payload_hash(splits),
        "splits": splits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def load_split(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if _payload_hash(payload["splits"]) != payload["sha256"]:
        raise ValueError(f"{path} 划分内容与 sha256 不符：文件可能被改动过")
    return payload
