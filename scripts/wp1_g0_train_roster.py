"""WP1：G0 train 侧新确认集 roster 冻结（PREREG #14）。

从官方 train 划分中枚举发布物可用、且从未进入 dualturn_prep 的会话（应为 1986 个），
以冻结种子 20260717 无放回抽取 300 个，写入 configs/splits/dualturn_train_confirm.json。
该文件生成后即冻结（脚本拒绝覆盖已存在文件），并由 prereg_fingerprint.py 自动纳入指纹。

用法：uv run python scripts/wp1_g0_train_roster.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from _bootstrap import REPO_ROOT

from floor_circuit.config import data_root, load_config, load_paths

ROSTER_RELATIVE = Path("configs") / "splits" / "dualturn_train_confirm.json"


def sample_roster(candidates: list[str], seed: int, k: int) -> list[str]:
    """对排序后的候选列表做确定性无放回抽样，返回按会话 id 排序的 roster。"""
    ordered = sorted(candidates)
    if len(set(ordered)) != len(ordered):
        raise ValueError("候选会话含重复 id")
    if len(ordered) < k:
        raise ValueError(f"候选 {len(ordered)} 个不足以抽取 {k} 个")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(ordered), size=k, replace=False)
    return sorted(ordered[int(index)] for index in picked)


def main() -> None:
    cfg = load_config("events")["g0"]["train_confirm"]
    seed = int(cfg["seed"])
    expected = int(cfg["expected_sessions"])
    roster_path = REPO_ROOT / ROSTER_RELATIVE
    if roster_path.exists():
        raise SystemExit(f"{roster_path} 已存在——roster 已冻结，拒绝覆盖")

    from floor_circuit.data.dualturn import load_splits, split_sessions

    dataset_root = Path(load_paths()["datasets"]["dualturn"])
    splits_path = dataset_root / "splits.json"
    splits_sha256 = hashlib.sha256(splits_path.read_bytes()).hexdigest()
    train_ids = set(split_sessions(dataset_root, "train"))
    _payload = load_splits(dataset_root)

    import pyarrow.parquet as pq

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        data_dir = dataset_root
    shard_glob = "train-*.parquet" if any(data_dir.glob("train-*.parquet")) else "*.parquet"
    release_ids: set[str] = set()
    for shard in sorted(data_dir.glob(shard_glob)):
        table = pq.read_table(shard, columns=["session_id"])
        release_ids.update(str(v) for v in table.column("session_id").to_pylist())

    prep_root = data_root() / "dualturn_prep"
    prepared = {d.name for d in prep_root.iterdir() if d.is_dir()} if prep_root.exists() else set()
    already_prepared = sorted(train_ids & prepared)
    if already_prepared:
        raise SystemExit(
            f"train 侧有 {len(already_prepared)} 个会话已被读取（新鲜性破坏）：{already_prepared[:5]}"
        )
    candidates = sorted(train_ids & release_ids)
    roster = sample_roster(candidates, seed, expected)

    payload = {
        "note": (
            "G0 train 侧一次性新确认集（PREREG #14）：生成即冻结；启封（prepare/decode/"
            "confirmation-train）前不得读取任何会话内容"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "n_pool": len(candidates),
        "n_sample": expected,
        "shard_glob": shard_glob,
        "splits_json_sha256": splits_sha256,
        "sessions": roster,
    }
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    roster_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"roster 冻结：{expected}/{len(candidates)} 会话（seed {seed}）→ {roster_path}\n"
        "下一步：git 提交本文件 + 本机重跑 prereg_fingerprint.py，然后才允许 prepare/decode"
    )


if __name__ == "__main__":
    main()
