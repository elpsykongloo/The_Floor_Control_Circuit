"""WP2：会话级划分冻结（生成即冻结，禁止覆盖；文档/00 §4.3）。

用法（CANDOR 索引与 SmoothConv 自检通过后）：
  uv run python scripts/wp2_freeze_splits.py --seed 20260717
产出：configs/splits/candor.json、smoothconv.json、duplexconv.json + reports/splits_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_paths
from floor_circuit.data.splits import (
    CANDOR_RATIOS,
    SMOOTHCONV_RATIOS,
    freeze_split,
    load_split,
    write_split,
)


def _frozen_summaries(out_dir: Path, seed: int) -> dict[str, dict]:
    """从已有冻结文件重建摘要，避免分数据集运行时覆盖先前结果。"""
    summaries: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        payload = load_split(path)
        dataset = str(payload.get("dataset", path.stem))
        frozen_seed = int(payload["seed"])
        if frozen_seed != seed:
            raise SystemExit(
                f"{path} 的冻结种子为 {frozen_seed}，本次为 {seed}；拒绝合并不同种子的摘要"
            )
        summaries[dataset] = {
            "counts": payload["counts"],
            "sha256": payload["sha256"],
        }
    return summaries


def _freeze_one(dataset: str, sessions: list[str], ratios: dict, seed: int, out_dir: Path) -> dict:
    if not sessions:
        raise SystemExit(f"{dataset}: 会话列表为空，拒绝冻结空划分（先解决数据形态，如 DuplexConv 的 tar 包）")
    splits = freeze_split(sessions, ratios, seed)
    payload = write_split(out_dir / f"{dataset}.json", dataset, splits, seed, ratios)
    return {"counts": payload["counts"], "sha256": payload["sha256"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument(
        "--dataset",
        choices=["candor", "smoothconv", "duplexconv", "all"],
        default="all",
        help="分数据集冻结（DuplexConv 待 tar 形态确认后单独冻结）",
    )
    args = ap.parse_args()
    out_dir = REPO_ROOT / "configs" / "splits"
    summary: dict = {"seed": args.seed, "datasets": _frozen_summaries(out_dir, args.seed)}

    if args.dataset in ("candor", "all"):
        index = pd.read_parquet(data_root() / "raw_index" / "candor_index.parquet")
        candor_sessions = sorted(index["session_id"].unique().tolist())
        summary["datasets"]["candor"] = _freeze_one("candor", candor_sessions, CANDOR_RATIOS, args.seed, out_dir)

    if args.dataset in ("smoothconv", "all"):
        sc_dir = Path(load_paths()["datasets"]["smoothconv"])
        sc_sessions = sorted(p.stem for p in sc_dir.rglob("*.json"))
        summary["datasets"]["smoothconv"] = _freeze_one(
            "smoothconv", sc_sessions, SMOOTHCONV_RATIOS, args.seed, out_dir
        )

    if args.dataset in ("duplexconv", "all"):
        dc_dir = Path(load_paths()["datasets"]["duplexconv"])
        dc_sessions = sorted(p.stem for p in dc_dir.rglob("*.json"))
        summary["datasets"]["duplexconv"] = _freeze_one(
            "duplexconv", dc_sessions, {"train": 1.0}, args.seed, out_dir
        )

    summary["note"] = "DualTurn 沿官方 splits.json，不在此生成；冻结后跑 prereg_fingerprint.py 回填哈希"
    write_report_json("splits_summary.json", summary)
    print("划分已冻结：", {k: v["counts"] for k, v in summary["datasets"].items()})


if __name__ == "__main__":
    main()
