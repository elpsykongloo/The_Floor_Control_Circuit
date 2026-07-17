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
from floor_circuit.data.splits import CANDOR_RATIOS, SMOOTHCONV_RATIOS, freeze_split, write_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()
    out_dir = REPO_ROOT / "configs" / "splits"
    summary: dict = {"seed": args.seed, "datasets": {}}

    index = pd.read_parquet(data_root() / "raw_index" / "candor_index.parquet")
    candor_sessions = sorted(index["session_id"].unique().tolist())
    splits = freeze_split(candor_sessions, CANDOR_RATIOS, args.seed)
    payload = write_split(out_dir / "candor.json", "candor", splits, args.seed, CANDOR_RATIOS)
    summary["datasets"]["candor"] = {"counts": payload["counts"], "sha256": payload["sha256"]}

    sc_dir = Path(load_paths()["datasets"]["smoothconv"])
    sc_sessions = sorted(p.stem for p in sc_dir.rglob("*.json"))
    splits = freeze_split(sc_sessions, SMOOTHCONV_RATIOS, args.seed)
    payload = write_split(out_dir / "smoothconv.json", "smoothconv", splits, args.seed, SMOOTHCONV_RATIOS)
    summary["datasets"]["smoothconv"] = {"counts": payload["counts"], "sha256": payload["sha256"]}

    dc_dir = Path(load_paths()["datasets"]["duplexconv"])
    dc_sessions = sorted(p.stem for p in dc_dir.rglob("*.json"))
    payload = write_split(
        out_dir / "duplexconv.json", "duplexconv", {"train": dc_sessions}, args.seed, {"train": 1.0}
    )
    summary["datasets"]["duplexconv"] = {"counts": payload["counts"], "sha256": payload["sha256"]}

    summary["note"] = "DualTurn 沿官方 splits.json，不在此生成；splits 哈希需登记入 PREREG.md"
    write_report_json("splits_summary.json", summary)
    print("划分已冻结：", {k: v["counts"] for k, v in summary["datasets"].items()})


if __name__ == "__main__":
    main()
