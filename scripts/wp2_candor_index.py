"""WP2：CANDOR 分卷 zip 索引（首跑必做）。

用法：uv run python scripts/wp2_candor_index.py
产出：<data_root>/raw_index/candor_index.parquet + reports/candor_index_summary.json
"""

from __future__ import annotations

import argparse

from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_paths
from floor_circuit.data.candor import index_zips, summarize_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candor-dir", default=None)
    args = ap.parse_args()
    candor_dir = args.candor_dir or load_paths()["datasets"]["candor"]
    print(f"索引 {candor_dir} ...")
    index = index_zips(candor_dir)
    out = data_root() / "raw_index"
    out.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out / "candor_index.parquet")
    summary = summarize_index(index)
    write_report_json("candor_index_summary.json", summary)
    print(f"索引行数 {len(index)}；会话数 {summary.get('n_sessions')}")
    if not summary.get("sessions_with_audio") and not summary.get("sessions_with_video"):
        print("警告：未发现音频/视频成员，请回传 reports/candor_index_summary.json 以调整分类规则")


if __name__ == "__main__":
    main()
