"""WP3-V2：盘点 dualturn-switchboard 发布物（Mimi 特征是离散码还是连续 embedding）。

用法：uv run python scripts/wp3_v2_inspect_dualturn.py
产出：reports/v1_v6/V2_dualturn_inventory.json —— 跑完后回传该文件，
帧级标签/特征载入器（data/dualturn.py）与 G0 音源方案据此冻结。
判读：peeks 中张量 dtype 为 int* → 离散码（可 teacher-force 进 Moshi，且可经 Mimi 解码出音频跑 G0）；
float*/bf16 → 连续 embedding（只作校准集与探针目标，G0 需另寻音源）。
"""

from __future__ import annotations

import argparse

from _bootstrap import write_report_json

from floor_circuit.config import load_paths
from floor_circuit.data.dualturn import inspect_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    root = args.dir or load_paths()["datasets"]["dualturn"]
    report = inspect_dir(root)
    write_report_json("v1_v6/V2_dualturn_inventory.json", report)
    print(f"文件数 {report['n_files']}；后缀分布 {report['by_suffix']}")
    print("请提交并推送 reports/v1_v6/V2_dualturn_inventory.json 以便远端定稿载入器")


if __name__ == "__main__":
    main()
