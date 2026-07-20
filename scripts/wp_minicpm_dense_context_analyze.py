"""联合分析 MiniCPM-o 密集全双工上下文运行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPO_ROOT

from floor_circuit.minicpm_dense_context import (
    analyze_dense_context_runs,
    load_dense_context_run,
    render_dense_context_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o 密集全双工上下文联合分析")
    parser.add_argument("--runs", nargs="+", required=True, help="运行目录，至少三条为宜")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    runs = [
        load_dense_context_run(
            Path(root),
            require_complete=not args.allow_partial,
        )
        for root in args.runs
    ]
    report = analyze_dense_context_runs(runs)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if not out_json.is_absolute():
        out_json = REPO_ROOT / out_json
    if not out_md.is_absolute():
        out_md = REPO_ROOT / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    out_md.write_text(
        render_dense_context_markdown(report),
        encoding="utf-8",
    )
    recommendation = report["recommendation"]
    print(f"[dense-context] JSON：{out_json}")
    print(f"[dense-context] Markdown：{out_md}")
    print(
        "[dense-context] "
        f"{recommendation['status']}；"
        f"2000 秒覆盖={recommendation['seconds_2000_covered']}，"
        f"干净={recommendation['seconds_2000_clean']}"
    )


if __name__ == "__main__":
    main()
