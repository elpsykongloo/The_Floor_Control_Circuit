"""联合分析 MiniCPM-o 因果双工与远端记忆上下文运行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPO_ROOT

from floor_circuit.minicpm_true_context import (
    analyze_true_context_runs,
    load_true_context_run,
    render_true_context_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-o 因果双工真实上下文联合分析"
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    runs = [
        load_true_context_run(
            Path(root),
            require_complete=not args.allow_partial,
        )
        for root in args.runs
    ]
    report = analyze_true_context_runs(runs)
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
        render_true_context_markdown(report),
        encoding="utf-8",
    )
    recommendation = report["recommendation"]
    print(f"[minicpm-true-context] JSON：{out_json}")
    print(f"[minicpm-true-context] Markdown：{out_md}")
    safe_checkpoint = recommendation["empirical_safe_checkpoint_seconds"]
    safe_text = (
        f"{safe_checkpoint} 秒"
        if safe_checkpoint is not None
        else "尚未定量"
    )
    print(
        "[minicpm-true-context] "
        f"{recommendation['status']}；"
        f"安全检查点={safe_text}；"
        f"首个失效={recommendation['first_confirmed_failure_seconds']}"
    )


if __name__ == "__main__":
    main()
