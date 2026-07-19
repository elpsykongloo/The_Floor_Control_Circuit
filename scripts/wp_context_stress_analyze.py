"""联合分析 MiniCPM-o 或 Freeze-Omni 的上下文应力运行。

示例：
  uv run python scripts/wp_context_stress_analyze.py \
      --runs D:\\data_storage\\The_Floor_Control_Circuit\\context_stress\\freeze_omni\\run0 \
             D:\\data_storage\\The_Floor_Control_Circuit\\context_stress\\freeze_omni\\run1 \
             D:\\data_storage\\The_Floor_Control_Circuit\\context_stress\\freeze_omni\\run2 \
      --out-json reports/freeze_omni_上下文应力.json \
      --out-md reports/freeze_omni_上下文应力.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPO_ROOT

from floor_circuit.context_stress import (
    analyze_context_stress_runs,
    load_context_stress_run,
    render_context_stress_markdown,
)


def _parse_boundaries(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--boundaries 不能为空")
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="上下文应力联合分析")
    parser.add_argument("--runs", nargs="+", required=True, help="一个或多个运行目录")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--boundaries", default=None, help="覆盖 manifest 的逗号分隔候选位置")
    parser.add_argument("--window-positions", type=int, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许读取因温度、显存或异常而中止的部分运行；正式结论不应使用",
    )
    args = parser.parse_args()

    runs = [
        load_context_stress_run(Path(root), require_complete=not args.allow_partial)
        for root in args.runs
    ]
    report = analyze_context_stress_runs(
        runs,
        boundaries=_parse_boundaries(args.boundaries),
        window_positions=args.window_positions,
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if not out_json.is_absolute():
        out_json = REPO_ROOT / out_json
    if not out_md.is_absolute():
        out_md = REPO_ROOT / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    out_md.write_text(render_context_stress_markdown(report), encoding="utf-8")
    print(f"[context-stress] JSON：{out_json}")
    print(f"[context-stress] Markdown：{out_md}")
    print(
        "[context-stress] "
        f"{report['model']}：{report['recommendation']['status']}；"
        f"600 秒覆盖={report['recommendation']['analysis_target_covered']}，"
        f"干净={report['recommendation']['analysis_target_clean']}"
    )


if __name__ == "__main__":
    main()
