"""WP2：SmoothConv / DuplexConv 解析自检与批量解析。

自检（首跑必做，回传 reports/*.json）：
  uv run python scripts/wp2_parse_smoothconv.py --self-check
  uv run python scripts/wp2_parse_smoothconv.py --self-check --dataset duplexconv
批量解析（自检通过后）：--parse-all 把统一 schema 写到 <data_root>/events/<dataset>_segments/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_paths
from floor_circuit.data.smoothconv import parse_file, self_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["smoothconv", "duplexconv"], default="smoothconv")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--n-files", type=int, default=20)
    ap.add_argument("--parse-all", action="store_true")
    args = ap.parse_args()

    src = Path(args.dir or load_paths()["datasets"][args.dataset])
    if args.self_check:
        report = self_check(src, n_files=args.n_files)
        write_report_json(f"{args.dataset}_selfcheck.json", report)
        print(
            f"{args.dataset}: files_ok={report['files_ok']}/{report['n_files_found']} "
            f"label_coverage={report['label_coverage']:.2%} text_coverage={report['text_coverage']:.2%}"
        )
        if report["errors"] or report["label_coverage"] < 0.95:
            print("存在解析问题：请回传 reports/ 下的自检 JSON，以收紧字段别名表")
        return
    if args.parse_all:
        out = data_root() / "events" / f"{args.dataset}_segments"
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(src.rglob("*.json"))
        n_ok, errors = 0, []
        for f in files:
            try:
                parse_file(f).to_parquet(out / (f.stem + ".parquet"))
                n_ok += 1
            except Exception as e:
                errors.append({"file": str(f), "error": repr(e)})
        write_report_json(
            f"{args.dataset}_parse_all.json",
            {"n_files": len(files), "n_ok": n_ok, "n_err": len(errors), "errors": errors[:50]},
        )
        print(f"解析完成 {n_ok}/{len(files)} → {out}")
        return
    ap.error("需要 --self-check 或 --parse-all")


if __name__ == "__main__":
    main()
