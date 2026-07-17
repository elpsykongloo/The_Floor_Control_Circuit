"""WP1：G0 准备 —— 导出 DualTurn 会话的离散码与金标轨，供 Mimi 解码与校准。

流程（V2 结论已冻结：发布物无原始音频 → G0 用 Mimi 解码音频）：
  1) uv run python scripts/wp1_g0_prepare.py --split test --limit 20
     → <data_root>/dualturn_prep/<sid>/codes_ch{0,1}.npy + gold_ch{0,1}.npz + meta.json
     并把数据集 README.md 抄录到 reports/v1_v6/V2_dualturn_README.md（核对标签语义）
  2) Moshi venv 解码：
     <moshi python> runners/moshi/decode_mimi.py --batch-root <data_root>/dualturn_prep --model-root <moshiko>
     （解码后请人工听任一会话 5 秒，确认帧主序 reshape 正确——听起来是正常语音即对）
  3) uv run python scripts/wp1_g0_calibrate.py [--split test] [--limit N]

划分语义（2026-07-17 审查后收紧）：splits.json 不可读或划分名不存在时**硬失败**并落报告，
绝不静默回退成全量导出（避免污染 dualturn_prep 与错误的 meta 溯源）；确需全量导出用 --all-sessions。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_paths
from floor_circuit.data.dualturn import iter_sessions, load_splits, split_sessions


def _fail(summary: dict, message: str) -> None:
    """硬失败前必落报告（两端协作反馈回路）。"""
    summary["error"] = message
    write_report_json("wp1_g0_prepare_summary.json", summary)
    raise SystemExit(message)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--all-sessions", action="store_true", help="绕过 splits.json，导出全部会话")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    root = Path(args.dir or load_paths()["datasets"]["dualturn"])
    out_root = data_root() / "dualturn_prep"
    split_tag = "all" if args.all_sessions else args.split
    summary: dict = {"split": split_tag, "sessions": [], "n_ok": 0}

    readme = root / "README.md"
    if readme.exists():
        dst = REPORTS_DIR / "v1_v6" / "V2_dualturn_README.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(readme.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        print(f"[report] 已抄录数据集 README → {dst}")

    # splits 只读一次；失败 = 硬失败（不再静默回退全量导出）
    wanted: set[str] | None = None
    payload: dict | None = None
    if not args.all_sessions:
        try:
            payload = load_splits(root)
            wanted = set(split_sessions(root, args.split))
        except Exception as e:
            _fail(
                summary,
                f"splits.json 读取或划分解析失败（{e!r}）；确需全量导出请显式加 --all-sessions",
            )
        if not wanted:
            _fail(summary, f"划分 '{args.split}' 解析到 0 个会话：splits 元素结构异常，请回传 splits.json 顶层样例")
        print(f"划分 {args.split}：{len(wanted)} 会话")

    for sess in iter_sessions(root, sessions=wanted, limit=args.limit):
        sdir = out_root / sess.session_id
        sdir.mkdir(parents=True, exist_ok=True)
        for ch in (0, 1):
            np.save(sdir / f"codes_ch{ch}.npy", sess.codes[ch])
            np.savez(
                sdir / f"gold_ch{ch}.npz",
                fvad=sess.fvad[ch],
                **{k: v for k, v in sess.tracks[ch].items()},
            )
        (sdir / "meta.json").write_text(
            json.dumps(
                {
                    "session_id": sess.session_id,
                    "dataset": sess.dataset,
                    "duration_s": sess.duration_s,
                    "num_frames": sess.num_frames,
                    "split": split_tag,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        summary["sessions"].append(
            {"session": sess.session_id, "frames": sess.num_frames, "dur_s": round(sess.duration_s, 1)}
        )
        summary["n_ok"] += 1
        print(f"{sess.session_id}: {sess.num_frames} 帧 / {sess.duration_s:.1f}s")
    if isinstance(payload, dict):
        summary["splits_meta"] = {
            k: v for k, v in payload.items() if k in ("total_sessions", "split_counts", "sessions_without_audio")
        }
    write_report_json("wp1_g0_prepare_summary.json", summary)
    print(f"完成 {summary['n_ok']} 会话 → {out_root}；下一步在 Moshi venv 跑 decode_mimi.py")


if __name__ == "__main__":
    main()
