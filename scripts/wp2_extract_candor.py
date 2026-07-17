"""WP2：CANDOR 按需解压 + 转 24 kHz 单声道对（MVE 输入）。

用法：
  uv run python scripts/wp2_extract_candor.py --sessions <id1,id2,...>
  uv run python scripts/wp2_extract_candor.py --from-split configs/splits/candor.json --group probe_train --limit 160
产出：<data_root>/candor_extracted/<session>/audio_ch{0,1}.wav（+原始转录成员）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from _bootstrap import write_report_json

from floor_circuit.config import data_root
from floor_circuit.data.candor import extract_members, media_to_dual_mono_24k, media_to_mono_24k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", help="逗号分隔的会话 id")
    ap.add_argument("--from-split", help="划分 json 路径")
    ap.add_argument("--group", default="probe_train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kinds", default="audio,video,transcript_backbiter,transcript")
    args = ap.parse_args()

    root = data_root()
    index = pd.read_parquet(root / "raw_index" / "candor_index.parquet")
    if args.sessions:
        session_ids = args.sessions.split(",")
    elif args.from_split:
        payload = json.loads(Path(args.from_split).read_text(encoding="utf-8"))
        session_ids = payload["splits"][args.group]
        if args.limit:
            session_ids = session_ids[: args.limit]
    else:
        raise SystemExit("需要 --sessions 或 --from-split")

    out_dir = root / "candor_extracted"
    written = extract_members(index, session_ids, args.kinds.split(","), out_dir)
    report: dict = {"n_sessions_requested": len(session_ids), "n_members": len(written), "converted": []}
    for sid in session_ids:
        sdir_candidates = [p for p in written if sid in p.parts]
        media = [p for p in sdir_candidates if p.suffix.lower() in
                 (".wav", ".flac", ".mp3", ".m4a", ".mp4", ".avi", ".mkv", ".webm")]
        sess_out = out_dir / sid
        try:
            if len(media) == 1:
                media_to_dual_mono_24k(media[0], sess_out)
                report["converted"].append({"session": sid, "mode": "stereo_split", "src": str(media[0])})
            elif len(media) >= 2:
                media_sorted = sorted(media)[:2]
                for ch, m in enumerate(media_sorted):
                    media_to_mono_24k(m, sess_out / f"audio_ch{ch}.wav")
                report["converted"].append(
                    {"session": sid, "mode": "two_files", "src": [str(m) for m in media_sorted]}
                )
            else:
                report["converted"].append({"session": sid, "mode": "no_media_found"})
        except Exception as e:
            report["converted"].append({"session": sid, "mode": "error", "error": repr(e)})
    n_ok = sum(1 for c in report["converted"] if c["mode"] in ("stereo_split", "two_files"))
    report["n_converted_ok"] = n_ok
    write_report_json("candor_extract_summary.json", report)
    print(f"完成：{n_ok}/{len(session_ids)} 会话已转 24 kHz 单声道对")
    if n_ok < len(session_ids):
        print("有失败样本：请回传 reports/candor_extract_summary.json（首跑用 --limit 3 验证声道映射）")


if __name__ == "__main__":
    main()
