"""WP2：CANDOR 按需解压 + 转 24 kHz 单声道对（MVE 输入）。

会话内成员结构（2026-07-17 索引已核实）：
  <sid>/processed/<participantA>.mp4、<participantB>.mp4、<sid>.mp3、<sid>.mp4、thumbnail.png
  <sid>/transcription/transcript_backbiter.csv 等
策略：先只解压会话 MP3 + Backbiter → 探测 MP3 声道数：
  立体声 → 直接左右声道拆分（省去解压视频）；单声道 → 再解压两个参与者 MP4 分别抽音轨。
声道与参与者对应写入每会话 channel_map.json（stereo 情形标注"待与 Backbiter 首段说话人核对"）。

用法：
  uv run python scripts/wp2_extract_candor.py --sessions <id1,id2>
  uv run python scripts/wp2_extract_candor.py --from-split configs/splits/candor.json --group probe_train --limit 160
首跑请 --limit 3 并回传 reports/candor_extract_summary.json（核对声道映射与模式）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from _bootstrap import write_report_json

from floor_circuit.config import data_root
from floor_circuit.data.candor import (
    audio_channels,
    extract_members,
    media_to_dual_mono_24k,
    media_to_mono_24k,
)


def _session_ids(args) -> list[str]:
    if args.sessions:
        return args.sessions.split(",")
    if args.from_split:
        payload = json.loads(Path(args.from_split).read_text(encoding="utf-8"))
        ids = payload["splits"][args.group]
        return ids[: args.limit] if args.limit else ids
    raise SystemExit("需要 --sessions 或 --from-split")


def _extract_one(index: pd.DataFrame, sid: str, out_dir: Path) -> dict:
    sess_out = out_dir / sid
    if (sess_out / "audio_ch0.wav").exists() and (sess_out / "audio_ch1.wav").exists():
        channel_map_path = sess_out / "channel_map.json"
        try:
            channel_map = json.loads(channel_map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            channel_map = {}
        return {
            "session": sid,
            "mode": str(channel_map.get("mode", "already_done")),
            "cached": True,
        }
    got = extract_members(index, [sid], ["audio", "transcript_backbiter"], out_dir)
    mp3s = [p for p in got if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")]
    if mp3s:
        n_ch = audio_channels(mp3s[0])
        if n_ch >= 2:
            media_to_dual_mono_24k(mp3s[0], sess_out)
            (sess_out / "channel_map.json").write_text(
                json.dumps(
                    {
                        "mode": "stereo_split",
                        "source": mp3s[0].name,
                        "note": "ch0=左声道 ch1=右声道；与参与者的对应待用 Backbiter 首段说话人核对",
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
            return {"session": sid, "mode": "stereo_split", "src": str(mp3s[0]), "channels": n_ch}
    # 单声道混音或无 MP3 → 参与者视频分别抽音轨
    vids = extract_members(index, [sid], ["video"], out_dir)
    participants = sorted(
        p for p in vids if p.suffix.lower() == ".mp4" and p.stem != sid and "thumbnail" not in p.stem.lower()
    )
    if len(participants) >= 2:
        chosen = participants[:2]
        for ch, m in enumerate(chosen):
            media_to_mono_24k(m, sess_out / f"audio_ch{ch}.wav")
        (sess_out / "channel_map.json").write_text(
            json.dumps(
                {"mode": "two_files", "audio_ch0": chosen[0].stem, "audio_ch1": chosen[1].stem},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        return {
            "session": sid,
            "mode": "two_files",
            "participants": [m.stem for m in chosen],
            "mp3_channels": audio_channels(mp3s[0]) if mp3s else None,
        }
    return {"session": sid, "mode": "no_media_found", "n_mp3": len(mp3s), "n_videos": len(participants)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", help="逗号分隔的会话 id")
    ap.add_argument("--from-split", help="划分 json 路径")
    ap.add_argument("--group", default="probe_train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = data_root()
    index = pd.read_parquet(root / "raw_index" / "candor_index.parquet")
    session_ids = _session_ids(args)
    out_dir = root / "candor_extracted"
    report: dict = {"n_sessions_requested": len(session_ids), "converted": []}
    for sid in session_ids:
        try:
            result = _extract_one(index, sid, out_dir)
        except Exception as e:
            result = {"session": sid, "mode": "error", "error": repr(e)}
        report["converted"].append(result)
        print(f"{sid}: {result['mode']}")
    ok_modes = ("stereo_split", "two_files", "already_done")
    report["n_converted_ok"] = sum(1 for c in report["converted"] if c["mode"] in ok_modes)
    write_report_json("candor_extract_summary.json", report)
    if args.from_split:
        write_report_json(f"candor_extract_{args.group}_summary.json", report)
    print(f"完成：{report['n_converted_ok']}/{len(session_ids)} 会话可用")
    if report["n_converted_ok"] < len(session_ids):
        print("有失败样本：请回传 reports/candor_extract_summary.json")


if __name__ == "__main__":
    main()
