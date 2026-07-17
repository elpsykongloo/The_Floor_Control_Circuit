"""WP1：事件管线批量运行（CANDOR 提取音频对 → 事件 + T1–T5 标签 parquet）。

用法（先完成 wp2_extract_candor）：
  uv run python scripts/wp1_run_events.py --sessions-dir <data_root>/candor_extracted --limit 5
产出：<data_root>/events/candor/<session>.events.parquet 与 .labels.parquet
     + reports/wp1_events_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.events.pipeline import (
    SessionChannel,
    labels_both_roles,
    masks_summary,
    process_session,
)
from floor_circuit.events.vad import SileroVad
from floor_circuit.schemas import events_to_dataframe
from floor_circuit.stimuli.qc import load_wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--clock", default="moshi", help="标签时钟（configs/grids.yaml clocks）")
    args = ap.parse_args()

    cfg = load_config("events")
    grids = load_config("grids")
    step_s = float(grids["clocks"][args.clock]["step_ms"]) / 1000.0
    deltas = list(grids["delta_grid"][f"{args.clock}_ms"])
    root = Path(args.sessions_dir or (data_root() / "candor_extracted"))
    out_dir = data_root() / "events" / "candor"
    out_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVad(cfg)

    sessions = sorted(p for p in root.iterdir() if (p / "audio_ch0.wav").exists())
    if args.limit:
        sessions = sessions[: args.limit]
    summary: dict = {"clock": args.clock, "n_sessions": len(sessions), "sessions": []}
    for sdir in sessions:
        sid = sdir.name
        wav0, sr0 = load_wav(sdir / "audio_ch0.wav")
        wav1, sr1 = load_wav(sdir / "audio_ch1.wav")
        total_dur = min(len(wav0) / sr0, len(wav1) / sr1)
        ch0 = SessionChannel(va_segs=vad.segments(wav0, sr0))
        ch1 = SessionChannel(va_segs=vad.segments(wav1, sr1))
        events, ctxs, dt = process_session(ch0, ch1, total_dur, cfg, lang=args.lang)
        events_to_dataframe(events).to_parquet(out_dir / f"{sid}.events.parquet")
        labels = labels_both_roles(events, ctxs, dt, total_dur, cfg, step_s, deltas)
        labels.to_parquet(out_dir / f"{sid}.labels.parquet")
        summary["sessions"].append(
            {"session": sid, "dur_s": round(total_dur, 1), "n_events": len(events), **masks_summary(ctxs, dt)}
        )
        print(f"{sid}: {total_dur:.0f}s，事件 {len(events)}，标签 {len(labels)} 行")
    write_report_json("wp1_events_summary.json", summary)


if __name__ == "__main__":
    main()
