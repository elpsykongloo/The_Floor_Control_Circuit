"""WP-E1：按精确会话清单增算事件与 T1–T5 标签。

本入口复用 `wp1_run_events.py` 的全部生产函数，同时不修改该权威脚本本身，因而
不会仅因增加 E1 调度能力就使既有事件完成标记的源码指纹失效。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wp1_run_events as wp1
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


def settings_sha256(
    cfg: dict,
    grids: dict,
    clock: str,
    lang: str,
    event_pipeline_code_sha256: str,
) -> tuple[float, list[int], str]:
    """复用权威入口的设置指纹字面量，并返回派生的时钟与 δ。"""
    step_s = float(grids["clocks"][clock]["step_ms"]) / 1000.0
    deltas = list(grids["delta_grid"][f"{clock}_ms"])
    digest = hashlib.sha256(
        json.dumps(
            {
                "events": cfg,
                "clock": clock,
                "step_s": step_s,
                "deltas": deltas,
                "lang": lang,
                "event_pipeline_code_sha256": event_pipeline_code_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return step_s, deltas, digest


def load_session_list(root: Path, path: Path, limit: int | None = None) -> list[Path]:
    """读取清单并保留顺序；拒绝重复、越界路径与缺双通道音频。"""
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"会话清单含重复项：{path}")
    resolved_root = root.resolve()
    sessions = [root / sid for sid in ids]
    invalid = [
        session.name
        for session in sessions
        if session.parent.resolve() != resolved_root
        or not (session / "audio_ch0.wav").is_file()
        or not (session / "audio_ch1.wav").is_file()
    ]
    if invalid:
        raise SystemExit(f"会话清单有 {len(invalid)} 项缺双通道音频：{invalid[:5]}")
    return sessions[:limit] if limit is not None else sessions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-list", type=Path, required=True)
    ap.add_argument("--sessions-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--clock", default="moshi")
    args = ap.parse_args()

    cfg = load_config("events")
    grids = load_config("grids")
    event_pipeline_code_sha256 = wp1._event_pipeline_code_sha256()
    step_s, deltas, settings_digest = settings_sha256(
        cfg, grids, args.clock, args.lang, event_pipeline_code_sha256
    )
    root = args.sessions_dir or (data_root() / "candor_extracted")
    sessions = load_session_list(root, args.session_list, args.limit)
    out_dir = data_root() / "events" / "candor"
    out_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVad(cfg)
    summary: dict = {
        "clock": args.clock,
        "session_list": str(args.session_list.resolve()),
        "n_sessions": len(sessions),
        "n_cached": 0,
        "n_processed": 0,
        "event_pipeline_code_sha256": event_pipeline_code_sha256,
        "sessions": [],
    }
    for sdir in sessions:
        sid = sdir.name
        fingerprint = wp1._session_fingerprint(
            sdir, settings_digest, event_pipeline_code_sha256
        )
        cached = wp1._cached_session_summary(sdir, out_dir, fingerprint)
        if cached is not None:
            summary["sessions"].append(cached)
            summary["n_cached"] += 1
            print(
                f"{sid}: 复用已校验输出，事件 {cached['n_events']}，"
                f"标签 {cached['n_labels']} 行"
            )
            continue
        wav0, sr0 = load_wav(sdir / "audio_ch0.wav")
        wav1, sr1 = load_wav(sdir / "audio_ch1.wav")
        total_dur = min(len(wav0) / sr0, len(wav1) / sr1)
        ch0 = SessionChannel(va_segs=vad.segments(wav0, sr0))
        ch1 = SessionChannel(va_segs=vad.segments(wav1, sr1))
        events, contexts, double_talk = process_session(
            ch0, ch1, total_dur, cfg, lang=args.lang
        )
        labels = labels_both_roles(
            events, contexts, double_talk, total_dur, cfg, step_s, deltas
        )
        wp1._write_session_outputs(
            events_to_dataframe(events), labels, sdir, out_dir, fingerprint
        )
        summary["sessions"].append(
            {
                "session": sid,
                "dur_s": round(total_dur, 1),
                "n_events": len(events),
                "n_labels": len(labels),
                "cached": False,
                **masks_summary(contexts, double_talk),
            }
        )
        summary["n_processed"] += 1
        print(f"{sid}: {total_dur:.0f}s，事件 {len(events)}，标签 {len(labels)} 行")
    write_report_json("wp_e1_missing_events_summary.json", summary)


if __name__ == "__main__":
    main()
