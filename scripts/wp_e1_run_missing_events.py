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


def shard_sessions(
    sessions: list[Path], num_shards: int, shard_id: int, limit: int | None = None
) -> list[Path]:
    """按清单固定顺序做互斥步长分片；limit 作用于分片结果。"""
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("要求 num_shards > 0 且 0 <= shard_id < num_shards")
    selected = sessions[shard_id::num_shards]
    return selected[:limit] if limit is not None else selected


def load_vad_channel(
    session_dir: Path, channel: int, vad: SileroVad
) -> tuple[SessionChannel, float]:
    """单独载入并处理一路波形，让完整双通道不会同时驻留内存。"""
    wav, sample_rate = load_wav(session_dir / f"audio_ch{channel}.wav")
    duration_s = len(wav) / sample_rate
    result = SessionChannel(va_segs=vad.segments(wav, sample_rate))
    return result, duration_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-list", type=Path, required=True)
    ap.add_argument("--sessions-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
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
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit 必须为正整数")
    try:
        sessions = shard_sessions(
            load_session_list(root, args.session_list),
            int(args.num_shards),
            int(args.shard_id),
            args.limit,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    out_dir = data_root() / "events" / "candor"
    out_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVad(cfg)
    summary: dict = {
        "clock": args.clock,
        "session_list": str(args.session_list.resolve()),
        "n_sessions": len(sessions),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
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
        ch0, dur0 = load_vad_channel(sdir, 0, vad)
        ch1, dur1 = load_vad_channel(sdir, 1, vad)
        total_dur = min(dur0, dur1)
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
    report_name = (
        "wp_e1_missing_events_summary.json"
        if int(args.num_shards) == 1
        else f"wp_e1_missing_events_summary_s{args.shard_id}_of_{args.num_shards}.json"
    )
    write_report_json(report_name, summary)


if __name__ == "__main__":
    main()
