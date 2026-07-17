"""WP1：事件管线批量运行（CANDOR 提取音频对 → 事件 + T1–T5 标签 parquet）。

用法（先完成 wp2_extract_candor）：
  uv run python scripts/wp1_run_events.py --sessions-dir <data_root>/candor_extracted --limit 5
产出：<data_root>/events/candor/<session>.events.parquet 与 .labels.parquet
     + reports/wp1_events_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import wave
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


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """完成标记最后发布；缺少标记的双文件永不视为可复用。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _session_fingerprint(sdir: Path, settings_sha256: str) -> dict:
    """快速绑定源音频版本与事件设置；输出文件另用内容哈希校验。"""
    audio = {}
    for channel in (0, 1):
        path = sdir / f"audio_ch{channel}.wav"
        stat = path.stat()
        audio[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"settings_sha256": settings_sha256, "source_audio": audio}


def _write_session_outputs(
    events_frame,
    labels_frame,
    sdir: Path,
    out_dir: Path,
    fingerprint: dict,
) -> None:
    """先写齐两个临时 parquet，再发布双文件，最后发布带哈希的完成标记。"""
    sid = sdir.name
    events_path = out_dir / f"{sid}.events.parquet"
    labels_path = out_dir / f"{sid}.labels.parquet"
    complete_path = out_dir / f"{sid}.complete.json"
    complete_path.unlink(missing_ok=True)
    nonce = f"{os.getpid()}.{time.time_ns()}"
    events_tmp = out_dir / f".{sid}.events.{nonce}.tmp.parquet"
    labels_tmp = out_dir / f".{sid}.labels.{nonce}.tmp.parquet"
    try:
        events_frame.to_parquet(events_tmp)
        labels_frame.to_parquet(labels_tmp)
        events_tmp.replace(events_path)
        labels_tmp.replace(labels_path)
        outputs = {
            "events": {
                "name": events_path.name,
                "size": events_path.stat().st_size,
                "sha256": _sha256_file(events_path),
            },
            "labels": {
                "name": labels_path.name,
                "size": labels_path.stat().st_size,
                "sha256": _sha256_file(labels_path),
            },
        }
        _write_json_atomic(
            complete_path,
            {
                "schema_version": 1,
                "session": sid,
                "input": fingerprint,
                "outputs": outputs,
            },
        )
    finally:
        events_tmp.unlink(missing_ok=True)
        labels_tmp.unlink(missing_ok=True)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def _cached_session_summary(
    sdir: Path,
    out_dir: Path,
    fingerprint: dict,
) -> dict | None:
    """同时验证完成标记、输入指纹和双文件哈希，才允许断点复用。"""
    import pyarrow.parquet as pq

    sid = sdir.name
    events_path = out_dir / f"{sid}.events.parquet"
    labels_path = out_dir / f"{sid}.labels.parquet"
    complete_path = out_dir / f"{sid}.complete.json"
    if not events_path.is_file() or not labels_path.is_file() or not complete_path.is_file():
        return None
    try:
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            marker.get("schema_version") != 1
            or marker.get("session") != sid
            or marker.get("input") != fingerprint
        ):
            return None
        for key, path in (("events", events_path), ("labels", labels_path)):
            declared = marker["outputs"][key]
            if (
                declared.get("name") != path.name
                or declared.get("size") != path.stat().st_size
                or declared.get("sha256") != _sha256_file(path)
            ):
                return None
        n_events = int(pq.ParquetFile(events_path).metadata.num_rows)
        n_labels = int(pq.ParquetFile(labels_path).metadata.num_rows)
        if n_labels <= 0:
            return None
        total_dur = min(
            _wav_duration(sdir / "audio_ch0.wav"),
            _wav_duration(sdir / "audio_ch1.wav"),
        )
    except (KeyError, OSError, EOFError, ValueError, json.JSONDecodeError, wave.Error):
        return None
    return {
        "session": sid,
        "dur_s": round(total_dur, 1),
        "n_events": n_events,
        "n_labels": n_labels,
        "cached": True,
    }


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
    settings_sha256 = hashlib.sha256(
        json.dumps(
            {
                "events": cfg,
                "clock": args.clock,
                "step_s": step_s,
                "deltas": deltas,
                "lang": args.lang,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    root = Path(args.sessions_dir or (data_root() / "candor_extracted"))
    out_dir = data_root() / "events" / "candor"
    out_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVad(cfg)

    sessions = sorted(p for p in root.iterdir() if (p / "audio_ch0.wav").exists())
    if args.limit:
        sessions = sessions[: args.limit]
    summary: dict = {
        "clock": args.clock,
        "n_sessions": len(sessions),
        "n_cached": 0,
        "n_processed": 0,
        "sessions": [],
    }
    for sdir in sessions:
        sid = sdir.name
        fingerprint = _session_fingerprint(sdir, settings_sha256)
        cached = _cached_session_summary(sdir, out_dir, fingerprint)
        if cached is not None:
            summary["sessions"].append(cached)
            summary["n_cached"] += 1
            print(f"{sid}: 复用已校验输出，事件 {cached['n_events']}，标签 {cached['n_labels']} 行")
            continue
        wav0, sr0 = load_wav(sdir / "audio_ch0.wav")
        wav1, sr1 = load_wav(sdir / "audio_ch1.wav")
        total_dur = min(len(wav0) / sr0, len(wav1) / sr1)
        ch0 = SessionChannel(va_segs=vad.segments(wav0, sr0))
        ch1 = SessionChannel(va_segs=vad.segments(wav1, sr1))
        events, ctxs, dt = process_session(ch0, ch1, total_dur, cfg, lang=args.lang)
        labels = labels_both_roles(events, ctxs, dt, total_dur, cfg, step_s, deltas)
        _write_session_outputs(
            events_to_dataframe(events),
            labels,
            sdir,
            out_dir,
            fingerprint,
        )
        summary["sessions"].append(
            {
                "session": sid,
                "dur_s": round(total_dur, 1),
                "n_events": len(events),
                "n_labels": len(labels),
                "cached": False,
                **masks_summary(ctxs, dt),
            }
        )
        summary["n_processed"] += 1
        print(f"{sid}: {total_dur:.0f}s，事件 {len(events)}，标签 {len(labels)} 行")
    write_report_json("wp1_events_summary.json", summary)


if __name__ == "__main__":
    main()
