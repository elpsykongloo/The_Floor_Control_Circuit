"""WP1：T4 标签风险审计 —— 启发式 TURNEND 事件 vs CANDOR Backbiter 话轮末点。

背景（PREREG #7）：CANDOR 的 T4 标签由 Silero VAD + 启发式事件链生成，生产链的
独立原始音频验证尚未完成；本脚本把撤回复盘中的只读交叉核验固化为可复跑工具，
以 Backbiter 话轮末点为**弱参考**（Backbiter 本身不是权威金标，结果只作风险证据，
不替代 PREREG 要求的人工原始音频验证）。

对每个已提取会话：
  - 从 <events_dir>/<sid>.events.parquet 取启发式 TURNEND 事件时刻（分通道）；
  - 从 <sessions_dir>/<sid>/**/ *backbiter*.csv 取各说话人话轮末点；
  - 在 ±tol 容差网格下做一对一最大匹配，输出 P/R/F1。
声道与说话人的对应存在不确定性（stereo_split 的 channel_map 未核对），因此同时
报告：合并双侧（channel-agnostic）与两种通道映射下的逐侧结果，取映射较优者。

用法：
  uv run python scripts/wp1_t4_audit.py [--limit 20] [--tolerances 0.16,0.5,1.0]
产出：reports/t4_label_audit.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import write_report_json

from floor_circuit.config import data_root
from floor_circuit.events.matching import macro_f1, match_sparse_times, precision_recall_f1
from floor_circuit.schemas import EventKind

START_COLUMNS = ("stop", "end", "turn_end", "stop_time", "end_time")
SPEAKER_COLUMNS = ("speaker", "user_id", "participant", "spkr")


def _find_backbiter_csv(session_dir: Path) -> Path | None:
    hits = sorted(session_dir.rglob("*backbiter*.csv"))
    return hits[0] if hits else None


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in frame.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def _backbiter_turn_ends(path: Path) -> tuple[dict[str, list[float]], dict]:
    frame = pd.read_csv(path)
    end_column = _pick_column(frame, START_COLUMNS)
    speaker_column = _pick_column(frame, SPEAKER_COLUMNS)
    meta = {
        "csv": str(path),
        "columns": [str(column) for column in frame.columns],
        "end_column": end_column,
        "speaker_column": speaker_column,
        "n_rows": len(frame),
    }
    if end_column is None:
        return {}, meta
    ends = pd.to_numeric(frame[end_column], errors="coerce")
    keep = ends.notna()
    if speaker_column is None:
        return {"_all": sorted(ends[keep].tolist())}, meta
    by_speaker: dict[str, list[float]] = {}
    for speaker, value in zip(frame.loc[keep, speaker_column], ends[keep], strict=True):
        by_speaker.setdefault(str(speaker), []).append(float(value))
    return {speaker: sorted(values) for speaker, values in by_speaker.items()}, meta


def _heuristic_turnends(events_path: Path) -> dict[int, list[float]]:
    frame = pd.read_parquet(events_path)
    turnend = frame[frame["kind"] == EventKind.TURNEND.value]
    return {
        channel: sorted(turnend.loc[turnend["channel"] == channel, "t"].astype(float).tolist())
        for channel in (0, 1)
    }


def _pooled_metrics(pred: list[float], gold: list[float], tol_s: float) -> dict:
    hits = match_sparse_times(pred, gold, tol_s)
    return precision_recall_f1(hits, len(pred), len(gold))


def _session_audit(
    pred_by_channel: dict[int, list[float]],
    gold_by_speaker: dict[str, list[float]],
    tolerances: list[float],
) -> dict:
    speakers = sorted(gold_by_speaker)
    pooled_pred = sorted(pred_by_channel[0] + pred_by_channel[1])
    pooled_gold = sorted(t for values in gold_by_speaker.values() for t in values)
    out: dict = {}
    for tol in tolerances:
        entry: dict = {"pooled": _pooled_metrics(pooled_pred, pooled_gold, tol)}
        if len(speakers) == 2:
            mappings = []
            for order in ((0, 1), (1, 0)):
                sides = [
                    _pooled_metrics(
                        pred_by_channel[channel],
                        gold_by_speaker[speakers[speaker_index]],
                        tol,
                    )
                    for channel, speaker_index in zip((0, 1), order, strict=True)
                ]
                mappings.append(
                    {
                        "mapping": {"ch0": speakers[order[0]], "ch1": speakers[order[1]]},
                        "macro_f1": macro_f1(sides),
                        "per_side": sides,
                    }
                )
            entry["best_mapping"] = max(mappings, key=lambda m: m["macro_f1"])
            entry["worst_mapping_macro_f1"] = min(m["macro_f1"] for m in mappings)
        out[f"tol_{tol:g}s"] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=None, help="默认 <data_root>/candor_extracted")
    ap.add_argument("--events-dir", default=None, help="默认 <data_root>/events/candor")
    ap.add_argument("--tolerances", default="0.16,0.5,1.0", help="逗号分隔的匹配容差（秒）")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sessions_dir = Path(args.sessions_dir or (data_root() / "candor_extracted"))
    events_dir = Path(args.events_dir or (data_root() / "events" / "candor"))
    tolerances = [float(value) for value in args.tolerances.split(",")]

    session_dirs = sorted(
        path
        for path in sessions_dir.iterdir()
        if path.is_dir() and (events_dir / f"{path.name}.events.parquet").is_file()
    )
    if args.limit:
        session_dirs = session_dirs[: args.limit]

    sessions: list[dict] = []
    skipped: list[dict] = []
    for session_dir in session_dirs:
        sid = session_dir.name
        csv_path = _find_backbiter_csv(session_dir)
        if csv_path is None:
            skipped.append({"session": sid, "reason": "无 backbiter csv"})
            continue
        gold_by_speaker, backbiter_meta = _backbiter_turn_ends(csv_path)
        if not gold_by_speaker:
            skipped.append({"session": sid, "reason": "backbiter 列无法解析", **backbiter_meta})
            continue
        pred_by_channel = _heuristic_turnends(events_dir / f"{sid}.events.parquet")
        sessions.append(
            {
                "session": sid,
                "n_speakers": len(gold_by_speaker),
                "n_pred": sum(len(v) for v in pred_by_channel.values()),
                "n_gold": sum(len(v) for v in gold_by_speaker.values()),
                "backbiter": backbiter_meta,
                "metrics": _session_audit(pred_by_channel, gold_by_speaker, tolerances),
            }
        )

    aggregate: dict = {}
    for tol in tolerances:
        key = f"tol_{tol:g}s"
        rows = [entry["metrics"][key]["pooled"] for entry in sessions]
        if rows:
            hits = sum(row["n_matched"] for row in rows)
            n_pred = sum(row["n_pred"] for row in rows)
            n_gold = sum(row["n_gold"] for row in rows)
            aggregate[key] = {
                "micro": precision_recall_f1(hits, n_pred, n_gold),
                "macro_f1": macro_f1(rows),
            }
    report = {
        "note": (
            "Backbiter 为弱参考，本报告仅作 T4 风险证据；"
            "PREREG 要求的生产链独立原始音频验证仍需人工完成"
        ),
        "n_sessions": len(sessions),
        "n_skipped": len(skipped),
        "tolerances_s": tolerances,
        "aggregate_pooled": aggregate,
        "sessions": sessions,
        "skipped": skipped,
    }
    write_report_json("t4_label_audit.json", report)
    for key, values in aggregate.items():
        micro = values["micro"]
        print(
            f"{key}: micro P={micro['precision']:.3f} R={micro['recall']:.3f} "
            f"F1={micro['f1']:.3f}（{len(sessions)} 会话）"
        )


if __name__ == "__main__":
    main()
