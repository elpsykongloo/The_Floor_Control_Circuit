"""WP1：T4 生产标签的人工原始音频核验（PREREG #1/#7 欠账，PREREG #13 抽样协议）。

生产链 TURNEND（T4 的事件源）与 Backbiter 弱参考分歧巨大（±0.16s micro-F1≈0.32，
±1s 仍只 ≈0.49），两者都是自动标注，只有人工听原始音频才能仲裁。本工具：

  sample：分层抽样三格事件 → 盲听清单 + 立体声片段（左=ch0，右=ch1，原始音频）
     - matched：生产 TURNEND 在 ±0.5 s 内有 Backbiter 末点（两链一致格，对照）
     - prod_only：生产 TURNEND 在 ±1.0 s 内无任何 Backbiter 末点（生产多报嫌疑）
     - backbiter_only：Backbiter 末点在 ±1.0 s 内无任何生产 TURNEND（生产漏报嫌疑）
     格标签写入密封 key.json（与清单分离，**听完之前不要打开**），清单行序随机化。
  ingest：回收填好的 manifest.csv → 逐格人工确认率 + Clopper-Pearson 95% CI
     → reports/t4_human_audit.json + reports/t4_人工核验报告.md。

判定问题（对每个片段）：**标记时刻（片段第 4 秒，见 event_time_in_clip_s）附近，
目标侧说话人是否真的结束了话轮**（停止说话且未在 ~1.5 s 内继续同一话轮；
对方随后接话或长静默都算结束）。verdict 填 y / n / u（不确定），可加 notes。

用法：
  uv run python scripts/wp1_t4_human_audit.py sample [--per-cell 40] [--seed 20260719]
  uv run python scripts/wp1_t4_human_audit.py ingest
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root
from floor_circuit.schemas import EventKind

MATCH_TOL_S = 0.5
ONLY_TOL_S = 1.0
CLIP_HALF_SPAN_S = 4.0
CELLS = ("matched", "prod_only", "backbiter_only")
START_COLUMNS = ("stop", "end", "turn_end", "stop_time", "end_time")
SPEAKER_COLUMNS = ("speaker", "user_id", "participant", "spkr")


def _audit_root() -> Path:
    return data_root() / "t4_human_audit"


def _find_backbiter_csv(session_dir: Path) -> Path | None:
    hits = sorted(session_dir.rglob("*backbiter*.csv"))
    return hits[0] if hits else None


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in frame.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def _backbiter_turn_ends(path: Path) -> dict[str, list[float]]:
    frame = pd.read_csv(path)
    end_column = _pick_column(frame, START_COLUMNS)
    speaker_column = _pick_column(frame, SPEAKER_COLUMNS)
    if end_column is None or speaker_column is None:
        return {}
    ends = pd.to_numeric(frame[end_column], errors="coerce")
    keep = ends.notna()
    by_speaker: dict[str, list[float]] = {}
    for speaker, value in zip(frame.loc[keep, speaker_column], ends[keep], strict=True):
        by_speaker.setdefault(str(speaker), []).append(float(value))
    return {speaker: sorted(values) for speaker, values in by_speaker.items()}


def _heuristic_turnends(events_path: Path) -> dict[int, list[float]]:
    frame = pd.read_parquet(events_path)
    turnend = frame[frame["kind"] == EventKind.TURNEND.value]
    return {
        channel: sorted(turnend.loc[turnend["channel"] == channel, "t"].astype(float).tolist())
        for channel in (0, 1)
    }


def _has_within(times: list[float], t: float, tol: float) -> bool:
    index = int(np.searchsorted(times, t))
    return any(0 <= j < len(times) and abs(times[j] - t) <= tol for j in (index - 1, index))


def _n_matched_greedy(pred: list[float], gold: list[float], tol: float) -> int:
    """按时间序贪心一对一匹配数（仅用于挑通道映射，无需最优匹配）。"""
    used: set[int] = set()
    count = 0
    for t in pred:
        index = int(np.searchsorted(gold, t))
        best_j, best_d = None, tol + 1.0
        for j in (index - 1, index):
            if 0 <= j < len(gold) and j not in used and abs(gold[j] - t) < best_d:
                best_j, best_d = j, abs(gold[j] - t)
        if best_j is not None and best_d <= tol:
            used.add(best_j)
            count += 1
    return count


def _best_channel_mapping(
    pred_by_channel: dict[int, list[float]],
    gold_by_speaker: dict[str, list[float]],
) -> dict[int, list[float]] | None:
    """两说话人 → 两通道的映射取匹配数较大者；说话人数 ≠ 2 时返回 None。"""
    speakers = sorted(gold_by_speaker)
    if len(speakers) != 2:
        return None
    scored = []
    for order in ((0, 1), (1, 0)):
        total = sum(
            _n_matched_greedy(pred_by_channel[ch], gold_by_speaker[speakers[idx]], MATCH_TOL_S)
            for ch, idx in zip((0, 1), order, strict=True)
        )
        scored.append((total, order))
    _best_total, best_order = max(scored)
    return {ch: gold_by_speaker[speakers[idx]] for ch, idx in zip((0, 1), best_order, strict=True)}


def _read_clip(session_dir: Path, center_s: float) -> tuple[np.ndarray, int, float]:
    """原始双通道音频 → 立体声片段（左=ch0，右=ch1），返回 (wav[n,2], sr, 事件在片段内秒)。"""
    import soundfile as sf

    waves = []
    rates = []
    for ch in (0, 1):
        with sf.SoundFile(str(session_dir / f"audio_ch{ch}.wav")) as handle:
            sr = int(handle.samplerate)
            start = max(0, int((center_s - CLIP_HALF_SPAN_S) * sr))
            stop = min(len(handle), int((center_s + CLIP_HALF_SPAN_S) * sr))
            handle.seek(start)
            wav = handle.read(frames=stop - start, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        waves.append(wav)
        rates.append(sr)
    if rates[0] != rates[1]:
        raise RuntimeError(f"{session_dir.name}: 双通道采样率不一致 {rates}")
    n = min(len(waves[0]), len(waves[1]))
    stereo = np.stack([waves[0][:n], waves[1][:n]], axis=1)
    event_offset = center_s - max(0.0, center_s - CLIP_HALF_SPAN_S)
    return stereo, rates[0], event_offset


def sample(per_cell: int, seed: int, per_session_cap: int) -> None:
    sessions_dir = data_root() / "candor_extracted"
    events_dir = data_root() / "events" / "candor"
    out_root = _audit_root()
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[dict]] = {cell: [] for cell in CELLS}
    n_sessions = 0
    n_skipped = 0
    for session_dir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
        sid = session_dir.name
        events_path = events_dir / f"{sid}.events.parquet"
        csv_path = _find_backbiter_csv(session_dir)
        if not events_path.is_file() or csv_path is None:
            continue
        gold_by_speaker = _backbiter_turn_ends(csv_path)
        pred_by_channel = _heuristic_turnends(events_path)
        mapping = _best_channel_mapping(pred_by_channel, gold_by_speaker)
        if mapping is None:
            n_skipped += 1
            continue
        n_sessions += 1
        for ch in (0, 1):
            gold = mapping[ch]
            for t in pred_by_channel[ch]:
                if _has_within(gold, t, MATCH_TOL_S):
                    candidates["matched"].append({"session": sid, "channel": ch, "t": t, "source": "production"})
                elif not _has_within(gold, t, ONLY_TOL_S):
                    candidates["prod_only"].append({"session": sid, "channel": ch, "t": t, "source": "production"})
            for t in gold:
                if not _has_within(pred_by_channel[ch], t, ONLY_TOL_S):
                    candidates["backbiter_only"].append({"session": sid, "channel": ch, "t": t, "source": "backbiter"})

    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    availability: dict[str, int] = {}
    for cell in CELLS:
        pool = sorted(candidates[cell], key=lambda r: (r["session"], r["channel"], r["t"]))
        availability[cell] = len(pool)
        order = rng.permutation(len(pool))
        per_session_count: dict[str, int] = {}
        for index in order:
            row = pool[int(index)]
            if per_session_count.get(row["session"], 0) >= per_session_cap:
                continue
            chosen.append({**row, "cell": cell})
            per_session_count[row["session"]] = per_session_count.get(row["session"], 0) + 1
            if sum(1 for r in chosen if r["cell"] == cell) >= per_cell:
                break

    import soundfile as sf

    rng.shuffle(chosen)
    key_rows = []
    manifest_rows = []
    for index, row in enumerate(chosen):
        clip_id = f"clip_{index:03d}"
        stereo, sr, event_offset = _read_clip(sessions_dir / row["session"], float(row["t"]))
        sf.write(str(clips_dir / f"{clip_id}.wav"), stereo, sr)
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "clip_file": f"clips/{clip_id}.wav",
                "target_side": "left(ch0)" if row["channel"] == 0 else "right(ch1)",
                "event_time_in_clip_s": round(event_offset, 3),
                "verdict": "",
                "notes": "",
            }
        )
        key_rows.append({"clip_id": clip_id, **row})

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_root / "key.json").write_text(
        json.dumps(
            {"seed": seed, "per_cell": per_cell, "availability": availability, "items": key_rows},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (out_root / "听审说明.md").write_text(
        "\n".join(
            [
                "# T4 人工核验听审说明（盲听：听完全部之前不要打开 key.json）",
                "",
                "1. 逐行打开 manifest.csv 中的片段（左声道=ch0，右声道=ch1）。",
                "2. 判定：`target_side` 一侧的说话人在片段第 `event_time_in_clip_s` 秒附近，",
                "   是否**结束了话轮**（停止说话且 ~1.5 秒内未继续同一话轮；对方接话或长静默都算结束）。",
                "3. verdict 填 y / n / u（不确定），可在 notes 备注。",
                "4. 全部填完后运行：uv run python scripts/wp1_t4_human_audit.py ingest",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"抽样完成：{len(chosen)} 片段（候选 {availability}；跳过非双说话人会话 {n_skipped}，"
        f"覆盖 {n_sessions} 会话）→ {out_root}"
    )


def _clopper_pearson(k: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def ingest() -> None:
    out_root = _audit_root()
    manifest_path = out_root / "manifest.csv"
    key = json.loads((out_root / "key.json").read_text(encoding="utf-8"))
    cell_by_clip = {row["clip_id"]: row for row in key["items"]}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unfilled = [row["clip_id"] for row in rows if str(row.get("verdict", "")).strip().lower() not in ("y", "n", "u")]
    if unfilled:
        raise SystemExit(f"仍有 {len(unfilled)} 行 verdict 未填（y/n/u）：{unfilled[:8]}")

    per_cell: dict[str, dict[str, int]] = {cell: {"y": 0, "n": 0, "u": 0} for cell in CELLS}
    detail = []
    for row in rows:
        verdict = str(row["verdict"]).strip().lower()
        info = cell_by_clip[row["clip_id"]]
        per_cell[info["cell"]][verdict] += 1
        detail.append({**info, "verdict": verdict, "notes": row.get("notes", "")})

    cells_out = {}
    for cell, counts in per_cell.items():
        judged = counts["y"] + counts["n"]
        rate = counts["y"] / judged if judged else float("nan")
        lo, hi = _clopper_pearson(counts["y"], judged)
        cells_out[cell] = {
            **counts,
            "n_judged": judged,
            "turn_end_confirmed_rate": rate,
            "ci95": [lo, hi],
        }
    report = {
        "note": (
            "人工原始音频核验（PREREG #13 抽样协议）：matched/prod_only 的确认率 → 生产 TURNEND 精确率证据；"
            "backbiter_only 的确认率 → 生产链漏报证据；u（不确定）不计入分母"
        ),
        "seed": key.get("seed"),
        "availability": key.get("availability"),
        "cells": cells_out,
        "items": detail,
    }
    write_report_json("t4_human_audit.json", report)
    lines = [
        "# T4 人工核验报告（PREREG #13）",
        "",
        "| 格 | y | n | u | 确认率 | 95% CI |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for cell, out in cells_out.items():
        lines.append(
            f"| {cell} | {out['y']} | {out['n']} | {out['u']} "
            f"| {out['turn_end_confirmed_rate']:.3f} | [{out['ci95'][0]:.3f}, {out['ci95'][1]:.3f}] |"
        )
    lines += ["", f"> {report['note']}", ""]
    (REPORTS_DIR / "t4_人工核验报告.md").write_text("\n".join(lines), encoding="utf-8")
    for cell, out in cells_out.items():
        print(f"{cell}: 确认率 {out['turn_end_confirmed_rate']:.3f}（y={out['y']}, n={out['n']}, u={out['u']}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sp = sub.add_parser("sample", help="分层抽样 + 片段导出 + 盲听清单")
    sp.add_argument("--per-cell", type=int, default=40)
    sp.add_argument("--seed", type=int, default=20260719)
    sp.add_argument("--per-session-cap", type=int, default=2)
    sub.add_parser("ingest", help="回收填好的 manifest.csv → 报告")
    args = ap.parse_args()
    if args.mode == "sample":
        sample(args.per_cell, args.seed, args.per_session_cap)
    else:
        ingest()


if __name__ == "__main__":
    main()
