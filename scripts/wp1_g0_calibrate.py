"""WP1：G0 校准（DualTurn-SWB 官方 12.5 Hz 金标二值轨 vs 我方事件管线）。

自测（无数据依赖）：uv run python scripts/wp1_g0_calibrate.py --self-test
真实校准（先 wp1_g0_prepare + decode_mimi 产出解码音频）：
  uv run python scripts/wp1_g0_calibrate.py [--root <data_root>/dualturn_prep] [--limit N]
判据：四类（eot/hold/bot/bc）语料级 micro macro-F1 ≥ 0.85（configs/events.yaml g0.f1_threshold）。
产出：reports/g0_校准报告.md + reports/g0_summary.json。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.events.detect import ChannelContext, detect_all
from floor_circuit.events.g0 import (
    G0_CLASSES,
    accumulate_counts,
    build_pred_tracks,
    events_to_frames,
    f1_report,
    finalize_counts,
    score_binary_tracks,
)
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.pipeline import SessionChannel, process_session
from floor_circuit.events.vad import SileroVad, rasterize
from floor_circuit.schemas import Seg


def self_test() -> dict:
    """构造两通道对话 → 管线事件 → 帧映射 → 评分器闭环（并非 Gate 本身）。"""
    cfg = load_config("events")
    dt = float(cfg["grid_dt_s"])
    hz = float(cfg["g0"]["frame_hz"])
    total = 30.0
    segs0 = [Seg(8.0, 10.0), Seg(10.5, 14.0), Seg(20.0, 24.0)]
    segs1 = [Seg(12.0, 12.6), Seg(15.0, 18.0), Seg(25.0, 27.0)]
    ipus0 = build_ipus(segs0, cfg["ipu"]["merge_gap_s"])
    ipus1 = build_ipus(segs1, cfg["ipu"]["merge_gap_s"])
    ctx0 = ChannelContext(mask=rasterize(ipus0, dt, total), ipus=ipus0, bc_flags=[False] * len(ipus0))
    ctx1 = ChannelContext(mask=rasterize(ipus1, dt, total), ipus=ipus1, bc_flags=[True, False, False])
    events = detect_all(ctx0, ctx1, dt, cfg)
    n_frames = int(total * hz)
    # 旧分类式闭环
    pred0 = events_to_frames(events, n_frames, hz, cfg["g0"]["mapping"], channel=0)
    rep_cat = f1_report(pred0, pred0.copy(), cfg["g0"]["tolerance_frames"])
    # 生产用二值轨闭环
    tracks0 = build_pred_tracks(ipus0, events, 0, n_frames, hz)
    rep_bin = score_binary_tracks(tracks0, {k: v.copy() for k, v in tracks0.items()}, cfg["g0"]["tolerance_frames"])
    return {
        "categorical_macro_f1": rep_cat["macro_f1"],
        "binary_macro_f1": rep_bin["macro_f1"],
        "self_test_events": sorted({e.kind.value for e in events}),
        "pred_track_counts": {k: int(v.sum()) for k, v in tracks0.items()},
    }


REQUIRED_FILES = ("audio_ch0.wav", "audio_ch1.wav", "gold_ch0.npz", "gold_ch1.npz")


def _session_split(sdir: Path) -> str | None:
    meta = sdir / "meta.json"
    if not meta.exists():
        return None
    try:
        import json

        return json.loads(meta.read_text(encoding="utf-8")).get("split")
    except Exception:
        return None


def calibrate(root: Path, limit: int | None, split: str | None) -> dict:
    cfg = load_config("events")
    hz = float(cfg["g0"]["frame_hz"])
    tol = int(cfg["g0"]["tolerance_frames"])
    vad = SileroVad(cfg)
    from floor_circuit.stimuli.qc import load_wav

    all_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    complete = [p for p in all_dirs if all((p / f).exists() for f in REQUIRED_FILES)]
    n_incomplete = sum(1 for p in all_dirs if (p / "codes_ch0.npy").exists() and p not in complete)
    if split is not None:
        skipped_split = [p for p in complete if _session_split(p) != split]
        complete = [p for p in complete if _session_split(p) == split]
        if skipped_split:
            print(f"按 --split {split} 过滤：排除 {len(skipped_split)} 个其他划分/无溯源的会话")
    if limit:
        complete = complete[:limit]
    if not complete:
        raise SystemExit(
            f"{root} 下没有四文件齐全的会话（先跑 wp1_g0_prepare 和 runners/moshi/decode_mimi.py；"
            f"半成品目录 {n_incomplete} 个——重跑 decode_mimi 补齐）"
        )
    if n_incomplete:
        print(f"警告：{n_incomplete} 个会话缺解码音频/金标（已跳过），建议重跑 decode_mimi 批量模式")
    totals = None
    per_session: list[dict] = []
    errors: list[dict] = []
    for sdir in complete:
        try:
            wav0, sr0 = load_wav(sdir / "audio_ch0.wav")
            wav1, sr1 = load_wav(sdir / "audio_ch1.wav")
            total_dur = min(len(wav0) / sr0, len(wav1) / sr1)
            ch0 = SessionChannel(va_segs=vad.segments(wav0, sr0))
            ch1 = SessionChannel(va_segs=vad.segments(wav1, sr1))
            events, ctxs, _dt = process_session(ch0, ch1, total_dur, cfg, lang="en")
            session_macro = []
            frames_mismatch = False
            for ch in (0, 1):
                gold_npz = np.load(sdir / f"gold_ch{ch}.npz", allow_pickle=False)
                gold = {k: gold_npz[k] for k in G0_CLASSES}
                n_frames = len(gold["eot"])
                if abs(total_dur * hz - n_frames) > 5:  # 解码时长与金标帧数明显不符 → 疑似截断
                    frames_mismatch = True
                pred = build_pred_tracks(ctxs[ch].ipus, events, ch, n_frames, hz)
                totals = accumulate_counts(totals, pred, gold, tol)
                session_macro.append(score_binary_tracks(pred, gold, tol)["macro_f1"])
            row = {"session": sdir.name, "macro_f1_mean": float(np.mean(session_macro))}
            if frames_mismatch:
                row["frames_mismatch"] = True
                print(f"警告 {sdir.name}: 解码时长与金标帧数不符（疑似截断 wav，建议删除后重解码）")
            per_session.append(row)
            print(f"{sdir.name}: 会话 macro-F1 ≈ {np.mean(session_macro):.3f}")
        except Exception as e:
            errors.append({"session": sdir.name, "error": repr(e)})
            print(f"错误 {sdir.name}: {e!r}（已跳过，继续）")
    if totals is None:
        raise SystemExit(f"全部 {len(complete)} 个会话处理失败，样例：{errors[:3]}")
    report = finalize_counts(totals)
    report["n_sessions"] = len(per_session)
    report["n_errors"] = len(errors)
    report["n_incomplete_skipped"] = n_incomplete
    report["per_session"] = per_session
    report["errors"] = errors
    report["split"] = split
    report["threshold"] = float(cfg["g0"]["f1_threshold"])
    report["g0_pass"] = bool(report["macro_f1"] >= report["threshold"])
    return report


def write_markdown(report: dict) -> None:
    lines = [
        "# G0 校准报告（DualTurn-SWB，Mimi 解码音频）",
        "",
        f"- 会话数：{report['n_sessions']}；判据：macro-F1 ≥ {report['threshold']}",
        "- **语料级 macro-F1 = {:.4f} → {}**".format(
            report["macro_f1"],
            "通过 ✅" if report["g0_pass"] else "未过 ❌（只修实现不动参数，见 文档/02 §WP1）",
        ),
        "",
        "| 类 | precision | recall | F1 | n_pred | n_gold |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in G0_CLASSES:
        cell = report["per_class"][c]
        lines.append(
            f"| {c} | {cell['precision']:.4f} | {cell['recall']:.4f} | {cell['f1']:.4f} "
            f"| {cell['n_pred']} | {cell['n_gold']} |"
        )
    worst = sorted(report["per_session"], key=lambda r: r["macro_f1_mean"])[:10]
    lines += ["", "最差 10 个会话（排查线索）：", ""]
    lines += [
        f"- {r['session']}: {r['macro_f1_mean']:.3f}"
        + ("（解码时长与金标帧数不符）" if r.get("frames_mismatch") else "")
        for r in worst
    ]
    if report.get("n_errors"):
        lines += ["", f"处理失败会话 {report['n_errors']} 个（详见 g0_summary.json 的 errors 节）"]
    if report.get("n_incomplete_skipped"):
        lines += ["", f"缺文件被跳过的会话 {report['n_incomplete_skipped']} 个（重跑 decode_mimi 补齐）"]
    (REPORTS_DIR / "g0_校准报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="按 meta.json 的划分溯源过滤（如 test）")
    args = ap.parse_args()
    if args.self_test:
        rep = self_test()
        ok = rep["categorical_macro_f1"] == 1.0 and rep["binary_macro_f1"] == 1.0
        write_report_json("g0_selftest.json", {"pass": bool(ok), **rep})
        print(
            f"G0 自测：分类式 {rep['categorical_macro_f1']:.3f} / 二值轨 {rep['binary_macro_f1']:.3f}"
            f"（均应 1.000）；事件 {rep['self_test_events']}"
        )
        return
    root = Path(args.root) if args.root else data_root() / "dualturn_prep"
    report = calibrate(root, args.limit, args.split)
    write_report_json("g0_summary.json", report)
    write_markdown(report)
    verdict = "通过" if report["g0_pass"] else "未过"
    print(f"G0：macro-F1 = {report['macro_f1']:.4f}（阈值 {report['threshold']}）→ {verdict}")


if __name__ == "__main__":
    main()
