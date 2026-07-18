"""WP1：G0 校准（三层协议，2026-07-17 语义定稿版）。

层 1 · 协议正确性（只需金标，无音频依赖；库版官方算法 vs 金标逐帧全等）：
  uv run python scripts/wp1_g0_calibrate.py --protocol-check [--split test] [--limit N]
  语义权威 = 官方 commit 2d0db21 的逐句移植（scripts/wp1_g0_reference_recompute.py 已在
  138/138 会话全等）；库版（events/g0_official.py）与其为两份独立实现，本检验即持续对拍。

层 2+3 · VAD 一致性 + 端到端（需解码音频；层 1 全等后再跑）：
  uv run python scripts/wp1_g0_calibrate.py [--split val] [--limit N]
  层 2：Silero(Mimi 解码音频) 12.5 Hz 帧化 vs 官方金标 VAD，逐通道 P/R/F1；
  层 3：Silero VAD → 官方算法 → vs 金标四类（±tolerance 稀疏匹配），报 macro-F1。

一次性确认（门槛已冻结，PREREG #9，用户 2026-07-18 裁决）：
  uv run python scripts/wp1_g0_calibrate.py --confirmation
  确认集 = test 除目录序前 20 探索会话外的 118 个会话，只允许运行一次；
  Gate 四条件：层1 全等 1.000 ∧ 层2 双通道 VAD F1 ≥ 0.90 ∧
  层3 语料级 macro-F1 ∈ [0.48, 0.56]（高出带 = 红旗调查）∧ 会话级 P10 ≥ 0.30。
  产出 reports/g0_confirmation.json + reports/g0_确认报告.md。

集合纪律（PREREG #1）：val 295 会话为门槛估计集（已完成）；--split 诊断路径保留。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.events.g0 import G0_CLASSES, accumulate_counts, finalize_counts, score_binary_tracks
from floor_circuit.events.g0_gate import evaluate_g0_gate
from floor_circuit.events.g0_official import (
    SOURCE_COMMIT,
    compute_official_labels,
    exact_mismatches,
    segments_to_frame_track,
    track_prf,
)
from floor_circuit.events.vad import SileroVad

REQUIRED_GOLD = ("gold_ch0.npz", "gold_ch1.npz")
REQUIRED_AUDIO = ("audio_ch0.wav", "audio_ch1.wav")


def _session_dirs(root: Path, need_audio: bool, split: str | None, limit: int | None) -> list[Path]:
    import json

    dirs = []
    for p in sorted(d for d in root.iterdir() if d.is_dir()):
        if not all((p / f).exists() for f in REQUIRED_GOLD):
            continue
        if need_audio and not all((p / f).exists() for f in REQUIRED_AUDIO):
            continue
        if split is not None:
            meta = p / "meta.json"
            try:
                if json.loads(meta.read_text(encoding="utf-8")).get("split") != split:
                    continue
            except Exception:
                continue
        dirs.append(p)
    return dirs[:limit] if limit else dirs


def _load_gold(sdir: Path, ch: int) -> dict[str, np.ndarray]:
    z = np.load(sdir / f"gold_ch{ch}.npz", allow_pickle=False)
    return {k: z[k] for k in (*G0_CLASSES, "vad")}


def _predict_pair(vad0: np.ndarray, vad1: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """双通道 VAD → 每通道四类轨（库版官方算法，一次算完两通道）。"""
    full = compute_official_labels(vad0, vad1)
    return {ch: {cls: full[f"{cls}_ch{ch}"] for cls in G0_CLASSES} for ch in (0, 1)}


def protocol_check(root: Path, limit: int | None, split: str | None) -> dict:
    """层 1：官方金标 VAD → 库版官方算法 vs 金标标签，逐帧全等检验。"""
    dirs = _session_dirs(root, need_audio=False, split=split, limit=limit)
    if not dirs:
        raise SystemExit(f"{root} 下没有含金标的会话（先跑 wp1_g0_prepare）")
    mism = dict.fromkeys(G0_CLASSES, 0)
    total_frames = 0
    bad_sessions = []
    for sdir in dirs:
        pair = {ch: _load_gold(sdir, ch) for ch in (0, 1)}
        pred = _predict_pair(pair[0]["vad"], pair[1]["vad"])
        session_mism = 0
        for ch in (0, 1):
            for cls, cnt in exact_mismatches(pred[ch], pair[ch]).items():
                mism[cls] += cnt
                session_mism += cnt
            total_frames += len(pair[ch]["vad"])
        if session_mism:
            bad_sessions.append({"session": sdir.name, "mismatch": session_mism})
    report = {
        "n_sessions": len(dirs),
        "source_commit": SOURCE_COMMIT,
        "mismatch": mism,
        "total_mismatch": sum(mism.values()),
        "n_channel_frames": total_frames,
        "exact_equal": sum(mism.values()) == 0,
        "mismatched_sessions": bad_sessions[:20],
    }
    write_report_json("g0_protocol_check.json", report)
    if report["exact_equal"]:
        print(f"层 1 协议全等 ✅（{len(dirs)} 会话，{total_frames} 通道帧，官方 commit {SOURCE_COMMIT[:8]}）")
    else:
        print(f"层 1 仍有 {report['total_mismatch']} 帧不等：{mism}——库版与参考脚本出现分歧，需回传排查")
    return report


def calibrate(
    root: Path,
    limit: int | None,
    split: str | None,
    session_dirs: list[Path] | None = None,
) -> dict:
    cfg = load_config("events")
    hz = float(cfg["g0"]["frame_hz"])
    tol = int(cfg["g0"]["tolerance_frames"])
    vad = SileroVad(cfg)
    from floor_circuit.stimuli.qc import load_wav

    dirs = session_dirs if session_dirs is not None else _session_dirs(root, need_audio=True, split=split, limit=limit)
    n_incomplete = len(_session_dirs(root, need_audio=False, split=split, limit=None)) - len(
        _session_dirs(root, need_audio=True, split=split, limit=None)
    )
    if not dirs:
        raise SystemExit(f"{root} 下没有四文件齐全的会话（半成品 {n_incomplete} 个——重跑 decode_mimi）")
    if n_incomplete:
        print(f"警告：{n_incomplete} 个会话缺解码音频（已跳过）")

    layer1 = None
    vad_stats: dict[str, dict] = {"ch0": None, "ch1": None}
    totals = None
    per_session: list[dict] = []
    errors: list[dict] = []
    for sdir in dirs:
        try:
            gold = {ch: _load_gold(sdir, ch) for ch in (0, 1)}
            wavs = {ch: load_wav(sdir / f"audio_ch{ch}.wav") for ch in (0, 1)}
            n_frames = {ch: len(gold[ch]["vad"]) for ch in (0, 1)}
            pred_vad = {}
            for ch in (0, 1):
                w, sr = wavs[ch]
                segs = vad.segments(w, sr)
                pred_vad[ch] = segments_to_frame_track(segs, n_frames[ch], hz, rule="majority")
            session_macro = []
            # 层 1（顺带累计）：金标 VAD → 官方算法 vs 金标标签（逐帧严格）
            l1_pred = _predict_pair(gold[0]["vad"], gold[1]["vad"])
            # 层 3：端到端（Silero VAD → 官方算法）
            e2e_pred = _predict_pair(pred_vad[0], pred_vad[1])
            for ch in (0, 1):
                layer1 = accumulate_counts(layer1, l1_pred[ch], gold[ch], 0, sparse=())
                # 层 2：VAD 一致性
                prf = track_prf(pred_vad[ch], gold[ch]["vad"])
                key = f"ch{ch}"
                if vad_stats[key] is None:
                    vad_stats[key] = {"tp": 0, "n_pred": 0, "n_gold": 0}
                vad_stats[key]["tp"] += round(prf["precision"] * prf["n_pred"])
                vad_stats[key]["n_pred"] += prf["n_pred"]
                vad_stats[key]["n_gold"] += prf["n_gold"]
                totals = accumulate_counts(totals, e2e_pred[ch], gold[ch], tol)
                session_macro.append(score_binary_tracks(e2e_pred[ch], gold[ch], tol)["macro_f1"])
            per_session.append({"session": sdir.name, "macro_f1_mean": float(np.mean(session_macro))})
            print(f"{sdir.name}: 端到端 macro-F1 ≈ {np.mean(session_macro):.3f}")
        except Exception as e:
            errors.append({"session": sdir.name, "error": repr(e)})
            print(f"错误 {sdir.name}: {e!r}（已跳过，继续）")
    if totals is None:
        raise SystemExit(f"全部会话处理失败，样例：{errors[:3]}")

    layer1_report = finalize_counts(layer1)
    vad_report = {}
    for key, st in vad_stats.items():
        prec = st["tp"] / st["n_pred"] if st["n_pred"] else 0.0
        rec = st["tp"] / st["n_gold"] if st["n_gold"] else 0.0
        vad_report[key] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        }
    report = finalize_counts(totals)
    report.update(
        n_sessions=len(per_session),
        n_errors=len(errors),
        n_incomplete_skipped=n_incomplete,
        per_session=per_session,
        errors=errors,
        split=split,
        gate_config=dict(cfg["g0"]["gate"]),
        layer1_protocol={"macro_f1": layer1_report["macro_f1"], "per_class": layer1_report["per_class"]},
        layer2_vad=vad_report,
        source_commit=SOURCE_COMMIT,
    )
    return report


def write_markdown(report: dict) -> None:
    lines = [
        "# G0 校准报告（三层协议，DualTurn-SWB，Mimi 解码音频）",
        "",
        f"- 会话数：{report['n_sessions']}；容差 ±2 帧；冻结门槛见 PREREG #9（本报告为诊断路径）",
        "- 层 1 协议正确性（金标 VAD → 官方算法）：macro-F1 = "
        f"{report['layer1_protocol']['macro_f1']:.4f}（目标 1.0000）",
        "- 层 2 VAD 一致性（Silero@Mimi 解码 vs 官方 VAD）："
        + "，".join(f"{k} F1 {v['f1']:.4f}" for k, v in report["layer2_vad"].items()),
        f"- **层 3 端到端 macro-F1 = {report['macro_f1']:.4f}**",
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
    lines += ["", "最差 10 个会话：", ""]
    lines += [f"- {r['session']}: {r['macro_f1_mean']:.3f}" for r in worst]
    if report.get("n_errors"):
        lines += ["", f"处理失败 {report['n_errors']} 个（见 g0_summary.json errors）"]
    (REPORTS_DIR / "g0_校准报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_confirmation(root: Path) -> dict:
    """一次性确认集运行（PREREG #9）：118 会话 + 冻结四条件 Gate 裁决。"""
    cfg = load_config("events")
    gate_cfg = dict(cfg["g0"]["gate"])
    n_exploration = int(gate_cfg["exploration_sessions"])
    expected = int(gate_cfg["confirmation_expected_sessions"])

    all_test = _session_dirs(root, need_audio=False, split="test", limit=None)
    exploration = all_test[:n_exploration]
    confirmation = all_test[n_exploration:]
    if len(confirmation) != expected:
        raise SystemExit(
            f"确认集会话数 {len(confirmation)} ≠ 冻结值 {expected}"
            f"（test 共 {len(all_test)}，探索集 {len(exploration)}）——不允许静默子集，先补齐再跑"
        )
    missing_audio = [d.name for d in confirmation if not all((d / f).exists() for f in REQUIRED_AUDIO)]
    if missing_audio:
        raise SystemExit(
            f"确认集有 {len(missing_audio)} 个会话缺解码音频（一次性运行要求 118 全齐）。"
            f"样例：{missing_audio[:5]}——先对 test 补跑 decode_mimi"
        )

    report = calibrate(root, limit=None, split="test", session_dirs=confirmation)
    if report["n_errors"]:
        raise SystemExit(
            f"确认集有 {report['n_errors']} 个会话处理失败，一次性运行不允许缺失：{report['errors'][:3]}"
        )
    gate = evaluate_g0_gate(
        layer1_macro_f1=report["layer1_protocol"]["macro_f1"],
        layer1_per_class_f1={
            cls: cell["f1"] for cls, cell in report["layer1_protocol"]["per_class"].items()
        },
        layer2_f1_by_channel={key: value["f1"] for key, value in report["layer2_vad"].items()},
        layer3_corpus_macro_f1=report["macro_f1"],
        layer3_session_macro_f1s=[row["macro_f1_mean"] for row in report["per_session"]],
        gate_cfg=gate_cfg,
    )
    report.update(
        gate=gate,
        gate_frozen=True,
        exploration_sessions=[d.name for d in exploration],
        confirmation_one_shot=True,
    )
    write_report_json("g0_confirmation.json", report)

    conditions = gate["conditions"]
    lines = [
        "# G0 一次性确认报告（PREREG #9 冻结门槛，118 会话）",
        "",
        f"- **Gate 裁决：`{gate['verdict']}`**",
        f"- 层1 协议全等：macro-F1 = {conditions['layer1_exact']['macro_f1']:.4f}"
        f"（要求 {conditions['layer1_exact']['required']:.4f}）→ "
        + ("✅" if conditions["layer1_exact"]["passed"] else "❌"),
        "- 层2 VAD F1："
        + "，".join(f"{k} {v:.4f}" for k, v in conditions["layer2_vad_f1"]["by_channel"].items())
        + f"（下限 {conditions['layer2_vad_f1']['required_min']}）→ "
        + ("✅" if conditions["layer2_vad_f1"]["passed"] else "❌"),
        f"- 层3 语料级 macro-F1 = {conditions['layer3_corpus_band']['corpus_macro_f1']:.4f}，"
        f"等价带 {conditions['layer3_corpus_band']['band']} → {conditions['layer3_corpus_band']['status']}",
        f"- 层3 会话级 P10 = {conditions['layer3_session_p10']['p10']:.4f}"
        f"（下限 {conditions['layer3_session_p10']['required_min']}，linear 分位，"
        f"{conditions['layer3_session_p10']['n_sessions']} 会话）→ "
        + ("✅" if conditions["layer3_session_p10"]["passed"] else "❌"),
        "",
    ]
    if gate["verdict"] == "red_flag_investigate":
        lines += [
            "> ⚠️ 语料级 macro-F1 **高出等价带**：按 PREREG #9 该情形为红旗——暂停 Gate、",
            "> 登记调查（探索侧渗漏 / 分布异常），不自动通过。",
            "",
        ]
    (REPORTS_DIR / "g0_确认报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"G0 确认裁决：{gate['verdict']}（详见 reports/g0_confirmation.json）")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-check", action="store_true", help="层 1：金标 VAD → 官方算法全等检验")
    ap.add_argument("--confirmation", action="store_true", help="一次性确认集运行 + 冻结 Gate 裁决（PREREG #9）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="按 meta.json 划分溯源过滤（如 val/test）")
    args = ap.parse_args()
    root = Path(args.root) if args.root else data_root() / "dualturn_prep"
    if args.protocol_check:
        protocol_check(root, args.limit, args.split)
        return
    if args.confirmation:
        if args.limit or args.split:
            ap.error("--confirmation 为一次性全量运行，不接受 --limit/--split")
        run_confirmation(root)
        return
    report = calibrate(root, args.limit, args.split)
    write_report_json("g0_summary.json", report)
    write_markdown(report)
    vad_str = "/".join(f"{v['f1']:.3f}" for v in report["layer2_vad"].values())
    print(
        f"层1 {report['layer1_protocol']['macro_f1']:.4f} | 层2 VAD {vad_str} | "
        f"层3 {report['macro_f1']:.4f}（诊断路径；Gate 裁决走 --confirmation）"
    )


if __name__ == "__main__":
    main()
