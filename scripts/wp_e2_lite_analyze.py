"""WP-E2L：E2-lite 行为分析（PREREG #34；仓库 uv 环境，依赖 silero VAD）。

对每个已完成运行：Silero VAD（冻结参数）→ IPU 合并 → dt 栅格掩码 →
floor 行为指标（e1x/behavior.py）。逐运行 VAD 结果缓存为 JSON，重复执行只聚合。

聚合：同会话跨条件配对差（条件 − baseline）→ 会话级 bootstrap CI + 符号计数；
主方向剂量-反应（α 单调性）；随机方向对照的效应量参照（自由度有限，仅作
数量级参照）。产出 reports/wp_e2_lite_summary.json + reports/e2_lite_行为报告.md。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1x.behavior import METRIC_KEYS, bootstrap_mean_ci, floor_metrics, paired_deltas
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.vad import SileroVad, rasterize

VAD_CACHE_SCHEMA = "e2_lite_vad_v1"


def _mask_from_wav(
    vad: SileroVad, wav_path: Path, sample_rate_expected: int, total_dur: float, dt: float, ipu_gap: float
) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate_expected:
        raise ValueError(f"{wav_path} 采样率 {sr} ≠ {sample_rate_expected}")
    segs = vad.segments(wav, sr)
    ipus = build_ipus(segs, ipu_gap)
    return rasterize(ipus, dt, total_dur)


def _analyze_run(
    run_dir: Path, plan: dict, events_cfg: dict, analysis_cfg: dict, vad: SileroVad, force: bool
) -> dict | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        return None
    cache_path = run_dir / "behavior.json"
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("schema") == VAD_CACHE_SCHEMA:
            return cached
    dt = 0.01
    total_dur = float(plan["window_s"])
    ipu_gap = float(events_cfg["ipu"]["merge_gap_s"])
    mask_agent = _mask_from_wav(
        vad, run_dir / "agent.wav", int(plan["sample_rate"]), total_dur, dt, ipu_gap
    )
    mask_user = _mask_from_wav(
        vad, Path(manifest["user_wav"]), int(plan["sample_rate"]), total_dur, dt, ipu_gap
    )
    n = min(len(mask_agent), len(mask_user))
    metrics = floor_metrics(
        mask_user[:n],
        mask_agent[:n],
        dt,
        events_cfg["events"],
        response_window_s=float(analysis_cfg["response_window_s"]),
        latency_max_s=float(analysis_cfg["latency_max_s"]),
        yield_windows_s=[float(v) for v in analysis_cfg["yield_windows_s"]],
    )
    payload = {
        "schema": VAD_CACHE_SCHEMA,
        "run_id": manifest["run_id"],
        "session_id": manifest["session_id"],
        "condition": manifest["condition"]["name"],
        "direction": manifest["condition"]["direction"],
        "alpha": manifest["condition"]["alpha"],
        "metrics": metrics,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-lite 行为分析（PREREG #34）")
    parser.add_argument("--plan", default=None, help="默认 <data_root>/e2_lite/e2_lite.plan.json")
    parser.add_argument("--force", action="store_true", help="忽略逐运行 VAD 缓存重算")
    args = parser.parse_args()

    grids = load_config("grids")
    cfg = grids["e1"]["e2_lite"]
    events_cfg = load_config("events")
    plan_path = Path(args.plan) if args.plan else data_root() / str(cfg["out_group"]) / "e2_lite.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs_root = Path(plan["out_root"]) / "runs"
    vad = SileroVad(events_cfg)

    records: list[dict] = []
    missing: list[str] = []
    for session in plan["sessions"]:
        for condition in plan["conditions"]:
            run_id = f"{session['session_id']}__{condition['name']}"
            record = _analyze_run(
                runs_root / run_id, plan, events_cfg, cfg["analysis"], vad, args.force
            )
            if record is None:
                missing.append(run_id)
            else:
                records.append(record)
    if not records:
        raise SystemExit(f"没有已完成的运行（缺 {len(missing)} 个）；先执行 run_steer.py")

    per_session: dict[str, dict[str, dict]] = {}
    for record in records:
        per_session.setdefault(record["session_id"], {})[record["condition"]] = record["metrics"]

    conditions = [c["name"] for c in plan["conditions"]]
    baseline = "baseline"
    if baseline not in conditions:
        raise SystemExit("计划缺 baseline（α=0）条件")
    aggregate: dict[str, dict] = {}
    for condition in conditions:
        if condition == baseline:
            continue
        aggregate[condition] = {}
        for metric in METRIC_KEYS:
            deltas = paired_deltas(per_session, condition, baseline, metric)
            aggregate[condition][metric] = bootstrap_mean_ci(deltas)

    # 主方向剂量-反应：α 与配对差的 Spearman（跨会话拼接）
    dose: dict[str, dict] = {}
    primary_conditions = [
        c for c in plan["conditions"] if c["direction"] == "probe_meanseed" and c["alpha"] != 0.0
    ]
    for metric in METRIC_KEYS:
        pairs: list[tuple[float, float]] = []
        for condition in primary_conditions:
            for delta in paired_deltas(per_session, condition["name"], baseline, metric):
                pairs.append((float(condition["alpha"]), delta))
        if len(pairs) >= 10:
            from scipy.stats import spearmanr

            rho, pval = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            dose[metric] = {"spearman_rho": float(rho), "p": float(pval), "n": len(pairs)}
        else:
            dose[metric] = {"n": len(pairs)}

    # 随机方向对照：|Δ| 参照（同 |α|）
    control: dict[str, dict] = {}
    max_alpha = max(abs(float(c["alpha"])) for c in primary_conditions) if primary_conditions else None
    if max_alpha is not None:
        for metric in METRIC_KEYS:
            primary_abs = [
                abs(aggregate[c["name"]][metric]["mean"])
                for c in primary_conditions
                if abs(float(c["alpha"])) == max_alpha and aggregate[c["name"]][metric]["mean"] is not None
            ]
            random_abs = [
                abs(aggregate[c["name"]][metric]["mean"])
                for c in plan["conditions"]
                if c["direction"].startswith("random_r")
                and abs(float(c["alpha"])) == max_alpha
                and c["name"] in aggregate
                and aggregate[c["name"]][metric]["mean"] is not None
            ]
            control[metric] = {
                "primary_abs_mean_delta": primary_abs,
                "random_abs_mean_deltas": random_abs,
                "note": "随机对照自由度有限，仅作数量级参照",
            }

    payload = {
        "schema": "e2_lite_summary_v1",
        "n_runs_analyzed": len(records),
        "n_runs_missing": len(missing),
        "missing_runs": missing[:20],
        "paired_deltas_vs_baseline": aggregate,
        "dose_response_primary": dose,
        "random_control_reference": control,
    }
    write_report_json("wp_e2_lite_summary.json", payload)

    lines = [
        "# E2-lite 行为报告（PREREG #34；探索性）",
        "",
        f"- 已分析运行：{len(records)}；缺失：{len(missing)}",
        f"- 注入：L{plan['layer']}，h ← h + α·s_v·v̂（s_v = 训练行投影标准差）",
        "",
        "## 主方向剂量-反应（配对差 vs baseline）",
        "",
    ]
    for condition in [c["name"] for c in primary_conditions]:
        lines.append(f"### {condition}")
        for metric in METRIC_KEYS:
            stats = aggregate[condition][metric]
            if stats["mean"] is None:
                lines.append(f"- {metric}：无有效配对")
            else:
                lines.append(
                    f"- {metric}：Δ={stats['mean']:+.4f} "
                    f"[{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}]"
                    f"（n={stats['n']}，+{stats['n_pos']}/−{stats['n_neg']}）"
                )
        lines.append("")
    lines += ["## α 单调性（Spearman）", ""]
    for metric, stats in dose.items():
        if "spearman_rho" in stats:
            lines.append(f"- {metric}：ρ={stats['spearman_rho']:+.3f}（p={stats['p']:.2e}，n={stats['n']}）")
        else:
            lines.append(f"- {metric}：样本不足（n={stats['n']}）")
    lines += ["", "## 差分均值方向与随机对照", ""]
    for condition in conditions:
        if condition == baseline or condition in {c["name"] for c in primary_conditions}:
            continue
        stats = aggregate.get(condition, {}).get("agent_speech_frac")
        if stats and stats["mean"] is not None:
            lines.append(
                f"- {condition}：agent_speech_frac Δ={stats['mean']:+.4f} "
                f"[{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}]（n={stats['n']}）"
            )
    report_path = Path(REPO_ROOT) / "reports" / "e2_lite_行为报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"分析完成：{report_path}（缺失 {len(missing)} 个运行）")


if __name__ == "__main__":
    main()
