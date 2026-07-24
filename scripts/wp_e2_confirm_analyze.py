"""WP-E2C：E2 确认臂行为分析（PREREG #40(b)；仓库 uv 环境）。

对确认臂运行计算 floor 行为指标 + 语境/SDT 分解，与复用的 E2-lite baseline
做同会话配对差；并把 E2-lite 的连续注入 probe_a±4 作为参照列一并计算，
按四个问题分组出报告：必要性（clamp）、时间特异性（respond/user_speech 门 vs
连续）、轴特异性（T1_d800/T5_SPEAK vs probe）、层特异性（L20/L31 vs L29）。

产出 reports/wp_e2_confirm_summary.json + reports/e2_confirm_行为报告.md。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1x import sdt as sx
from floor_circuit.e1x.behavior import METRIC_KEYS, bootstrap_mean_ci, floor_metrics, paired_deltas
from floor_circuit.e1x.mask_cache import cached_mask, shifted_agent_mask
from floor_circuit.events.vad import SileroVad

SCHEMA = "e2_confirm_behavior_v1"
DT = 0.01
EXTRA_METRICS = ("d_prime", "criterion_c", "respond_share", "during_user_rate_per_min")


def _fingerprint(plan: dict, events_cfg: dict, analysis_cfg: dict, deep_cfg: dict) -> str:
    payload = json.dumps(
        {
            "schema": SCHEMA,
            "dt": DT,
            "window_s": float(plan["window_s"]),
            "vad": events_cfg["vad"],
            "ipu": events_cfg["ipu"],
            "events": events_cfg["events"],
            "analysis": analysis_cfg,
            "sdt": {k: deep_cfg[k] for k in ("sdt_window_s", "sdt_noise_stride_s", "sdt_noise_guard_s")},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _run_metrics(
    run_dir: Path,
    mask_user: np.ndarray,
    plan: dict,
    events_cfg: dict,
    analysis_cfg: dict,
    deep_cfg: dict,
    vad: SileroVad,
    mask_root: Path,
    fingerprint: str,
    force: bool,
) -> dict | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        return None
    cache_path = run_dir / "confirm_behavior.json"
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return cached["metrics"]
    agent_local = cached_mask(
        mask_root, vad, run_dir / "agent.wav", events_cfg,
        total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
    )
    first_emitted = int(manifest.get("first_emitted_frame") or 0)
    mask_agent = shifted_agent_mask(agent_local, first_emitted, 1.0 / float(plan["frame_hz"]), DT)
    ev = events_cfg["events"]
    metrics = floor_metrics(
        mask_user,
        mask_agent,
        DT,
        ev,
        response_window_s=float(analysis_cfg["response_window_s"]),
        latency_max_s=float(analysis_cfg["latency_max_s"]),
        yield_windows_s=[float(v) for v in analysis_cfg["yield_windows_s"]],
    )
    context = sx.onset_context_split(
        mask_user, mask_agent, DT, ev, respond_window_s=float(deep_cfg["sdt_window_s"])
    )
    decision = sx.sdt_decision_stats(
        mask_user, mask_agent, DT, ev,
        window_s=float(deep_cfg["sdt_window_s"]),
        noise_stride_s=float(deep_cfg["sdt_noise_stride_s"]),
        noise_guard_s=float(deep_cfg["sdt_noise_guard_s"]),
    )
    metrics.update(
        {
            "d_prime": decision["d_prime"],
            "criterion_c": decision["criterion_c"],
            "respond_share": context["respond_share"],
            "during_user_rate_per_min": context["during_user_rate_per_min"],
        }
    )
    cache_path.write_text(
        json.dumps({"fingerprint": fingerprint, "metrics": metrics}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 确认臂行为分析（PREREG #40(b)）")
    parser.add_argument("--plan", default=None, help="默认 <data_root>/e2_confirm/e2_confirm.plan.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    grids = load_config("grids")
    confirm_cfg = grids["e1"]["e2_confirm"]
    lite_cfg = grids["e1"]["e2_lite"]
    deep_cfg = grids["e1"]["e1x"]["deep"]
    analysis_cfg = lite_cfg["analysis"]
    events_cfg = load_config("events")
    base = data_root()
    plan_path = Path(args.plan) if args.plan else base / str(confirm_cfg["out_group"]) / "e2_confirm.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    lite_plan = json.loads(Path(plan["baseline"]["lite_plan"]).read_text(encoding="utf-8"))
    baseline_root = Path(plan["baseline"]["runs_root"])
    confirm_root = Path(plan["out_root"]) / "runs"
    mask_root = Path(plan["out_root"]) / "deep_masks"
    fingerprint = _fingerprint(plan, events_cfg, analysis_cfg, deep_cfg)
    vad = SileroVad(events_cfg)

    # E2-lite 连续注入参照条件（同剂量 ±4；存在才纳入）
    lite_reference_names = [
        c["name"]
        for c in lite_plan["conditions"]
        if c["direction"] == "probe_meanseed" and abs(float(c["alpha"])) == 4.0 and float(c["alpha"]) != 0.0
    ]

    per_session: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for session in plan["sessions"]:
        sid = session["session_id"]
        mask_user = cached_mask(
            mask_root, vad, Path(session["user_wav"]), events_cfg,
            total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
        )
        jobs = [("baseline", baseline_root / f"{sid}__baseline")]
        jobs += [(f"lite::{name}", baseline_root / f"{sid}__{name}") for name in lite_reference_names]
        jobs += [(c["name"], confirm_root / f"{sid}__{c['name']}") for c in plan["conditions"]]
        for label, run_dir in jobs:
            metrics = _run_metrics(
                run_dir, mask_user, plan, events_cfg, analysis_cfg, deep_cfg,
                vad, mask_root, fingerprint, args.force,
            )
            if metrics is None:
                missing.append(f"{sid}__{label}")
            else:
                per_session.setdefault(sid, {})[label] = metrics
    analyzed = sum(len(v) for v in per_session.values())
    if analyzed == 0:
        raise SystemExit(f"没有可分析的运行（缺 {len(missing)} 个）")

    all_labels = [f"lite::{name}" for name in lite_reference_names] + [c["name"] for c in plan["conditions"]]
    metric_names = tuple(METRIC_KEYS) + EXTRA_METRICS
    aggregate = {
        label: {
            metric: bootstrap_mean_ci(paired_deltas(per_session, label, "baseline", metric))
            for metric in metric_names
        }
        for label in all_labels
    }

    groups = {
        "necessity": [c["name"] for c in plan["conditions"] if c["mode"] == "clamp"],
        "timing": [c["name"] for c in plan["conditions"] if c["gate"] != "none"]
        + [f"lite::{name}" for name in lite_reference_names],
        "axis": [c["name"] for c in plan["conditions"] if c["direction"] != "probe_meanseed"]
        + [f"lite::{name}" for name in lite_reference_names],
        "layer": [c["name"] for c in plan["conditions"] if int(c["layer"]) != int(plan["layer"])]
        + [f"lite::{name}" for name in lite_reference_names],
    }

    payload = {
        "schema": SCHEMA,
        "plan_path": str(plan_path),
        "baseline_runs_root": str(baseline_root),
        "n_sessions": len(per_session),
        "n_missing": len(missing),
        "missing_runs": missing[:30],
        "paired_deltas_vs_baseline": aggregate,
        "groups": groups,
    }
    write_report_json("wp_e2_confirm_summary.json", payload)

    def cell(stats: dict) -> str:
        if stats["mean"] is None:
            return "—"
        return f"{stats['mean']:+.3f} [{stats['ci95'][0]:+.3f},{stats['ci95'][1]:+.3f}]"

    key_metrics = (
        "agent_speech_frac",
        "d_prime",
        "criterion_c",
        "respond_share",
        "during_user_rate_per_min",
        "yield_rate_400ms",
    )
    lines = [
        "# E2 确认臂行为报告（PREREG #40(b)；探索性）",
        "",
        f"- 基线复用：`{baseline_root}`（E2-lite baseline，同会话同种子）",
        f"- 会话 {len(per_session)}；缺失运行 {len(missing)}",
        "",
    ]
    titles = {
        "necessity": "## 必要性：投影钳制消融（clamp）",
        "timing": "## 时间特异性：respond / user_speech 门 vs 连续注入",
        "axis": "## 轴特异性：T1_d800 / T5:SPEAK vs probe 方向",
        "layer": "## 层特异性：L20 / L31 vs L29",
    }
    for group, labels in groups.items():
        lines += [titles[group], "", "| 条件 | " + " | ".join(key_metrics) + " |",
                  "| --- |" + " --- |" * len(key_metrics)]
        for label in labels:
            row = aggregate.get(label)
            if row is None:
                continue
            lines.append(
                f"| {label} | " + " | ".join(cell(row[m]) for m in key_metrics) + " |"
            )
        lines.append("")
    report_path = Path(REPO_ROOT) / "reports" / "e2_confirm_行为报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"确认臂分析完成：{report_path}（缺失 {len(missing)}）")


if __name__ == "__main__":
    main()
