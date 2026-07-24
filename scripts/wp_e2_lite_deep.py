"""WP-E2L-deep：E2-lite 二次分析（PREREG #40(a)；零 GPU，仓库 uv 环境）。

对已完成的 260 段生成运行做三件事：
  1. 语境分解 + SDT：agent 合格 onset 按「合宜接话 / 用户话中闯入 / 静默自起」
     三分；信号检测分解 d′（时机分辨力）与 criterion c（发言倾向）——判别
     "α 旋钮平移的是判据还是分辨力"。
  2. 文本质量粗查：text_tokens 的 PAD 占比 / 熵 / 非 PAD 连续段长逐 α 曲线，
     检查高 α 的发声增量是否伴随文本流退化。
  3. 分布对齐：交接间隙与重叠时长直方图 vs 同 20 会话人类双方（原始 ch0/ch1）
     的参考分布，JSD 逐条件。

逐运行结果缓存于 run_dir/deep.json（内容寻址指纹），VAD 掩码缓存共享
<out_root>/deep_masks/。产出 reports/wp_e2_lite_deep.json + e2_lite_深挖报告.md。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1x import sdt as sx
from floor_circuit.e1x.behavior import bootstrap_mean_ci, paired_deltas
from floor_circuit.e1x.mask_cache import cached_mask, shifted_agent_mask
from floor_circuit.events.vad import SileroVad

DEEP_SCHEMA = "e2_lite_deep_v1"
DT = 0.01

DELTA_METRICS = (
    "d_prime",
    "criterion_c",
    "hit_rate",
    "fa_rate",
    "respond_share",
    "during_user_share",
    "during_user_rate_per_min",
    "gap_rate_per_min",
    "pad_frac",
    "entropy_bits",
    "mean_nonpad_run",
    "jsd_gap_vs_human",
    "jsd_overlap_vs_human",
)


def _resolve_runs_root(plan: dict, override: str | None) -> Path:
    """默认自动优先已验收的 optimized 运行根（完成的 baseline 数更多者胜）。"""
    if override:
        return Path(override)
    base = Path(plan["out_root"]) / "runs"
    optimized = Path(str(plan["out_root"]) + "_optimized") / "runs"

    def n_completed_baselines(root: Path) -> int:
        count = 0
        for session in plan["sessions"]:
            manifest = root / f"{session['session_id']}__baseline" / "manifest.json"
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("completed"):
                    count += 1
            except (OSError, json.JSONDecodeError):
                continue
        return count

    if optimized.is_dir() and n_completed_baselines(optimized) >= n_completed_baselines(base):
        return optimized
    return base


def _fingerprint(plan: dict, events_cfg: dict, deep_cfg: dict) -> str:
    import hashlib

    payload = json.dumps(
        {
            "schema": DEEP_SCHEMA,
            "dt": DT,
            "window_s": float(plan["window_s"]),
            "vad": events_cfg["vad"],
            "ipu": events_cfg["ipu"],
            "events": events_cfg["events"],
            "deep": deep_cfg,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _run_payload(
    run_dir: Path,
    manifest: dict,
    mask_user: np.ndarray,
    mask_agent: np.ndarray,
    events_cfg: dict,
    deep_cfg: dict,
    pad_id: int,
) -> dict:
    ev = events_cfg["events"]
    context = sx.onset_context_split(
        mask_user, mask_agent, DT, ev, respond_window_s=float(deep_cfg["sdt_window_s"])
    )
    decision = sx.sdt_decision_stats(
        mask_user,
        mask_agent,
        DT,
        ev,
        window_s=float(deep_cfg["sdt_window_s"]),
        noise_stride_s=float(deep_cfg["sdt_noise_stride_s"]),
        noise_guard_s=float(deep_cfg["sdt_noise_guard_s"]),
    )
    tokens = np.load(run_dir / "text_tokens.npy", allow_pickle=False)
    token_quality = sx.token_stats(tokens, pad_id)
    gaps = sx.exchange_gaps(mask_user, mask_agent, DT, ev, max_gap_s=float(deep_cfg["gap_max_s"]))
    overlaps = sx.overlap_durations(mask_user, mask_agent, DT)
    return {
        "schema": DEEP_SCHEMA,
        "run_id": manifest["run_id"],
        "session_id": manifest["session_id"],
        "condition": manifest["condition"]["name"],
        "direction": manifest["condition"]["direction"],
        "alpha": manifest["condition"]["alpha"],
        "context": context,
        "sdt": decision,
        "tokens": token_quality,
        "gaps_s": gaps,
        "overlaps_s": overlaps,
    }


def _human_reference(
    plan: dict,
    vad: SileroVad,
    events_cfg: dict,
    deep_cfg: dict,
    mask_root: Path,
) -> dict:
    """同 20 会话真实双方（ch0 用户 + ch1 人类对话者）前 240 s 的参考统计。"""
    ev = events_cfg["events"]
    gaps_all: list[float] = []
    overlaps_all: list[float] = []
    per_session = {}
    for session in plan["sessions"]:
        user_wav = Path(session["user_wav"])
        partner_channel = 1 - int(session["user_channel"])
        partner_wav = user_wav.with_name(f"audio_ch{partner_channel}.wav")
        if not partner_wav.is_file():
            raise SystemExit(f"缺人类参考通道音频：{partner_wav}")
        mask_user = cached_mask(
            mask_root, vad, user_wav, events_cfg,
            total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
        )
        mask_partner = cached_mask(
            mask_root, vad, partner_wav, events_cfg,
            total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
        )
        gaps = sx.exchange_gaps(mask_user, mask_partner, DT, ev, max_gap_s=float(deep_cfg["gap_max_s"]))
        overlaps = sx.overlap_durations(mask_user, mask_partner, DT)
        gaps_all.extend(gaps)
        overlaps_all.extend(overlaps)
        context = sx.onset_context_split(
            mask_user, mask_partner, DT, ev, respond_window_s=float(deep_cfg["sdt_window_s"])
        )
        decision = sx.sdt_decision_stats(
            mask_user, mask_partner, DT, ev,
            window_s=float(deep_cfg["sdt_window_s"]),
            noise_stride_s=float(deep_cfg["sdt_noise_stride_s"]),
            noise_guard_s=float(deep_cfg["sdt_noise_guard_s"]),
        )
        per_session[session["session_id"]] = {
            "partner_speech_frac": float(mask_partner.mean()),
            "context": context,
            "sdt": decision,
        }
    gap_hist = sx.histogram_counts(gaps_all, [float(v) for v in deep_cfg["gap_hist_bins_s"]])
    overlap_hist = sx.histogram_counts(overlaps_all, [float(v) for v in deep_cfg["overlap_hist_bins_s"]])
    d_primes = [v["sdt"]["d_prime"] for v in per_session.values() if v["sdt"]["d_prime"] is not None]
    criteria = [v["sdt"]["criterion_c"] for v in per_session.values() if v["sdt"]["criterion_c"] is not None]
    return {
        "n_sessions": len(per_session),
        "gap_hist": gap_hist.tolist(),
        "overlap_hist": overlap_hist.tolist(),
        "n_gaps": len(gaps_all),
        "n_overlaps": len(overlaps_all),
        "median_gap_s": float(np.median(gaps_all)) if gaps_all else None,
        "human_d_prime_mean": float(np.mean(d_primes)) if d_primes else None,
        "human_criterion_mean": float(np.mean(criteria)) if criteria else None,
        "per_session": per_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-lite 二次分析（PREREG #40(a)）")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--runs-root", default=None, help="覆盖运行根（默认自动优先 *_optimized）")
    parser.add_argument("--pad-token-id", type=int, default=None, help="覆盖 PAD token 假定值")
    parser.add_argument("--force", action="store_true", help="忽略逐运行 deep 缓存重算")
    args = parser.parse_args()

    grids = load_config("grids")
    lite_cfg = grids["e1"]["e2_lite"]
    deep_cfg = dict(grids["e1"]["e1x"]["deep"])
    events_cfg = load_config("events")
    plan_path = Path(args.plan) if args.plan else data_root() / str(lite_cfg["out_group"]) / "e2_lite.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs_root = _resolve_runs_root(plan, args.runs_root)
    pad_id = int(args.pad_token_id if args.pad_token_id is not None else deep_cfg["pad_token_id"])
    deep_cfg["pad_token_id"] = pad_id
    mask_root = Path(plan["out_root"]) / "deep_masks"
    fingerprint = _fingerprint(plan, events_cfg, deep_cfg)
    vad = SileroVad(events_cfg)
    print(f"运行根：{runs_root}")

    human = _human_reference(plan, vad, events_cfg, deep_cfg, mask_root)
    gap_bins = [float(v) for v in deep_cfg["gap_hist_bins_s"]]
    overlap_bins = [float(v) for v in deep_cfg["overlap_hist_bins_s"]]
    human_gap_hist = np.asarray(human["gap_hist"])
    human_overlap_hist = np.asarray(human["overlap_hist"])

    records: list[dict] = []
    missing: list[str] = []
    for session in plan["sessions"]:
        user_mask = cached_mask(
            mask_root, vad, Path(session["user_wav"]), events_cfg,
            total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
        )
        for condition in plan["conditions"]:
            run_id = f"{session['session_id']}__{condition['name']}"
            run_dir = runs_root / run_id
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                missing.append(run_id)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("completed"):
                missing.append(run_id)
                continue
            cache_path = run_dir / "deep.json"
            if cache_path.is_file() and not args.force:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    records.append(cached["payload"])
                    continue
            agent_local = cached_mask(
                mask_root, vad, run_dir / "agent.wav", events_cfg,
                total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
            )
            first_emitted = int(manifest.get("first_emitted_frame") or 0)
            agent_mask = shifted_agent_mask(agent_local, first_emitted, 1.0 / float(plan["frame_hz"]), DT)
            payload = _run_payload(run_dir, manifest, user_mask, agent_mask, events_cfg, deep_cfg, pad_id)
            payload["jsd_gap_vs_human"] = sx.jensen_shannon_divergence(
                sx.histogram_counts(payload["gaps_s"], gap_bins), human_gap_hist
            )
            payload["jsd_overlap_vs_human"] = sx.jensen_shannon_divergence(
                sx.histogram_counts(payload["overlaps_s"], overlap_bins), human_overlap_hist
            )
            cache_path.write_text(
                json.dumps({"fingerprint": fingerprint, "payload": payload}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            records.append(payload)
    if not records:
        raise SystemExit(f"没有可分析的运行（缺 {len(missing)} 个）")

    # 平铺出配对指标表：sid -> condition -> {metric: value}
    per_session_metrics: dict[str, dict[str, dict]] = {}
    condition_pooled: dict[str, dict[str, list[float]]] = {}
    for record in records:
        flat = {
            "d_prime": record["sdt"]["d_prime"],
            "criterion_c": record["sdt"]["criterion_c"],
            "hit_rate": record["sdt"]["hit_rate"],
            "fa_rate": record["sdt"]["fa_rate"],
            "respond_share": record["context"]["respond_share"],
            "during_user_share": record["context"]["during_user_share"],
            "during_user_rate_per_min": record["context"]["during_user_rate_per_min"],
            "gap_rate_per_min": record["context"]["gap_rate_per_min"],
            "pad_frac": record["tokens"]["pad_frac"],
            "entropy_bits": record["tokens"]["entropy_bits"],
            "mean_nonpad_run": record["tokens"]["mean_nonpad_run"],
            "jsd_gap_vs_human": record["jsd_gap_vs_human"],
            "jsd_overlap_vs_human": record["jsd_overlap_vs_human"],
        }
        per_session_metrics.setdefault(record["session_id"], {})[record["condition"]] = flat
        pool = condition_pooled.setdefault(record["condition"], {"gaps": [], "overlaps": []})
        pool["gaps"].extend(record["gaps_s"])
        pool["overlaps"].extend(record["overlaps_s"])

    conditions = [c["name"] for c in plan["conditions"]]
    baseline = "baseline"
    aggregate: dict[str, dict] = {}
    for condition in conditions:
        if condition == baseline:
            continue
        aggregate[condition] = {
            metric: bootstrap_mean_ci(paired_deltas(per_session_metrics, condition, baseline, metric))
            for metric in DELTA_METRICS
        }
    baseline_absolute = {}
    for metric in DELTA_METRICS:
        values = [
            by_condition[baseline][metric]
            for by_condition in per_session_metrics.values()
            if baseline in by_condition and by_condition[baseline][metric] is not None
        ]
        baseline_absolute[metric] = float(np.mean(values)) if values else None

    # 主方向 α 单调性
    primary = [c for c in plan["conditions"] if c["direction"] == "probe_meanseed" and c["alpha"] != 0.0]
    dose: dict[str, dict] = {}
    for metric in DELTA_METRICS:
        pairs: list[tuple[float, float]] = []
        for condition in primary:
            for delta in paired_deltas(per_session_metrics, condition["name"], baseline, metric):
                pairs.append((float(condition["alpha"]), delta))
        if len(pairs) >= 10:
            from scipy.stats import spearmanr

            rho, pval = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            dose[metric] = {"spearman_rho": float(rho), "p": float(pval), "n": len(pairs)}
        else:
            dose[metric] = {"n": len(pairs)}

    condition_jsd = {
        name: {
            "jsd_gap_pooled_vs_human": sx.jensen_shannon_divergence(
                sx.histogram_counts(pool["gaps"], gap_bins), human_gap_hist
            ),
            "jsd_overlap_pooled_vs_human": sx.jensen_shannon_divergence(
                sx.histogram_counts(pool["overlaps"], overlap_bins), human_overlap_hist
            ),
            "n_gaps": len(pool["gaps"]),
        }
        for name, pool in condition_pooled.items()
    }

    baseline_top_tokens = [
        r["tokens"]["top_tokens"] for r in records if r["condition"] == baseline
    ]
    payload = {
        "schema": DEEP_SCHEMA,
        "runs_root": str(runs_root),
        "pad_token_id_assumed": pad_id,
        "n_runs_analyzed": len(records),
        "n_runs_missing": len(missing),
        "missing_runs": missing[:20],
        "human_reference": {k: v for k, v in human.items() if k != "per_session"},
        "baseline_absolute": baseline_absolute,
        "baseline_top_tokens_first3": baseline_top_tokens[:3],
        "paired_deltas_vs_baseline": aggregate,
        "dose_response_primary": dose,
        "condition_pooled_jsd": condition_jsd,
    }
    write_report_json("wp_e2_lite_deep.json", payload)

    lines = [
        "# E2-lite 深挖报告（PREREG #40(a)；探索性）",
        "",
        f"- 运行根：`{runs_root}`；已析 {len(records)}，缺失 {len(missing)}",
        f"- PAD token 假定 = {pad_id}（以 baseline top-token 核验，见 JSON）",
        f"- 人类参考（同 20 会话真实双方）：d′≈{human['human_d_prime_mean']}, "
        f"c≈{human['human_criterion_mean']}，中位交接间隙 {human['median_gap_s']} s",
        "",
        "## SDT 分解与语境分解（配对 Δ vs baseline）",
        "",
        "| 条件 | Δd′ | Δc | Δrespond_share | Δduring_user/min | Δpad_frac |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def cell(stats: dict) -> str:
        if stats["mean"] is None:
            return "—"
        return f"{stats['mean']:+.3f} [{stats['ci95'][0]:+.3f},{stats['ci95'][1]:+.3f}]"

    for condition in conditions:
        if condition == baseline:
            continue
        row = aggregate[condition]
        lines.append(
            f"| {condition} | {cell(row['d_prime'])} | {cell(row['criterion_c'])} | "
            f"{cell(row['respond_share'])} | {cell(row['during_user_rate_per_min'])} | "
            f"{cell(row['pad_frac'])} |"
        )
    lines += [
        "",
        "## 主方向 α 单调性（Spearman）",
        "",
    ]
    for metric, stats in dose.items():
        if "spearman_rho" in stats:
            lines.append(f"- {metric}: ρ={stats['spearman_rho']:+.3f}（p={stats['p']:.2e}，n={stats['n']}）")
    lines += ["", "## 逐条件与人类分布的 JSD（池化直方图）", ""]
    for name in conditions:
        entry = condition_jsd.get(name)
        if entry:
            lines.append(
                f"- {name}: gap JSD={entry['jsd_gap_pooled_vs_human']}, "
                f"overlap JSD={entry['jsd_overlap_pooled_vs_human']}（{entry['n_gaps']} 次交接）"
            )
    report_path = Path(REPO_ROOT) / "reports" / "e2_lite_深挖报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"深挖分析完成：{report_path}")


if __name__ == "__main__":
    main()
