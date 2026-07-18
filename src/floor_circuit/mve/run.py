"""MVE 编排与 G1 裁决：探针网格（层 × 目标 × 种子）+ 基线族 + 成对优势 bootstrap + 报告。

数据获取被依赖注入（providers 返回 SessionData / PerSession），
真实装配在 scripts/wp7_run_mve.py；本模块只含可单测的编排与报告逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from floor_circuit.probes.linear import SessionData, fit_probe, score_sessions
from floor_circuit.probes.stats import (
    PerSession,
    ScoreCollection,
    SeededPerSession,
    cluster_bootstrap_seed_mean_auc,
    g1_verdict,
    paired_seed_mean_advantage_bootstrap,
    pooled_metrics,
    seed_mean_metrics,
    shuffle_labels_within_session,
)


@dataclass
class ProbeCell:
    layer: int
    target: str
    seed: int
    metrics: dict
    per_session: PerSession = field(repr=False, default_factory=dict)


def probe_grid(
    data_by_layer: dict[int, SessionData],
    train_sids: list[str],
    eval_sids: list[str],
    seeds: list[int],
    c_grid: list[float],
    neg_ratio: int,
    target: str,
) -> list[ProbeCell]:
    cells = []
    for layer, data in sorted(data_by_layer.items()):
        for seed in seeds:
            fit = fit_probe(data, train_sids, eval_sids, c_grid, seed, neg_ratio)
            per_session = score_sessions(fit, data, eval_sids)
            metrics = pooled_metrics(per_session)
            metrics["best_c"] = fit.best_c
            cells.append(ProbeCell(layer, target, seed, metrics, per_session))
    return cells


def average_over_seeds(cells: list[ProbeCell]) -> dict[int, dict]:
    """按层聚合种子指标，并保留每个种子的会话分数供统一统计。"""

    out: dict[int, dict] = {}
    layers = sorted({c.layer for c in cells})
    for layer in layers:
        grp = sorted((c for c in cells if c.layer == layer), key=lambda cell: cell.seed)
        if len({cell.seed for cell in grp}) != len(grp):
            raise ValueError(f"L{layer} 含重复探针种子")
        aucs = np.array([c.metrics["auc"] for c in grp])
        auprcs = np.array([c.metrics["auprc"] for c in grp])
        balanced_accs = np.array([c.metrics["balanced_acc"] for c in grp])
        out[layer] = {
            "n_seeds": len(grp),
            "auc_mean": float(aucs.mean()),
            "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "auprc_mean": float(auprcs.mean()),
            "auprc_sd": float(auprcs.std(ddof=1)) if len(auprcs) > 1 else 0.0,
            "balanced_acc_mean": float(balanced_accs.mean()),
            "balanced_acc_sd": (
                float(balanced_accs.std(ddof=1)) if len(balanced_accs) > 1 else 0.0
            ),
            "auc_by_seed": {cell.seed: float(cell.metrics["auc"]) for cell in grp},
            "best_c_by_seed": {
                cell.seed: float(cell.metrics["best_c"])
                for cell in grp
            },
            "per_seed": {cell.seed: cell.per_session for cell in grp},
        }
        if all("selection_auc" in cell.metrics for cell in grp):
            selection_aucs = np.array([c.metrics["selection_auc"] for c in grp])
            out[layer]["selection_auc_mean"] = float(selection_aucs.mean())
            out[layer]["selection_auc_by_seed"] = {
                cell.seed: float(cell.metrics["selection_auc"]) for cell in grp
            }
    return out


def evaluate_target(
    layer_summary: dict[int, dict],
    baselines: dict[str, ScoreCollection],
    n_boot: int,
    full_thr: float,
    backup_thr: float,
    boot_seed: int = 0,
    *,
    best_layer: int,
) -> dict:
    """对**外部选定**的最优层做成对优势 bootstrap → 该目标的 G1 材料。

    PREREG #7：最优层与各种子的 C 必须在 probe_train 内层划分上选择后传入；
    本函数不得再在报告用的评估集分数上做任何选择。
    """

    if best_layer not in layer_summary:
        raise ValueError(f"外部选定层 L{best_layer} 不在层摘要 {sorted(layer_summary)} 中")
    probes: SeededPerSession = layer_summary[best_layer]["per_seed"]
    adv = paired_seed_mean_advantage_bootstrap(
        probes,
        baselines,
        n_boot=n_boot,
        seed=boot_seed,
    )
    probe_ci = cluster_bootstrap_seed_mean_auc(
        probes,
        n_boot=n_boot,
        seed=boot_seed,
    )
    expected_probe_auc = float(layer_summary[best_layer]["auc_mean"])
    if not np.isclose(adv["probe_auc"], expected_probe_auc, rtol=0, atol=1e-12):
        raise RuntimeError(
            f"最优层点估计 {adv['probe_auc']} 与种子均值 {expected_probe_auc} 不一致"
        )
    if not np.isclose(probe_ci["point"], expected_probe_auc, rtol=0, atol=1e-12):
        raise RuntimeError(
            f"最优层 CI 点估计 {probe_ci['point']} 与种子均值 {expected_probe_auc} 不一致"
        )
    shuffled = {
        seed: shuffle_labels_within_session(scores, seed=boot_seed)
        for seed, scores in probes.items()
    }
    shuffled_metrics = seed_mean_metrics(shuffled)
    return {
        "best_layer": int(best_layer),
        "layer_summary": {
            int(k): {kk: vv for kk, vv in v.items() if kk != "per_seed"}
            for k, v in layer_summary.items()
        },
        "advantage": adv,
        "baseline_metrics": {
            name: seed_mean_metrics(scores)
            for name, scores in baselines.items()
        },
        "probe_ci": probe_ci,
        "shuffled_auc": shuffled_metrics["auc_mean"],
        "shuffled_auc_sd": shuffled_metrics["auc_sd"],
        "shuffled_n_seeds": shuffled_metrics["n_seeds"],
        "ci_scope": (
            "层与各种子 C 在 probe_train 内层划分上选择，probe_val 仅用于最终报告；"
            "会话级 bootstrap CI 对选定模型无报告集选择泄漏（目标间取优仍为冻结决策规则）"
        ),
        "selection_disjoint_from_report": True,
        "verdict": g1_verdict(adv["advantage_point"], adv["ci_lo"], full_thr, backup_thr),
    }


def overall_g1(per_target: dict[str, dict], full_thr: float, backup_thr: float) -> dict:
    """G1 总裁决：取各目标中优势较大者（文档/02 §WP7 冻结口径）。"""
    best_target = max(per_target, key=lambda t: per_target[t]["advantage"]["advantage_point"])
    m = per_target[best_target]
    return {
        "decisive_target": best_target,
        "advantage_point": m["advantage"]["advantage_point"],
        "ci_lo": m["advantage"]["ci_lo"],
        "verdict": g1_verdict(
            m["advantage"]["advantage_point"], m["advantage"]["ci_lo"], full_thr, backup_thr
        ),
    }


_VERDICT_TEXT = {
    "full_e1": "≥ +0.05 且 CI 下界 > 0 → 全量 E1（W3 起五模型展开）",
    "backup_mve": "+0.02 ~ +0.05 → 备胎 MVE（PersonaPlex 与 MiniCPM-o 各补一次，+3 天）",
    "n1": "全部 < +0.02 → 分支 N1：止损改组（文档/00 §12.3）",
}


def render_report(per_target: dict[str, dict], overall: dict, meta: dict) -> str:
    """生成 reports/mve_报告.md 正文。"""
    title = "# MVE 报告（Moshi R1，G1 裁决）"
    if meta.get("ablation"):
        title = f"# MVE 消融报告（Moshi R1，{meta['ablation']}；**非正式 G1 裁决**）"
    lines = [
        title,
        "",
        f"- 配置：层 {meta.get('layers')}，目标 {list(per_target)}，种子 {meta.get('seeds')}，"
        f"bootstrap {meta.get('bootstrap_n')} 次（会话级）；文本流 text_mode={meta.get('text_mode')}；"
        f"T1 判据点 δ={meta.get('t1_delta_ms')} ms（PREREG #8 净毫秒读法）",
        f"- 数据：训练 {meta.get('n_train_sessions')} 会话 / 评估 {meta.get('n_eval_sessions')} 会话；"
        f"内层选择划分 inner_train {meta.get('n_inner_train_sessions')} / inner_val {meta.get('n_inner_val_sessions')}",
        "- 时间对齐（PREREG #7）：标签步 s 的观测截止统一为 s·τ；acts 读行 s、"
        "Mimi/hazard/声学读行 s−1，step 0 全表征剔除。",
        "- 置信区间口径：层与各种子 C 在 probe_train 内层划分上选择，probe_val 只用于最终报告；"
        "会话级 bootstrap CI 对选定模型无报告集选择泄漏（目标间取优为冻结决策规则）。",
        "",
    ]
    score_bundle = meta.get("score_bundle")
    if score_bundle:
        lines += [
            "## 独立复算材料",
            "",
            f"- 分数包绝对路径：`{score_bundle['absolute_path']}`",
            f"- 分数包相对路径：`{score_bundle['relative_path']}`",
            f"- 最终 manifest SHA-256：`{score_bundle['manifest_sha256']}`",
            "",
        ]
    for target, m in per_target.items():
        lines += [
            f"## 目标 {target}",
            "",
            "| 层 | 种子数 | AUC（均值±SD） | AUPRC（均值±SD） | balanced acc（均值±SD） |",
            "| --- | ---: | --- | --- | --- |",
        ]
        for layer, s in sorted(m["layer_summary"].items()):
            lines.append(
                f"| L{layer} | {s['n_seeds']} | {s['auc_mean']:.4f} ± {s['auc_sd']:.4f} | "
                f"{s['auprc_mean']:.4f} ± {s['auprc_sd']:.4f} | "
                f"{s['balanced_acc_mean']:.4f} ± {s['balanced_acc_sd']:.4f} |"
            )
        adv = m["advantage"]
        selected = m["layer_summary"][m["best_layer"]]
        selection_note = (
            f"（内层选择 AUC {selected['selection_auc_mean']:.4f}）"
            if "selection_auc_mean" in selected
            else ""
        )
        lines += [
            "",
            f"- 最优层：L{m['best_layer']}（在 inner_val 上选定{selection_note}）；探针 AUC "
            f"{adv['probe_auc']:.4f} ± {selected['auc_sd']:.4f}"
            f"（{selected['n_seeds']} 个种子；95% CI "
            f"[{m['probe_ci']['ci_lo']:.4f}, {m['probe_ci']['ci_hi']:.4f}]）",
            "",
            "| 基线 | 种子数 | AUC（均值±SD） | AUPRC（均值±SD） | balanced acc（均值±SD） |",
            "| --- | ---: | --- | --- | --- |",
        ]
        for name, metrics in sorted(m["baseline_metrics"].items()):
            lines.append(
                f"| {name} | {metrics['n_seeds']} | "
                f"{metrics['auc_mean']:.4f} ± {metrics['auc_sd']:.4f} | "
                f"{metrics['auprc_mean']:.4f} ± {metrics['auprc_sd']:.4f} | "
                f"{metrics['balanced_acc_mean']:.4f} ± {metrics['balanced_acc_sd']:.4f} |"
            )
        lines += [
            "",
            f"- shuffled-labels sanity AUC：{m['shuffled_auc']:.4f} ± "
            f"{m['shuffled_auc_sd']:.4f}（{m['shuffled_n_seeds']} 个种子；期望 ≈ 0.5）",
            f"- **优势 = {adv['advantage_point']:+.4f}**（95% CI [{adv['ci_lo']:+.4f}, {adv['ci_hi']:+.4f}]）",
            f"- 该目标裁决：`{m['verdict']}`",
            "",
        ]
    descriptive = meta.get("descriptive") or {}
    descriptive_entries = descriptive.get("T1") or {}
    if descriptive_entries:
        lines += [
            "## 描述性附表：T1 前瞻衰减（非判据，PREREG #8）",
            "",
            "| δ (ms) | 净前瞻 (ms) | 最优层 | 探针 AUC | 最大基线 AUC | 优势（95% CI） |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for delta_key in sorted(descriptive_entries, key=int):
            entry = descriptive_entries[delta_key]
            adv = entry["advantage"]
            max_baseline = adv["probe_auc"] - adv["advantage_point"]
            lines.append(
                f"| {entry['delta_ms']} | [{entry['net_lead_ms'][0]}, {entry['net_lead_ms'][1]}) "
                f"| L{entry['best_layer']} | {adv['probe_auc']:.4f} | {max_baseline:.4f} "
                f"| {adv['advantage_point']:+.4f} [{adv['ci_lo']:+.4f}, {adv['ci_hi']:+.4f}] |"
            )
        lines += [
            "",
            f"> {descriptive.get('note', '')}",
            "",
        ]
    heading = "## G1 总裁决" if not meta.get("ablation") else "## 消融结论（不构成 G1 裁决）"
    lines += [
        heading,
        "",
        f"- 决定性目标：{overall['decisive_target']}；优势 {overall['advantage_point']:+.4f}"
        f"（CI 下界 {overall['ci_lo']:+.4f}）",
        f"- **裁决：`{overall['verdict']}`** —— {_VERDICT_TEXT[overall['verdict']]}",
        "",
    ]
    if meta.get("ablation"):
        lines += [
            f"> ⚠️ 本报告为 **{meta['ablation']}** 消融变体，按 PREREG 变更记录 #7 不得作为正式 G1 依据。",
            "",
        ]
    return "\n".join(lines)
