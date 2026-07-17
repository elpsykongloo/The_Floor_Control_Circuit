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
    cluster_bootstrap_auc,
    g1_verdict,
    paired_advantage_bootstrap,
    pooled_metrics,
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
    """按层聚合：AUC 的种子均值±SD + 以中位种子的 per_session 作为该层代表（用于 bootstrap）。"""
    out: dict[int, dict] = {}
    layers = sorted({c.layer for c in cells})
    for layer in layers:
        grp = [c for c in cells if c.layer == layer]
        aucs = np.array([c.metrics["auc"] for c in grp])
        rep = grp[int(np.argsort(aucs)[len(aucs) // 2])]
        out[layer] = {
            "auc_mean": float(aucs.mean()),
            "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "auprc_mean": float(np.mean([c.metrics["auprc"] for c in grp])),
            "balanced_acc_mean": float(np.mean([c.metrics["balanced_acc"] for c in grp])),
            "rep_seed": rep.seed,
            "rep_per_session": rep.per_session,
        }
    return out


def evaluate_target(
    layer_summary: dict[int, dict],
    baselines: dict[str, PerSession],
    n_boot: int,
    full_thr: float,
    backup_thr: float,
    boot_seed: int = 0,
) -> dict:
    """选最优层 → 成对优势 bootstrap → 该目标的 G1 材料。"""
    best_layer = max(layer_summary, key=lambda ell: layer_summary[ell]["auc_mean"])
    rep = layer_summary[best_layer]["rep_per_session"]
    adv = paired_advantage_bootstrap(rep, baselines, n_boot=n_boot, seed=boot_seed)
    shuffled = shuffle_labels_within_session(rep, seed=boot_seed)
    return {
        "best_layer": int(best_layer),
        "layer_summary": {
            int(k): {kk: vv for kk, vv in v.items() if kk != "rep_per_session"}
            for k, v in layer_summary.items()
        },
        "advantage": adv,
        "probe_ci": cluster_bootstrap_auc(rep, n_boot=n_boot, seed=boot_seed),
        "shuffled_auc": pooled_metrics(shuffled)["auc"],
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
    lines = [
        "# MVE 报告（Moshi R1，G1 裁决）",
        "",
        f"- 配置：层 {meta.get('layers')}，目标 {list(per_target)}，种子 {meta.get('seeds')}，"
        f"bootstrap {meta.get('bootstrap_n')} 次（会话级）",
        f"- 数据：训练 {meta.get('n_train_sessions')} 会话 / 评估 {meta.get('n_eval_sessions')} 会话",
        "",
    ]
    for target, m in per_target.items():
        lines += [f"## 目标 {target}", "", "| 层 | AUC (mean±sd) | AUPRC | balanced acc |", "| --- | --- | --- | --- |"]
        for layer, s in sorted(m["layer_summary"].items()):
            lines.append(
                f"| L{layer} | {s['auc_mean']:.4f} ± {s['auc_sd']:.4f} | "
                f"{s['auprc_mean']:.4f} | {s['balanced_acc_mean']:.4f} |"
            )
        adv = m["advantage"]
        lines += [
            "",
            f"- 最优层：L{m['best_layer']}；探针 AUC {adv['probe_auc']:.4f}"
            f"（95% CI [{m['probe_ci']['ci_lo']:.4f}, {m['probe_ci']['ci_hi']:.4f}]）",
            "- 基线 AUC：" + "，".join(f"{k} {v:.4f}" for k, v in adv["baseline_aucs"].items()),
            f"- shuffled-labels sanity AUC：{m['shuffled_auc']:.4f}（期望 ≈ 0.5）",
            f"- **优势 = {adv['advantage_point']:+.4f}**（95% CI [{adv['ci_lo']:+.4f}, {adv['ci_hi']:+.4f}]）",
            f"- 该目标裁决：`{m['verdict']}`",
            "",
        ]
    lines += [
        "## G1 总裁决",
        "",
        f"- 决定性目标：{overall['decisive_target']}；优势 {overall['advantage_point']:+.4f}"
        f"（CI 下界 {overall['ci_lo']:+.4f}）",
        f"- **裁决：`{overall['verdict']}`** —— {_VERDICT_TEXT[overall['verdict']]}",
        "",
    ]
    return "\n".join(lines)
