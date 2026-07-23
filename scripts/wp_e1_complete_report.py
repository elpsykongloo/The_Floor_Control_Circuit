"""汇总 Moshi R1 的 E1 正式网格、G2 裁决与事后有效秩诊断。"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from _bootstrap import REPO_ROOT

REPORTS_DIR = REPO_ROOT / "reports"
TITLE = "E1 完整实验结论（Moshi R1）"
SPEC_ORDER = [
    "T1_d0",
    "T1_d80",
    "T1_d160",
    "T1_d240",
    "T1_d400",
    "T1_d800",
    "T2",
    "T3",
    "T4",
    "T5",
]
SPEC_LABELS = {
    "T1_d0": "T1·0 ms",
    "T1_d80": "T1·80 ms",
    "T1_d160": "T1·160 ms",
    "T1_d240": "T1·240 ms",
    "T1_d400": "T1·400 ms",
    "T1_d800": "T1·800 ms",
    "T2": "T2·让位",
    "T3": "T3·来话",
    "T4": "T4·终点",
    "T5": "T5·状态",
}
BASELINE_LABELS = {"mimi": "Mimi", "hazard": "hazard", "gru": "声学 GRU"}
SQL_QUERIES = {
    "headline": "SELECT * FROM headline;",
    "spec_results": "SELECT * FROM spec_results ORDER BY sort_order;",
    "t1_horizon": "SELECT * FROM t1_horizon ORDER BY delta_ms;",
    "t4_layers": "SELECT * FROM t4_layers ORDER BY layer;",
    "formal_rank_curve": "SELECT * FROM formal_rank_curve ORDER BY rank;",
    "rank_diagnostics": "SELECT * FROM rank_diagnostics ORDER BY layer;",
    "g2_conditions": "SELECT * FROM g2_conditions ORDER BY sort_order;",
    "quality_checks": "SELECT * FROM quality_checks ORDER BY sort_order;",
}


def _load_json(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少 E1 报告：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _format_ci(values: list[float]) -> str:
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def _add_mobile_breaks(text: str) -> str:
    """为便携式阅读器补充不可见的安全换行点。"""
    text = re.sub(r"([，。；：！？、）】])", lambda match: match.group(1) + "\u200b", text)
    text = re.sub(
        r"([\u4e00-\u9fff]{8})(?=[\u4e00-\u9fff])",
        lambda match: match.group(1) + "\u200b",
        text,
    )
    return re.sub(
        r"([/×=∈≥≤+])",
        lambda match: match.group(1) + "\u200b",
        text,
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_type(values: list[Any]) -> str:
    non_null = [value for value in values if value is not None]
    if non_null and all(isinstance(value, bool | int) for value in non_null):
        return "INTEGER"
    if non_null and all(isinstance(value, bool | int | float) for value in non_null):
        return "REAL"
    return "TEXT"


def _materialize_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """通过报告中声明的 SQLite 查询生成最终快照。"""
    _require(set(datasets) == set(SQL_QUERIES), "报告数据集与 SQL 查询集合不一致")
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for dataset, rows in datasets.items():
            _require(bool(rows), f"报告数据集 {dataset} 为空")
            columns = list(rows[0])
            definitions = ", ".join(
                f"{_quote_identifier(column)} "
                f"{_sqlite_type([row.get(column) for row in rows])}"
                for column in columns
            )
            connection.execute(f"CREATE TABLE {_quote_identifier(dataset)} ({definitions})")
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {_quote_identifier(dataset)} VALUES ({placeholders})",
                [[row.get(column) for column in columns] for row in rows],
            )
        return {
            dataset: [
                dict(row) for row in connection.execute(SQL_QUERIES[dataset]).fetchall()
            ]
            for dataset in SQL_QUERIES
        }
    finally:
        connection.close()


def _build_spec_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_spec = summary["per_spec"]
    _require(set(per_spec) == set(SPEC_ORDER), "正式汇总的规格集合与冻结十规格不一致")
    for spec_name in SPEC_ORDER:
        entry = per_spec[spec_name]
        advantage = entry["advantage_vs_strongest"]
        baseline_name = advantage["strongest_baseline"]
        selected_layer = int(entry["selected_layer"])
        seed_aucs = [
            float(entry["auc_by_seed_layer"][str(seed)][str(selected_layer)])
            for seed in (0, 1, 2)
        ]
        top3_sets = [set(entry["top3_by_seed"][str(seed)]) for seed in (0, 1, 2)]
        overlap = set.intersection(*top3_sets)
        rows.append(
            {
                "spec": spec_name,
                "sort_order": SPEC_ORDER.index(spec_name),
                "spec_label": SPEC_LABELS[spec_name],
                "n_classes": int(entry["n_classes"]),
                "selected_layer": selected_layer,
                "probe_auc": float(entry["probe_auc_seed_mean_at_selected"]),
                "probe_auc_min": min(seed_aucs),
                "probe_auc_max": max(seed_aucs),
                "seed_range": max(seed_aucs) - min(seed_aucs),
                "baseline": BASELINE_LABELS[baseline_name],
                "baseline_auc": float(entry["baseline_pooled_auc"][baseline_name]),
                "advantage": float(advantage["advantage"]),
                "ci_low": float(advantage["ci95"][0]),
                "ci_high": float(advantage["ci95"][1]),
                "ci_text": _format_ci(advantage["ci95"]),
                "significant": "是" if float(advantage["ci95"][0]) > 0 else "否",
                "top3_overlap": len(overlap),
            }
        )
    return rows


def _build_t1_rows(spec_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in spec_rows[:6]:
        delta_ms = int(row["spec"].split("d", maxsplit=1)[1])
        rows.append(
            {
                "delta_ms": delta_ms,
                "probe_auc": row["probe_auc"],
                "baseline_auc": row["baseline_auc"],
                "advantage": row["advantage"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "selected_layer": row["selected_layer"],
                "significant": row["significant"],
            }
        )
    return rows


def _build_t4_layer_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    entry = summary["per_spec"]["T4"]
    mimi_auc = float(entry["baseline_pooled_auc"]["mimi"])
    rows: list[dict[str, Any]] = []
    for layer in range(32):
        aucs = [
            float(entry["auc_by_seed_layer"][str(seed)][str(layer)])
            for seed in (0, 1, 2)
        ]
        rows.append(
            {
                "layer": layer,
                "probe_auc_mean": mean(aucs),
                "probe_auc_min": min(aucs),
                "probe_auc_max": max(aucs),
                "mimi_auc": mimi_auc,
                "is_shared_top3": "是" if layer in {29, 30, 31} else "否",
            }
        )
    return rows


def _build_formal_rank_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_seed = summary["g2"]["effective_rank_by_seed"]
    ks = sorted(int(value) for value in by_seed["0"]["curve"])
    rows: list[dict[str, Any]] = []
    for k in ks:
        aucs = [float(by_seed[str(seed)]["curve"][str(k)]) for seed in (0, 1, 2)]
        thresholds = [
            0.5
            + float(by_seed[str(seed)]["retention"])
            * (float(by_seed[str(seed)]["auc_full"]) - 0.5)
            for seed in (0, 1, 2)
        ]
        rows.append(
            {
                "rank": k,
                "auc_mean": mean(aucs),
                "auc_min": min(aucs),
                "auc_max": max(aucs),
                "retention_threshold_mean": mean(thresholds),
                "all_seeds_pass": "是"
                if all(auc >= threshold for auc, threshold in zip(aucs, thresholds, strict=True))
                else "否",
            }
        )
    return rows


def _build_diagnostic_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in (29, 30, 31):
        entry = diagnostics["layers"][str(layer)]
        by_seed = entry["by_seed"]
        rows.append(
            {
                "layer": f"L{layer}",
                "fixed_rank": int(entry["fixed_c_conservative_rank"]),
                "nested_rank": int(entry["nested_c_conservative_rank"]),
                "gate_max": 16,
                "fixed_seed_ranks": "/".join(
                    str(by_seed[str(seed)]["fixed_c_rank"]) for seed in (0, 1, 2)
                ),
                "nested_seed_ranks": "/".join(
                    str(by_seed[str(seed)]["nested_c_rank"]) for seed in (0, 1, 2)
                ),
                "full_auc_mean": mean(
                    float(by_seed[str(seed)]["full_auc"]) for seed in (0, 1, 2)
                ),
                "pca_var_k16_mean": mean(
                    float(by_seed[str(seed)]["pca_cumulative_variance_at_k16"])
                    for seed in (0, 1, 2)
                ),
            }
        )
    return rows


def _build_g2_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = summary["g2"]["verdict"]["conditions"]
    overlap = conditions["top3_overlap"]
    rank = conditions["effective_rank"]
    cosine = conditions["direction_cosine"]
    return [
        {
            "sort_order": 0,
            "condition": "三种子 top-3 交集",
            "observed": len(overlap["overlap"]),
            "criterion": f"≥ {overlap['required_min']}",
            "status": "通过" if overlap["passed"] else "失败",
            "detail": ", ".join(f"L{layer}" for layer in overlap["overlap"]),
        },
        {
            "sort_order": 1,
            "condition": "保守有效秩",
            "observed": float(rank["value"]),
            "criterion": f"≤ {rank['required_max']:.0f}",
            "status": "通过" if rank["passed"] else "失败",
            "detail": "正式预设 k 网格的首次共同过线点",
        },
        {
            "sort_order": 2,
            "condition": "方向最小 |cos|",
            "observed": float(cosine["min"]),
            "criterion": f"≥ {cosine['required_min']}",
            "status": "通过" if cosine["passed"] else "失败",
            "detail": "三对种子方向的最小绝对余弦",
        },
    ]


def _build_quality_rows(
    *,
    cache_plan: dict[str, Any],
    cache_audit: dict[str, Any],
    cache_parity: dict[str, Any],
    ingest: dict[str, Any],
    labels: dict[str, Any],
    probe_parity: dict[str, Any],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    parity_cells = probe_parity["cells"]
    max_auc_delta = max(float(cell["abs_auc_diff"]) for cell in parity_cells.values())
    min_direction_cos = min(
        float(cell["direction_abs_cos"]) for cell in parity_cells.values()
    )
    return [
        {
            "sort_order": 0,
            "check": "冻结计划",
            "observed": f"{cache_plan['n_sessions']} 会话 / {cache_plan['n_roles']} 角色",
            "status": "通过",
        },
        {
            "sort_order": 1,
            "check": "激活缓存审计",
            "observed": f"{cache_audit['n_passed']}/{cache_audit['n_roles']} 路",
            "status": "通过" if cache_audit["verdict"] == "passed" else "失败",
        },
        {
            "sort_order": 2,
            "check": "Zarr 摄取",
            "observed": f"{ingest['ok']} 新写 + {ingest['skipped']} 断点复用",
            "status": "通过" if ingest["verdict"] == "passed" else "失败",
        },
        {
            "sort_order": 3,
            "check": "标签闭合",
            "observed": f"{labels['n_sessions']} 会话 / {labels['n_roles']} 角色",
            "status": "通过",
        },
        {
            "sort_order": 4,
            "check": "缓存历史前缀奇偶校验",
            "observed": f"{cache_parity['n_all_equal']}/{cache_parity['n_compared']} 精确全等",
            "status": "通过" if cache_parity["verdict"] == "equal" else "失败",
        },
        {
            "sort_order": 5,
            "check": "GPU/CPU 探针奇偶校验",
            "observed": f"最大 |ΔAUC|={max_auc_delta:.6f}；最小 |cos|={min_direction_cos:.6f}",
            "status": "通过" if probe_parity["verdict"] == "passed" else "失败",
        },
        {
            "sort_order": 6,
            "check": "有效秩诊断收敛",
            "observed": (
                f"{diagnostics['fit_audit']['records']} 断点 / "
                f"{diagnostics['fit_audit']['candidate_fits']} 候选拟合 / 0 未收敛"
            ),
            "status": "通过"
            if diagnostics["fit_audit"]["candidate_nonconverged"] == 0
            else "失败",
        },
    ]


def _build_sources() -> list[dict[str, Any]]:
    file_sources: list[dict[str, Any]] = [
        {
            "id": "formal_summary",
            "label": "E1 正式探针与 G2 汇总",
            "path": "reports/wp_e1_probe_summary.json",
        },
        {
            "id": "formal_grid_report",
            "label": "E1 正式探针网格简表",
            "path": "reports/e1_探针网格报告.md",
        },
        {
            "id": "rank_diagnostics",
            "label": "有效秩事后诊断",
            "path": "reports/wp_e1_effective_rank_diagnostics.json",
        },
        {
            "id": "cache_audit",
            "label": "E1 激活缓存审计",
            "path": "reports/wp_e1_cache_audit.json",
        },
        {
            "id": "cache_parity",
            "label": "E1 缓存奇偶校验",
            "path": "reports/wp_e1_cache_parity.json",
        },
        {
            "id": "probe_parity",
            "label": "E1 探针训练器奇偶校验",
            "path": "reports/wp_e1_probe_parity.json",
        },
        {
            "id": "ingest",
            "label": "E1 Zarr 摄取报告",
            "path": "reports/wp_e1_ingest.json",
        },
        {
            "id": "labels",
            "label": "E1 标签闭合报告",
            "path": "reports/wp_e1_probe_labels.json",
        },
        {
            "id": "mve_summary",
            "label": "E1 前置 MVE 正式汇总",
            "path": "reports/mve_summary.json",
        },
        {"id": "prereg", "label": "预注册与变更登记", "path": "PREREG.md"},
    ]
    query_labels = {
        "headline": "E1 核心指标查询",
        "spec_results": "十规格正式结果查询",
        "t1_horizon": "T1 前瞻曲线查询",
        "t4_layers": "T4 跨层曲线查询",
        "formal_rank_curve": "正式有效秩曲线查询",
        "rank_diagnostics": "事后有效秩诊断查询",
        "g2_conditions": "G2 条件对账查询",
        "quality_checks": "质量门对账查询",
    }
    metric_definitions = [
        "AUC：二分类 ROC AUC；多分类规格为逐类一对多 AUC 的宏平均，评估缺类时跳过该类并登记。",
        "优势：选定层三种子探针相对同规格最强可用基线的会话级主指标差。",
        "95% CI：以会话为聚类单位的 1000 次 bootstrap 百分位区间。",
        "有效秩：满足 (AUC_k−0.5)≥0.95×(AUC_full−0.5) 的最小 PCA 投影秩，G2 取三种子最大值。",
    ]
    query_sources = [
        {
            "id": f"{dataset}_sql",
            "label": label,
            "path": "scripts/wp_e1_complete_report.py",
            "query": {
                "id": f"e1_complete_{dataset}",
                "language": "sql",
                "engine": "SQLite",
                "description": f"从已复核的 E1 正式 JSON 派生 {label}。",
                "sql": SQL_QUERIES[dataset],
                "tables_used": [dataset],
                "metric_definitions": metric_definitions,
            },
        }
        for dataset, label in query_labels.items()
    ]
    return [*file_sources, *query_sources]


def build_artifact() -> dict[str, Any]:
    summary = _load_json("wp_e1_probe_summary.json")
    diagnostics = _load_json("wp_e1_effective_rank_diagnostics.json")
    cache_plan = _load_json("wp_e1_cache_plan.json")
    cache_audit = _load_json("wp_e1_cache_audit.json")
    cache_parity = _load_json("wp_e1_cache_parity.json")
    probe_parity = _load_json("wp_e1_probe_parity.json")
    ingest = _load_json("wp_e1_ingest.json")
    labels = _load_json("wp_e1_probe_labels.json")
    mve = _load_json("mve_summary.json")

    _require(summary["g2"]["primary_target"] == "T4", "G2 主目标不再是预冻结的 T4")
    _require(summary["g2"]["verdict"]["verdict"] == "fail", "G2 正式裁决与报告预期不一致")
    _require(diagnostics["formal_g2_unchanged"] is True, "事后诊断未声明保持正式 G2")
    _require(cache_audit["verdict"] == "passed", "激活缓存审计未通过")
    _require(cache_parity["verdict"] == "equal", "缓存奇偶校验未通过")
    _require(probe_parity["verdict"] == "passed", "探针奇偶校验未通过")
    _require(ingest["verdict"] == "passed", "Zarr 摄取未通过")
    _require(labels["n_sessions"] == 500 and labels["n_roles"] == 1000, "标签集合未闭合")
    _require(mve["overall"]["verdict"] == "full_e1", "E1 前置 G1 未通过")

    spec_rows = _build_spec_rows(summary)
    t1_rows = _build_t1_rows(spec_rows)
    t4_layer_rows = _build_t4_layer_rows(summary)
    formal_rank_rows = _build_formal_rank_rows(summary)
    diagnostic_rows = _build_diagnostic_rows(diagnostics)
    g2_rows = _build_g2_rows(summary)
    quality_rows = _build_quality_rows(
        cache_plan=cache_plan,
        cache_audit=cache_audit,
        cache_parity=cache_parity,
        ingest=ingest,
        labels=labels,
        probe_parity=probe_parity,
        diagnostics=diagnostics,
    )
    t4 = next(row for row in spec_rows if row["spec"] == "T4")
    g2 = summary["g2"]
    mlp_gaps = [float(value["gap"]) for value in g2["mlp_contrast"].values()]
    shuffled = [float(value) for value in g2["shuffled_auc_by_seed"].values()]
    generated_at = datetime.now(UTC).isoformat()

    headline = [
        {
            "g2_verdict": "fail",
            "primary_target": "T4",
            "selected_layer": int(g2["selected_layer"]),
            "t4_advantage": t4["advantage"],
            "t4_ci_low": t4["ci_low"],
            "direction_min_cos": min(float(value) for value in g2["direction_abs_cosines"].values()),
            "direction_threshold": 0.8,
            "formal_effective_rank": int(g2["effective_rank_conservative"]),
            "rank_threshold": 16,
            "diagnostic_best_rank": min(row["nested_rank"] for row in diagnostic_rows),
            "diagnostic_excess": min(row["nested_rank"] for row in diagnostic_rows) - 16,
            "sessions": int(cache_plan["n_sessions"]),
            "roles": int(cache_plan["n_roles"]),
            "cache_gb": float(cache_audit["telemetry"]["output_bytes_total"]) / 1_000_000_000,
        }
    ]

    cards = [
        {
            "id": "t4_card",
            "description": "T4 在最早稳定高分层 L29 相对最强基线 Mimi 的 AUC 优势。",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {"label": "T4 AUC 优势", "field": "t4_advantage", "format": "number"},
                {"label": "95% CI 下界", "field": "t4_ci_low", "format": "number"},
                {"label": "选定层", "field": "selected_layer", "format": "number"},
            ],
        },
        {
            "id": "stability_card",
            "description": "同一 L29 上三种子方向的最小绝对余弦。",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {"label": "方向最小 |cos|", "field": "direction_min_cos", "format": "number"},
                {"label": "门限", "field": "direction_threshold", "format": "number"},
            ],
        },
        {
            "id": "rank_card",
            "description": "正式预设 k 网格的保守有效秩及冻结上限。",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {"label": "正式有效秩", "field": "formal_effective_rank", "format": "number"},
                {"label": "门限", "field": "rank_threshold", "format": "number"},
            ],
        },
        {
            "id": "diagnostic_card",
            "description": "事后逐整数细扫与逐 k 嵌套选 C 得到的最小保守秩，仅作解释。",
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {"label": "诊断最小秩", "field": "diagnostic_best_rank", "format": "number"},
                {"label": "仍高于门限", "field": "diagnostic_excess", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "spec_auc_comparison",
            "title": "十项规格的探针与最强基线 AUC",
            "subtitle": "T4 的独立优势最突出；T2、T3、T5 与 Mimi 基本重合",
            "intent": "comparison",
            "question": "各目标的最优稳定层相对同规格最强基线表现如何？",
            "rationale": "十项规格使用同一 AUC 尺度，分组柱形图可同时呈现绝对可读性与基线差距。",
            "comparisonContext": {
                "baseline": "同规格最强可用基线",
                "denominator": "冻结主评估会话的有效目标行",
                "grain": "目标规格",
                "unit": "AUC 或 macro-OVR AUC",
            },
            "type": "bar",
            "dataset": "spec_results",
            "sourceId": "spec_results_sql",
            "encodings": {
                "x": {"field": "spec_label", "type": "nominal", "label": "目标规格"},
                "y": {
                    "fields": ["probe_auc", "baseline_auc"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "AUC",
                },
                "tooltip": [
                    {"field": "selected_layer", "type": "quantitative", "label": "选定层"},
                    {"field": "advantage", "type": "quantitative", "label": "优势"},
                    {"field": "ci_text", "type": "text", "label": "优势 95% CI"},
                    {"field": "baseline", "type": "text", "label": "最强基线"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "labels": {"values": "auto"},
            "settings": {"groupMode": "grouped", "sort": "custom", "showValues": False},
        },
        {
            "id": "t1_horizon_curve",
            "title": "T1 发言起始预测随前瞻量变化",
            "subtitle": "前瞻越远，绝对 AUC 下降；探针相对 Mimi 的优势逐步扩大",
            "intent": "trend",
            "question": "发言起始预测的绝对性能与相对基线优势如何随 δ 变化？",
            "rationale": "δ 是有序时间轴，双线能直接比较模型内部表征与 Mimi 的衰减速度。",
            "comparisonContext": {
                "baseline": "Mimi 同规格探针",
                "denominator": "T1 冻结行域",
                "grain": "前瞻量",
                "unit": "AUC",
            },
            "type": "line",
            "dataset": "t1_horizon",
            "sourceId": "t1_horizon_sql",
            "encodings": {
                "x": {
                    "field": "delta_ms",
                    "type": "quantitative",
                    "format": "number",
                    "label": "δ（毫秒）",
                },
                "y": {
                    "fields": ["probe_auc", "baseline_auc"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "AUC",
                },
                "tooltip": [
                    {"field": "advantage", "type": "quantitative", "label": "优势"},
                    {"field": "ci_low", "type": "quantitative", "label": "CI 下界"},
                    {"field": "ci_high", "type": "quantitative", "label": "CI 上界"},
                    {"field": "selected_layer", "type": "quantitative", "label": "选定层"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "sort": "spec", "title": "读出"},
        },
        {
            "id": "t4_layer_curve",
            "title": "T4 跨层 AUC 曲线",
            "subtitle": "L29–L31 在三种子中共同占据前三；L29 是交集内最早层",
            "intent": "trend",
            "question": "T4 信息沿 32 层如何浮现，稳定高分区间在哪里？",
            "rationale": "逐层曲线可区分单点尖峰与连续晚层平台。",
            "comparisonContext": {
                "baseline": "Mimi T4 AUC",
                "denominator": "T4 冻结评估行域",
                "grain": "Transformer 层",
                "unit": "AUC",
            },
            "type": "line",
            "dataset": "t4_layers",
            "sourceId": "t4_layers_sql",
            "encodings": {
                "x": {"field": "layer", "type": "quantitative", "label": "层号"},
                "y": {
                    "fields": ["probe_auc_mean", "mimi_auc"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "AUC",
                },
                "tooltip": [
                    {"field": "probe_auc_min", "type": "quantitative", "label": "种子最小值"},
                    {"field": "probe_auc_max", "type": "quantitative", "label": "种子最大值"},
                    {"field": "is_shared_top3", "type": "text", "label": "共同 top-3"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "sort": "spec", "title": "读出"},
        },
        {
            "id": "formal_rank_curve",
            "title": "L29 正式有效秩曲线",
            "subtitle": "k=64 仍未达到 95% 保留线；原预设网格直到 k=128 才共同过线",
            "intent": "trend",
            "question": "T4 线性信号需要多少个 PCA 方向才能保留 95% 的超随机 AUC？",
            "rationale": "AUC 曲线与逐种子门线的均值同图，直接展示正式秩裁决。",
            "comparisonContext": {
                "baseline": "随机 AUC=0.5；保留 95% 的超随机部分",
                "denominator": "L29 全维 T4 AUC",
                "grain": "PCA 投影秩",
                "unit": "AUC",
            },
            "type": "line",
            "dataset": "formal_rank_curve",
            "sourceId": "formal_rank_curve_sql",
            "encodings": {
                "x": {"field": "rank", "type": "quantitative", "label": "PCA 秩 k"},
                "y": {
                    "fields": ["auc_mean", "retention_threshold_mean"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "AUC",
                },
                "tooltip": [
                    {"field": "auc_min", "type": "quantitative", "label": "种子最小 AUC"},
                    {"field": "auc_max", "type": "quantitative", "label": "种子最大 AUC"},
                    {"field": "all_seeds_pass", "type": "text", "label": "三种子均过线"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "sort": "spec", "title": "曲线"},
        },
        {
            "id": "diagnostic_rank_comparison",
            "title": "L29–L31 的事后有效秩复核",
            "subtitle": "最优诊断值为 L30 嵌套选 C 的 57，仍是冻结上限 16 的 3.56 倍",
            "intent": "comparison",
            "question": "固定 C、低维重选 C 与相邻层能否解释正式有效秩失败？",
            "rationale": "三层两种训练口径与冻结上限处在同一维数尺度，可直接比较。",
            "comparisonContext": {
                "baseline": "G2 有效秩上限 16",
                "denominator": "三种子最小过线 k 的最大值",
                "grain": "层",
                "unit": "维",
            },
            "type": "bar",
            "dataset": "rank_diagnostics",
            "sourceId": "rank_diagnostics_sql",
            "encodings": {
                "x": {"field": "layer", "type": "nominal", "label": "层"},
                "y": {
                    "fields": ["fixed_rank", "nested_rank", "gate_max"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "保守有效秩",
                },
                "tooltip": [
                    {"field": "fixed_seed_ranks", "type": "text", "label": "固定 C 三种子"},
                    {"field": "nested_seed_ranks", "type": "text", "label": "嵌套 C 三种子"},
                    {"field": "pca_var_k16_mean", "type": "quantitative", "label": "k=16 方差占比"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "labels": {"values": "auto"},
            "settings": {"groupMode": "grouped", "sort": "custom", "showValues": True},
        },
    ]
    for chart in charts:
        chart["surface"] = {
            "surface": "compact",
            "compact": True,
            "showControls": False,
            "viewMode": "visualization",
        }

    tables = [
        {
            "id": "spec_table",
            "title": "十项规格的正式结果",
            "subtitle": "优势与区间均相对同规格最强可用基线",
            "dataset": "spec_results",
            "sourceId": "spec_results_sql",
            "defaultSort": {"field": "advantage", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "spec_label", "label": "规格", "type": "text"},
                {"field": "selected_layer", "label": "ℓ*", "format": "number"},
                {"field": "probe_auc", "label": "探针 AUC", "format": "number"},
                {"field": "baseline", "label": "最强基线", "type": "text"},
                {"field": "baseline_auc", "label": "基线 AUC", "format": "number"},
                {
                    "field": "advantage",
                    "label": "优势",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                },
                {"field": "ci_text", "label": "95% CI", "type": "text"},
                {"field": "significant", "label": "下界 > 0", "type": "text"},
            ],
        },
        {
            "id": "g2_table",
            "title": "G2 三条件对账",
            "subtitle": "总裁决要求三项同时通过",
            "dataset": "g2_conditions",
            "sourceId": "g2_conditions_sql",
            "defaultSort": {"field": "condition", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "condition", "label": "条件", "type": "text"},
                {"field": "observed", "label": "观察值", "format": "number"},
                {"field": "criterion", "label": "冻结判据", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
                {"field": "detail", "label": "说明", "type": "text"},
            ],
        },
        {
            "id": "quality_table",
            "title": "数据与数值质量门",
            "subtitle": "正式结果所依赖的缓存、标签、训练器与诊断收敛检查",
            "dataset": "quality_checks",
            "sourceId": "quality_checks_sql",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "复核项", "type": "text"},
                {"field": "observed", "label": "观察值", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 技术摘要\n\n"
                "**G2=`fail`：仅有效秩失败，128 > 16。** "
                "T4·L29 的 AUC 为 0.8348，相对 Mimi +0.0659 "
                "[+0.0581,+0.0743]；共同 top-3=L29/L30/L31，|cos|≥0.9757。"
                "细扫最低秩 57；T1 有小幅增量，T2/T3/T5 无确认增量。"
                "当前范围仅为 Moshi R1；完整章节与精确表见同名 Markdown。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["t4_card", "stability_card", "rank_card", "diagnostic_card"],
            "layout": "full",
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 已完成范围\n\n"
                "本轮覆盖 Moshi 7B、R1 双通道复放、500 个冻结 CANDOR 会话："
                "probe_train 前 400 个用于训练，probe_val[40:140] 的 100 个全新会话用于主评估。"
                "每个会话取前 240 秒、双角色，共 1000 路；缓存 32 层、每层 3000 步、"
                "隐藏维度 4096。正式探针包含 10 个规格 × 32 层 × 3 种子 = 960 个单元，"
                "每单元扫描五档 C 后正式重训。多分类使用 macro-OVR AUC，T1 使用 5:1 负采样，"
                "T5 使用 stride=4。"
            ),
        },
        {"id": "spec_chart_block", "type": "chart", "chartId": "spec_auc_comparison", "layout": "full"},
        {
            "id": "target_findings",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 各目标的实验结论\n\n"
                "- **T4：** 唯一跨过 +0.05 且区间下界为正的强增量结果，内部晚层含有 Mimi、"
                "hazard 与声学 GRU 均未覆盖的话轮完整性信息。\n"
                "- **T1：** 80–800 ms 的优势区间均高于 0，模型晚层持续携带发言起始前瞻信息；"
                "增量最高为 800 ms 的 +0.0238，未达到冻结的 +0.05 强效应线。0 ms 规格区间跨 0，"
                "且选层 L4，稳定性较弱。\n"
                "- **T2：** 探针 AUC 0.7962，略低于 Mimi 0.7972，缺少内部增量证据。\n"
                "- **T3：** macro-OVR AUC 0.9709，但只比 Mimi 高 0.0001，属于高绝对性能下的"
                "输入端表征饱和。\n"
                "- **T5：** macro-OVR AUC 0.9942，与 Mimi 的差距仅 0.0002 且区间跨 0；"
                "作为状态解码健全性检查通过，不能视为骨干新增信息。"
            ),
        },
        {"id": "spec_table_block", "type": "table", "tableId": "spec_table", "layout": "full"},
        {
            "id": "t1_section",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## T1 前瞻曲线\n\n"
                "绝对 AUC 从 δ=0 的 0.9589 下降到 δ=800 ms 的 0.8678；Mimi 同期从 0.9484 "
                "下降到 0.8441。探针优势因此随距离总体扩大。δ≥80 ms 时共同指向晚层 L29，"
                "说明较远期的发言起始信息主要在骨干后段形成。按照冻结 H1 的强效应口径，"
                "这些结果支持可靠但中小幅的前瞻增量，尚不足以宣称 ≥0.05 的长时域独立优势。"
            ),
        },
        {"id": "t1_chart_block", "type": "chart", "chartId": "t1_horizon_curve", "layout": "full"},
        {
            "id": "layer_section",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 层位定位\n\n"
                "T4 的三种子 top-3 完全一致，顺序均为 L31、L30、L29。正式规则从共同交集中取"
                "最早层，因此 ℓ*=L29；它代表最早稳定高分层，并非全层 AUC 的单点最大值。"
                "T2 选 L28，T3/T5 选 L30，T1 的 80–800 ms 均选 L29。整体证据把决策相关信息"
                "定位到 Transformer 末端三至四层的宽平台，未呈现脆弱的单层尖峰。"
            ),
        },
        {"id": "layer_chart_block", "type": "chart", "chartId": "t4_layer_curve", "layout": "full"},
        {
            "id": "g2_section",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## G2 正式裁决\n\n"
                "G2 要求层稳定、有效秩与方向稳定三项同时通过。层交集为 3≥2，方向最小 "
                "|cos|=0.9757≥0.8，均有充足余量；正式保守有效秩为 128，超过上限 16，"
                "因此总裁决必须保持 `fail`。失败集中在几何紧凑性条件，不能解读为目标不可读、"
                "层位不稳定或方向不一致。"
            ),
        },
        {"id": "g2_table_block", "type": "table", "tableId": "g2_table", "layout": "full"},
        {"id": "rank_chart_block", "type": "chart", "chartId": "formal_rank_curve", "layout": "full"},
        {
            "id": "controls",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 线性度与负对照\n\n"
                f"两层 MLP 在三种子上的 AUC 相对线性探针分别为 "
                f"{mlp_gaps[0]:+.4f}、{mlp_gaps[1]:+.4f}、{mlp_gaps[2]:+.4f}，"
                "均更低；没有证据表明当前残差流需要非线性读出才能获得 T4 信号。"
                f"打乱标签 AUC 为 {shuffled[0]:.4f}/{shuffled[1]:.4f}/{shuffled[2]:.4f}，"
                "围绕随机水平波动，削弱了标签泄漏、会话抽样捷径或评估实现偏差的解释。"
            ),
        },
        {
            "id": "diagnostic_section",
            "type": "markdown",
            "sourceId": "rank_diagnostics",
            "layout": "full",
            "body": (
                "## 有效秩失败的三项事后诊断\n\n"
                "PREREG #30 的非裁决复核同时检查低维重选 C、相邻高分层与 64–128 逐整数细扫。"
                "L29 的固定/嵌套保守秩均为 84；L30 为 68/57；L31 为 66/66。"
                "低维正则失配只能解释 L30 的 11 维差异，相邻层可把结果降至 57，仍比门限多 41 维。"
                "前 16 个主成分只解释 38.3%–44.9% 总方差，实际过线点对应 54.1%–60.8%。"
                "因此最稳妥的几何解释是：T4 信号线性、稳定且分布式存在，未压缩为一个"
                "≤16 维的高方差 PCA 子空间。正式粗网格结果 128 与 G2 失败均不回写。"
            ),
        },
        {
            "id": "diagnostic_chart_block",
            "type": "chart",
            "chartId": "diagnostic_rank_comparison",
            "layout": "full",
        },
        {
            "id": "quality_section",
            "type": "markdown",
            "sourceId": "cache_audit",
            "layout": "full",
            "body": (
                "## 数据与数值可信度\n\n"
                f"缓存审计覆盖 1000/1000 路，输出 {headline[0]['cache_gb']:.2f} GB；"
                "历史 MVE 前缀的 2 次复核逐位全等。GPU 训练器相对 sklearn 的最大 "
                "|ΔAUC|=0.000267、方向最小 |cos|=0.999997，均通过硬门。"
                "Zarr 摄取与 500 会话标签清单闭合；事后诊断的 3705 个候选拟合与全部重训"
                "未出现不收敛。独立质量证据支持把 G2 失败归因于观察到的表示几何，"
                "缺少缓存损坏或求解器偏差的迹象。"
            ),
        },
        {
            "id": "quality_table_block",
            "type": "table",
            "tableId": "quality_table",
            "layout": "full",
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "prereg",
            "layout": "full",
            "body": (
                "## 方法与判据\n\n"
                "线性探针为标准化后的 L2-logistic，C∈{1e-4,1e-3,0.01,0.1,1.0}，"
                "inner_val 固定前 80 个训练会话；每种子从后续训练池做 90% 会话级无放回抽样。"
                "主评估保持 100 个会话不变，优势区间使用 1000 次会话级 cluster bootstrap。"
                "有效秩定义为最小 k，使 (AUC_k−0.5)≥0.95×(AUC_full−0.5)，G2 取三种子最大值。"
                "T4 预先冻结为 G2 主目标，避免看到其余九规格后择优。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 限制与解释边界\n\n"
                "- 当前结果只覆盖 Moshi 的 R1 表征，尚不能推广到 PersonaPlex、MiniCPM-o、"
                "Freeze-Omni、dGSLM 或 R2 自主生成。\n"
                "- 探针建立预测关联，不提供因果性；G2 未通过时尤其不能直接进入残差流 E2 主位点。\n"
                "- CANDOR 没有人工话轮标签，T4 依赖冻结事件管线；已有人工核验、MVE 优势和"
                "负对照支撑效度，但仍弱于逐事件金标。\n"
                "- T3/T5 多分类只有 Mimi 输入端基线；声学 GRU 的现有接口仅覆盖二分类。\n"
                "- 正式有效秩受 PCA 排序、95% 定义与离散 k 网格影响；事后细扫提高了解析度，"
                "不能追溯改变 Gate。\n"
                "- 诊断首次过线的最小正裕量仅 8.4e-06，具体整数秩可能有约一维数值波动；"
                "它与上限 16 的巨大间隔不受该不确定性影响。"
            ),
        },
        {
            "id": "implications",
            "type": "markdown",
            "sourceId": "formal_summary",
            "layout": "full",
            "body": (
                "## 科学含义\n\n"
                "Moshi 晚层确实形成了输入编解码表征之外的话轮终点投射信息，并且该方向跨训练"
                "子样本高度一致。其组织方式更接近分布式线性读出：线性探针已足够，信号却需要"
                "数十个 PCA 主方向共同保留。发言起始前瞻也存在稳定增量，但幅度属于中小效应。"
                "因此本轮同时支持“晚层含有 floor 相关信息”和“该信息未形成预注册的低秩紧凑"
                "残差流电路”两项结论。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "prereg",
            "layout": "full",
            "body": (
                "## 下一步建议\n\n"
                "1. 冻结并保留当前 G2=`fail`，不改有效秩门限或正式主目标。\n"
                "2. 按原计划登记一次换位点重试：优先检查 L29–L31 的 attention 输出与 MLP 输出，"
                "沿同一 T4 行域、三种子和基线复用判据。\n"
                "3. 非线性 MLP 已无正收益，重试资源应优先投向位点分解与子模块方向，"
                "同时保留范数匹配和打乱标签对照。\n"
                "4. 继续完成其余模型、R2、泛化矩阵与外部上界；这些属于 E1 全计划的未完成部分，"
                "不应由 Moshi R1 单项结果替代。\n"
                "5. 若换位点后仍无法满足 ≤16，则按预注册转入 N1′，保留 T4 强可读性、"
                "稳定方向与分布式几何作为核心描述性结果。"
            ),
        },
    ]
    for block in blocks:
        if block.get("type") == "markdown":
            block["body"] = _add_mobile_breaks(block["body"])
    # 便携式阅读器在窄屏上无法安全承载八列正式表；精确表保留在 Markdown 与快照中。
    blocks = [
        block
        for block in blocks
        if block["id"] in {"title", "summary", "diagnostic_chart_block"}
    ]
    tables = []

    sources = _build_sources()
    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Moshi R1 的 E1 正式探针网格、G2 裁决、质量审计与有效秩事后诊断。",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    raw_datasets = {
        "headline": headline,
        "spec_results": spec_rows,
        "t1_horizon": t1_rows,
        "t4_layers": t4_layer_rows,
        "formal_rank_curve": formal_rank_rows,
        "rank_diagnostics": diagnostic_rows,
        "g2_conditions": g2_rows,
        "quality_checks": quality_rows,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": _materialize_datasets(raw_datasets),
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def build_markdown(artifact: dict[str, Any]) -> str:
    datasets = artifact["snapshot"]["datasets"]
    headline = datasets["headline"][0]
    spec_rows = datasets["spec_results"]
    g2_rows = datasets["g2_conditions"]
    diagnostic_rows = datasets["rank_diagnostics"]
    quality_rows = datasets["quality_checks"]

    lines = [
        f"# {TITLE}",
        "",
        "> 正式主结果以 `wp_e1_probe_summary.json` 为准；有效秩细扫属于 PREREG #30 的事后、"
        "严格非裁决诊断。",
        "",
        "## 一句话结论",
        "",
        "**Moshi 晚层存在稳定、线性且显著超越 Mimi 的 T4 话轮终点投射信息；"
        "该信息未压缩进预注册要求的 ≤16 维 PCA 子空间，因此 G2 正式裁决为 `fail`。**",
        "",
        "## 核心数字",
        "",
        f"- G2：`{headline['g2_verdict']}`；主目标 T4；正式层 L{headline['selected_layer']}。",
        f"- T4 优势：{headline['t4_advantage']:+.6f}，95% CI 下界 "
        f"{headline['t4_ci_low']:+.6f}。",
        f"- 方向最小 |cos|：{headline['direction_min_cos']:.6f}。",
        f"- 正式保守有效秩：{headline['formal_effective_rank']}；门限 16。",
        f"- 事后诊断最佳保守秩：{headline['diagnostic_best_rank']}；仍高于门限 41 维。",
        f"- 数据规模：{headline['sessions']} 会话、{headline['roles']} 角色路、"
        f"{headline['cache_gb']:.2f} GB 激活缓存。",
        "",
        "## 十项规格正式结果",
        "",
        "| 规格 | ℓ* | 探针 AUC | 最强基线 | 基线 AUC | 优势 | 95% CI | 下界>0 |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in spec_rows:
        lines.append(
            f"| {row['spec_label']} | L{row['selected_layer']} | {row['probe_auc']:.4f} | "
            f"{row['baseline']} | {row['baseline_auc']:.4f} | {row['advantage']:+.4f} | "
            f"{row['ci_text']} | {row['significant']} |"
        )

    lines.extend(
        [
            "",
            "## 逐目标结论",
            "",
            "- **T4：** AUC 0.8348，相对 Mimi +0.0659 "
            "[+0.0581,+0.0743]，是唯一决定性增量结果。",
            "- **T1：** δ=80–800 ms 的优势区间均高于 0，但最大优势仅 +0.0238；"
            "支持中小幅前瞻增量，未达到 +0.05 强效应线。δ=0 的区间跨 0。",
            "- **T2：** 探针略低于 Mimi，缺少新增信息证据。",
            "- **T3：** 绝对 macro-OVR AUC 很高，与 Mimi 基本等价。",
            "- **T5：** 状态解码接近饱和，与 Mimi 基本等价，只支持健全性检查。",
            "",
            "## 层位与几何",
            "",
            "- T4 三种子 top-3 均为 L31/L30/L29；正式规则取共同交集中的最早层 L29。",
            "- 三对方向 |cos| 为 0.9800、0.9777、0.9757，方向稳定性有充足余量。",
            "- MLP−线性 AUC 为 −0.0211、−0.0175、−0.0202；当前信号线性可读。",
            "- 打乱标签 AUC 为 0.4824、0.4897、0.5158，围绕随机水平。",
            "",
            "## G2 三条件",
            "",
            "| 条件 | 观察值 | 判据 | 状态 |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in g2_rows:
        lines.append(
            f"| {row['condition']} | {row['observed']:.4f} | {row['criterion']} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## 有效秩事后诊断",
            "",
            "| 层 | 固定正式 C | 逐 k 嵌套选 C | k=16 平均方差占比 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            f"| {row['layer']} | {row['fixed_rank']} | {row['nested_rank']} | "
            f"{row['pca_var_k16_mean']:.1%} |"
        )

    lines.extend(
        [
            "",
            "细扫说明正式 128 是预设粗网格的首次共同过线点；更精确的数值位置为 57–84。"
            "低维正则失配和层位差异都无法把结果降到 16。信号分布在数十个中等排序主成分上，"
            "G2 失败保持不变。",
            "",
            "## 数据与数值质量门",
            "",
            "| 复核项 | 观察值 | 状态 |",
            "| --- | --- | --- |",
        ]
    )
    for row in quality_rows:
        lines.append(f"| {row['check']} | {row['observed']} | {row['status']} |")

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 当前覆盖 Moshi R1；五模型、R2、跨语料、跨语言和外部上界尚未完成。",
            "- 探针结果属于预测关联，不能替代因果干预。",
            "- CANDOR 的 T4 来自冻结事件管线；人工核验、MVE 与负对照提供支持，"
            "标签强度仍低于逐事件人工金标。",
            "- 事后有效秩诊断只解释失败机制，不参与正式裁决。",
            "",
            "## 后续动作",
            "",
            "1. 冻结 G2=`fail`，不改正式门限。",
            "2. 预注册并执行 L29–L31 attention 输出与 MLP 输出的换位点重试。",
            "3. 非线性 MLP 已无正收益，资源优先用于子模块位点分解。",
            "4. 继续其余模型、R2 与泛化矩阵，完成 E1 全计划。",
            "5. 换位点仍失败时按预注册转入 N1′。",
            "",
            "## 权威来源",
            "",
            "- `reports/wp_e1_probe_summary.json`：正式十规格与 G2。",
            "- `reports/wp_e1_effective_rank_diagnostics.json`：PREREG #30 事后诊断。",
            "- `reports/wp_e1_cache_audit.json`、`wp_e1_cache_parity.json`、"
            "`wp_e1_probe_parity.json`：数据与数值质量门。",
            "- `PREREG.md`、`文档/00_原始计划.md`：冻结判据与解释边界。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=REPORTS_DIR / "e1_完整实验结论.artifact.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPORTS_DIR / "e1_完整实验结论.md",
    )
    args = parser.parse_args()
    artifact = build_artifact()
    args.artifact_output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(build_markdown(artifact), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(args.artifact_output),
                "markdown": str(args.markdown_output),
                "datasets": {
                    name: len(rows)
                    for name, rows in artifact["snapshot"]["datasets"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
