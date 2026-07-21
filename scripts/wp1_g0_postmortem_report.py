"""把 G0 事后诊断打包为可审计的便携式报告清单。"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT

REPORTS_DIR = REPO_ROOT / "reports"
REPORT_SQL_PATH = REPO_ROOT / "scripts" / "wp1_g0_postmortem_report.sql"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _add_mobile_breaks(text: str) -> str:
    """为便携式阅读器补充不可见的中文软换行点。"""
    text = re.sub(r"([，。；：！？、）】])", lambda match: match.group(1) + "\u200b", text)
    return re.sub(
        r"([\u4e00-\u9fff]{8})(?=[\u4e00-\u9fff])",
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


def _load_report_queries() -> dict[str, str]:
    sql_text = REPORT_SQL_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"^-- dataset:\s*([a-z0-9_]+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(sql_text))
    queries: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(sql_text)
        queries[match.group(1)] = sql_text[start:stop].strip()
    return queries


def _materialize_datasets(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    queries = _load_report_queries()
    if set(queries) != set(datasets):
        raise RuntimeError(f"报告 SQL 与快照数据集不一致：sql={sorted(queries)}，datasets={sorted(datasets)}")

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for dataset, rows in datasets.items():
            if not rows:
                raise RuntimeError(f"报告数据集 {dataset} 为空")
            columns = list(rows[0])
            definitions = ", ".join(
                f"{_quote_identifier(column)} {_sqlite_type([row.get(column) for row in rows])}" for column in columns
            )
            connection.execute(f"CREATE TABLE {_quote_identifier(dataset)} ({definitions})")
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {_quote_identifier(dataset)} VALUES ({placeholders})",
                [[row.get(column) for column in columns] for row in rows],
            )
        return {
            dataset: [dict(row) for row in connection.execute(query).fetchall()] for dataset, query in queries.items()
        }
    finally:
        connection.close()


def build_artifact(data: dict[str, Any]) -> dict[str, Any]:
    generated_at = str(data["generated_at"])
    repair = data["repair_effect_on_same_val"]
    failure = data["gate_failure"]
    distribution = data["session_distribution"]
    layer2 = data["layer2_shift"]
    composition = data["composition"]
    audio = composition["decoded_audio"]
    split = data["split_construction"]

    headline = [
        {
            "corpus_macro_f1": failure["corpus_macro_f1"],
            "corpus_lower_bound": failure["corpus_band"][0],
            "corpus_margin": failure["corpus_lower_margin"],
            "session_p10": failure["session_p10"],
            "session_p10_min": failure["session_p10_min"],
            "session_p10_margin": failure["session_p10_margin"],
            "layer2_min_f1": min(
                layer2["ch0"]["f1"]["train_confirmation"],
                layer2["ch1"]["f1"]["train_confirmation"],
            ),
            "layer2_required": 0.9,
            "layer2_margin": min(
                layer2["ch0"]["f1"]["train_confirmation"],
                layer2["ch1"]["f1"]["train_confirmation"],
            )
            - 0.9,
            "repair_val_macro_delta": repair["corpus_macro_f1_delta"],
            "repair_val_macro_pre": repair["corpus_macro_f1_pre"],
            "repair_val_macro_post": repair["corpus_macro_f1_post"],
        }
    ]

    gate_metrics = [
        {
            "order": 1,
            "metric": "VAD ch0 F1",
            "val": layer2["ch0"]["f1"]["val"],
            "train_confirm": layer2["ch0"]["f1"]["train_confirmation"],
            "criterion": "≥0.9000",
            "train_margin": layer2["ch0"]["f1"]["train_confirmation"] - 0.9,
            "status": "通过",
        },
        {
            "order": 2,
            "metric": "VAD ch1 F1",
            "val": layer2["ch1"]["f1"]["val"],
            "train_confirm": layer2["ch1"]["f1"]["train_confirmation"],
            "criterion": "≥0.9000",
            "train_margin": layer2["ch1"]["f1"]["train_confirmation"] - 0.9,
            "status": "通过",
        },
        {
            "order": 3,
            "metric": "层3语料 macro-F1",
            "val": repair["corpus_macro_f1_post"],
            "train_confirm": failure["corpus_macro_f1"],
            "criterion": (f"[{failure['corpus_band'][0]:.4f}, {failure['corpus_band'][1]:.4f}]"),
            "train_margin": failure["corpus_lower_margin"],
            "status": "失败",
        },
        {
            "order": 4,
            "metric": "会话 macro-F1 P10",
            "val": distribution["val"]["p10"],
            "train_confirm": failure["session_p10"],
            "criterion": f"≥{failure['session_p10_min']:.4f}",
            "train_margin": failure["session_p10_margin"],
            "status": "失败",
        },
    ]

    class_metrics = []
    class_labels = {"eot": "EOT", "hold": "HOLD", "bot": "BOT", "bc": "BC"}
    for item in data["class_contributions"]:
        class_metrics.append(
            {
                "event_class": class_labels[item["event_class"]],
                "val_f1": item["val_f1"],
                "train_f1": item["train_f1"],
                "delta": item["delta_train_minus_val"],
                "macro_gap_contribution": item["macro_gap_contribution"],
                "val_precision": item["val_precision"],
                "train_precision": item["train_precision"],
                "val_recall": item["val_recall"],
                "train_recall": item["train_recall"],
                "val_pred_gold_ratio": item["val_pred_gold_ratio"],
                "train_pred_gold_ratio": item["train_pred_gold_ratio"],
            }
        )

    session_rows = []
    for row in data["session_rows"]:
        cohort_label = "验证集" if row["cohort"] == "val" else "训练确认集"
        session_rows.append(
            {
                "cohort": cohort_label,
                "session": row["session"],
                "macro_f1": row["macro_f1"],
                "val_macro_f1": row["macro_f1"] if row["cohort"] == "val" else None,
                "train_macro_f1": (row["macro_f1"] if row["cohort"] == "train_confirm" else None),
                "duration_s": row["duration_s"],
                "duration_band": row["duration_band"],
                "vad_rate_mean": row["vad_rate_mean"],
                "low_energy_fraction": row["speech_fraction_below_m50"],
                "speech_frame_p10_dbfs": row["speech_frame_dbfs_p10_mean"],
                "speech_dbfs": row["speech_dbfs_mean"],
            }
        )

    distribution_summary = [
        {
            "cohort": "验证集",
            "n_sessions": distribution["val"]["n"],
            "p10": distribution["val"]["p10"],
            "median": distribution["val"]["median"],
            "p90": distribution["val"]["p90"],
            "mean": distribution["val"]["mean"],
            "sd": distribution["val"]["sd"],
        },
        {
            "cohort": "训练确认集",
            "n_sessions": distribution["train_confirmation"]["n"],
            "p10": distribution["train_confirmation"]["p10"],
            "median": distribution["train_confirmation"]["median"],
            "p90": distribution["train_confirmation"]["p90"],
            "mean": distribution["train_confirmation"]["mean"],
            "sd": distribution["train_confirmation"]["sd"],
        },
    ]

    low_energy = [
        {
            "threshold": "<−50 dBFS",
            "val_fraction": audio["val_low_energy_tail"]["fraction_below_minus_50_dbfs"],
            "train_fraction": audio["train_low_energy_tail"]["fraction_below_minus_50_dbfs"],
        },
        {
            "threshold": "<−45 dBFS",
            "val_fraction": audio["val_low_energy_tail"]["fraction_below_minus_45_dbfs"],
            "train_fraction": audio["train_low_energy_tail"]["fraction_below_minus_45_dbfs"],
        },
        {
            "threshold": "<−40 dBFS",
            "val_fraction": audio["val_low_energy_tail"]["fraction_below_minus_40_dbfs"],
            "train_fraction": audio["train_low_energy_tail"]["fraction_below_minus_40_dbfs"],
        },
    ]

    qa_rows = [
        {
            "check": label,
            "status": "通过" if data["qa_checks"][key] else "失败",
        }
        for key, label in (
            ("zero_processing_errors", "两批处理错误与缺失均为 0"),
            ("layer1_exact_both", "两批层1协议逐帧全等"),
            ("unique_sessions_both", "两批会话无重复"),
            ("train_rows_match_roster", "确认结果顺序与冻结 roster 完全一致"),
            ("class_contributions_reconcile", "四类贡献与层3总差额精确对账"),
            ("gold_counts_reconcile", "金标事件计数与正式报告精确对账"),
            ("audio_lengths_verified_during_energy_scan", "能量扫描逐会话核对音频长度"),
        )
    ]

    sources = [
        {
            "id": "postmortem",
            "label": "G0 事后诊断数据",
            "path": "reports/g0_postmortem.json",
        },
        {
            "id": "analysis_code",
            "label": "事后诊断复算脚本",
            "path": "scripts/wp1_g0_postmortem.py",
        },
        {
            "id": "report_sql",
            "label": "报告快照最终查询",
            "path": "scripts/wp1_g0_postmortem_report.sql",
        },
        {
            "id": "val_report",
            "label": "修复后验证集报告",
            "path": "reports/g0_summary.json",
        },
        {
            "id": "train_report",
            "label": "训练侧正式确认报告",
            "path": "reports/g0_train_confirmation.json",
        },
        {
            "id": "gate_derivation",
            "label": "Gate 推导报告",
            "path": "reports/g0_gate_derivation.json",
        },
        {
            "id": "prereg",
            "label": "预注册变更记录",
            "path": "PREREG.md",
        },
    ]

    charts = [
        {
            "id": "gate_comparison",
            "title": "Gate 指标对照",
            "subtitle": "同一阈值 0.40；层3与 P10 在训练确认集发生明显下移",
            "intent": "comparison",
            "question": "修复后两批数据的四项 Gate 指标相差多少？",
            "rationale": "四项指标处于同一 0–1 数值尺度，分组柱形图便于直接比较两批数据。",
            "comparisonContext": {
                "baseline": "修复后验证集",
                "denominator": "各指标各自冻结定义",
                "grain": "语料级指标与会话级 P10",
                "unit": "F1 或分位数",
            },
            "type": "bar",
            "dataset": "gate_metrics",
            "sourceId": "postmortem",
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "指标"},
                "y": {
                    "fields": ["val", "train_confirm"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "数值",
                },
                "tooltip": [
                    {"field": "criterion", "type": "text", "label": "冻结判据"},
                    {"field": "train_margin", "type": "quantitative", "label": "确认集差额"},
                    {"field": "status", "type": "text", "label": "确认状态"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "labels": {"values": "auto"},
            "settings": {"groupMode": "grouped", "sort": "custom", "showValues": True},
            "surface": {
                "surface": "compact",
                "compact": True,
                "showControls": False,
                "viewMode": "visualization",
            },
        },
        {
            "id": "session_distribution",
            "title": "会话 macro-F1 分位数",
            "subtitle": "P10、中位数与 P90；验证集 295 会话，训练确认集 300 会话",
            "intent": "comparison",
            "question": "训练确认集的失败来自少数异常会话，还是整体分布下移？",
            "rationale": "三处分位数同时下移可区分整体分布变化与少数异常，并适配便携式报告的紧凑布局。",
            "comparisonContext": {
                "baseline": "修复后验证集",
                "denominator": "每会话四类 F1 算术均值",
                "grain": "会话",
                "unit": "macro-F1",
            },
            "type": "bar",
            "dataset": "distribution_summary",
            "sourceId": "postmortem",
            "encodings": {
                "x": {"field": "cohort", "type": "nominal", "label": "集合"},
                "y": {
                    "fields": ["p10", "median", "p90"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "会话 macro-F1 分位数",
                },
                "tooltip": [
                    {"field": "n_sessions", "type": "quantitative", "label": "会话数"},
                    {"field": "mean", "type": "quantitative", "label": "均值"},
                    {"field": "sd", "type": "quantitative", "label": "标准差"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "labels": {"values": "auto"},
            "settings": {"groupMode": "grouped", "sort": "custom", "showValues": True},
            "surface": {
                "surface": "compact",
                "compact": True,
                "showControls": False,
                "viewMode": "visualization",
            },
        },
        {
            "id": "energy_relationship",
            "title": "低能量占比与会话 F1",
            "subtitle": "低能量定义为金标语音 80 ms 帧 RMS < −50 dBFS；每点一会话",
            "intent": "relationship",
            "question": "低能量语音尾部与层3表现之间是否存在稳定关联？",
            "rationale": "595 个同粒度会话提供充分散点，双系列显式区分验证集与训练确认集。",
            "comparisonContext": {
                "baseline": "修复后验证集",
                "denominator": "每会话全部金标语音帧",
                "grain": "会话",
                "unit": "占比与 macro-F1",
            },
            "type": "scatter",
            "dataset": "session_rows",
            "sourceId": "postmortem",
            "encodings": {
                "x": {
                    "field": "low_energy_fraction",
                    "type": "quantitative",
                    "format": "percent",
                    "label": "低能量金标语音帧占比",
                },
                "y": {
                    "fields": ["val_macro_f1", "train_macro_f1"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "会话 macro-F1",
                },
                "tooltip": [
                    {"field": "session", "type": "text", "label": "会话"},
                    {"field": "cohort", "type": "nominal", "label": "集合"},
                    {"field": "vad_rate_mean", "type": "quantitative", "label": "金标语音活动率"},
                    {"field": "speech_frame_p10_dbfs", "type": "quantitative", "label": "语音帧能量 P10"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "sort": "spec", "title": "集合"},
            "surface": {
                "surface": "compact",
                "compact": True,
                "showControls": False,
                "viewMode": "visualization",
            },
        },
        {
            "id": "class_comparison",
            "title": "四类事件层3 F1",
            "subtitle": "四类均下降；BOT 降幅最大",
            "intent": "comparison",
            "question": "层3总差距由单一事件类驱动，还是四类共同下滑？",
            "rationale": "四类数量少且处于同一 F1 尺度，分组柱形图可直接检查系统性下滑。",
            "comparisonContext": {
                "baseline": "修复后验证集",
                "denominator": "各事件类的语料级匹配计数",
                "grain": "事件类",
                "unit": "F1",
            },
            "type": "bar",
            "dataset": "class_metrics",
            "sourceId": "postmortem",
            "encodings": {
                "x": {"field": "event_class", "type": "nominal", "label": "事件类"},
                "y": {
                    "fields": ["val_f1", "train_f1"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "F1",
                },
                "tooltip": [
                    {"field": "delta", "type": "quantitative", "label": "训练−验证"},
                    {
                        "field": "macro_gap_contribution",
                        "type": "quantitative",
                        "label": "对 macro 差距的贡献",
                    },
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "labels": {"values": "auto"},
            "settings": {"groupMode": "grouped", "sort": "custom", "showValues": True},
            "surface": {
                "surface": "compact",
                "compact": True,
                "showControls": False,
                "viewMode": "visualization",
            },
        },
    ]

    tables = [
        {
            "id": "gate_table",
            "title": "正式确认的 Gate 对账",
            "subtitle": "差额按训练确认值减冻结下限计算",
            "dataset": "gate_metrics",
            "sourceId": "postmortem",
            "defaultSort": {"field": "metric", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "指标", "type": "text"},
                {"field": "val", "label": "验证集", "format": "number"},
                {"field": "train_confirm", "label": "训练确认集", "format": "number"},
                {"field": "criterion", "label": "冻结判据", "type": "text"},
                {
                    "field": "train_margin",
                    "label": "确认集差额",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                },
                {"field": "status", "label": "状态", "type": "text"},
            ],
        },
        {
            "id": "low_energy_table",
            "title": "金标语音帧的低能量尾部",
            "subtitle": "分母为各集合全部金标语音帧",
            "dataset": "low_energy",
            "sourceId": "postmortem",
            "defaultSort": {"field": "threshold", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "threshold", "label": "帧能量阈值", "type": "text"},
                {"field": "val_fraction", "label": "验证集占比", "format": "percent"},
                {
                    "field": "train_fraction",
                    "label": "训练确认集占比",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "qa_table",
            "title": "数据与计算复核",
            "subtitle": "高影响口径全部通过独立对账",
            "dataset": "qa_checks",
            "sourceId": "postmortem",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "复核项", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
        },
        {
            "id": "class_table",
            "title": "四类事件的 F1 与差距贡献",
            "subtitle": "差额为训练确认集减验证集",
            "dataset": "class_metrics",
            "sourceId": "postmortem",
            "defaultSort": {"field": "delta", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "event_class", "label": "事件类", "type": "text"},
                {"field": "val_f1", "label": "验证 F1", "format": "number"},
                {"field": "train_f1", "label": "确认 F1", "format": "number"},
                {
                    "field": "delta",
                    "label": "F1 差额",
                    "format": "number",
                    "movement": True,
                    "role": "movement",
                },
                {
                    "field": "macro_gap_contribution",
                    "label": "总差距贡献",
                    "format": "number",
                },
            ],
        },
    ]
    # 精确值已保留在图表数据源与正文；便携式窄屏报告不放置宽表，避免全页横向滚动。
    tables = []

    val_mean = distribution["val"]["mean"]
    train_mean = distribution["train_confirmation"]["mean"]
    mean_ci = distribution["train_minus_val_mean_bootstrap"]
    low_energy_ci = audio["train_minus_val_fraction_below_minus_50_dbfs_bootstrap"]
    frame_p10_ci = audio["train_minus_val_speech_frame_p10_dbfs_bootstrap"]
    short_val = composition["short_duration_comparison"]["val"]["mean"]
    short_train = composition["short_duration_comparison"]["train_confirmation"]["mean"]
    roster_ks = split["roster_vs_release_available_train_id_ks"]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# G0 修复后失败诊断", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 技术摘要\n\n"
                "**结论：G0 仍为 `fail`。** 修复本身确实生效，但没有稳定迁移到训练确认集。"
                f"同一验证集上，阈值 0.50→0.40 使语料级 macro-F1 从 "
                f"{_format(repair['corpus_macro_f1_pre'])} 升至 "
                f"{_format(repair['corpus_macro_f1_post'])}，295 个会话中 "
                f"{repair['paired_session_fraction_improved']:.1%} 改善。正式训练确认集仍降至 "
                f"{_format(failure['corpus_macro_f1'])}，并以 "
                f"{_format(failure['corpus_lower_margin'])} 低于层3下界；会话 P10 也低 "
                f"{_format(abs(failure['session_p10_margin']))}。\n\n"
                "主因是跨官方分段的声学与会话结构漂移：训练确认集的低能量语音尾部更重、"
                "金标语音活动更密，造成 VAD 精确率保持很高而召回率明显回落；事件算法随后把"
                "这部分边界误差放大为四类同步下滑。零处理错误、层1全等、roster精确匹配与"
                "逐项对账均通过，因此工程损坏或抽样脚本偏斜缺乏证据。当前 Gate 正确捕获了"
                "跨分段不可迁移性，不能通过事后放宽带宽或再次抽样来改判。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["corpus_card", "p10_card", "layer2_card", "repair_card"],
            "layout": "full",
        },
        {
            "id": "repair_section",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 修复有效，但未跨域迁移\n\n"
                f"阈值调整在同一验证集上的会话级平均提升为 "
                f"{repair['paired_session_mean_delta']['estimate']:+.4f}，95% bootstrap 区间 "
                f"[{repair['paired_session_mean_delta']['ci95_low']:+.4f}, "
                f"{repair['paired_session_mean_delta']['ci95_high']:+.4f}]。双通道 VAD 召回各提升约 "
                "2.18 个百分点，四类 F1 全部提高。这说明首轮故障诊断与修复方向成立。\n\n"
                "训练确认集仍只在层2最低 F1 上留下 +0.0042 的通过余量；层3和尾部护栏同时失败。"
                "因此更准确的表述是：**修复有效，跨域鲁棒性仍失败。**"
            ),
        },
        {"id": "gate_chart_block", "type": "chart", "chartId": "gate_comparison", "layout": "full"},
        {"id": "distribution_chart_block", "type": "chart", "chartId": "session_distribution", "layout": "full"},
        {
            "id": "distribution_section",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 会话分布整体下移\n\n"
                f"会话级 macro-F1 均值从验证集 {_format(val_mean)} 降至训练确认集 "
                f"{_format(train_mean)}；差值为 {mean_ci['estimate']:+.4f}，95% 区间 "
                f"[{mean_ci['ci95_low']:+.4f}, {mean_ci['ci95_high']:+.4f}]。P10 同样下移 "
                f"{distribution['train_minus_val_p10_bootstrap']['estimate']:+.4f}。"
                "这类同时影响中心与尾部的变化不符合“少数坏文件拖累”的模式。\n\n"
                f"验证集 295 个会话均约 300 秒；训练确认集含 160 个约 300 秒会话。即使只比较"
                f"这一时长层，均值仍为 {_format(short_train)} 对 {_format(short_val)}，差距约 "
                f"{short_train - short_val:+.4f}。会话时长具有影响，但并非主因。"
            ),
        },
        {
            "id": "energy_section",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 低能量尾部拖累召回\n\n"
                f"训练确认集中低于 −50 dBFS 的金标语音帧占 "
                f"{audio['train_low_energy_tail']['fraction_below_minus_50_dbfs']:.1%}，验证集为 "
                f"{audio['val_low_energy_tail']['fraction_below_minus_50_dbfs']:.1%}。按会话重采样，"
                f"占比差为 {low_energy_ci['estimate']:+.1%}，95% 区间 "
                f"[{low_energy_ci['ci95_low']:+.1%}, {low_energy_ci['ci95_high']:+.1%}]；金标语音帧"
                f"能量 P10 平均低 {abs(frame_p10_ci['estimate']):.2f} dB。\n\n"
                f"同时，训练确认集 VAD 精确率比验证集略高，但召回率在 ch0/ch1 分别低 "
                f"{abs(layer2['ch0']['recall']['delta_train_minus_val']):.1%}/"
                f"{abs(layer2['ch1']['recall']['delta_train_minus_val']):.1%}。这种“精确率高、召回率低”"
                "的组合与低能量语音漏检一致。会话整体语音 RMS 仅低约 0.52 dB且区间跨 0，"
                "真正变化集中在局部低能量尾部。散点关系属于事后关联证据，不能单独证明因果。"
            ),
        },
        {"id": "energy_chart_block", "type": "chart", "chartId": "energy_relationship", "layout": "full"},
        {
            "id": "class_section",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 四类事件同步下滑\n\n"
                "EOT、HOLD、BOT、BC 的 F1 差额分别为 −0.0739、−0.0798、−0.0874、−0.0701。"
                "BOT 降幅最大，但没有单一类别足以解释整体失败。BC 还呈现不同误差形态："
                "召回率接近验证集，精确率却从 0.5480 降至 0.4011，预测/金标数量比从 0.779 升至 "
                "1.046。单纯继续降低全局 VAD 阈值，很可能进一步增加 BC 假阳性。"
            ),
        },
        {"id": "class_chart_block", "type": "chart", "chartId": "class_comparison", "layout": "full"},
        {
            "id": "split_section",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 官方划分存在分段漂移\n\n"
                "官方 train 为 sw02001–sw04139，val 从 sw04140 开始，test 从 sw04637 开始；"
                "这是连续编号分段。冻结 roster 对 1,986 个发布物可用 train 会话的编号分布没有"
                f"可见偏斜（KS={roster_ks['statistic']:.3f}，描述性 p={roster_ks['p_value']:.3f}），"
                "所以再随机抽一批相同管线的 train 会话，缺乏能够扭转结果的机制依据。"
            ),
        },
        {
            "id": "interpretation_section",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## Gate 的正确解释\n\n"
                "当前四条件 Gate 中，层1证明官方事件语义实现正确；层2确认总体 VAD F1 达到最低线；"
                "层3与 P10 检查修复后验证分布能否迁移到全新官方分段。前两项通过、后两项失败，"
                "最合理的裁决是：**基础保真度通过，可迁移性失败。** 总体 G0 仍必须保持 `fail`。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 下一步建议\n\n"
                "1. **冻结当前裁决。** 在 PREREG 结果区登记本次 `fail`，保留既有哈希与报告；"
                "不得改宽等价带、继续降低阈值或重跑这 300 个会话来寻求通过。\n"
                "2. **把 G0 拆成两个透明状态。** 建议下一版预注册区分 G0-A（协议/最低保真度）"
                "与 G0-B（跨分段事件可迁移性）。当前证据支持 G0-A 通过、G0-B 失败；旧版总体"
                "裁决不追溯改写。\n"
                "3. **停止只调一个全局阈值。** 在已启封数据上做探索性误差分层，比较 Silero"
                "后验、局部能量、语音活动密度与间隙长度；优先考察能量分层校准与时序滞回/短间隙"
                "闭合，目标同时恢复安静语音召回并抑制 BC 假阳性。任何候选参数先登记。\n"
                "4. **建立新的开发/最终确认隔离。** 当前 300 会话已转为失败分析集。若继续修复，"
                "应在读取剩余 train 会话内容前先冻结开发集与最后一次确认集，并按连续编号块、"
                "时长及可预先获得的会话属性做分层；最终确认只运行一次。\n"
                "5. **下游分级推进。** 不依赖 Mimi 解码事件端点的 E1/表示分析可继续；依赖层3"
                "事件分布的 R2/R3 行为端点暂缓主结论，或按预注册方案同时报告金标/多种 VAD"
                "敏感性结果。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "postmortem",
            "layout": "full",
            "body": (
                "## 限制与不确定性\n\n"
                "本报告是看到正式失败后的诊断，所有新增分层只能用于机制解释和下一版设计，"
                "不能用于当前 Gate 重判。bootstrap 区间以会话为抽样单位，描述观察到的两个集合，"
                "不消除连续编号分段带来的选择偏差。低能量尾部与 macro-F1 的相关性较强，"
                "但训练确认集未重新运行逐会话预测 VAD 分解，因此机制结论仍保留关联性限制。"
                "修复前 test 与修复后 train 属于不同集合，历史对照不能替代同会话配对实验。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 待回答问题\n\n"
                "- 低能量帧增加主要来自 Mimi 解码、原始通道串音，还是官方金标 VAD 的语义边界？\n"
                "- 在能量与活动率分层后，官方编号块仍剩余多少事件分布差异？\n"
                "- 新版 G0 的目标应保持严格跨分段可迁移性，还是将最低保真度与可迁移性拆成"
                "两个独立 Gate？这个选择需要在下一轮实现前由研究目标裁决。"
            ),
        },
    ]
    compact_bodies = {
        "summary": (
            "## 技术摘要\n\n"
            "- 裁决仍为 `fail`：层3低 0.0378，P10 低 0.0162。\n"
            "- 修复在 val 有效：macro-F1 提升 0.0415。\n"
            "- 训练低能量语音更多，VAD 召回低 3.7–4.5 点。\n"
            "- 四类均降；零错误、层1与 roster 对账均通过。\n"
            "- 冻结裁决；下一版拆分基础保真度与可迁移性。"
        ),
        "repair_section": (
            "## 修复有效，但未跨域迁移\n\n"
            f"- val 语料 F1：{repair['corpus_macro_f1_pre']:.4f}→{repair['corpus_macro_f1_post']:.4f}。\n"
            f"- 会话均值增量：{repair['paired_session_mean_delta']['estimate']:+.4f}。\n"
            f"- 95% 区间：[{repair['paired_session_mean_delta']['ci95_low']:+.4f}, "
            f"{repair['paired_session_mean_delta']['ci95_high']:+.4f}]。\n"
            f"- 改善会话占比：{repair['paired_session_fraction_improved']:.1%}。\n"
            "- 双通道 VAD 召回各提升约 2.18 个百分点。\n"
            "- 结论：验证域修复成立，跨域鲁棒性仍失败。"
        ),
        "distribution_section": (
            "## 会话分布整体下移\n\n"
            f"- val 均值：{val_mean:.4f}。\n"
            f"- 训练确认均值：{train_mean:.4f}。\n"
            f"- 均值差：{mean_ci['estimate']:+.4f}。\n"
            f"- 95% 区间：[{mean_ci['ci95_low']:+.4f}, {mean_ci['ci95_high']:+.4f}]。\n"
            f"- P10 差：{distribution['train_minus_val_p10_bootstrap']['estimate']:+.4f}。\n"
            f"- 约 300 秒会话均值：{short_train:.4f} 对 {short_val:.4f}。\n"
            "- 时长只能解释一小部分差距。"
        ),
        "energy_section": (
            "## 低能量尾部拖累召回\n\n"
            f"- 低于 −50 dBFS：训练 {audio['train_low_energy_tail']['fraction_below_minus_50_dbfs']:.1%}。\n"
            f"- 低于 −50 dBFS：验证 {audio['val_low_energy_tail']['fraction_below_minus_50_dbfs']:.1%}。\n"
            f"- 会话级占比差：{low_energy_ci['estimate']:+.1%}。\n"
            f"- 95% 区间：[{low_energy_ci['ci95_low']:+.1%}, {low_energy_ci['ci95_high']:+.1%}]。\n"
            f"- 语音帧能量 P10 低 {abs(frame_p10_ci['estimate']):.2f} dB。\n"
            "- 整体语音 RMS 仅低约 0.52 dB。\n"
            "- 精确率略升，召回率明显下降。\n"
            "- 该关系属于事后关联证据。"
        ),
        "class_section": (
            "## 四类事件同步下滑\n\n"
            "- EOT：−0.0739。\n"
            "- HOLD：−0.0798。\n"
            "- BOT：−0.0874。\n"
            "- BC：−0.0701。\n"
            "- BOT 降幅最大，四类都参与总失败。\n"
            "- BC 精确率从 0.5480 降至 0.4011。\n"
            "- BC 预测/金标比从 0.779 升至 1.046。\n"
            "- 继续降低全局阈值可能增加 BC 假阳性。"
        ),
        "split_section": (
            "## 官方划分存在分段漂移\n\n"
            "- train：sw02001–sw04139。\n"
            "- val：从 sw04140 开始。\n"
            "- test：从 sw04637 开始。\n"
            f"- roster 来自 {split['release_available_train_n']:,} 个可用 train 会话。\n"
            f"- roster 编号分布 KS：{roster_ks['statistic']:.3f}。\n"
            f"- 描述性 p 值：{roster_ks['p_value']:.3f}。\n"
            "- 再抽同域样本缺乏扭转结果的机制依据。"
        ),
        "interpretation_section": (
            "## Gate 的正确解释\n\n"
            "- 层1通过：官方事件语义实现正确。\n"
            "- 层2通过：最低 VAD 保真度达线。\n"
            "- 层3失败：事件分布未跨分段迁移。\n"
            "- P10失败：低端会话稳定性不足。\n"
            "- 总体 G0 必须保持 `fail`。\n"
            "- 可概括为：基础保真度通过，可迁移性失败。"
        ),
        "next_steps": (
            "## 下一步建议\n\n"
            "1. 冻结当前裁决，并登记本次 `fail`。\n"
            "2. 禁止改宽带宽、降阈值或重跑 300 会话。\n"
            "3. 下一版拆分 G0-A 与 G0-B。\n"
            "4. G0-A 表示协议与最低保真度。\n"
            "5. G0-B 表示跨分段可迁移性。\n"
            "6. 停止只调一个全局 VAD 阈值。\n"
            "7. 探索能量分层校准与时序滞回。\n"
            "8. 同时约束安静语音召回与 BC 假阳性。\n"
            "9. 先冻结开发集与最后确认集，再读取内容。\n"
            "10. E1 可继续；依赖层3端点的 R2/R3 暂缓。"
        ),
        "limitations": (
            "## 限制与不确定性\n\n"
            "- 本报告属于看到失败后的事后诊断。\n"
            "- 新增分层不能用于当前 Gate 重判。\n"
            "- bootstrap 只描述观察到的两个集合。\n"
            "- 连续编号分段的选择偏差仍然存在。\n"
            "- 未重新运行逐会话预测 VAD 分解。\n"
            "- 历史 test 与新 train 不是配对对照。"
        ),
        "questions": (
            "## 待回答问题\n\n"
            "- 低能量尾部来自解码、串音还是金标边界？\n"
            "- 分层后还剩多少编号块差异？\n"
            "- 下一版应保持严格迁移 Gate 吗？\n"
            "- 或将最低保真度与迁移性分开裁决？"
        ),
    }
    for block in blocks:
        if block["id"] in compact_bodies:
            block["body"] = compact_bodies[block["id"]]
        if block.get("type") == "markdown":
            block["body"] = _add_mobile_breaks(block["body"])
            block.pop("sourceId", None)
    blocks = [
        block
        for block in blocks
        if block.get("type") != "metric-strip"
        and (block.get("type") != "chart" or block.get("chartId") == "gate_comparison")
    ]
    blocks = [
        block
        for block in blocks
        if block["id"]
        in {
            "title",
            "summary",
            "gate_chart_block",
        }
    ]

    cards = [
        {
            "id": "corpus_card",
            "description": "训练确认集层3语料级 macro-F1 及冻结下界差额。",
            "dataset": "headline",
            "sourceId": "postmortem",
            "metrics": [
                {"label": "层3语料 F1", "field": "corpus_macro_f1", "format": "number"},
                {"label": "下界", "field": "corpus_lower_bound", "format": "number"},
                {
                    "label": "差额",
                    "field": "corpus_margin",
                    "format": "number",
                    "signed": True,
                },
            ],
        },
        {
            "id": "p10_card",
            "description": "训练确认集会话级 macro-F1 第 10 百分位及最低线差额。",
            "dataset": "headline",
            "sourceId": "postmortem",
            "metrics": [
                {"label": "会话 P10", "field": "session_p10", "format": "number"},
                {"label": "最低线", "field": "session_p10_min", "format": "number"},
                {
                    "label": "差额",
                    "field": "session_p10_margin",
                    "format": "number",
                    "signed": True,
                },
            ],
        },
        {
            "id": "layer2_card",
            "description": "训练确认集双通道中较低的 VAD F1。",
            "dataset": "headline",
            "sourceId": "postmortem",
            "metrics": [
                {"label": "层2最低 F1", "field": "layer2_min_f1", "format": "number"},
                {"label": "最低线", "field": "layer2_required", "format": "number"},
                {
                    "label": "余量",
                    "field": "layer2_margin",
                    "format": "number",
                    "signed": True,
                },
            ],
        },
        {
            "id": "repair_card",
            "description": "同一验证集阈值 0.50→0.40 后的语料级 macro-F1 增量。",
            "dataset": "headline",
            "sourceId": "postmortem",
            "metrics": [
                {
                    "label": "验证集修复增量",
                    "field": "repair_val_macro_delta",
                    "format": "number",
                    "signed": True,
                },
                {"label": "修复前", "field": "repair_val_macro_pre", "format": "number"},
                {"label": "修复后", "field": "repair_val_macro_post", "format": "number"},
            ],
        },
    ]

    chart_map = [
        {
            "section": "Gate 对照",
            "question": "四项 Gate 指标跨集合变化多少",
            "family": "比较",
            "chart_type": "bar",
            "fields": "metric; val; train_confirm",
            "palette": "blue-orange",
        },
        {
            "section": "会话分布",
            "question": "整体分布下移还是少数异常",
            "family": "分布",
            "chart_type": "bar",
            "fields": "cohort; p10; median; p90",
            "palette": "blue-orange",
        },
        {
            "section": "低能量机制",
            "question": "低能量尾部与会话表现的关系",
            "family": "关系",
            "chart_type": "scatter",
            "fields": "low_energy_fraction; val_macro_f1; train_macro_f1",
            "palette": "blue-orange",
        },
        {
            "section": "类别贡献",
            "question": "差距是否由单一类别驱动",
            "family": "比较",
            "chart_type": "bar",
            "fields": "event_class; val_f1; train_f1",
            "palette": "blue-orange",
        },
    ]

    for item in [*cards, *charts, *tables]:
        item["sourceId"] = "report_sql"

    raw_datasets = {
        "headline": headline,
        "gate_metrics": gate_metrics,
        "class_metrics": class_metrics,
        "session_rows": session_rows,
        "distribution_summary": distribution_summary,
        "low_energy": low_energy,
        "qa_checks": qa_rows,
        "history": data["history"],
        "chart_map": chart_map,
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "G0 修复后失败诊断",
        "description": "解释阈值修复为何在验证集有效、却在训练侧正式确认中仍未通过。",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPORTS_DIR / "g0_postmortem.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "g0_postmortem_artifact.json",
    )
    args = parser.parse_args()
    artifact = build_artifact(_load_json(args.input))
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写出报告清单：{args.output}")


if __name__ == "__main__":
    main()
