"""生成 E1-X 与 E2-lite 的综合技术报告。

输入均为仓库内已经复核的小型 JSON 报告；大体积音频、激活和运行缓存不进入 Git。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "reports"
REPORT_TITLE = "E1-X 与 E2-lite 实验综合结论"


def _read_json(name: str) -> dict:
    return json.loads((REPORTS_ROOT / name).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source(
    source_id: str,
    label: str,
    path: str,
    *,
    query: dict | None = None,
) -> dict:
    source = {"id": source_id, "label": label, "path": path}
    if query:
        source["query"] = query
    return source


def _markdown_block(
    block_id: str,
    body: str,
    *,
    source_id: str | None = None,
) -> dict:
    block = {
        "id": block_id,
        "type": "markdown",
        "body": body,
        "layout": "full",
    }
    if source_id:
        block["sourceId"] = source_id
    return block


def _build_markdown() -> str:
    return """# E1-X 与 E2-lite 实验综合结论

## 技术摘要

E1-X 表明，Moshi 的 L29 残差流包含一个跨种子稳定、与高方差主成分错位的话轮完整性判别方向。
该方向在对方语音片段末端前至少 2 秒仍可读，并保留 Mimi 序列声学基线无法解释的判别信息。
E2-lite 随后给出方向级充分性证据：沿主探针方向连续注入时，模型发声占比随剂量单调变化（ρ=0.871），
α=−4 在 20/20 会话中降低发声，α=+4 在 20/20 会话中提高发声；三个随机方向的效应接近零。

这条证据链支持“稳定的 L29 T4 方向参与控制发言倾向”。正式 G2 仍按预注册记为 `fail`，
因为 PCA 有效秩 128 超过门槛 16；E1-X 与 E2-lite 均为探索性支线，尚不裁决正式 G3。

## E1-X：判别信息至少提前两秒存在

先知曲线在共享锚点集合上把读取位置从话轮末端向前平移。
探针 AUC 从末端的 0.8359 平滑下降至提前 2 秒的 0.6816；Mimi 从 0.7694 降至 0.6554。
全部 13 个时间档的探针优势置信区间都高于零，说明额外判别信息贯穿当前两秒窗口。
当前窗口左边界仍已显著，因此只能给出“至少两秒”的下界。

## E1-X：额外信息无法由 Mimi 线性重建

T4@L29 的 AUC 约 0.835，25 步 Mimi-GRU 序列基线约 0.789，
配对优势为 +0.0464 [+0.0378,+0.0559]。用 Mimi 双通道表征线性重建 L29 后，
残差探针仍达到约 0.699。该结果排除了“L29 优势完全来自可线性重建的 Mimi 声学状态”，
但仍允许更长上下文、非线性声学加工和语义结构等多种来源。

方向几何进一步解释了正式有效秩失败：前三种子方向余弦约 0.975–0.980；
前 16 个 PCA 主成分解释约 39% 激活方差，却只承载约 21% 的 T4 方向质量。
有效秩 128 衡量的是判别方向与方差主轴的错位，不能解释为 128 个独立决策变量。

## E2-lite：主探针方向产生双向剂量效应

20 个会话各运行 13 个条件，共 260 次生成。
主方向相对 α=0 基线的发声占比变化为：α=−4 时 −0.1208，α=−2 时 −0.0823，
α=+2 时 +0.1033，α=+4 时 +0.3000。剂量单调性 ρ=0.871（p=7.52×10⁻²⁶）。

效应同时扩展到交互风格：重叠占比的剂量相关为 ρ=0.749，抢话率为 ρ=0.638；
α=+4 使重叠占比增加 0.0587，并使每分钟用户语音对应的抢话率增加约 13.0。
响应延迟相关约为零，说明当前连续注入主要改变发言倾向与侵入性，
尚未形成更精准的响应时机旋钮。

## 对照表明效应具有方向特异性

三个范数匹配随机方向在 ±4 剂量下的发声占比变化绝对值均不超过约 0.0293，
置信区间普遍跨零。差分均值方向的正负注入都降低发声占比，
未出现主探针方向的双向单调关系。主探针方向因此是当前最有力的后续干预候选。

## 范围、设计与指标定义

E1-X 只读冻结的 E1 行域、正式拟合和激活缓存，不访问 `causal_eval`。
E2-lite 使用主评估集保留段中的 20 个会话；每个会话固定输入前 240 秒用户音频，
并以相同采样种子生成各条件的 agent 音频。
每个生成步在 L29 注入 α·s_v·v̂，α 以训练行投影标准差为单位。

行为指标由冻结 Silero VAD、IPU 合并和合格 onset/offset 规则产生；
条件效应采用同会话“条件减基线”配对差，并以会话为单位进行重采样置信区间。
随机方向只有三个，作为数量级参照，不承担正式显著性检验。

## 等价性与稳健性验收

完整行为分析覆盖 260/260 运行，缺失为零。
优化执行在一个完整 13 条件会话上实现 13/13 逐位一致：
文本令牌、16 位 PCM 音频、基线 L28–L31 激活和 3000 次注入计数全部相同；
冷启动加速为 3.45 倍。现有两个参考会话共 24 个条件也已全部逐位通过。

## 局限与解释边界

- E2-lite 证明方向级充分性，尚未证明该方向对行为的必要性。
- 连续全窗注入无法定位真正起作用的事件窗口，也无法区分注意力头与 MLP 的贡献。
- 响应延迟没有剂量效应，当前旋钮更接近“发言倾向/侵入性”控制。
- 语义与声学来源仍需 E4 的受控最小对实验裁决。
- 结果限于 Moshi、L29 和 20 个探索会话，不能直接外推到其他模型、语言或闭环对话。

## 推荐下一步

1. 进入完整 E2：执行事件锁定注入、反向消融与捐体—受体换位，检验必要性并缩小有效时间窗。
2. 恢复无条件组件级定位：对注意力头和 MLP 做预筛与真实换位，判断信号来源的组件结构。
3. 优先扫描 α∈[−2,+2] 的细粒度安全工作区，并同步评价语义内容、音频质量和侵入性代价。
4. 在 PersonaPlex 上复现 L29 邻域方向，并测试同架构方向移植。

## 仍待回答的问题

- 方向效应来自少数门控组件，还是由多个可替代组件共同实现？
- 事件锁定短窗注入能否保留发言控制，同时降低抢话与重叠副作用？
- L29 方向主要编码语义完整性、韵律预测，还是更长程对话结构？
"""


def main() -> None:
    e1x = _read_json("wp_e1x_summary.json")
    e2 = _read_json("wp_e2_lite_summary_optimized.json")
    validation = _read_json("wp_e2_lite_optimization_validation_final_session13.json")

    leadtime_rows = [
        {
            "lead_ms": int(item["lead_ms"]),
            "probe_auc": float(item["probe_auc_mean"]),
            "mimi_auc": float(item["mimi_auc_mean"]),
            "advantage": float(item["advantage"]["advantage"]),
            "ci_low": float(item["advantage"]["ci95"][0]),
            "ci_high": float(item["advantage"]["ci95"][1]),
            "bootstrap_replicates": int(item["advantage"]["n_boot_effective"]),
        }
        for item in e1x["leadtime"]["curve"]
    ]

    dose_rows = [{"alpha": 0, "alpha_label": "0", "agent_speech_delta": 0.0}]
    for condition, alpha in (
        ("probe_a-4", -4),
        ("probe_a-2", -2),
        ("probe_a+2", 2),
        ("probe_a+4", 4),
    ):
        stats = e2["paired_deltas_vs_baseline"][condition]["agent_speech_frac"]
        dose_rows.append(
            {
                "condition": condition,
                "alpha": alpha,
                "alpha_label": f"{alpha:+d}",
                "agent_speech_delta": float(stats["mean"]),
                "ci_low": float(stats["ci95"][0]),
                "ci_high": float(stats["ci95"][1]),
                "n_sessions": int(stats["n"]),
                "n_positive": int(stats["n_pos"]),
                "n_negative": int(stats["n_neg"]),
            }
        )
    dose_rows.sort(key=lambda row: row["alpha"])

    generated_at = datetime.now(UTC).isoformat()
    leadtime_query = {
        "engine": "duckdb",
        "description": "从 E1-X 先知曲线结果展开 13 个时间档。",
        "tables_used": ["reports/wp_e1x_leadtime.json"],
        "metric_definitions": [
            "probe_auc 与 mimi_auc 为三个训练种子的均值。",
            "advantage 为共享锚点上的探针 AUC 减 Mimi AUC；区间按会话重采样。",
        ],
        "sql": (
            "SELECT c.lead_ms,\n"
            "       c.probe_auc_mean AS probe_auc,\n"
            "       c.mimi_auc_mean AS mimi_auc,\n"
            "       c.advantage.advantage AS advantage,\n"
            "       c.advantage.ci95[1] AS ci_low,\n"
            "       c.advantage.ci95[2] AS ci_high,\n"
            "       c.advantage.n_boot_effective AS bootstrap_replicates\n"
            "FROM read_json_auto('reports/wp_e1x_leadtime.json') AS source,\n"
            "     UNNEST(source.curve) AS item(c)\n"
            "ORDER BY c.lead_ms"
        ),
    }
    dose_query = {
        "engine": "duckdb",
        "description": "从 E2-lite 汇总提取主探针方向各剂量相对基线的发声占比配对差。",
        "tables_used": ["reports/wp_e2_lite_summary_optimized.json"],
        "metric_definitions": [
            "agent_speech_delta 为同会话条件减 α=0 基线后的 20 会话均值。",
            "ci_low 与 ci_high 为按会话重采样的 95% 置信区间。",
        ],
        "sql": (
            "WITH source AS (\n"
            "  SELECT paired_deltas_vs_baseline AS d\n"
            "  FROM read_json_auto('reports/wp_e2_lite_summary_optimized.json')\n"
            "), dose AS (\n"
            "  SELECT 'probe_a-4' AS condition, -4 AS alpha, '-4' AS alpha_label,\n"
            '         d."probe_a-4".agent_speech_frac AS m FROM source\n'
            "  UNION ALL\n"
            "  SELECT 'probe_a-2', -2, '-2', d.\"probe_a-2\".agent_speech_frac FROM source\n"
            "  UNION ALL\n"
            "  SELECT 'probe_a+2', 2, '+2', d.\"probe_a+2\".agent_speech_frac FROM source\n"
            "  UNION ALL\n"
            "  SELECT 'probe_a+4', 4, '+4', d.\"probe_a+4\".agent_speech_frac FROM source\n"
            ")\n"
            "SELECT condition, alpha, alpha_label, m.mean AS agent_speech_delta,\n"
            "       m.ci95[1] AS ci_low, m.ci95[2] AS ci_high,\n"
            "       m.n AS n_sessions, m.n_pos AS n_positive, m.n_neg AS n_negative\n"
            "FROM dose\n"
            "UNION ALL SELECT NULL, 0, '0', 0.0, NULL, NULL, NULL, NULL, NULL\n"
            "ORDER BY alpha"
        ),
    }
    sources = [
        _source("e1x_summary", "E1-X 探索套件机器可读汇总", "reports/wp_e1x_summary.json"),
        _source(
            "e1x_leadtime",
            "E1-X 先知曲线机器可读结果",
            "reports/wp_e1x_leadtime.json",
            query=leadtime_query,
        ),
        _source(
            "e2_summary",
            "E2-lite 完整行为汇总",
            "reports/wp_e2_lite_summary_optimized.json",
        ),
        _source(
            "e2_dose",
            "E2-lite 主探针方向剂量效应",
            "reports/wp_e2_lite_summary_optimized.json",
            query=dose_query,
        ),
        _source(
            "e2_validation",
            "E2-lite 优化版逐位等价验收",
            "reports/wp_e2_lite_optimization_validation_final_session13.json",
        ),
        _source("prereg", "预注册与探索性偏离登记", "PREREG.md"),
    ]

    charts = [
        {
            "id": "leadtime_curve",
            "title": "T4 先知曲线",
            "subtitle": "共享锚点集合；13 个时间档，三种子均值",
            "intent": "trend",
            "question": "L29 的 T4 判别信息在话轮末端前多早仍可读？",
            "rationale": "13 个有序时间档适合用双序列折线比较探针与 Mimi 的衰减形状。",
            "type": "line",
            "dataset": "leadtime",
            "sourceId": "e1x_leadtime",
            "encodings": {
                "x": {
                    "field": "lead_ms",
                    "type": "quantitative",
                    "label": "相对锚点提前量（毫秒）",
                },
                "y": {
                    "fields": ["probe_auc", "mimi_auc"],
                    "type": "quantitative",
                    "format": "number",
                    "label": "AUC",
                },
                "tooltip": [
                    {"field": "advantage", "type": "quantitative", "label": "探针优势"},
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI 下界"},
                    {"field": "ci_high", "type": "quantitative", "label": "95% CI 上界"},
                ],
            },
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "settings": {"showValues": False},
        },
        {
            "id": "dose_speech",
            "title": "L29 主方向剂量与发声占比变化",
            "subtitle": "20 个会话内配对差；α=0 为同路径基线",
            "intent": "comparison",
            "question": "主探针方向是否以剂量单调方式改变模型发声倾向？",
            "rationale": "五个预设离散剂量使用带零基线的柱形图，直接呈现方向与幅度。",
            "type": "bar",
            "dataset": "e2_dose",
            "sourceId": "e2_dose",
            "encodings": {
                "x": {
                    "field": "alpha_label",
                    "type": "nominal",
                    "label": "注入剂量 α",
                },
                "y": {
                    "field": "agent_speech_delta",
                    "type": "quantitative",
                    "format": "number",
                    "label": "发声占比配对差",
                },
                "tooltip": [
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI 下界"},
                    {"field": "ci_high", "type": "quantitative", "label": "95% CI 上界"},
                    {"field": "n_sessions", "type": "quantitative", "label": "会话数"},
                    {"field": "n_positive", "type": "quantitative", "label": "正向会话"},
                    {"field": "n_negative", "type": "quantitative", "label": "负向会话"},
                ],
            },
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"groupMode": "grouped", "sort": "custom", "showValues": True},
        },
    ]

    blocks = [
        _markdown_block("title", f"# {REPORT_TITLE}"),
        _markdown_block(
            "technical_summary",
            "## 技术摘要\n\n"
            "**L29 的稳定 T4 方向同时获得提前可读与方向级因果证据。** "
            "E1-X 显示该方向至少提前 2 秒可读，并保留 Mimi 序列基线无法解释的信息；"
            "E2-lite 显示发声占比随注入剂量单调变化（ρ=0.871），"
            "α=−4 与 α=+4 分别在 20/20 会话中产生相反效应，随机方向接近零。"
            "正式 G2 仍为 `fail`，正式 G3 尚未裁决。",
        ),
        _markdown_block(
            "e1_lead_finding",
            "## 判别信息至少提前两秒存在\n\n"
            "探针 AUC 从末端的 0.8359 平滑下降至提前 2 秒的 0.6816；"
            "Mimi 从 0.7694 降至 0.6554。全部 13 个时间档的探针优势置信区间高于零。"
            "当前左边界仍显著，因此“两秒”是下界。",
            source_id="e1x_summary",
        ),
        {"id": "leadtime_chart_block", "type": "chart", "chartId": "leadtime_curve", "layout": "full"},
        _markdown_block(
            "e1_decompose_finding",
            "## 额外信息无法由 Mimi 线性重建\n\n"
            "L29 相对 25 步 Mimi-GRU 的 AUC 优势为 +0.0464 [+0.0378,+0.0559]；"
            "减去 Mimi 可线性重建部分后，残差探针仍约为 0.699。"
            "前 16 个 PCA 主成分只承载约 21% 的 T4 方向质量，解释了有效秩 128 的错位形态。",
            source_id="e1x_summary",
        ),
        _markdown_block(
            "e2_dose_finding",
            "## 主探针方向产生双向剂量效应\n\n"
            "相对 α=0，发声占比在 α=−4、−2、+2、+4 下分别变化 "
            "−0.1208、−0.0823、+0.1033、+0.3000，剂量单调性 "
            "ρ=0.871（p=7.52×10⁻²⁶）。重叠与抢话也随 α 增大，响应延迟没有单调变化。",
            source_id="e2_summary",
        ),
        {"id": "dose_chart_block", "type": "chart", "chartId": "dose_speech", "layout": "full"},
        _markdown_block(
            "control_finding",
            "## 随机对照支持方向特异性\n\n"
            "三个随机方向在 ±4 下的发声占比变化绝对值均不超过约 0.0293，"
            "远小于主方向两端的 0.1208 与 0.3000。差分均值方向正负注入都降低发声，"
            "没有形成主方向的双向单调关系。",
            source_id="e2_summary",
        ),
        _markdown_block(
            "scope_design",
            "## 设计范围与指标\n\n"
            "E2-lite 使用 20 个会话、13 个条件和每会话 240 秒固定用户流，共 260 次生成。"
            "每步在 L29 注入 α·s_v·v̂；效应采用同会话条件减基线的配对差，"
            "置信区间按会话重采样。探索支线未读取 `causal_eval`。",
            source_id="prereg",
        ),
        _markdown_block(
            "validation",
            "## 优化执行保持逐位等价\n\n"
            f"完整分析覆盖 {e2['n_runs_analyzed']}/260，缺失 {e2['n_runs_missing']}。"
            f"完整 13 条件会话中 {validation['exact_runs']}/{validation['requested_runs']} "
            f"逐位一致，冷启动加速 {validation['cold_speedup']:.2f} 倍。"
            "文本、PCM、基线 L28–L31 激活和注入计数均通过。",
            source_id="e2_validation",
        ),
        _markdown_block(
            "limitations",
            "## 结论边界\n\n"
            "- 当前证据支持方向级充分性，尚未检验必要性。\n"
            "- 连续全窗注入无法定位有效事件窗口或具体注意力头/MLP。\n"
            "- 结果限于 Moshi、L29 和 20 个探索会话；正式 G3 保持未裁决。\n"
            "- 语义与声学来源仍需 E4 的受控最小对实验。",
        ),
        _markdown_block(
            "next_steps",
            "## 下一步应转向必要性、时间窗与组件定位\n\n"
            "1. 执行事件锁定注入、反向消融和捐体—受体换位。\n"
            "2. 对注意力头与 MLP 做预筛和真实换位。\n"
            "3. 细扫 α∈[−2,+2]，同步测量语义、音频质量与侵入性代价。\n"
            "4. 在 PersonaPlex 上复现并测试同架构方向移植。",
        ),
        _markdown_block(
            "further_questions",
            "## 仍待回答\n\n"
            "- 哪些组件承载并读取该方向？\n"
            "- 短窗注入能否保留控制力并降低抢话副作用？\n"
            "- 该方向主要依赖语义完整性、韵律预测，还是长程对话结构？",
        ),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "Moshi 话语权方向的探索性定位、声学分解与方向级因果试点。",
            "generatedAt": generated_at,
            "cards": [],
            "charts": charts,
            "tables": [],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"leadtime": leadtime_rows, "e2_dose": dose_rows},
        },
        "sources": sources,
        "package_info": {
            "report_kind": "technical",
            "chart_map": [
                {
                    "section": "E1-X 先知曲线",
                    "family": "trend",
                    "type": "line",
                    "fields": ["lead_ms", "probe_auc", "mimi_auc"],
                    "claim": "探针优势贯穿两秒窗口",
                    "palette": "blue-orange",
                },
                {
                    "section": "E2-lite 剂量反应",
                    "family": "comparison",
                    "type": "bar",
                    "fields": ["alpha_label", "agent_speech_delta"],
                    "claim": "发声占比随 α 双向单调变化",
                    "palette": "blue",
                },
            ],
        },
    }

    (REPORTS_ROOT / "e1x_e2_lite_综合结论.md").write_text(
        _build_markdown(),
        encoding="utf-8",
    )
    _write_json(REPORTS_ROOT / "e1x_e2_lite_综合结论.artifact.json", artifact)
    print("已生成 E1-X 与 E2-lite 综合结论 Markdown 和分析工件")


if __name__ == "__main__":
    main()
