"""G0 冻结门槛评估（PREREG 变更记录 #9，用户 2026-07-18 裁决）。

四条件（configs/events.yaml g0.gate，确认集 = test 除目录序前 20 探索集外的 118 会话）：
  层1 协议全等 macro-F1 = 1.000（且逐类 F1 = 1.000）；
  层2 双通道 VAD F1 均 ≥ layer2_vad_f1_min；
  层3 语料级 macro-F1 落在等价带 layer3_corpus_band（方案 2：低出带 = fail，
      **高出带 = red_flag_investigate**——暂停 Gate、登记调查，不自动通过）；
  层3 会话级 macro-F1 的 P10 ≥ layer3_session_p10_min（方案 4 尾部护栏；
      分位估计冻结为 numpy percentile method="linear"）。
"""

from __future__ import annotations

import numpy as np


def evaluate_g0_gate(
    *,
    layer1_macro_f1: float,
    layer1_per_class_f1: dict[str, float],
    layer2_f1_by_channel: dict[str, float],
    layer3_corpus_macro_f1: float,
    layer3_session_macro_f1s: list[float],
    gate_cfg: dict,
) -> dict:
    """返回逐条件结果与总裁决 verdict ∈ {pass, fail, red_flag_investigate}。"""

    band_lo, band_hi = (float(value) for value in gate_cfg["layer3_corpus_band"])
    if not band_lo < band_hi:
        raise ValueError(f"等价带非法：[{band_lo}, {band_hi}]")
    if not layer3_session_macro_f1s:
        raise ValueError("会话级 macro-F1 列表为空，无法评估 P10")

    layer1_pass = float(layer1_macro_f1) == float(gate_cfg["layer1_exact_macro_f1"]) and all(
        float(value) == 1.0 for value in layer1_per_class_f1.values()
    )
    layer2_min = min(float(value) for value in layer2_f1_by_channel.values())
    layer2_pass = layer2_min >= float(gate_cfg["layer2_vad_f1_min"])

    corpus = float(layer3_corpus_macro_f1)
    if corpus < band_lo:
        band_status = "below_band"
    elif corpus > band_hi:
        band_status = "above_band"
    else:
        band_status = "in_band"

    p10 = float(
        np.percentile(
            np.asarray(layer3_session_macro_f1s, dtype=np.float64),
            10,
            method="linear",
        )
    )
    p10_pass = p10 >= float(gate_cfg["layer3_session_p10_min"])

    hard_failures = []
    if not layer1_pass:
        hard_failures.append("layer1_exact")
    if not layer2_pass:
        hard_failures.append("layer2_vad_f1")
    if band_status == "below_band":
        hard_failures.append("layer3_below_band")
    if not p10_pass:
        hard_failures.append("layer3_session_p10")

    if hard_failures:
        verdict = "fail"
    elif band_status == "above_band":
        # 方案 2 上界红旗：等价带的职能是一致性检验，异常偏高同样需要调查
        verdict = "red_flag_investigate"
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "hard_failures": hard_failures,
        "conditions": {
            "layer1_exact": {
                "macro_f1": float(layer1_macro_f1),
                "required": float(gate_cfg["layer1_exact_macro_f1"]),
                "passed": layer1_pass,
            },
            "layer2_vad_f1": {
                "by_channel": {key: float(value) for key, value in layer2_f1_by_channel.items()},
                "min": layer2_min,
                "required_min": float(gate_cfg["layer2_vad_f1_min"]),
                "passed": layer2_pass,
            },
            "layer3_corpus_band": {
                "corpus_macro_f1": corpus,
                "band": [band_lo, band_hi],
                "status": band_status,
            },
            "layer3_session_p10": {
                "p10": p10,
                "required_min": float(gate_cfg["layer3_session_p10_min"]),
                "n_sessions": len(layer3_session_macro_f1s),
                "percentile_method": "linear",
                "passed": p10_pass,
            },
        },
    }
