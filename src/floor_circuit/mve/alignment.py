"""MVE 表征时间对齐（PREREG 变更记录 #7 + #8，2026-07-18）。

锚定（用户 2026-07-18 依 LMGen 源码裁定，变更记录 #8）：**标签行 step s 对应
"在线系统刚接收完对方音频帧 s 的决策状态"，观测截止 = (s+1)·τ**。

依据（本机 Moshi venv 实测，moshi/models/lm.py `LMGen._step`）：在 offset p 先把
用户 token 写入 p+delay，再读取位置 p 送入 transformer；Moshiko 对方流 delays =
[0,1,...,1]，故位置 p 已携带对方 q0 帧 p（q1–q7 为帧 p−1）；官方 run_inference
对首帧调用两次以覆盖 initial。R1 复放的激活数组首位保留 initial（acts[0]），
因此**在线接收完对方帧 s 后的等价状态是 acts[s+1]**，而非 acts[s]。

各表征在自身行索引 i 上的观测截止（物理事实，不随锚定变化）：

| 表征 | rep[i] 观测截止 | step s 应读的行 |
| --- | --- | --- |
| Moshi acts（首位 initial 的 teacher-forced 复放） | i·τ（q0 语义流口径） | i = s + 1 |
| Mimi 连续潜表征（流式编码，latent[i] 编码帧 i） | (i+1)·τ | i = s |
| hazard 特征（由 T5 状态 0..i 构造，状态 i 覆盖到 (i+1)·τ） | (i+1)·τ | i = s |
| 声学帧特征 / GRU 窗口（窗尾帧 i 覆盖到 (i+1)·τ） | (i+1)·τ | i = s |

acts[0]（纯 initial、未观测任何音频）永不使用；末标签步 n_steps−1 因无对应
acts[n_steps] 而丢弃——可用标签步为 0..n_steps−2（`usable_label_steps`）。
细码本注记：acts[s+1] 的 q1–q7 只到帧 s−1（≤ s·τ），这是 Moshi 自身延迟架构的
物理事实，两种锚定下同样存在，不构成表征间不公平。

δ 注记（诚实登记）：T1 的 δ=240 ms 仍为标签步索引整 3 步（y[s] = onset 落在步
s+3，即 onset ∈ [(s+3)τ, (s+4)τ)）；相对观测截止 (s+1)·τ 的**净前瞻为
[160, 240) ms（2–3 个时钟步）**。H1 "≥3 时钟步前瞻"的步索引读法满足，
净毫秒读法见 PREREG #8 登记。
"""

from __future__ import annotations

import numpy as np

# 标签步 s 的观测截止 = (s + LABEL_STEP_OBSERVED_THROUGH_OFFSET)·τ（#8 锚定）
LABEL_STEP_OBSERVED_THROUGH_OFFSET = 1

# rep[i] 的观测截止 = (i + offset)·τ（物理事实）
FEATURE_OBSERVED_THROUGH_OFFSET: dict[str, int] = {
    "acts": 0,
    "mimi": 1,
    "hazard": 1,
    "acoustic": 1,
}

# 最小合法标签步（#8 锚定下基线行 = s ≥ 0，step 0 合法）
MIN_ELIGIBLE_STEP = 0

# 供 runner manifest 与 preflight 双向核验的机器可读声明（物理事实，#8 不变更）
RUNNER_TIME_ALIGNMENT = {
    "initial_token_position": 0,
    "acts_observed_through_offset_steps": 0,
}

# 供 summary.protocol 与独立审计比对的分析侧锚定声明
ANALYSIS_TIME_ALIGNMENT = {
    **RUNNER_TIME_ALIGNMENT,
    "label_step_observed_through_offset_steps": LABEL_STEP_OBSERVED_THROUGH_OFFSET,
    "acts_row_for_step": "s+1",
    "baseline_row_for_step": "s",
    "min_eligible_step": MIN_ELIGIBLE_STEP,
    "last_label_step_dropped": True,
}


def usable_label_steps(n_steps: int) -> int:
    """可用标签步数：步 0..n_steps−2（acts 需要行 s+1 ≤ n_steps−1）。"""
    return max(int(n_steps) - 1, 0)


def feature_row_indices(feature: str, steps: np.ndarray) -> np.ndarray:
    """标签步号 → 该表征应读取的行索引（观测截止统一为 (s+1)·τ）。

    acts → s+1；mimi/hazard/acoustic → s。步号越下界直接硬失败；上界
    （s ≤ n_steps−2）由调用方以 ``usable_label_steps`` 截断后由数组校验兜底。
    """
    if feature not in FEATURE_OBSERVED_THROUGH_OFFSET:
        raise ValueError(f"未知表征类型：{feature}")
    steps = np.asarray(steps, dtype=np.int64)
    if steps.ndim != 1:
        raise ValueError("steps 必须是一维数组")
    if len(steps) and int(steps.min()) < MIN_ELIGIBLE_STEP:
        raise ValueError(
            f"{feature} 收到步号 {int(steps.min())} < MIN_ELIGIBLE_STEP={MIN_ELIGIBLE_STEP}"
        )
    offset = LABEL_STEP_OBSERVED_THROUGH_OFFSET - FEATURE_OBSERVED_THROUGH_OFFSET[feature]
    return steps + offset
