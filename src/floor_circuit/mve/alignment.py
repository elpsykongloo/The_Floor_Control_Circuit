"""MVE 表征时间对齐（PREREG 变更记录 #7，2026-07-18）。

统一约定：**标签行 step s 表示"决策发生在 s·τ 时刻，只允许观测 ≤ s·τ 的音频"**。
T1 的 δ=240 ms 冻结实现为 y[s]=onset 落在步 s+3（即 onset ∈ [(s+3)τ, (s+4)τ)），
相对观测截止 s·τ 的前瞻窗为半开区间 **[240, 320) ms**。

各表征在自身行索引 i 上的观测截止时刻（撤回前的旧实现直接用 rep[s]，导致基线
比主探针多看约一帧未来音频，见 PREREG #7）：

| 表征 | rep[i] 观测截止 | step s 应读的行 |
| --- | --- | --- |
| Moshi acts（首位插入初始 token 的 teacher-forced 复放） | i·τ | i = s |
| Mimi 连续潜表征（流式编码，latent[i] 编码帧 i） | (i+1)·τ | i = s − 1 |
| hazard 特征（由 T5 状态 0..i 构造，状态 i 覆盖到 (i+1)·τ） | (i+1)·τ | i = s − 1 |
| 声学帧特征 / GRU 窗口（窗尾帧 i 覆盖到 (i+1)·τ） | (i+1)·τ | i = s − 1 |

step 0 对基线不存在合法行（且 acts[0] 是纯初始 token 状态、未观测任何音频），
因此全部表征统一剔除 step < MIN_ELIGIBLE_STEP 的标签行，保证四路使用完全相同的
标签行集合（preflight 的基线对齐校验依赖这一点）。
"""

from __future__ import annotations

import numpy as np

# rep[i] 的观测截止 = (i + offset)·τ
FEATURE_OBSERVED_THROUGH_OFFSET: dict[str, int] = {
    "acts": 0,
    "mimi": 1,
    "hazard": 1,
    "acoustic": 1,
}

# 所有表征共同的最小合法标签步（保证 step − offset ≥ 0 且剔除纯初始 token 状态）
MIN_ELIGIBLE_STEP = 1

# 供 runner manifest 与 preflight 双向核验的机器可读声明
RUNNER_TIME_ALIGNMENT = {
    "initial_token_position": 0,
    "acts_observed_through_offset_steps": 0,
}


def feature_row_indices(feature: str, steps: np.ndarray) -> np.ndarray:
    """标签步号 → 该表征应读取的行索引；步号低于 MIN_ELIGIBLE_STEP 直接硬失败。"""
    if feature not in FEATURE_OBSERVED_THROUGH_OFFSET:
        raise ValueError(f"未知表征类型：{feature}")
    steps = np.asarray(steps, dtype=np.int64)
    if steps.ndim != 1:
        raise ValueError("steps 必须是一维数组")
    if len(steps) and int(steps.min()) < MIN_ELIGIBLE_STEP:
        raise ValueError(
            f"{feature} 收到步号 {int(steps.min())} < MIN_ELIGIBLE_STEP={MIN_ELIGIBLE_STEP}；"
            "标签行必须先经 min_step 过滤"
        )
    return steps - FEATURE_OBSERVED_THROUGH_OFFSET[feature]
