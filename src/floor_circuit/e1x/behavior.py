"""E2-lite 行为读出：双通道 VA 掩码 → floor 行为指标（探索性）。

掩码为 dt 栅格布尔序列（与事件管线同源：Silero VAD → IPU 合并 → rasterize）。
onset/offset 合格判据沿 configs/events.yaml 冻结值（onset 前静默 ≥0.4 s、
offset 后静默 ≥0.4 s），复用 events/detect.py 的权威实现。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.events.detect import qualified_offsets, qualified_onsets


def floor_metrics(
    mask_user: np.ndarray,
    mask_agent: np.ndarray,
    dt: float,
    ev_cfg: dict,
    *,
    response_window_s: float,
    latency_max_s: float,
    yield_windows_s: list[float],
) -> dict:
    """单次运行的 floor 行为指标。所有率的分母同时返回，供跨条件配对聚合。"""
    mask_user = np.asarray(mask_user, dtype=bool)
    mask_agent = np.asarray(mask_agent, dtype=bool)
    if mask_user.shape != mask_agent.shape:
        raise ValueError("双通道掩码长度不一致")
    total_s = len(mask_user) * dt
    agent_onsets = qualified_onsets(mask_agent, dt, ev_cfg["onset_pre_silence_s"])
    agent_offsets = qualified_offsets(mask_agent, dt, ev_cfg["offset_post_silence_s"])
    user_offsets = qualified_offsets(mask_user, dt, ev_cfg["offset_post_silence_s"])
    user_onsets = qualified_onsets(mask_user, dt, ev_cfg["onset_pre_silence_s"])

    # 响应延迟：用户合格 offset → 下一个 agent 合格 onset
    latencies: list[float] = []
    responded = 0
    for t_off in user_offsets:
        candidates = [t for t in agent_onsets if t > t_off]
        if not candidates:
            continue
        latency = min(candidates) - t_off
        if latency <= latency_max_s:
            latencies.append(float(latency))
        if latency <= response_window_s:
            responded += 1

    # 抢话率：agent 合格 onset 时用户仍在发声
    interruptions = 0
    for t_on in agent_onsets:
        idx = min(len(mask_user) - 1, round(t_on / dt))
        if mask_user[idx]:
            interruptions += 1

    # 让位率：用户在 agent 发声中 onset → agent 是否在窗口内合格 offset
    incursions = []
    for t_on in user_onsets:
        idx = min(len(mask_agent) - 1, round(t_on / dt))
        if mask_agent[idx]:
            incursions.append(t_on)
    yield_rates = {}
    for window in yield_windows_s:
        if incursions:
            yielded = sum(
                1 for t_on in incursions if any(t_on < t <= t_on + window for t in agent_offsets)
            )
            yield_rates[f"yield_rate_{round(window * 1000)}ms"] = yielded / len(incursions)
        else:
            yield_rates[f"yield_rate_{round(window * 1000)}ms"] = None

    both = mask_user & mask_agent
    user_speech_s = float(mask_user.sum() * dt)
    return {
        "total_s": float(total_s),
        "agent_speech_frac": float(mask_agent.mean()),
        "overlap_frac": float(both.mean()),
        "n_agent_qualified_onsets": len(agent_onsets),
        "n_user_qualified_offsets": len(user_offsets),
        "median_response_latency_s": float(np.median(latencies)) if latencies else None,
        "n_latency_samples": len(latencies),
        "response_rate": (responded / len(user_offsets)) if user_offsets else None,
        "interruption_rate_per_min_user_speech": (
            interruptions / (user_speech_s / 60.0) if user_speech_s > 0 else None
        ),
        "n_user_incursions": len(incursions),
        **yield_rates,
    }


METRIC_KEYS = (
    "agent_speech_frac",
    "overlap_frac",
    "median_response_latency_s",
    "response_rate",
    "interruption_rate_per_min_user_speech",
    "yield_rate_400ms",
    "yield_rate_1000ms",
)


def paired_deltas(
    per_session_metrics: dict[str, dict[str, dict]],
    condition: str,
    baseline: str,
    metric: str,
) -> list[float]:
    """同会话配对差：condition − baseline；任一侧缺失/None 的会话剔除。"""
    deltas: list[float] = []
    for by_condition in per_session_metrics.values():
        a = by_condition.get(condition, {}).get(metric)
        b = by_condition.get(baseline, {}).get(metric)
        if a is None or b is None:
            continue
        deltas.append(float(a) - float(b))
    return deltas


def bootstrap_mean_ci(values: list[float], *, n_boot: int = 2000, seed: int = 20260723) -> dict:
    """会话级配对差的均值 bootstrap CI 与符号计数。"""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": None, "ci95": None, "n_pos": 0, "n_neg": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(int(n_boot), len(array)))
    samples = array[draws].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "ci95": [float(lo), float(hi)],
        "n_pos": int((array > 0).sum()),
        "n_neg": int((array < 0).sum()),
    }
