"""E2-lite 二次分析（PREREG #40(a)）：语境分解、SDT 判别/判据、文本质量与分布距离。

全部函数只依赖 dt 栅格布尔掩码与整数 token 序列，纯 numpy、可测。
合格 onset/offset 沿 events/detect.py 权威实现（冻结判据）。
"""

from __future__ import annotations

import math

import numpy as np

from floor_circuit.events.detect import qualified_offsets, qualified_onsets
from floor_circuit.events.vad import mask_to_segments


def onset_context_split(
    mask_user: np.ndarray,
    mask_agent: np.ndarray,
    dt: float,
    ev_cfg: dict,
    *,
    respond_window_s: float,
) -> dict:
    """把 agent 合格 onset 按语境三分：合宜接话 / 用户话中闯入 / 静默自起。

    respond = onset 落在某个用户合格 offset 后 (t, t+W]；user_speech = onset 时刻
    用户掩码为真；gap = 其余。分母（机会数）一并返回：用户合格 offset 数、
    用户发声分钟数、双方皆静默分钟数。
    """
    mask_user = np.asarray(mask_user, dtype=bool)
    mask_agent = np.asarray(mask_agent, dtype=bool)
    if mask_user.shape != mask_agent.shape:
        raise ValueError("双通道掩码长度不一致")
    agent_onsets = qualified_onsets(mask_agent, dt, ev_cfg["onset_pre_silence_s"])
    user_offsets = qualified_offsets(mask_user, dt, ev_cfg["offset_post_silence_s"])
    n_respond = 0
    n_during_user = 0
    n_gap = 0
    for t_on in agent_onsets:
        index = min(len(mask_user) - 1, round(t_on / dt))
        if mask_user[index]:
            n_during_user += 1
        elif any(t_off < t_on <= t_off + respond_window_s for t_off in user_offsets):
            n_respond += 1
        else:
            n_gap += 1
    user_speech_min = float(mask_user.sum() * dt / 60.0)
    silence_min = float((~mask_user & ~mask_agent).sum() * dt / 60.0)
    total = len(agent_onsets)
    return {
        "n_agent_onsets": total,
        "n_respond": n_respond,
        "n_during_user": n_during_user,
        "n_gap": n_gap,
        "respond_share": (n_respond / total) if total else None,
        "during_user_share": (n_during_user / total) if total else None,
        "n_user_offsets": len(user_offsets),
        "during_user_rate_per_min": (n_during_user / user_speech_min) if user_speech_min > 0 else None,
        "gap_rate_per_min": (n_gap / silence_min) if silence_min > 0 else None,
        "user_speech_min": user_speech_min,
        "silence_min": silence_min,
    }


def _z(p: float) -> float:
    """标准正态分位数（Acklam 有理逼近；|误差|<1.2e-8，SDT 用途足够）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("z 变换要求 0<p<1")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= 1 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def sdt_decision_stats(
    mask_user: np.ndarray,
    mask_agent: np.ndarray,
    dt: float,
    ev_cfg: dict,
    *,
    window_s: float,
    noise_stride_s: float,
    noise_guard_s: float,
) -> dict:
    """接话决策的信号检测分解：d′（时机分辨力）与 criterion c（发言倾向）。

    信号试次 = 用户合格 offset（话轮让出点）：W 秒内 agent 合格 onset → 命中。
    噪声试次 = 用户段内部、距该段末 ≥ noise_guard_s 的等距探针点（话轮明确未
    结束）：W 秒内 agent onset → 误报。命中/误报率用 (k+0.5)/(n+1) 校正防 0/1。
    """
    mask_user = np.asarray(mask_user, dtype=bool)
    mask_agent = np.asarray(mask_agent, dtype=bool)
    agent_onsets = qualified_onsets(mask_agent, dt, ev_cfg["onset_pre_silence_s"])
    user_offsets = qualified_offsets(mask_user, dt, ev_cfg["offset_post_silence_s"])

    def onset_within(t0: float) -> bool:
        return any(t0 < t <= t0 + window_s for t in agent_onsets)

    hits = sum(1 for t_off in user_offsets if onset_within(t_off))
    n_signal = len(user_offsets)

    noise_points: list[float] = []
    for seg in mask_to_segments(mask_user, dt):
        t = seg.start + noise_stride_s
        while t <= seg.end - noise_guard_s:
            noise_points.append(t)
            t += noise_stride_s
    false_alarms = sum(1 for t0 in noise_points if onset_within(t0))
    n_noise = len(noise_points)

    if n_signal == 0 or n_noise == 0:
        return {
            "n_signal": n_signal,
            "n_noise": n_noise,
            "hits": hits,
            "false_alarms": false_alarms,
            "hit_rate": None,
            "fa_rate": None,
            "d_prime": None,
            "criterion_c": None,
        }
    hit_rate = (hits + 0.5) / (n_signal + 1)
    fa_rate = (false_alarms + 0.5) / (n_noise + 1)
    z_hit = _z(hit_rate)
    z_fa = _z(fa_rate)
    return {
        "n_signal": n_signal,
        "n_noise": n_noise,
        "hits": hits,
        "false_alarms": false_alarms,
        "hit_rate": float(hit_rate),
        "fa_rate": float(fa_rate),
        "d_prime": float(z_hit - z_fa),
        "criterion_c": float(-0.5 * (z_hit + z_fa)),
    }


# ---------------------------------------------------------------------------
# 文本 token 质量
# ---------------------------------------------------------------------------


def token_stats(tokens: np.ndarray, pad_id: int, *, top_n: int = 5) -> dict:
    """PAD 占比、全序列熵（bit）、非 PAD 连续段长与 top-N token 直方图。"""
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.size == 0:
        raise ValueError("token 序列为空")
    values, counts = np.unique(tokens, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    pad_frac = float((tokens == int(pad_id)).mean())
    non_pad = tokens != int(pad_id)
    runs: list[int] = []
    run = 0
    for flag in non_pad:
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    order = np.argsort(counts)[::-1][:top_n]
    return {
        "n_tokens": tokens.size,
        "pad_id_assumed": int(pad_id),
        "pad_frac": pad_frac,
        "entropy_bits": entropy,
        "n_unique": len(values),
        "mean_nonpad_run": float(np.mean(runs)) if runs else 0.0,
        "max_nonpad_run": int(max(runs)) if runs else 0,
        "top_tokens": {int(values[i]): int(counts[i]) for i in order},
    }


# ---------------------------------------------------------------------------
# 交接间隙 / 重叠时长分布与 JSD
# ---------------------------------------------------------------------------


def exchange_gaps(
    mask_from: np.ndarray,
    mask_to: np.ndarray,
    dt: float,
    ev_cfg: dict,
    *,
    max_gap_s: float,
) -> list[float]:
    """话轮交接间隙：mask_from 的合格 offset → mask_to 的下一个合格 onset。"""
    offsets = qualified_offsets(np.asarray(mask_from, dtype=bool), dt, ev_cfg["offset_post_silence_s"])
    onsets = qualified_onsets(np.asarray(mask_to, dtype=bool), dt, ev_cfg["onset_pre_silence_s"])
    gaps: list[float] = []
    for t_off in offsets:
        candidates = [t for t in onsets if t > t_off]
        if candidates and (candidates[0] - t_off) <= max_gap_s:
            gaps.append(float(candidates[0] - t_off))
    return gaps


def overlap_durations(mask_a: np.ndarray, mask_b: np.ndarray, dt: float) -> list[float]:
    both = np.asarray(mask_a, dtype=bool) & np.asarray(mask_b, dtype=bool)
    return [float(seg.end - seg.start) for seg in mask_to_segments(both, dt)]


def histogram_counts(values: list[float], bin_edges: list[float]) -> np.ndarray:
    """固定右开区间直方图；≥ 最后边界的值入末桶（保证计数守恒）。"""
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("直方图边界必须严格递增且至少两点")
    clipped = np.clip(
        np.asarray(values, dtype=np.float64), edges[0], np.nextafter(edges[-1], -np.inf)
    )
    counts, _ = np.histogram(clipped, bins=edges)
    return counts.astype(np.int64)


def jensen_shannon_divergence(counts_p: np.ndarray, counts_q: np.ndarray) -> float | None:
    """以 2 为底的 JSD ∈ [0,1]；任一侧总数为 0 返回 None。"""
    p = np.asarray(counts_p, dtype=np.float64)
    q = np.asarray(counts_q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("JSD 两侧桶数不一致")
    if p.sum() == 0 or q.sum() == 0:
        return None
    p = p / p.sum()
    q = q / q.sum()
    mixture = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * kl(p, mixture) + 0.5 * kl(q, mixture))
