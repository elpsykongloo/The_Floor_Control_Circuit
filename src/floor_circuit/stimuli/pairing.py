"""S1 拆臂（PREREG 变更记录 #2，修改版已批准，2026-07-17）：

- **S1-A 确证臂**：跨条目配对 complete_i vs incomplete_j（i ≠ j），要求总时长与
  语音活动时长均在容差内匹配——消除时长/能量混淆，可作 H4/G4 确证证据；
- **S1-B 探索臂**：原有真实前缀对（同条目 complete vs incomplete），保留方向性
  时长比、韵律指标与人工抽听，仅作探索性分析，不单独作为确证证据。

本模块实现 S1-A 的一维贪心最优配对（排序后最早可行匹配，一维区间二部图上最优）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StimulusClip:
    id: str
    duration_s: float
    speech_s: float | None = None  # VAD 语音活动总时长；None 时跳过该维过滤


def _rel_diff_pct(a: float, b: float) -> float:
    return abs(a - b) / max(a, b, 1e-9) * 100.0


def greedy_duration_pairing(
    completes: list[StimulusClip],
    incompletes: list[StimulusClip],
    tol_pct: float,
    forbid_same_id: bool = True,
) -> tuple[list[dict], dict]:
    """S1-A 配对：总时长相对差 ≤ tol_pct 的最大一对一匹配（排序贪心），
    再按语音活动时长同容差过滤（双方都有 speech_s 时）。

    forbid_same_id：禁止同条目自配（同条目对属 S1-B 前缀臂）。
    返回 (pairs, stats)；pairs 行含两侧 id、双维差异百分比。
    """
    cs = sorted(completes, key=lambda c: c.duration_s)
    used = [False] * len(cs)
    pairs: list[dict] = []
    n_speech_dropped = 0
    lo = 0
    for inc in sorted(incompletes, key=lambda c: c.duration_s):
        while lo < len(cs) and (
            used[lo]
            or (cs[lo].duration_s < inc.duration_s and _rel_diff_pct(cs[lo].duration_s, inc.duration_s) > tol_pct)
        ):
            lo += 1
        j = lo
        chosen = None
        while j < len(cs):
            c = cs[j]
            if c.duration_s > inc.duration_s and _rel_diff_pct(c.duration_s, inc.duration_s) > tol_pct:
                break  # 后面只会更大，提前终止
            feasible = not used[j] and _rel_diff_pct(c.duration_s, inc.duration_s) <= tol_pct
            if feasible and not (forbid_same_id and c.id == inc.id):
                chosen = j
                break
            j += 1
        if chosen is None:
            continue
        c = cs[chosen]
        row = {
            "complete_id": c.id,
            "incomplete_id": inc.id,
            "duration_complete_s": c.duration_s,
            "duration_incomplete_s": inc.duration_s,
            "duration_diff_pct": _rel_diff_pct(c.duration_s, inc.duration_s),
        }
        if c.speech_s is not None and inc.speech_s is not None:
            speech_diff = _rel_diff_pct(c.speech_s, inc.speech_s)
            if speech_diff > tol_pct:
                n_speech_dropped += 1
                used[chosen] = True  # 该 complete 已被消耗（保持贪心最优性口径一致）
                continue
            row["speech_complete_s"] = c.speech_s
            row["speech_incomplete_s"] = inc.speech_s
            row["speech_diff_pct"] = speech_diff
        used[chosen] = True
        pairs.append(row)
    stats = {
        "n_completes": len(completes),
        "n_incompletes": len(incompletes),
        "n_pairs": len(pairs),
        "n_speech_dropped": n_speech_dropped,
        "tol_pct": tol_pct,
    }
    return pairs, stats
