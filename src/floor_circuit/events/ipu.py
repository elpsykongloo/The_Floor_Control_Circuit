"""IPU 构建：同通道连续语音，内部间隙 < merge_gap_s（冻结 180 ms）合并。"""

from __future__ import annotations

from floor_circuit.schemas import Seg


def build_ipus(segs: list[Seg], merge_gap_s: float) -> list[Seg]:
    if not segs:
        return []
    segs = sorted(segs)
    out = [segs[0]]
    for s in segs[1:]:
        prev = out[-1]
        # 严格小于（带浮点容差）：间隙恰为 merge_gap_s 时不合并
        if (merge_gap_s - (s.start - prev.end)) > 1e-9:
            out[-1] = Seg(prev.start, max(prev.end, s.end))
        else:
            out.append(s)
    return out
