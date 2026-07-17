"""Turn 构建与 backchannel 判据。

EN 启发式（向 CANDOR Backbiter 对齐）：同说话人相邻 IPU 间隙 < 1.0 s 且期间对方无
非-backchannel IPU 则合并；对方 ≤ 1.0 s 且命中 backchannel 判据的 IPU 不切断 turn。
ZH 一律以 SmoothConv/DuplexConv 专家 turn 金标为准（见 data/smoothconv.py），不走本启发式。
"""

from __future__ import annotations

import re

from floor_circuit.schemas import Seg, Turn

_PUNCT_RE = re.compile(r"[^\w\s一-鿿-]", re.UNICODE)


def normalize_text(text: str) -> str:
    t = _PUNCT_RE.sub(" ", text.lower())
    return " ".join(t.split())


def is_backchannel(
    ipu: Seg,
    text: str | None,
    lexicon: list[str],
    max_s: float,
) -> bool:
    """backchannel 判据：时长 ≤ max_s，且（有转录时）词面命中词表。

    无转录时退化为"时长-only 代理"（其代价由 G0 校准量化并写入校准报告）。
    """
    if ipu.dur > max_s:
        return False
    if text is None:
        return True
    norm = normalize_text(text)
    if not norm:
        return False
    lex = {normalize_text(w) for w in lexicon}
    if norm in lex:
        return True
    # EN 多词情形："yeah yeah"、"oh okay" 等全部 token 均在词表内也算命中
    tokens = norm.split()
    return all(tok in lex for tok in tokens)


def build_turns_en(
    ipus_x: list[Seg],
    ipus_y: list[Seg],
    bc_flags_y: list[bool],
    merge_gap_s: float,
) -> list[Turn]:
    """X 通道的 EN 启发式 turn。bc_flags_y 与 ipus_y 一一对应（是否 backchannel）。

    合并条件：相邻 IPU 间隙 < merge_gap_s 且间隙区间内没有与之重叠的非-bc 对方 IPU。
    """
    if len(ipus_y) != len(bc_flags_y):
        raise ValueError("ipus_y 与 bc_flags_y 长度不一致")
    if not ipus_x:
        return []
    non_bc_y = [seg for seg, bc in zip(ipus_y, bc_flags_y, strict=True) if not bc]

    def gap_blocked(gap_start: float, gap_end: float) -> bool:
        return any(seg.start < gap_end and seg.end > gap_start for seg in non_bc_y)

    turns: list[Turn] = []
    cur = [0]
    for i in range(1, len(ipus_x)):
        prev, nxt = ipus_x[i - 1], ipus_x[i]
        gap = nxt.start - prev.end
        if gap < merge_gap_s and not gap_blocked(prev.end, nxt.start):
            cur.append(i)
        else:
            turns.append(_mk_turn(ipus_x, cur))
            cur = [i]
    turns.append(_mk_turn(ipus_x, cur))
    return turns


def _mk_turn(ipus: list[Seg], idx: list[int]) -> Turn:
    return Turn(
        channel=-1,  # 由调用方回填
        start=ipus[idx[0]].start,
        end=ipus[idx[-1]].end,
        ipu_indices=list(idx),
    )
