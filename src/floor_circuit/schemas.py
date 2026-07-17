"""统一数据结构：事件本体与语料 schema 的共享类型。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

import pandas as pd


class EventKind(StrEnum):
    ONSET = "ONSET"
    OFFSET = "OFFSET"
    YIELD = "YIELD"
    HOLD = "HOLD"
    BC = "BC"
    GRAB = "GRAB"
    PAUSE = "PAUSE"
    TURNEND = "TURNEND"


class TurnLabel(StrEnum):
    """SmoothConv/DuplexConv 专家标注的 turn 四分类。"""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    BACKCHANNEL = "backchannel"
    WAIT = "wait"


class State(int, Enum):
    """T5 五分类状态（agent 视角）。OVERLAP_UNRESOLVED 不入探针，仅供审计。"""

    SPEAK = 0
    LISTEN = 1
    OVERLAP_YIELD = 2
    OVERLAP_HOLD = 3
    GAP = 4
    OVERLAP_UNRESOLVED = 5


@dataclass(frozen=True, order=True)
class Seg:
    """左闭右开时间段，单位秒。"""

    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Seg end < start: {self}")


@dataclass
class Turn:
    channel: int
    start: float
    end: float
    label: str | None = None  # TurnLabel 值或 None（EN 启发式无标签）
    ipu_indices: list[int] = field(default_factory=list)


@dataclass
class Event:
    kind: EventKind
    channel: int  # 事件归属通道（YIELD_X 归 X；BC_Y/GRAB_Y 归 Y）
    t: float  # 事件锚点时刻（秒）
    t_end: float | None = None  # 有跨度的事件（BC、PAUSE）的结束时刻
    aux: dict[str, Any] = field(default_factory=dict)


def events_to_dataframe(events: list[Event]) -> pd.DataFrame:
    rows = [
        {
            "kind": e.kind.value,
            "channel": e.channel,
            "t": e.t,
            "t_end": e.t_end,
            "aux": json.dumps(e.aux, ensure_ascii=False, sort_keys=True),
        }
        for e in events
    ]
    df = pd.DataFrame(rows, columns=["kind", "channel", "t", "t_end", "aux"])
    return df.sort_values(["t", "kind"], kind="stable").reset_index(drop=True)


def dataframe_to_events(df: pd.DataFrame) -> list[Event]:
    out = []
    for row in df.itertuples(index=False):
        t_end = None if pd.isna(row.t_end) else float(row.t_end)
        out.append(
            Event(
                kind=EventKind(row.kind),
                channel=int(row.channel),
                t=float(row.t),
                t_end=t_end,
                aux=json.loads(row.aux) if row.aux else {},
            )
        )
    return out
