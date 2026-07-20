"""E1 冻结会话集合（PREREG #15，用户 2026-07-19 裁决 D2）。

- E1 训练集 = probe_train[0:400]
- E1 主评估集 = probe_val[40:140]（全新 100 会话）
- MVE 历史评估集 = probe_val[0:40]（仅历史复现，不并入 E1 主置信区间）
- 后备池 = probe_val[140:248]（108 会话；任何使用必须先登记 PREREG 变更）
- causal_eval 25% 侧在 E2 前绝不读取。

冻结前缀切片依据：candor.json 生成时已按种子 20260717 随机分池、池内按 UUIDv4
字典序排列——前缀近似内容无关的固定抽样，无需新随机源。
"""

from __future__ import annotations

from dataclasses import dataclass

E1_N_TRAIN = 400
E1_EVAL_SLICE = (40, 140)
MVE_HIST_SLICE = (0, 40)


@dataclass(frozen=True)
class E1Sessions:
    train: tuple[str, ...]
    eval: tuple[str, ...]
    mve_hist: tuple[str, ...]
    n_reserve: int


def e1_sessions(split_payload: dict) -> E1Sessions:
    """从冻结 candor.json 载荷解析 E1 集合；任何越界/重叠立即硬失败。"""

    splits = split_payload["splits"]
    probe_train = [str(value) for value in splits["probe_train"]]
    probe_val = [str(value) for value in splits["probe_val"]]
    if len(probe_train) < E1_N_TRAIN:
        raise ValueError(f"probe_train 仅 {len(probe_train)} 会话，不足 E1 训练集 {E1_N_TRAIN}")
    if len(probe_val) < E1_EVAL_SLICE[1]:
        raise ValueError(f"probe_val 仅 {len(probe_val)} 会话，不足评估切片上界 {E1_EVAL_SLICE[1]}")
    train = tuple(probe_train[:E1_N_TRAIN])
    evals = tuple(probe_val[E1_EVAL_SLICE[0] : E1_EVAL_SLICE[1]])
    mve_hist = tuple(probe_val[MVE_HIST_SLICE[0] : MVE_HIST_SLICE[1]])
    if len(evals) != E1_EVAL_SLICE[1] - E1_EVAL_SLICE[0]:
        raise ValueError("E1 主评估集切片长度异常")
    for name, group in (("train", train), ("eval", evals), ("mve_hist", mve_hist)):
        if len(set(group)) != len(group):
            raise ValueError(f"E1 {name} 集合含重复会话")
    if set(train) & set(evals) or set(train) & set(mve_hist):
        raise ValueError("E1 训练集与评估/历史集重叠——冻结划分被破坏")
    if set(evals) & set(mve_hist):
        raise ValueError("E1 主评估集与 MVE 历史集重叠——切片起点必须为 40")
    return E1Sessions(
        train=train,
        eval=evals,
        mve_hist=mve_hist,
        n_reserve=len(probe_val) - E1_EVAL_SLICE[1],
    )
