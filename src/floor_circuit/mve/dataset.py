"""MVE 数据集装配：zarr 激活 + 标签 parquet → 按会话组织的 (X, y)。

R1 复放每会话跑两次（agent=ch0 / agent=ch1），run 目录命名 <session>_agent{ch}；
bootstrap 的 cluster 单位始终是会话（两份角色数据并入同一 sid）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from floor_circuit.cachelib.zarr_io import read_acts, read_array
from floor_circuit.probes.linear import SessionData


def eligible_rows(
    labels: pd.DataFrame, target: str, delta_ms: int | None, agent_channel: int
) -> pd.DataFrame:
    sub = labels[(labels["target"] == target) & (labels["agent_channel"] == agent_channel)]
    if target == "T1":
        if delta_ms is None:
            raise ValueError("T1 需要 delta_ms")
        sub = sub[sub["delta_ms"] == delta_ms]
    return sub


def run_dir_for(runs_root: str | Path, session_id: str, agent_channel: int) -> Path:
    return Path(runs_root) / f"{session_id}_agent{agent_channel}"


def load_role_xy(
    runs_root: str | Path,
    labels: pd.DataFrame,
    session_id: str,
    agent_channel: int,
    layer: int,
    target: str,
    delta_ms: int | None,
    feature: str = "acts",
) -> tuple[np.ndarray, np.ndarray]:
    """单会话单角色的 (X, y)。feature="acts" 读 acts_L{layer}；"mimi" 读 mimi_latent。"""
    rd = run_dir_for(runs_root, session_id, agent_channel)
    acts = read_acts(rd, layer) if feature == "acts" else read_array(rd, "mimi_latent")
    rows = eligible_rows(labels, target, delta_ms, agent_channel)
    steps = rows["step"].to_numpy(dtype=np.int64)
    keep = steps < acts.shape[0]
    steps, y = steps[keep], rows["label"].to_numpy(dtype=np.int64)[keep]
    return acts[steps].astype(np.float32), y


def build_session_data(
    runs_root: str | Path,
    labels_root: str | Path,
    session_ids: list[str],
    layer: int,
    target: str,
    delta_ms: int | None,
    feature: str = "acts",
    agent_channels: tuple[int, ...] = (0, 1),
) -> SessionData:
    """sid -> (X, y)，两角色纵向拼接；标签文件 <labels_root>/<session>.parquet。"""
    data: SessionData = {}
    for sid in session_ids:
        labels = pd.read_parquet(Path(labels_root) / f"{sid}.parquet")
        xs, ys = [], []
        for ch in agent_channels:
            X, y = load_role_xy(runs_root, labels, sid, ch, layer, target, delta_ms, feature)
            if len(y):
                xs.append(X)
                ys.append(y)
        if xs:
            data[sid] = (np.concatenate(xs), np.concatenate(ys))
    return data
