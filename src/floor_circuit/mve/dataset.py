"""MVE 数据集装配：zarr 激活 + 标签 parquet → 按会话组织的 (X, y)。

R1 复放每会话跑两次（agent=ch0 / agent=ch1），run 目录命名 <session>_agent{ch}；
bootstrap 的 cluster 单位始终是会话（两份角色数据并入同一 sid）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from floor_circuit.cachelib.zarr_io import read_acts, read_array
from floor_circuit.probes.linear import SessionData

MAX_TRAIN_FEATURE_BYTES = 8 * 1024**3
MAX_SESSION_FEATURE_BYTES = 1024**3


@dataclass(frozen=True)
class RoleSampleSelection:
    """一个会话角色经过全局负类抽样后需要读取的特征行。"""

    session_id: str
    agent_channel: int
    n_steps: int
    steps: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class TrainingSamplePlan:
    """只含标签与步号的训练抽样计划；构建阶段不触碰高维特征。"""

    target: str
    delta_ms: int | None
    seed: int
    neg_ratio: int
    n_available: int
    n_positive: int
    n_negative_available: int
    roles: tuple[RoleSampleSelection, ...]

    @property
    def n_selected(self) -> int:
        return sum(len(role.labels) for role in self.roles)

    @property
    def n_selected_positive(self) -> int:
        return sum(int(role.labels.sum()) for role in self.roles)


def eligible_rows(
    labels: pd.DataFrame,
    target: str,
    delta_ms: int | None,
    agent_channel: int,
    max_steps: int | None = None,
) -> pd.DataFrame:
    sub = labels[(labels["target"] == target) & (labels["agent_channel"] == agent_channel)]
    if target == "T1":
        if delta_ms is None:
            raise ValueError("T1 需要 delta_ms")
        sub = sub[sub["delta_ms"] == delta_ms]
    if max_steps is not None:
        sub = sub[sub["step"] < int(max_steps)]
    return sub.sort_values("step", kind="stable")


def run_dir_for(runs_root: str | Path, session_id: str, agent_channel: int) -> Path:
    return Path(runs_root) / f"{session_id}_agent{agent_channel}"


def build_training_sample_plan(
    labels_root: str | Path,
    session_ids: list[str],
    run_specs: dict[tuple[str, int], Any],
    target: str,
    delta_ms: int | None,
    neg_ratio: int,
    seed: int,
) -> TrainingSamplePlan:
    """先在低维标签域完成全局确定性负类抽样。

    全局行序严格保持“会话列表 → agent0/agent1 → step”的既有顺序，因此抽样结果与
    ``fit_probe`` 先拼接再抽样的旧实现一致，同时避免先读取全部高维激活。
    """

    if neg_ratio < 0:
        raise ValueError("neg_ratio 不能小于 0")
    role_rows: list[tuple[str, int, int, np.ndarray, np.ndarray, int, int]] = []
    label_parts: list[np.ndarray] = []
    offset = 0
    labels_root = Path(labels_root)
    for sid in session_ids:
        labels = pd.read_parquet(labels_root / f"{sid}.parquet")
        for channel in (0, 1):
            spec = run_specs[(sid, channel)]
            rows = eligible_rows(
                labels,
                target,
                delta_ms,
                channel,
                max_steps=int(spec.n_steps),
            )
            steps = rows["step"].to_numpy(dtype=np.int64)
            values = rows["label"].to_numpy(dtype=np.int64)
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"{sid}/agent{channel}/{target} 含非二值标签")
            stop = offset + len(values)
            role_rows.append(
                (
                    sid,
                    channel,
                    int(spec.n_steps),
                    steps,
                    values,
                    offset,
                    stop,
                )
            )
            label_parts.append(values)
            offset = stop
    if not label_parts:
        raise ValueError(f"{target} 训练标签为空")
    all_labels = np.concatenate(label_parts)
    positive = np.flatnonzero(all_labels == 1)
    negative = np.flatnonzero(all_labels == 0)
    n_keep_negative = min(len(negative), neg_ratio * max(len(positive), 1))
    rng = np.random.default_rng(seed)
    kept_negative = (
        rng.choice(negative, size=n_keep_negative, replace=False)
        if len(negative) > n_keep_negative
        else negative
    )
    selected = np.sort(np.concatenate([positive, kept_negative]))

    roles: list[RoleSampleSelection] = []
    for sid, channel, n_steps, steps, values, start, stop in role_rows:
        left = int(np.searchsorted(selected, start, side="left"))
        right = int(np.searchsorted(selected, stop, side="left"))
        local = selected[left:right] - start
        roles.append(
            RoleSampleSelection(
                session_id=sid,
                agent_channel=channel,
                n_steps=n_steps,
                steps=steps[local],
                labels=values[local],
            )
        )
    return TrainingSamplePlan(
        target=target,
        delta_ms=delta_ms,
        seed=seed,
        neg_ratio=neg_ratio,
        n_available=len(all_labels),
        n_positive=len(positive),
        n_negative_available=len(negative),
        roles=tuple(roles),
    )


def _feature_name(layer: int, feature: str) -> str:
    if feature == "acts":
        return f"acts_L{layer}"
    if feature == "mimi":
        return "mimi_latent"
    raise ValueError(f"未知特征类型：{feature}")


def _feature_array(
    runs_root: str | Path,
    session_id: str,
    agent_channel: int,
    layer: int,
    feature: str,
) -> zarr.Array:
    run_dir = run_dir_for(runs_root, session_id, agent_channel)
    group = zarr.open_group(str(run_dir), mode="r")
    name = _feature_name(layer, feature)
    try:
        array = group[name]
    except Exception as exc:
        raise KeyError(f"{run_dir} 缺少特征数组 {name}") from exc
    if not isinstance(array, zarr.Array) or len(array.shape) != 2:
        raise ValueError(f"{run_dir}/{name} 必须是二维 zarr 数组")
    return array


def _role_feature_arrays(
    runs_root: str | Path,
    session_id: str,
    agent_channel: int,
    layer: int,
    feature: str,
) -> tuple[zarr.Array, ...]:
    """返回角色顺序固定的特征数组；Mimi 为 ``[self, other]``。"""

    own = _feature_array(runs_root, session_id, agent_channel, layer, feature)
    if feature == "acts":
        return (own,)
    if feature == "mimi":
        other = _feature_array(
            runs_root,
            session_id,
            1 - agent_channel,
            layer,
            feature,
        )
        return own, other
    raise ValueError(f"未知特征类型：{feature}")


def _validate_role_feature_arrays(
    arrays: tuple[zarr.Array, ...],
    role: RoleSampleSelection,
    feature: str,
) -> int:
    """校验角色数组的时间轴、步号和维度，返回拼接后的特征维度。"""

    shapes = [tuple(int(value) for value in array.shape) for array in arrays]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError(
            f"{role.session_id}/agent{role.agent_channel}/{feature} 特征必须是二维数组"
        )
    if any(shape[0] != role.n_steps for shape in shapes):
        raise ValueError(
            f"{role.session_id}/agent{role.agent_channel}/{feature} 时间长度 {shapes}，"
            f"期望每路 {role.n_steps} 步"
        )
    if feature == "mimi" and (
        len(shapes) != 2 or shapes[0][0] != shapes[1][0] or shapes[0][1] != shapes[1][1]
    ):
        raise ValueError(
            f"{role.session_id}/agent{role.agent_channel}/mimi 双通道形状不一致：{shapes}"
        )
    if len(role.steps):
        if int(role.steps[0]) < 0 or int(role.steps[-1]) >= role.n_steps:
            raise ValueError(
                f"{role.session_id}/agent{role.agent_channel} 的步号越过特征时间域"
            )
        if np.any(np.diff(role.steps) < 0):
            raise ValueError(
                f"{role.session_id}/agent{role.agent_channel} 的步号未按时间升序排列"
            )
    return sum(shape[1] for shape in shapes)


def _read_feature_rows(array: zarr.Array, steps: np.ndarray) -> np.ndarray:
    """只读取指定步，禁止先把整个 run 转成内存数组。"""

    if len(steps) == 0:
        return np.empty((0, int(array.shape[1])), dtype=np.float32)
    values = array.get_orthogonal_selection((steps, slice(None)))
    return np.asarray(values, dtype=np.float32)


def _checked_feature_dim(
    runs_root: str | Path,
    roles: tuple[RoleSampleSelection, ...],
    layer: int,
    feature: str,
) -> tuple[int, int]:
    dims: set[int] = set()
    max_role_rows = 0
    for role in roles:
        arrays = _role_feature_arrays(
            runs_root,
            role.session_id,
            role.agent_channel,
            layer,
            feature,
        )
        dims.add(_validate_role_feature_arrays(arrays, role, feature))
        max_role_rows = max(max_role_rows, len(role.steps))
    if len(dims) != 1:
        raise ValueError(f"特征维度不唯一或没有可读数组：{sorted(dims)}")
    return dims.pop(), max_role_rows


def _write_role_feature_rows(
    destination: np.ndarray,
    arrays: tuple[zarr.Array, ...],
    steps: np.ndarray,
) -> None:
    """逐通道写入预分配矩阵，避免为双通道拼接创建额外整块副本。"""

    column = 0
    for array in arrays:
        width = int(array.shape[1])
        destination[:, column : column + width] = _read_feature_rows(array, steps)
        column += width
    if column != destination.shape[1]:
        raise RuntimeError(f"角色特征只写入 {column}/{destination.shape[1]} 列")


def load_training_sample(
    runs_root: str | Path,
    plan: TrainingSamplePlan,
    layer: int,
    feature: str = "acts",
    max_bytes: int = MAX_TRAIN_FEATURE_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """按抽样计划读取单层训练矩阵，峰值仅为结果矩阵加一个角色的小块。"""

    dim, max_role_rows = _checked_feature_dim(runs_root, plan.roles, layer, feature)
    estimated_peak = (plan.n_selected + max_role_rows) * dim * np.dtype(np.float32).itemsize
    if estimated_peak > max_bytes:
        raise MemoryError(
            f"{plan.target} 单层训练特征预计峰值 {estimated_peak / 1024**3:.2f} GiB，"
            f"超过上限 {max_bytes / 1024**3:.2f} GiB"
        )
    features = np.empty((plan.n_selected, dim), dtype=np.float32)
    labels = np.empty(plan.n_selected, dtype=np.int64)
    offset = 0
    for role in plan.roles:
        n_rows = len(role.steps)
        if not n_rows:
            continue
        arrays = _role_feature_arrays(
            runs_root,
            role.session_id,
            role.agent_channel,
            layer,
            feature,
        )
        _write_role_feature_rows(
            features[offset : offset + n_rows],
            arrays,
            role.steps,
        )
        labels[offset : offset + n_rows] = role.labels
        offset += n_rows
    if offset != plan.n_selected:
        raise RuntimeError(f"训练特征只写入 {offset}/{plan.n_selected} 行")
    return features, labels


def load_session_feature(
    runs_root: str | Path,
    labels_root: str | Path,
    session_id: str,
    run_specs: dict[tuple[str, int], Any],
    layer: int,
    target: str,
    delta_ms: int | None,
    feature: str = "acts",
    max_bytes: int = MAX_SESSION_FEATURE_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """只装配一个评估会话；超过显式内存上限时硬失败，不裁剪冻结样本。"""

    labels_frame = pd.read_parquet(Path(labels_root) / f"{session_id}.parquet")
    roles: list[RoleSampleSelection] = []
    for channel in (0, 1):
        spec = run_specs[(session_id, channel)]
        rows = eligible_rows(
            labels_frame,
            target,
            delta_ms,
            channel,
            max_steps=int(spec.n_steps),
        )
        roles.append(
            RoleSampleSelection(
                session_id=session_id,
                agent_channel=channel,
                n_steps=int(spec.n_steps),
                steps=rows["step"].to_numpy(dtype=np.int64),
                labels=rows["label"].to_numpy(dtype=np.int64),
            )
        )
    role_tuple = tuple(roles)
    dim, max_role_rows = _checked_feature_dim(runs_root, role_tuple, layer, feature)
    n_rows = sum(len(role.steps) for role in roles)
    estimated_peak = (n_rows + max_role_rows) * dim * np.dtype(np.float32).itemsize
    if estimated_peak > max_bytes:
        raise MemoryError(
            f"{session_id}/{target} 评估特征预计峰值 {estimated_peak / 1024**3:.2f} GiB，"
            f"超过上限 {max_bytes / 1024**3:.2f} GiB"
        )
    features = np.empty((n_rows, dim), dtype=np.float32)
    values = np.empty(n_rows, dtype=np.int64)
    offset = 0
    for role in roles:
        count = len(role.steps)
        if not count:
            continue
        arrays = _role_feature_arrays(
            runs_root,
            session_id,
            role.agent_channel,
            layer,
            feature,
        )
        _write_role_feature_rows(
            features[offset : offset + count],
            arrays,
            role.steps,
        )
        values[offset : offset + count] = role.labels
        offset += count
    if offset != n_rows:
        raise RuntimeError(f"{session_id} 评估特征只写入 {offset}/{n_rows} 行")
    return features, values


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
    """单会话单角色的 ``(X, y)``；Mimi 固定拼接 ``[self, other]``。"""

    rd = run_dir_for(runs_root, session_id, agent_channel)
    if feature == "acts":
        features = (read_acts(rd, layer),)
    elif feature == "mimi":
        other_dir = run_dir_for(runs_root, session_id, 1 - agent_channel)
        features = (
            read_array(rd, "mimi_latent"),
            read_array(other_dir, "mimi_latent"),
        )
        shapes = [tuple(int(value) for value in array.shape) for array in features]
        if any(len(shape) != 2 for shape in shapes) or shapes[0] != shapes[1]:
            raise ValueError(
                f"{session_id}/agent{agent_channel}/mimi 双通道形状不一致：{shapes}"
            )
    else:
        raise ValueError(f"未知特征类型：{feature}")
    rows = eligible_rows(labels, target, delta_ms, agent_channel)
    steps = rows["step"].to_numpy(dtype=np.int64)
    if feature == "mimi" and len(steps) and (
        int(steps[0]) < 0 or int(steps[-1]) >= int(features[0].shape[0])
    ):
        raise ValueError(f"{session_id}/agent{agent_channel} 的步号越过 Mimi 双通道时间域")
    keep = steps < features[0].shape[0]
    steps, y = steps[keep], rows["label"].to_numpy(dtype=np.int64)[keep]
    selected = [array[steps].astype(np.float32) for array in features]
    return np.concatenate(selected, axis=1) if len(selected) > 1 else selected[0], y


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
