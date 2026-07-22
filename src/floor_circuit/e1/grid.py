"""E1 探针网格核心（PREREG #18）：规格展开、抽样计划、特征装配与格子评估。

规格（10 个）：T1×6δ、T2、T3、T4、T5。行域统一 #11 截断（usable_label_steps）
与 #8 对齐（acts 读行 s+1，Mimi/hazard/声学读行 s；mve/alignment.py 唯一权威）。
种子语义（#18(b)）：训练池 = inner_val(80 固定) ∪ rng(seed) 从 probe_train[80:400]
无放回抽 288；T1 另做全局 5:1 负类抽样；T5 训练与评估同用 stride=4 行栅。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from floor_circuit.mve.alignment import (
    MIN_ELIGIBLE_STEP,
    feature_row_indices,
    usable_label_steps,
)

T1_DELTAS_MS = (0, 80, 160, 240, 400, 800)
N_CLASSES = {"T1": 2, "T2": 2, "T3": 3, "T4": 2, "T5": 5}


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    target: str
    delta_ms: int | None
    n_classes: int
    sampling: str  # neg5 / none / stride


def expand_specs(probe_cfg: dict) -> list[ProbeSpec]:
    """冻结的 10 个目标规格（#18(c)）。"""
    specs = [
        ProbeSpec(f"T1_d{delta}", "T1", delta, 2, "neg5") for delta in T1_DELTAS_MS
    ]
    specs.append(ProbeSpec("T2", "T2", None, 2, "none"))
    specs.append(ProbeSpec("T3", "T3", None, 3, "none"))
    specs.append(ProbeSpec("T4", "T4", None, 2, "none"))
    specs.append(ProbeSpec("T5", "T5", None, 5, "stride"))
    if len(specs) != 10:
        raise RuntimeError("规格展开必须恰为 10 个")
    _ = probe_cfg["t5_step_stride"]  # 配置存在性校验
    return specs


def seed_train_sessions(train_sessions: list[str], probe_cfg: dict, seed: int) -> list[str]:
    """#18(b)：inner_val 固定前 80；其余池按种子 90% 会话级子抽样。"""
    inner_n = int(probe_cfg["inner_val_sessions"])
    lo, hi = (int(v) for v in probe_cfg["seed_subsample_pool"])
    take_n = int(probe_cfg["seed_subsample_n"])
    if len(train_sessions) < hi:
        raise ValueError(f"训练池 {len(train_sessions)} < 切片上界 {hi}")
    if lo != inner_n:
        raise ValueError("seed_subsample_pool 下界必须等于 inner_val_sessions")
    pool = train_sessions[lo:hi]
    if take_n > len(pool):
        raise ValueError("seed_subsample_n 超过池大小")
    rng = np.random.default_rng(int(seed))
    picked = sorted(rng.choice(len(pool), size=take_n, replace=False).tolist())
    return train_sessions[:inner_n] + [pool[i] for i in picked]


@dataclass(frozen=True)
class RoleRows:
    session_id: str
    agent_channel: int
    steps: np.ndarray
    labels: np.ndarray


def _spec_rows(labels: pd.DataFrame, spec: ProbeSpec, channel: int, n_steps: int, stride: int) -> pd.DataFrame:
    sub = labels[(labels["target"] == spec.target) & (labels["agent_channel"] == channel)]
    if spec.target == "T1":
        sub = sub[sub["delta_ms"] == spec.delta_ms]
    sub = sub[(sub["step"] < usable_label_steps(n_steps)) & (sub["step"] >= MIN_ELIGIBLE_STEP)]
    if spec.sampling == "stride":
        sub = sub[sub["step"] % stride == 0]
    return sub.sort_values("step", kind="stable")


def build_rows(
    labels_root: Path,
    sessions: list[str],
    n_steps_by_role: dict[tuple[str, int], int],
    spec: ProbeSpec,
    probe_cfg: dict,
    seed: int,
    *,
    downsample: bool,
) -> list[RoleRows]:
    """按会话→角色→步的全局固定顺序取行；T1 训练侧做全局 5:1 负类抽样。"""
    roles = build_rows_multi(
        labels_root, sessions, n_steps_by_role, [spec], probe_cfg
    )[spec.name]
    return sample_role_rows(roles, spec, probe_cfg, seed, downsample=downsample)


def build_rows_multi(
    labels_root: Path,
    sessions: list[str],
    n_steps_by_role: dict[tuple[str, int], int],
    specs: list[ProbeSpec],
    probe_cfg: dict,
) -> dict[str, list[RoleRows]]:
    """每个会话只读一次 parquet，同时构建全部规格的未下采样行域。"""
    stride = int(probe_cfg["t5_step_stride"])
    by_spec: dict[str, list[RoleRows]] = {spec.name: [] for spec in specs}

    def read_one(sid: str) -> tuple[str, pd.DataFrame]:
        return sid, pd.read_parquet(labels_root / f"{sid}.parquet")

    with ThreadPoolExecutor(
        max_workers=io_jobs(len(sessions)), thread_name_prefix="e1-labels"
    ) as pool:
        loaded = pool.map(read_one, sessions)
        for sid, frame in loaded:
            for channel in (0, 1):
                for spec in specs:
                    rows = _spec_rows(
                        frame,
                        spec,
                        channel,
                        n_steps_by_role[(sid, channel)],
                        stride,
                    )
                    values = rows["label"].to_numpy(dtype=np.int64)
                    if values.size and (
                        values.min() < 0 or values.max() >= spec.n_classes
                    ):
                        raise ValueError(f"{sid}/agent{channel}/{spec.name} 标签越界")
                    if len(values):
                        by_spec[spec.name].append(
                            RoleRows(
                                sid,
                                channel,
                                rows["step"].to_numpy(dtype=np.int64),
                                values,
                            )
                        )
    return by_spec


def sample_role_rows(
    roles: list[RoleRows],
    spec: ProbeSpec,
    probe_cfg: dict,
    seed: int,
    *,
    downsample: bool,
) -> list[RoleRows]:
    """在已构建行域上执行冻结的 T1 全局负类下采样。"""
    if downsample and spec.sampling == "neg5":
        ratio = int(probe_cfg["neg_ratio_t1"])
        all_labels = np.concatenate([r.labels for r in roles]) if roles else np.empty(0, np.int64)
        positive = np.flatnonzero(all_labels == 1)
        negative = np.flatnonzero(all_labels == 0)
        keep_n = min(len(negative), ratio * max(len(positive), 1))
        rng = np.random.default_rng(int(seed))
        kept_negative = (
            rng.choice(negative, size=keep_n, replace=False)
            if len(negative) > keep_n
            else negative
        )
        selected = np.sort(np.concatenate([positive, kept_negative]))
        out: list[RoleRows] = []
        offset = 0
        for role in roles:
            stop = offset + len(role.labels)
            local = selected[(selected >= offset) & (selected < stop)] - offset
            out.append(RoleRows(role.session_id, role.agent_channel, role.steps[local], role.labels[local]))
            offset = stop
        roles = out
    return [r for r in roles if len(r.steps)]


def io_jobs(n_tasks: int) -> int:
    try:
        jobs = int(os.environ.get("FLOOR_CIRCUIT_IO_JOBS", "8"))
    except ValueError as exc:
        raise ValueError("FLOOR_CIRCUIT_IO_JOBS 必须为正整数") from exc
    return max(1, min(jobs, max(1, n_tasks)))


def preload_layer(
    runs_root: Path, run_keys: list[tuple[str, int]], layer: int
) -> dict[tuple[str, int], np.ndarray]:
    """载入指定角色的某层 zarr 数组；run 分训练 800 路与评估 200 路调用。"""
    import zarr

    def read_one(key: tuple[str, int]) -> tuple[tuple[str, int], np.ndarray]:
        sid, channel = key
        group = zarr.open_group(str(runs_root / f"{sid}_agent{channel}"), mode="r")
        return key, np.asarray(group[f"acts_L{layer}"][:], dtype=np.float16)

    loaded: dict[tuple[str, int], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=io_jobs(len(run_keys)), thread_name_prefix="e1-io") as pool:
        for key, array in pool.map(read_one, run_keys):
            loaded[key] = array
    return loaded


def preload_mimi(
    runs_root: Path, run_keys: list[tuple[str, int]]
) -> dict[tuple[str, int], np.ndarray]:
    """Mimi 潜表征常驻（[self, other] 拼接在装配时进行；此处按角色缓存单路）。"""
    import zarr

    def read_one(key: tuple[str, int]) -> tuple[tuple[str, int], np.ndarray]:
        sid, channel = key
        group = zarr.open_group(str(runs_root / f"{sid}_agent{channel}"), mode="r")
        return key, np.asarray(group["mimi_latent"][:], dtype=np.float16)

    loaded: dict[tuple[str, int], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=io_jobs(len(run_keys)), thread_name_prefix="e1-mimi") as pool:
        for key, array in pool.map(read_one, run_keys):
            loaded[key] = array
    return loaded


def assemble(
    roles: list[RoleRows],
    feature: str,
    store: dict[tuple[str, int], np.ndarray],
    *,
    dtype=np.float16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 #8 行映射装配 (X, y, session_index)；mimi 拼接 [self, other]。"""
    n_rows, dims = feature_layout(roles, feature, store)
    features = np.empty((n_rows, dims), dtype=dtype)
    y = np.empty(n_rows, dtype=np.int64)
    sid_codes = np.empty(n_rows, dtype=np.int64)
    offset = 0
    for role, block in feature_blocks(roles, feature, store):
        stop = offset + len(role.labels)
        features[offset:stop] = block
        y[offset:stop] = role.labels
        sid_codes[offset:stop] = hash(role.session_id) % (1 << 62)
        offset = stop
    return features, y, sid_codes


def feature_layout(
    roles: list[RoleRows],
    feature: str,
    store: dict[tuple[str, int], np.ndarray],
) -> tuple[int, int]:
    """返回角色行域装配后的总行数与特征维度，不分配大矩阵。"""
    if not roles:
        return 0, 0
    sample = store[(roles[0].session_id, roles[0].agent_channel)]
    if feature == "acts":
        dims = int(sample.shape[1])
    elif feature == "mimi":
        dims = int(sample.shape[1]) * 2
    else:
        raise ValueError(f"未知特征：{feature}")
    return sum(len(role.labels) for role in roles), dims


def feature_blocks(
    roles: list[RoleRows],
    feature: str,
    store: dict[tuple[str, int], np.ndarray],
):
    """逐角色生成对齐特征块；训练器可直接写入设备，避免主存整矩阵副本。"""
    dims = None
    for role in roles:
        rows = feature_row_indices("acts" if feature == "acts" else "mimi", role.steps)
        if feature == "acts":
            block = store[(role.session_id, role.agent_channel)][rows]
        elif feature == "mimi":
            own = store[(role.session_id, role.agent_channel)][rows]
            other = store[(role.session_id, 1 - role.agent_channel)][rows]
            block = np.concatenate([own, other], axis=1)
        else:
            raise ValueError(f"未知特征：{feature}")
        if dims is None:
            dims = block.shape[1]
        elif dims != block.shape[1]:
            raise ValueError("特征维度跨角色不一致")
        yield role, block


def t5_state_array(labels_frame: pd.DataFrame, channel: int, n_steps: int) -> np.ndarray:
    """从标签表还原逐步 T5 状态（hazard 特征输入）；缺步硬失败。"""
    sub = labels_frame[
        (labels_frame["target"] == "T5") & (labels_frame["agent_channel"] == channel)
    ].sort_values("step", kind="stable")
    steps = sub["step"].to_numpy(dtype=np.int64)
    values = sub["label"].to_numpy(dtype=np.int64)
    states = np.full(n_steps, -1, dtype=np.int64)
    keep = steps < n_steps
    states[steps[keep]] = values[keep]
    if (states < 0).any():
        missing = int((states < 0).sum())
        raise ValueError(f"T5 状态覆盖缺 {missing} 步（agent{channel}）")
    return states


def eval_cell_scores(
    probe,
    roles: list[RoleRows],
    feature: str,
    store: dict[tuple[str, int], np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """逐会话评估：返回 sid -> (y, probs)，两角色纵向拼接（cluster=会话）。"""
    by_session: dict[str, list[RoleRows]] = {}
    for role in roles:
        by_session.setdefault(role.session_id, []).append(role)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid, group in by_session.items():
        features, y, _ = assemble(group, feature, store)
        if not len(y):
            continue
        out[sid] = (y, probe.predict_proba(np.asarray(features, dtype=np.float32)))
    return out


def eval_cell_scores_many(
    probes: list,
    roles: list[RoleRows],
    feature: str,
    store: dict[tuple[str, int], np.ndarray],
    *,
    device: str,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """一次装配与搬运同时评估多个探针，仍按会话保留 cluster 边界。"""
    from floor_circuit.e1.probe_gpu import LinearProbeBatchPredictor

    predictor = LinearProbeBatchPredictor(probes, device=device)
    by_session: dict[str, list[RoleRows]] = {}
    for role in roles:
        by_session.setdefault(role.session_id, []).append(role)
    outputs: list[dict[str, tuple[np.ndarray, np.ndarray]]] = [
        {} for _ in probes
    ]
    for sid, group in by_session.items():
        features, y, _ = assemble(group, feature, store)
        if not len(y):
            continue
        probabilities = predictor.predict_proba(features)
        for out, probs in zip(outputs, probabilities, strict=True):
            out[sid] = (y.copy(), probs)
    return outputs


def pooled_primary_metric(scores: dict[str, tuple[np.ndarray, np.ndarray]], n_classes: int) -> float:
    from floor_circuit.e1.probe_gpu import primary_metric

    ys = np.concatenate([y for y, _ in scores.values()])
    ps = np.concatenate([p for _, p in scores.values()])
    return primary_metric(ys, ps, n_classes)
