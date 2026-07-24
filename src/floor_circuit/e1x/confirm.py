"""E2 确认臂纯函数（PREREG #40(b)）：条件网格、事件锁定门、投影钳制参考实现。

确认臂四问：必要性（投影钳制消融）、时间特异性（respond / user_speech 门控注入）、
轴特异性（T1_d800 / T5:SPEAK 方向对照）、层特异性（错层注入）。基线不重跑——
复用 E2-lite 的 baseline 运行（同会话同种子，公共随机数配对）。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.events.detect import qualified_offsets

GATE_NONE = "none"
GATE_RESPOND = "respond"
GATE_USER_SPEECH = "user_speech"
MODE_INJECT = "inject"
MODE_CLAMP = "clamp"


def build_confirm_conditions(cfg: dict) -> list[dict]:
    """确定性条件网格。字段：name/mode/direction/alpha/layer/gate。"""
    layer = int(cfg["layer"])
    alphas = [float(a) for a in cfg["alphas"]]
    if any(a == 0.0 for a in alphas):
        raise ValueError("确认臂 α 网格不含 0（基线复用 E2-lite baseline）")
    conditions: list[dict] = []
    if bool(cfg.get("clamp", True)):
        conditions.append(
            {
                "name": f"clamp_L{layer}",
                "mode": MODE_CLAMP,
                "direction": "probe_meanseed",
                "alpha": 0.0,
                "layer": layer,
                "gate": GATE_NONE,
            }
        )
    for gate in (GATE_RESPOND, GATE_USER_SPEECH):
        for alpha in alphas:
            conditions.append(
                {
                    "name": f"probe_a{alpha:+g}_gate_{gate}",
                    "mode": MODE_INJECT,
                    "direction": "probe_meanseed",
                    "alpha": alpha,
                    "layer": layer,
                    "gate": gate,
                }
            )
    for direction in [str(name) for name in cfg["axis_directions"]]:
        for alpha in alphas:
            conditions.append(
                {
                    "name": f"{direction}_a{alpha:+g}",
                    "mode": MODE_INJECT,
                    "direction": direction,
                    "alpha": alpha,
                    "layer": layer,
                    "gate": GATE_NONE,
                }
            )
    for wrong_layer in [int(v) for v in cfg["wrong_layers"]]:
        if wrong_layer == layer:
            raise ValueError("错层列表不得包含主层")
        for alpha in alphas:
            conditions.append(
                {
                    "name": f"probe_a{alpha:+g}_L{wrong_layer}",
                    "mode": MODE_INJECT,
                    "direction": "probe_meanseed",
                    "alpha": alpha,
                    "layer": wrong_layer,
                    "gate": GATE_NONE,
                }
            )
    names = [c["name"] for c in conditions]
    if len(names) != len(set(names)):
        raise ValueError("确认臂条件名重复")
    return conditions


# ---------------------------------------------------------------------------
# 事件锁定门（决策帧栅格，80 ms/帧；由用户通道 VAD 预计算，与生成无循环依赖）
# ---------------------------------------------------------------------------


def _dt_mask_to_frame_gate(mask_dt: np.ndarray, dt: float, n_frames: int, frame_s: float) -> np.ndarray:
    """dt 栅格布尔掩码 → 帧门（帧内任一 dt 样本活跃即活跃）。"""
    mask_dt = np.asarray(mask_dt, dtype=bool)
    per = frame_s / dt
    gate = np.zeros(n_frames, dtype=np.uint8)
    for frame in range(n_frames):
        lo = round(frame * per)
        hi = round((frame + 1) * per)
        if lo >= len(mask_dt):
            break
        gate[frame] = 1 if mask_dt[lo : max(hi, lo + 1)].any() else 0
    return gate


def respond_gate_frames(
    mask_user: np.ndarray,
    dt: float,
    ev_cfg: dict,
    *,
    window_s: float,
    n_frames: int,
    frame_s: float,
) -> np.ndarray:
    """respond 门：每个用户合格 offset 后 (t, t+W] 的帧置 1。"""
    mask_user = np.asarray(mask_user, dtype=bool)
    windows = np.zeros(len(mask_user), dtype=bool)
    for t_off in qualified_offsets(mask_user, dt, ev_cfg["offset_post_silence_s"]):
        lo = round(t_off / dt)
        hi = min(len(windows), round((t_off + window_s) / dt))
        windows[lo:hi] = True
    return _dt_mask_to_frame_gate(windows, dt, n_frames, frame_s)


def user_speech_gate_frames(
    mask_user: np.ndarray,
    dt: float,
    *,
    n_frames: int,
    frame_s: float,
) -> np.ndarray:
    """user_speech 门：用户发声帧置 1。"""
    return _dt_mask_to_frame_gate(np.asarray(mask_user, dtype=bool), dt, n_frames, frame_s)


# ---------------------------------------------------------------------------
# 投影钳制（必要性消融）的数值参考实现——runner 中的设备端实现须与此逐项一致
# ---------------------------------------------------------------------------


def apply_clamp(
    hidden: np.ndarray,
    unit_direction: np.ndarray,
    target: float,
    gate: float = 1.0,
) -> np.ndarray:
    """h ← h + g·(μ_v − h·v̂)·v̂：把 v̂ 分量钳到 μ_v；g=0 时恒等。"""
    v = np.asarray(unit_direction, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("钳制要求单位方向")
    h = np.asarray(hidden, dtype=np.float64)
    projection = h @ v
    return h + float(gate) * (float(target) - projection)[..., None] * v
