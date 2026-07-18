"""基线族（文档/00 §6-E1 步骤 2）：声学特征、hazard-rate 生存模型。
（Mimi 潜表征基线 = 对 runner 导出的 mimi_latent 复用 linear.py 的同一探针协议；
声学 GRU 见 gru.py。）"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from floor_circuit.schemas import State

ACOUSTIC_FEATURES = ("rms", "f0", "spectral_flux", "zcr")


def acoustic_frames(
    wav: np.ndarray,
    sr: int,
    hop_ms: float = 80.0,
    *,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """80 ms 帧声学特征 [T, 4]：RMS、F0、谱通量、ZCR（顺序冻结为 ACOUSTIC_FEATURES）。

    时间对齐前提（PREREG #7，mve/alignment.py 依赖）：帧 i 的观测不越过 (i+1)·hop。
    四个特征均为 frame≤2·hop + center=True（足迹 [i·hop−hop, i·hop+hop)）或
    parselmouth ±20 ms 分析窗，逐帧因果性由 tests/test_time_alignment.py 扰动核验。
    return_meta=True 时返回 (feats, {"f0_backend": ...})，供缓存元数据登记后端。
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    hop = round(sr * hop_ms / 1000.0)
    frame = 2 * hop
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=frame, hop_length=hop, center=True)[0]
    stft = np.abs(librosa.stft(wav, n_fft=frame, hop_length=hop, center=True))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1, prepend=stft[:, :1]) ** 2, axis=0))
    f0, f0_backend = _f0_track(wav, sr, hop)
    n = min(len(rms), len(zcr), len(flux), len(f0))
    feats = np.stack([rms[:n], f0[:n], flux[:n], zcr[:n]], axis=1).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    if return_meta:
        return feats, {"f0_backend": f0_backend}
    return feats


def _f0_track(wav: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, str]:
    """F0 轨迹：优先 parselmouth，异常时**显式警告**后退 librosa.yin。无声帧记 0。

    两条路径都必须满足"帧 i 观测 ≤ (i+1)·hop"：parselmouth 分析窗 3/75≈40 ms
    （帧时刻 ±20 ms < hop）；yin 用 frame_length=2·hop + center=True，与 RMS/ZCR
    同足迹（撤回前的 4·hop 会多看一帧未来，见 PREREG #7 审查记录）。
    返回 (f0, backend)，backend ∈ {"parselmouth", "yin"} 供缓存元数据登记。
    """
    n_frames = 1 + len(wav) // hop
    try:
        import parselmouth

        snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=hop / sr, pitch_floor=75.0, pitch_ceiling=500.0)
        values = pitch.selected_array["frequency"]
        t_pitch = pitch.xs()
        t_frames = np.arange(n_frames) * hop / sr
        f0 = np.interp(t_frames, t_pitch, values, left=0.0, right=0.0)
        f0[~np.isfinite(f0)] = 0.0
        return f0.astype(np.float32), "parselmouth"
    except Exception as exc:
        import sys

        import librosa

        print(
            f"[acoustic] 警告：parselmouth F0 失败（{exc!r}），退用 librosa.yin"
            "（足迹 2·hop，因果对齐保持）",
            file=sys.stderr,
        )
        f0 = librosa.yin(wav, fmin=75.0, fmax=500.0, sr=sr, frame_length=2 * hop, hop_length=hop)
        f0[~np.isfinite(f0)] = 0.0
        return f0[:n_frames].astype(np.float32), "yin"


def hazard_features(states: np.ndarray, step_s: float) -> np.ndarray:
    """离散时间 hazard 协变量 [T, 8]：
    自上次任一 ONSET、任一 OFFSET、说话人切换的时长，当前态时长，以及当前态四类 one-hot。

    T5 的 YIELD/HOLD/UNRESOLVED 都表示当前双通道同时活跃；为避免把未来重叠结局泄漏给
    hazard 基线，三者统一并入 OVERLAP。说话人切换指当前独占说话人相对最近一次独占
    说话人发生变化，中间经过 GAP 或 OVERLAP 仍可触发。
    """

    states = np.asarray(states, dtype=np.int64)
    if states.ndim != 1:
        raise ValueError("states 必须是一维数组")
    if step_s <= 0:
        raise ValueError("step_s 必须大于 0")
    valid = {state.value for state in State}
    unknown = sorted(set(np.unique(states).tolist()) - valid)
    if unknown:
        raise ValueError(f"states 含未知状态：{unknown}")

    n = len(states)
    if n == 0:
        return np.empty((0, 8), dtype=np.float32)
    overlap_states = {
        State.OVERLAP_YIELD.value,
        State.OVERLAP_HOLD.value,
        State.OVERLAP_UNRESOLVED.value,
    }
    agent_active = np.isin(states, [State.SPEAK.value, *overlap_states])
    other_active = np.isin(states, [State.LISTEN.value, *overlap_states])
    previous_agent = np.r_[False, agent_active[:-1]]
    previous_other = np.r_[False, other_active[:-1]]
    onset = (agent_active & ~previous_agent) | (other_active & ~previous_other)
    offset = (~agent_active & previous_agent) | (~other_active & previous_other)

    speaker_switch = np.zeros(n, dtype=bool)
    last_exclusive: int | None = None
    for index, (agent, other) in enumerate(zip(agent_active, other_active, strict=True)):
        current_exclusive = 0 if agent and not other else 1 if other and not agent else None
        if current_exclusive is None:
            continue
        if last_exclusive is not None and current_exclusive != last_exclusive:
            speaker_switch[index] = True
        last_exclusive = current_exclusive

    canonical = np.full(n, 3, dtype=np.int8)  # GAP
    canonical[(agent_active) & (~other_active)] = 0  # SPEAK
    canonical[(~agent_active) & (other_active)] = 1  # LISTEN
    canonical[agent_active & other_active] = 2  # OVERLAP

    current_duration = np.zeros(n, dtype=np.float32)
    last_change = 0
    for i in range(1, n):
        if canonical[i] != canonical[i - 1]:
            last_change = i
        current_duration[i] = (i - last_change) * step_s

    def elapsed_since(events: np.ndarray) -> np.ndarray:
        elapsed = np.zeros(n, dtype=np.float32)
        last_event = 0
        for i in range(1, n):
            if events[i]:
                last_event = i
            elapsed[i] = (i - last_event) * step_s
        return elapsed

    onehot = np.zeros((n, 4), dtype=np.float32)
    onehot[np.arange(n), canonical] = 1.0
    return np.concatenate(
        [
            elapsed_since(onset)[:, None],
            elapsed_since(offset)[:, None],
            elapsed_since(speaker_switch)[:, None],
            current_duration[:, None],
            onehot,
        ],
        axis=1,
    )


def fit_hazard(
    X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0
) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf
