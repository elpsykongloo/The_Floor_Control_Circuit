"""基线族（文档/00 §6-E1 步骤 2）：声学特征、hazard-rate 生存模型。
（Mimi 潜表征基线 = 对 runner 导出的 mimi_latent 复用 linear.py 的同一探针协议；
声学 GRU 见 gru.py。）"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

ACOUSTIC_FEATURES = ("rms", "f0", "spectral_flux", "zcr")


def acoustic_frames(wav: np.ndarray, sr: int, hop_ms: float = 80.0) -> np.ndarray:
    """80 ms 帧声学特征 [T, 4]：RMS、F0、谱通量、ZCR（顺序冻结为 ACOUSTIC_FEATURES）。"""
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    hop = round(sr * hop_ms / 1000.0)
    frame = 2 * hop
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=frame, hop_length=hop, center=True)[0]
    stft = np.abs(librosa.stft(wav, n_fft=frame, hop_length=hop, center=True))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1, prepend=stft[:, :1]) ** 2, axis=0))
    f0 = _f0_track(wav, sr, hop)
    n = min(len(rms), len(zcr), len(flux), len(f0))
    feats = np.stack([rms[:n], f0[:n], flux[:n], zcr[:n]], axis=1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _f0_track(wav: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """F0 轨迹：优先 parselmouth（快、稳），缺失时退回 librosa.yin。无声帧记 0。"""
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
        return f0.astype(np.float32)
    except Exception:
        import librosa

        f0 = librosa.yin(wav, fmin=75.0, fmax=500.0, sr=sr, frame_length=4 * hop, hop_length=hop)
        f0[~np.isfinite(f0)] = 0.0
        return f0[:n_frames].astype(np.float32)


def hazard_features(states: np.ndarray, step_s: float) -> np.ndarray:
    """离散时间 hazard 协变量 [T, 8]：
    当前态时长、自上次状态切换时长、自上次任一方发声段结束时长（近似）、
    以及当前态 one-hot（5 类，UNRESOLVED 并入 GAP 桶之外的第 5 位）。"""
    n = len(states)
    cur_dur = np.zeros(n, dtype=np.float32)
    since_change = np.zeros(n, dtype=np.float32)
    last_change = 0
    for i in range(1, n):
        if states[i] != states[i - 1]:
            last_change = i
        cur_dur[i] = (i - last_change) * step_s
        since_change[i] = cur_dur[i]
    onehot = np.zeros((n, 5), dtype=np.float32)
    clipped = np.clip(states, 0, 4)
    onehot[np.arange(n), clipped] = 1.0
    t_abs = (np.arange(n, dtype=np.float32) * step_s)[:, None]
    return np.concatenate([cur_dur[:, None], since_change[:, None], np.log1p(t_abs), onehot], axis=1)


def fit_hazard(
    X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0
) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(X_tr, y_tr)
    return clf
