"""刺激音频操作：倒放、SNR 混合、响度、F0 拉平、变调、变速（S1/S2/S5）。
重依赖（pyloudnorm/parselmouth/librosa）全部延迟导入。"""

from __future__ import annotations

import numpy as np


def reverse(wav: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(wav[::-1])


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)))) if len(wav) else 0.0


def clip_ratio(wav: np.ndarray, thresh: float = 0.999) -> float:
    return float(np.mean(np.abs(wav) >= thresh)) if len(wav) else 0.0


def fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """噪声裁剪/平铺到目标长度（随机起点）。"""
    if len(noise) == 0:
        raise ValueError("空噪声")
    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    start = int(rng.integers(0, len(noise) - n + 1)) if len(noise) > n else 0
    return noise[start : start + n].copy()


def mix_at_snr(
    signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """按 SNR 混合（噪声整形到目标长度后缩放）。"""
    noise = fit_length(np.asarray(noise, dtype=np.float32), len(signal), rng)
    rs, rn = rms(signal), rms(noise)
    if rn == 0:
        raise ValueError("噪声 RMS 为 0")
    gain = rs / (rn * (10.0 ** (snr_db / 20.0)))
    out = signal + gain * noise
    peak = np.max(np.abs(out))
    if peak > 0.999:  # 防削波，整体回缩
        out = out * (0.999 / peak)
    return out.astype(np.float32)


def energy_match(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """把 target 的 RMS 缩放到与 reference 一致（S2b 能量匹配 babble）。"""
    rt, rr = rms(target), rms(reference)
    if rt == 0:
        raise ValueError("目标 RMS 为 0")
    return (target * (rr / rt)).astype(np.float32)


def loudness_lufs(wav: np.ndarray, sr: int) -> float:
    import pyloudnorm

    meter = pyloudnorm.Meter(sr)
    return float(meter.integrated_loudness(np.asarray(wav, dtype=np.float64)))


def normalize_lufs(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    import pyloudnorm

    meter = pyloudnorm.Meter(sr)
    cur = meter.integrated_loudness(np.asarray(wav, dtype=np.float64))
    out = pyloudnorm.normalize.loudness(np.asarray(wav, dtype=np.float64), cur, target_lufs)
    peak = np.max(np.abs(out))
    if peak > 0.999:
        out = out * (0.999 / peak)
    return out.astype(np.float32)


def _to_sound(wav: np.ndarray, sr: int):
    import parselmouth

    return parselmouth.Sound(np.asarray(wav, dtype=np.float64), sampling_frequency=sr)


def f0_flatten(wav: np.ndarray, sr: int) -> np.ndarray:
    """F0 拉平到全句中位数（praat-parselmouth 操纵重合成，overlap-add）。"""
    from parselmouth.praat import call

    snd = _to_sound(wav, sr)
    manipulation = call(snd, "To Manipulation", 0.01, 75.0, 500.0)
    pitch_tier = call(manipulation, "Extract pitch tier")
    values = []
    n_pts = int(call(pitch_tier, "Get number of points"))
    for i in range(1, n_pts + 1):
        values.append(float(call(pitch_tier, "Get value at index", i)))
    if not values:
        return np.asarray(wav, dtype=np.float32)  # 全句无声/无基频，原样返回
    median = float(np.median(values))
    flat = call(snd, "Create PitchTier", "flat", 0.0, snd.duration)
    call(flat, "Add point", snd.duration / 2.0, median)
    call([manipulation, flat], "Replace pitch tier")
    out = call(manipulation, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32)


def pitch_shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """变调 ±半音（PSOLA；失败时退回 librosa 相位声码器）。"""
    try:
        from parselmouth.praat import call

        snd = _to_sound(wav, sr)
        manipulation = call(snd, "To Manipulation", 0.01, 75.0, 500.0)
        pitch_tier = call(manipulation, "Extract pitch tier")
        factor = 2.0 ** (semitones / 12.0)
        call(pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, factor)
        call([manipulation, pitch_tier], "Replace pitch tier")
        out = call(manipulation, "Get resynthesis (overlap-add)")
        return out.values[0].astype(np.float32)
    except Exception:
        import librosa

        return librosa.effects.pitch_shift(
            np.asarray(wav, dtype=np.float32), sr=sr, n_steps=semitones
        ).astype(np.float32)


def time_stretch(wav: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """变速（rate>1 加快）。PSOLA 时长域实现；失败退回 librosa。"""
    try:
        from parselmouth.praat import call

        snd = _to_sound(wav, sr)
        manipulation = call(snd, "To Manipulation", 0.01, 75.0, 500.0)
        tier = call(manipulation, "Extract duration tier")
        call(tier, "Add point", snd.duration / 2.0, 1.0 / rate)
        call([manipulation, tier], "Replace duration tier")
        out = call(manipulation, "Get resynthesis (overlap-add)")
        return out.values[0].astype(np.float32)
    except Exception:
        import librosa

        return librosa.effects.time_stretch(np.asarray(wav, dtype=np.float32), rate=rate).astype(
            np.float32
        )
