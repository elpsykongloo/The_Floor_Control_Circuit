"""双通道 VAD 后端与 VA 时间线工具。

Silero-VAD 仅在真正调用时导入（依赖 torch）；测试与下游状态机只依赖
纯 numpy 的段/栅格互转工具。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.schemas import Seg


def rasterize(segs: list[Seg], dt: float, total_dur: float) -> np.ndarray:
    """把时间段列表栅格化为 dt 步长的布尔数组（左闭右开）。"""
    n = round(total_dur / dt)
    mask = np.zeros(n, dtype=bool)
    for s in segs:
        i0 = max(0, int(np.floor(s.start / dt + 1e-9)))
        i1 = min(n, int(np.ceil(s.end / dt - 1e-9)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def mask_to_segments(mask: np.ndarray, dt: float) -> list[Seg]:
    """布尔数组还原为段列表。"""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return [Seg(float(i * dt), float(j * dt)) for i, j in zip(starts, ends, strict=True)]


class SileroVad:
    """Silero-VAD v5 封装（冻结参数：threshold 0.5 / min_speech 120 ms / min_silence 180 ms）。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg["vad"]
        self._model = None

    def _load(self):
        if self._model is None:
            from silero_vad import load_silero_vad  # 延迟导入：依赖 torch

            self._model = load_silero_vad()
        return self._model

    def segments(self, wav: np.ndarray, sr: int) -> list[Seg]:
        """单通道波形 → 语音段列表（秒）。输入任意采样率，内部重采样至 16 kHz。"""
        import torch  # 延迟导入
        from silero_vad import get_speech_timestamps

        target_sr = int(self.cfg["sample_rate"])
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"期望单通道波形，得到 shape={wav.shape}")
        if sr != target_sr:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        model = self._load()
        stamps = get_speech_timestamps(
            torch.from_numpy(wav),
            model,
            threshold=float(self.cfg["threshold"]),
            min_speech_duration_ms=int(self.cfg["min_speech_ms"]),
            min_silence_duration_ms=int(self.cfg["min_silence_ms"]),
            sampling_rate=target_sr,
        )
        return [Seg(s["start"] / target_sr, s["end"] / target_sr) for s in stamps]
