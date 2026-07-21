"""WAV 前缀 PCM 指纹（PREREG #16(d)）：计划器与审计侧的输入溯源。

与 runner 侧 ``runners/_shared/moshi_family.py::pcm_prefix_digest`` 逐字对齐
（跨环境不 import；``tests/test_e1_cache_plan.py`` 以同文件对拍保证两实现一致）。
指纹对象是文件里的原始 PCM 字节前缀，与读库（soundfile/wave）无关。
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path


def pcm_prefix_digest(path: str | Path, seconds: float) -> dict:
    """对 WAV 前 N 秒原始 PCM 字节做 sha256；返回含帧数与采样率的指纹字典。"""
    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        n_channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        compression = reader.getcomptype()
        if compression != "NONE":
            raise ValueError(f"{path} 非 PCM WAV（压缩类型 {compression}），无法计算前缀指纹")
        n_frames = min(int(reader.getnframes()), round(float(seconds) * sample_rate))
        hasher = hashlib.sha256()
        hasher.update(
            f"pcm:{sample_rate}:{n_channels}:{sample_width}:{n_frames}".encode("ascii") + b"\0"
        )
        remaining = n_frames
        while remaining > 0:
            data = reader.readframes(min(remaining, 1 << 16))
            if not data:
                raise ValueError(f"{path} PCM 数据提前结束：仍缺 {remaining} 帧")
            got, tail = divmod(len(data), n_channels * sample_width)
            if tail:
                raise ValueError(f"{path} PCM 块字节数 {len(data)} 不是整帧")
            hasher.update(data)
            remaining -= got
    return {
        "sha256": hasher.hexdigest(),
        "n_frames": int(n_frames),
        "sample_rate": int(sample_rate),
        "seconds": float(seconds),
    }
