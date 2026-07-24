"""共享 VAD 掩码缓存（PREREG #40(a)/(d)）：wav → IPU 合并 dt 栅格掩码，按内容寻址落盘。

供 wp_e2_lite_deep 与 wp_e2_lite_r2_probe 复用；与 wp_e2_lite_analyze 的
逐运行 behavior 缓存互不干扰（那边只存指标不存掩码）。缓存键 = 文件 sha256
前缀 + 分析契约摘要，任何 VAD/IPU/窗口参数变化自动失效。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.vad import SileroVad, rasterize

MASK_CACHE_SCHEMA = "e1x_vad_mask_v1"
VAD_LOOKAHEAD_S = 1.0  # 边界 offset 合格性判定需要的前视（与 wp_e2_lite_analyze 一致）


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract(events_cfg: dict, total_dur: float, dt: float, sample_rate: int) -> dict:
    return {
        "schema": MASK_CACHE_SCHEMA,
        "vad": events_cfg["vad"],
        "ipu": events_cfg["ipu"],
        "total_dur": float(total_dur),
        "dt": float(dt),
        "sample_rate": int(sample_rate),
        "lookahead_s": VAD_LOOKAHEAD_S,
    }


def _contract_digest(contract: dict) -> str:
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_mask(
    vad: SileroVad,
    wav_path: Path,
    events_cfg: dict,
    *,
    total_dur: float,
    dt: float,
    sample_rate: int,
) -> np.ndarray:
    import soundfile as sf

    max_frames = round((float(total_dur) + VAD_LOOKAHEAD_S) * sample_rate)
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False, frames=max_frames)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(f"{wav_path} 采样率 {sr} ≠ {sample_rate}")
    segments = vad.segments(np.asarray(wav, dtype=np.float32), sr)
    ipus = build_ipus(segments, float(events_cfg["ipu"]["merge_gap_s"]))
    return rasterize(ipus, dt, float(total_dur))


def cached_mask(
    cache_root: Path,
    vad: SileroVad,
    wav_path: Path,
    events_cfg: dict,
    *,
    total_dur: float,
    dt: float,
    sample_rate: int,
    force: bool = False,
) -> np.ndarray:
    """带内容寻址缓存的掩码计算；缓存损坏/契约不符时自动重算覆盖。"""
    wav_path = Path(wav_path)
    contract = _contract(events_cfg, total_dur, dt, sample_rate)
    key = f"{file_sha256(wav_path)[:24]}__{_contract_digest(contract)[:12]}"
    cache_root = Path(cache_root)
    npy_path = cache_root / f"{key}.npy"
    expected_steps = round(float(total_dur) / dt)
    if not force and npy_path.is_file():
        try:
            mask = np.load(npy_path, allow_pickle=False)
            if mask.shape == (expected_steps,) and mask.dtype == np.bool_:
                return mask
        except (OSError, ValueError):
            pass
    mask = compute_mask(
        vad, wav_path, events_cfg, total_dur=total_dur, dt=dt, sample_rate=sample_rate
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = npy_path.with_name(f".{npy_path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, mask.astype(np.bool_), allow_pickle=False)
        tmp.replace(npy_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return mask.astype(np.bool_)


def shifted_agent_mask(
    mask_agent_local: np.ndarray, first_emitted_frame: int, frame_s: float, dt: float
) -> np.ndarray:
    """把生成音频的本地掩码平移到输入时间轴（agent.wav 起点 = 首发射帧时刻）。

    R2 生成的 agent.wav 样本 0 对应输入帧 first_emitted 的起始时刻——
    对齐行为标签与激活行时必须加此偏移。
    """
    mask_agent_local = np.asarray(mask_agent_local, dtype=bool)
    offset = round(int(first_emitted_frame) * float(frame_s) / float(dt))
    out = np.zeros(len(mask_agent_local), dtype=bool)
    if offset >= len(out):
        return out
    take = len(out) - offset
    out[offset:] = mask_agent_local[:take]
    return out
