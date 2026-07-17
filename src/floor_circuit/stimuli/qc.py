"""刺激质检（configs/stimuli.yaml qc 节）：采样率、最小对时长 ±5%、响度 ±0.5 LU、削波。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from floor_circuit.stimuli.audio_ops import clip_ratio, loudness_lufs


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def qc_single(path: str | Path, qc_cfg: dict, expect_sr: int) -> dict:
    wav, sr = load_wav(path)
    return {
        "path": str(path),
        "sr": sr,
        "sr_ok": sr == expect_sr,
        "duration_s": len(wav) / sr if sr else 0.0,
        "lufs": loudness_lufs(wav, sr),
        "clip_ratio": clip_ratio(wav),
        "clip_ok": clip_ratio(wav) <= float(qc_cfg["max_clip_ratio"]),
    }


def qc_pair(path_a: str | Path, path_b: str | Path, qc_cfg: dict, expect_sr: int) -> dict:
    """最小对（如 S1 完整/不完整、S2a/S2b 能量匹配对）的配平检查。"""
    a, b = qc_single(path_a, qc_cfg, expect_sr), qc_single(path_b, qc_cfg, expect_sr)
    dur_ref = max(a["duration_s"], b["duration_s"], 1e-9)
    dur_diff_pct = abs(a["duration_s"] - b["duration_s"]) / dur_ref * 100.0
    lufs_diff = abs(a["lufs"] - b["lufs"])
    return {
        "a": a["path"],
        "b": b["path"],
        "sr_ok": a["sr_ok"] and b["sr_ok"],
        "clip_ok": a["clip_ok"] and b["clip_ok"],
        "duration_diff_pct": dur_diff_pct,
        "duration_ok": dur_diff_pct <= float(qc_cfg["duration_tol_pct"]),
        "lufs_diff": lufs_diff,
        "loudness_ok": lufs_diff <= float(qc_cfg["loudness_tol_lu"]),
    }


def qc_report(pair_rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(pair_rows)
    if df.empty:
        return df, {"n_pairs": 0, "pass_rate": 0.0}
    df["pass"] = df["sr_ok"] & df["clip_ok"] & df["duration_ok"] & df["loudness_ok"]
    summary = {
        "n_pairs": len(df),
        "pass_rate": float(df["pass"].mean()),
        "fail_duration": int((~df["duration_ok"]).sum()),
        "fail_loudness": int((~df["loudness_ok"]).sum()),
        "fail_sr": int((~df["sr_ok"]).sum()),
        "fail_clip": int((~df["clip_ok"]).sum()),
    }
    return df, summary
