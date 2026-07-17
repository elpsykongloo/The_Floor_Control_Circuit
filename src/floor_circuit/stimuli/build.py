"""S1–S5 生产编排。TTS 经 configs/paths.windows.yaml 的 tts.synth_command 模板调用
（V6 冻结音色与命令后启用；模板可为参数列表或字符串，占位符
{python} {text} {voice} {out_wav} {seed}）。质检见 qc.py。"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from floor_circuit.stimuli.textgen import generate_s1


class TtsNotConfigured(RuntimeError):
    pass


def _render_command(paths_cfg: dict, text: str, voice: str, out_wav: Path, seed: int) -> list[str]:
    tts = paths_cfg.get("tts", {})
    tpl = tts.get("synth_command")
    if not tpl:
        raise TtsNotConfigured(
            "tts.synth_command 未配置：V6 定选音色后，在 configs/paths.windows.yaml 填入 "
            "qwen3tts-custom 的实际合成命令模板（建议参数列表形式）"
        )
    mapping = {
        "python": tts.get("venv_python", "python"),
        "text": text,
        "voice": voice,
        "out_wav": str(out_wav),
        "seed": str(seed),
    }
    if isinstance(tpl, list):
        return [str(part).format(**mapping) for part in tpl]
    rendered = str(tpl).format(**mapping)
    return shlex.split(rendered, posix=(os.name != "nt"))


def synthesize(paths_cfg: dict, text: str, voice: str, out_wav: str | Path, seed: int) -> Path:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = _render_command(paths_cfg, text, voice, out_wav, seed)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(f"TTS 失败（exit {proc.returncode}）：{cmd}\nstderr: {proc.stderr[-2000:]}")
    return out_wav


def s1_manifest(lang: str, out_root: str | Path, stimuli_cfg: dict) -> pd.DataFrame:
    """S1 文本清单（不含音频）：texts 阶段产物，供 synth 阶段消费。"""
    items = generate_s1(lang)
    if len(items) != int(stimuli_cfg["s1"]["pairs_per_lang"]):
        raise ValueError("S1 条目数与冻结规格不符")
    out_root = Path(out_root)
    rows = []
    for it in items:
        base = out_root / "s1" / lang / it["id"]
        rows.append(
            {
                **it,
                "wav_complete": str(base / "complete.wav"),
                "wav_incomplete": str(base / "incomplete.wav"),
                "wav_complete_f0flat": str(base / "complete_f0flat.wav"),
                "wav_incomplete_f0flat": str(base / "incomplete_f0flat.wav"),
            }
        )
    return pd.DataFrame(rows)


def s2_trials_manifest(lang: str, out_root: str | Path, stimuli_cfg: dict, seed: int) -> pd.DataFrame:
    """S2 试次清单：每类 trials_per_type 条，onset 在宿主语句 20–80% 均匀采样（种子固定）。
    音频池：instruct/backchannel 由 TTS 合成（多种子变体循环），babble 取 MUSAN，
    reversed 为 instruct 倒放，crosslingual 直接复用另一语言 instruct 池。"""
    s2 = stimuli_cfg["s2"]
    n = int(s2["trials_per_type"])
    lo, hi = (float(x) for x in s2["onset_range_pct"])
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    rows = []
    n_variants = 20
    for typ in s2["types"]:
        src_lang = ("zh" if lang == "en" else "en") if typ == "crosslingual" else lang
        pool = "instruct" if typ in ("reversed", "crosslingual") else typ
        for i in range(n):
            variant = i % n_variants
            rows.append(
                {
                    "trial_id": f"s2_{lang}_{typ}_{i:03d}",
                    "lang": lang,
                    "type": typ,
                    "source_lang": src_lang,
                    "variant": variant,
                    "onset_pct": float(rng.uniform(lo, hi)),
                    "wav": str(out_root / "s2" / src_lang / pool / f"v{variant:02d}.wav"),
                    "reversed": typ == "reversed",
                }
            )
    return pd.DataFrame(rows)


def s2_pool_texts(lang: str, stimuli_cfg: dict) -> dict[str, str]:
    s2 = stimuli_cfg["s2"]
    return {"instruct": s2["interrupt_text"][lang], "backchannel": s2["backchannel_text"][lang]}


def cut_pause_scenario(
    wav: np.ndarray,
    sr: int,
    pause_start_s: float,
    lead_in_s: float,
    tail_silence_s: float,
) -> np.ndarray:
    """S3：截到停顿起点为止的语音 + 人工静默尾（供"停顿处理"场景）。"""
    i1 = round(pause_start_s * sr)
    i0 = max(0, i1 - round(lead_in_s * sr))
    clip = wav[i0:i1]
    tail = np.zeros(round(tail_silence_s * sr), dtype=np.float32)
    return np.concatenate([clip, tail]).astype(np.float32)


def cut_turnend_scenario(
    wav: np.ndarray, sr: int, turn_start_s: float, turn_end_s: float, tail_silence_s: float
) -> np.ndarray:
    """S4：完整问句/完整 turn + 静默尾（供"应答"场景）。"""
    i0 = max(0, round(turn_start_s * sr))
    i1 = round(turn_end_s * sr)
    clip = wav[i0:i1]
    tail = np.zeros(round(tail_silence_s * sr), dtype=np.float32)
    return np.concatenate([clip, tail]).astype(np.float32)
