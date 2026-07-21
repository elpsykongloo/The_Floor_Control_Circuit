"""G0 修复后失败的事后诊断。

本脚本只读取已经生成的校准、确认与金标元数据，不调用 VAD，不改写 Gate 配置，
也不产生新的确认裁决。输出仅用于解释修复为何未能跨集合泛化。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from _bootstrap import REPO_ROOT
from scipy import stats
from scipy.io import wavfile

from floor_circuit.config import data_root, load_paths
from floor_circuit.events.g0 import G0_CLASSES

ROOT = REPO_ROOT
REPORTS_DIR = ROOT / "reports"
DEFAULT_PRE_REPAIR_REF = "a1a11f1^"
BOOTSTRAP_SEED = 20260721
N_BOOTSTRAP = 20_000
FRAME_HZ = 12.5


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_git_json(ref: str, relative_path: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _session_number(session: str) -> int:
    return int(session.removeprefix("sw"))


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10, method="linear")),
        "p25": float(np.percentile(values, 25, method="linear")),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75, method="linear")),
        "p90": float(np.percentile(values, 90, method="linear")),
        "max": float(np.max(values)),
    }


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
) -> dict[str, float]:
    """返回 left - right 的独立样本 bootstrap 区间。"""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        left_sample = left[rng.integers(0, left.size, size=left.size)]
        right_sample = right[rng.integers(0, right.size, size=right.size)]
        samples[index] = statistic(left_sample) - statistic(right_sample)
    return {
        "estimate": float(statistic(left) - statistic(right)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _bootstrap_paired(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        sample = values[rng.integers(0, values.size, size=values.size)]
        samples[index] = np.mean(sample)
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _duration_band(duration_s: float) -> str:
    if duration_s <= 330:
        return "≤330秒"
    if duration_s <= 480:
        return "331–480秒"
    return ">480秒"


def _id_band(session_id: int) -> str:
    if session_id < 2500:
        return "2000–2499"
    if session_id < 3000:
        return "2500–2999"
    if session_id < 3500:
        return "3000–3499"
    if session_id < 4000:
        return "3500–3999"
    if session_id < 4140:
        return "4000–4139"
    if session_id < 4200:
        return "4140–4199"
    if session_id < 4300:
        return "4200–4299"
    if session_id < 4400:
        return "4300–4399"
    if session_id < 4500:
        return "4400–4499"
    return "4500及以上"


def _load_session_rows(report: dict[str, Any], cohort: str, prep_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report["per_session"]:
        session = str(item["session"])
        meta = _load_json(prep_root / session / "meta.json")
        duration_s = float(meta["duration_s"])
        gold_counts = dict.fromkeys(G0_CLASSES, 0)
        vad_rates: list[float] = []
        for channel in (0, 1):
            with np.load(prep_root / session / f"gold_ch{channel}.npz") as gold:
                vad_rates.append(float(np.mean(gold["vad"])))
                for event_class in G0_CLASSES:
                    gold_counts[event_class] += int(np.sum(gold[event_class]))

        rows.append(
            {
                "cohort": cohort,
                "session": session,
                "session_id": _session_number(session),
                "macro_f1": float(item["macro_f1_mean"]),
                "duration_s": duration_s,
                "duration_band": _duration_band(duration_s),
                "id_band": _id_band(_session_number(session)),
                "num_frames": int(meta["num_frames"]),
                "vad_rate_ch0": vad_rates[0],
                "vad_rate_ch1": vad_rates[1],
                "vad_rate_mean": float(np.mean(vad_rates)),
                **{f"gold_{event_class}": gold_counts[event_class] for event_class in G0_CLASSES},
                **{
                    f"gold_{event_class}_per_hour": gold_counts[event_class] / (duration_s / 3600)
                    for event_class in G0_CLASSES
                },
            }
        )
    return rows


def _power_dbfs(sum_squares: float, n_samples: int) -> float:
    if n_samples <= 0 or sum_squares <= 0:
        return float("-inf")
    return float(10 * np.log10(sum_squares / n_samples / (32768.0**2)))


def _channel_audio_stats(prep_root: Path, session: str, channel: int) -> dict[str, float]:
    """在金标 VAD 区域内计算解码波形能量，不调用预测 VAD。"""
    with np.load(prep_root / session / f"gold_ch{channel}.npz") as gold:
        vad = np.asarray(gold["vad"], dtype=bool)

    sample_rate, audio = wavfile.read(prep_root / session / f"audio_ch{channel}.wav", mmap=True)
    if audio.ndim != 1 or audio.dtype != np.int16:
        raise RuntimeError(f"{session}/ch{channel} 音频制式异常：shape={audio.shape}, dtype={audio.dtype}")
    samples_per_frame = round(sample_rate / FRAME_HZ)
    expected_samples = vad.size * samples_per_frame
    if audio.size != expected_samples:
        raise RuntimeError(f"{session}/ch{channel} 音频长度异常：{audio.size} != {vad.size}*{samples_per_frame}")

    frames = audio.reshape(vad.size, samples_per_frame)
    speech_sum_squares = 0.0
    nonspeech_sum_squares = 0.0
    speech_samples = 0
    nonspeech_samples = 0
    speech_frame_dbfs_parts: list[np.ndarray] = []
    for start in range(0, vad.size, 250):
        stop = min(start + 250, vad.size)
        block = frames[start:stop]
        block_vad = vad[start:stop]
        block_float = block.astype(np.float64, copy=False)
        frame_sum_squares = np.einsum("ij,ij->i", block_float, block_float)
        if np.any(block_vad):
            speech_frame_sums = frame_sum_squares[block_vad]
            speech_sum_squares += float(np.sum(speech_frame_sums))
            speech_samples += int(speech_frame_sums.size * samples_per_frame)
            speech_frame_dbfs_parts.append(
                10
                * np.log10(
                    np.maximum(
                        speech_frame_sums / samples_per_frame / (32768.0**2),
                        np.finfo(np.float64).tiny,
                    )
                )
            )
        if np.any(~block_vad):
            nonspeech_frame_sums = frame_sum_squares[~block_vad]
            nonspeech_sum_squares += float(np.sum(nonspeech_frame_sums))
            nonspeech_samples += int(nonspeech_frame_sums.size * samples_per_frame)

    speech_frame_dbfs = np.concatenate(speech_frame_dbfs_parts)

    return {
        "speech_dbfs": _power_dbfs(speech_sum_squares, speech_samples),
        "nonspeech_dbfs": _power_dbfs(nonspeech_sum_squares, nonspeech_samples),
        "all_dbfs": _power_dbfs(
            speech_sum_squares + nonspeech_sum_squares,
            speech_samples + nonspeech_samples,
        ),
        "speech_frame_count": float(speech_frame_dbfs.size),
        "speech_frame_dbfs_p10": float(np.percentile(speech_frame_dbfs, 10, method="linear")),
        "speech_frame_dbfs_p25": float(np.percentile(speech_frame_dbfs, 25, method="linear")),
        "speech_frame_dbfs_median": float(np.median(speech_frame_dbfs)),
        "speech_frames_below_m50": float(np.sum(speech_frame_dbfs < -50)),
        "speech_frames_below_m45": float(np.sum(speech_frame_dbfs < -45)),
        "speech_frames_below_m40": float(np.sum(speech_frame_dbfs < -40)),
    }


def _session_audio_stats(task: tuple[str, str]) -> tuple[str, dict[str, float]]:
    prep_root_text, session = task
    prep_root = Path(prep_root_text)
    channels = [_channel_audio_stats(prep_root, session, channel) for channel in (0, 1)]
    speech_frame_count = sum(item["speech_frame_count"] for item in channels)
    result = {
        "speech_dbfs_ch0": channels[0]["speech_dbfs"],
        "speech_dbfs_ch1": channels[1]["speech_dbfs"],
        "speech_dbfs_mean": float(np.mean([item["speech_dbfs"] for item in channels])),
        "nonspeech_dbfs_ch0": channels[0]["nonspeech_dbfs"],
        "nonspeech_dbfs_ch1": channels[1]["nonspeech_dbfs"],
        "nonspeech_dbfs_mean": float(np.mean([item["nonspeech_dbfs"] for item in channels])),
        "all_dbfs_ch0": channels[0]["all_dbfs"],
        "all_dbfs_ch1": channels[1]["all_dbfs"],
        "all_dbfs_mean": float(np.mean([item["all_dbfs"] for item in channels])),
        "speech_frame_count": float(speech_frame_count),
        "speech_frame_dbfs_p10_mean": float(np.mean([item["speech_frame_dbfs_p10"] for item in channels])),
        "speech_frame_dbfs_p25_mean": float(np.mean([item["speech_frame_dbfs_p25"] for item in channels])),
        "speech_frame_dbfs_median_mean": float(np.mean([item["speech_frame_dbfs_median"] for item in channels])),
    }
    for threshold in (50, 45, 40):
        count = sum(item[f"speech_frames_below_m{threshold}"] for item in channels)
        result[f"speech_frames_below_m{threshold}"] = float(count)
        result[f"speech_fraction_below_m{threshold}"] = float(count / speech_frame_count)
    return session, result


def _attach_audio_stats(rows: list[dict[str, Any]], prep_root: Path, jobs: int) -> None:
    _attach_audio_stats_with_cache(rows, prep_root, jobs, {})


def _attach_audio_stats_with_cache(
    rows: list[dict[str, Any]],
    prep_root: Path,
    jobs: int,
    cache: dict[str, dict[str, Any]],
) -> None:
    required_key = "speech_fraction_below_m50"
    missing_rows = [
        row for row in rows if str(row["session"]) not in cache or required_key not in cache[str(row["session"])]
    ]
    audio_by_session: dict[str, dict[str, float]] = {}
    for row in rows:
        session = str(row["session"])
        if session in cache and required_key in cache[session]:
            audio_by_session[session] = {
                key: value
                for key, value in cache[session].items()
                if key.startswith(("speech_", "nonspeech_", "all_dbfs"))
            }

    if missing_rows:
        worker_count = max(1, min(jobs, len(missing_rows)))
        tasks = [(str(prep_root), str(row["session"])) for row in missing_rows]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            audio_by_session.update(dict(executor.map(_session_audio_stats, tasks, chunksize=1)))

    for row in rows:
        row.update(audio_by_session[str(row["session"])])


def _group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)

    output: list[dict[str, Any]] = []
    for group, items in groups.items():
        macros = np.array([float(item["macro_f1"]) for item in items])
        durations = np.array([float(item["duration_s"]) for item in items])
        output.append(
            {
                "group": group,
                **_summary(macros),
                "duration_hours": float(np.sum(durations) / 3600),
                "mean_vad_rate": float(np.mean([float(item["vad_rate_mean"]) for item in items])),
            }
        )
    return output


def _spearman(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    x = [float(row[field]) for row in rows]
    y = [float(row["macro_f1"]) for row in rows]
    result = stats.spearmanr(x, y)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def _aggregate_gold_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration_hours = sum(float(row["duration_s"]) for row in rows) / 3600
    return {
        "n_sessions": len(rows),
        "duration_hours": float(duration_hours),
        "mean_vad_rate_ch0": float(np.mean([float(row["vad_rate_ch0"]) for row in rows])),
        "mean_vad_rate_ch1": float(np.mean([float(row["vad_rate_ch1"]) for row in rows])),
        "gold_events_per_hour": {
            event_class: float(sum(int(row[f"gold_{event_class}"]) for row in rows) / duration_hours)
            for event_class in G0_CLASSES
        },
    }


def _aggregate_low_energy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_count = sum(float(row["speech_frame_count"]) for row in rows)
    return {
        "speech_frame_count": int(frame_count),
        **{
            f"fraction_below_minus_{threshold}_dbfs": float(
                sum(float(row[f"speech_frames_below_m{threshold}"]) for row in rows) / frame_count
            )
            for threshold in (50, 45, 40)
        },
        "per_session_speech_frame_p10_dbfs": _summary(
            np.array([float(row["speech_frame_dbfs_p10_mean"]) for row in rows])
        ),
    }


def _class_contributions(val: dict[str, Any], train: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_class in G0_CLASSES:
        val_item = val["per_class"][event_class]
        train_item = train["per_class"][event_class]
        delta = float(train_item["f1"] - val_item["f1"])
        rows.append(
            {
                "event_class": event_class,
                "val_f1": float(val_item["f1"]),
                "train_f1": float(train_item["f1"]),
                "delta_train_minus_val": delta,
                "macro_gap_contribution": delta / len(G0_CLASSES),
                "val_precision": float(val_item["precision"]),
                "train_precision": float(train_item["precision"]),
                "val_recall": float(val_item["recall"]),
                "train_recall": float(train_item["recall"]),
                "val_n_gold": int(val_item["n_gold"]),
                "train_n_gold": int(train_item["n_gold"]),
                "val_pred_gold_ratio": float(val_item["n_pred"] / val_item["n_gold"]),
                "train_pred_gold_ratio": float(train_item["n_pred"] / train_item["n_gold"]),
            }
        )
    return rows


def _repair_effects(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    pre_sessions = {item["session"]: float(item["macro_f1_mean"]) for item in pre["per_session"]}
    post_sessions = {item["session"]: float(item["macro_f1_mean"]) for item in post["per_session"]}
    if pre_sessions.keys() != post_sessions.keys():
        raise RuntimeError("修复前后 val 会话集合不一致，无法做配对比较")
    paired_delta = np.array([post_sessions[key] - pre_sessions[key] for key in sorted(pre_sessions)])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        "corpus_macro_f1_pre": float(pre["macro_f1"]),
        "corpus_macro_f1_post": float(post["macro_f1"]),
        "corpus_macro_f1_delta": float(post["macro_f1"] - pre["macro_f1"]),
        "paired_session_mean_delta": _bootstrap_paired(paired_delta, rng),
        "paired_session_fraction_improved": float(np.mean(paired_delta > 0)),
        "paired_session_fraction_unchanged": float(np.mean(paired_delta == 0)),
        "paired_session_fraction_worsened": float(np.mean(paired_delta < 0)),
        "paired_session_delta_p10": float(np.percentile(paired_delta, 10, method="linear")),
        "paired_session_delta_median": float(np.median(paired_delta)),
        "paired_session_delta_p90": float(np.percentile(paired_delta, 90, method="linear")),
        "per_class": {
            event_class: {
                "pre_f1": float(pre["per_class"][event_class]["f1"]),
                "post_f1": float(post["per_class"][event_class]["f1"]),
                "delta": float(post["per_class"][event_class]["f1"] - pre["per_class"][event_class]["f1"]),
            }
            for event_class in G0_CLASSES
        },
        "layer2": {
            channel: {
                metric: {
                    "pre": float(pre["layer2_vad"][channel][metric]),
                    "post": float(post["layer2_vad"][channel][metric]),
                    "delta": float(post["layer2_vad"][channel][metric] - pre["layer2_vad"][channel][metric]),
                }
                for metric in ("precision", "recall", "f1")
            }
            for channel in ("ch0", "ch1")
        },
    }


def _official_split_ids() -> dict[str, np.ndarray]:
    split_path = Path(load_paths()["datasets"]["dualturn"]) / "splits.json"
    payload = _load_json(split_path)
    groups: dict[str, list[int]] = {}
    for session, split in payload["splits"].items():
        groups.setdefault(str(split), []).append(_session_number(str(session)))
    return {split: np.sort(np.asarray(ids, dtype=np.int64)) for split, ids in groups.items()}


def _release_train_ids(official_train_ids: np.ndarray) -> np.ndarray:
    dataset_root = Path(load_paths()["datasets"]["dualturn"])
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        data_dir = dataset_root
    shards = sorted(data_dir.glob("train-*.parquet"))
    if not shards:
        shards = sorted(data_dir.glob("*.parquet"))
    release_ids: set[int] = set()
    for shard in shards:
        table = pq.read_table(shard, columns=["session_id"])
        release_ids.update(_session_number(str(item)) for item in table.column("session_id").to_pylist())
    return np.asarray(sorted(set(official_train_ids.tolist()) & release_ids), dtype=np.int64)


def _split_ranges(groups: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, values in sorted(groups.items()):
        rows.append(
            {
                "split": split,
                "n": int(values.size),
                "min_id": int(values[0]),
                "p10_id": int(np.percentile(values, 10, method="nearest")),
                "median_id": int(np.percentile(values, 50, method="nearest")),
                "p90_id": int(np.percentile(values, 90, method="nearest")),
                "max_id": int(values[-1]),
            }
        )
    return rows


def build_report(
    pre_repair_ref: str,
    audio_jobs: int,
    reuse_output: Path | None = None,
) -> dict[str, Any]:
    val = _load_json(REPORTS_DIR / "g0_summary.json")
    train = _load_json(REPORTS_DIR / "g0_train_confirmation.json")
    test = _load_json(REPORTS_DIR / "g0_confirmation.json")
    pre_val = _load_git_json(pre_repair_ref, "reports/g0_summary.json")
    roster = _load_json(ROOT / "configs" / "splits" / "dualturn_train_confirm.json")
    prep_root = data_root() / "dualturn_prep"

    val_rows = _load_session_rows(val, "val", prep_root)
    train_rows = _load_session_rows(train, "train_confirm", prep_root)
    cache: dict[str, dict[str, Any]] = {}
    if reuse_output is not None and reuse_output.exists():
        previous = _load_json(reuse_output)
        cache = {str(row["session"]): row for row in previous.get("session_rows", [])}
    _attach_audio_stats_with_cache(val_rows + train_rows, prep_root, audio_jobs, cache)
    val_values = np.array([float(row["macro_f1"]) for row in val_rows])
    train_values = np.array([float(row["macro_f1"]) for row in train_rows])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ks = stats.ks_2samp(train_values, val_values, alternative="two-sided", method="auto")
    mw = stats.mannwhitneyu(train_values, val_values, alternative="two-sided", method="auto")

    official_ids = _official_split_ids()
    release_train_ids = _release_train_ids(official_ids["train"])
    roster_ids = np.asarray([_session_number(session) for session in roster["sessions"]], dtype=np.int64)
    roster_ks = stats.ks_2samp(roster_ids, release_train_ids, alternative="two-sided", method="auto")

    val_short = [row for row in val_rows if row["duration_band"] == "≤330秒"]
    train_short = [row for row in train_rows if row["duration_band"] == "≤330秒"]
    train_near_boundary = [row for row in train_rows if int(row["session_id"]) >= 3500]

    class_rows = _class_contributions(val, train)
    history = [
        {
            "pipeline": "修复前",
            "split": "val",
            "n_sessions": int(pre_val["n_sessions"]),
            "layer2_ch0_f1": float(pre_val["layer2_vad"]["ch0"]["f1"]),
            "layer2_ch1_f1": float(pre_val["layer2_vad"]["ch1"]["f1"]),
            "layer3_corpus_macro_f1": float(pre_val["macro_f1"]),
            "session_p10": float(
                np.percentile(
                    [item["macro_f1_mean"] for item in pre_val["per_session"]],
                    10,
                    method="linear",
                )
            ),
        },
        {
            "pipeline": "修复前",
            "split": "test确认",
            "n_sessions": int(test["n_sessions"]),
            "layer2_ch0_f1": float(test["layer2_vad"]["ch0"]["f1"]),
            "layer2_ch1_f1": float(test["layer2_vad"]["ch1"]["f1"]),
            "layer3_corpus_macro_f1": float(test["macro_f1"]),
            "session_p10": float(test["gate"]["conditions"]["layer3_session_p10"]["p10"]),
        },
        {
            "pipeline": "阈值0.40",
            "split": "val",
            "n_sessions": int(val["n_sessions"]),
            "layer2_ch0_f1": float(val["layer2_vad"]["ch0"]["f1"]),
            "layer2_ch1_f1": float(val["layer2_vad"]["ch1"]["f1"]),
            "layer3_corpus_macro_f1": float(val["macro_f1"]),
            "session_p10": float(np.percentile(val_values, 10, method="linear")),
        },
        {
            "pipeline": "阈值0.40",
            "split": "train确认",
            "n_sessions": int(train["n_sessions"]),
            "layer2_ch0_f1": float(train["layer2_vad"]["ch0"]["f1"]),
            "layer2_ch1_f1": float(train["layer2_vad"]["ch1"]["f1"]),
            "layer3_corpus_macro_f1": float(train["macro_f1"]),
            "session_p10": float(train["gate"]["conditions"]["layer3_session_p10"]["p10"]),
        },
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "analysis_type": "事后诊断，非判据，不得用于重判当前确认结果",
        "sources": {
            "val_post_repair": "reports/g0_summary.json",
            "train_confirmation": "reports/g0_train_confirmation.json",
            "test_pre_repair": "reports/g0_confirmation.json",
            "val_pre_repair": f"git:{pre_repair_ref}:reports/g0_summary.json",
            "roster": "configs/splits/dualturn_train_confirm.json",
            "official_splits": "DualTurn本机发布物/splits.json",
            "gold_metadata": "D盘dualturn_prep中的meta.json与gold_ch*.npz",
        },
        "repair_effect_on_same_val": _repair_effects(pre_val, val),
        "history": history,
        "gate_failure": {
            "verdict": train["gate"]["verdict"],
            "hard_failures": train["gate"]["hard_failures"],
            "corpus_macro_f1": float(train["macro_f1"]),
            "corpus_band": train["gate"]["conditions"]["layer3_corpus_band"]["band"],
            "corpus_lower_margin": float(
                train["macro_f1"] - train["gate"]["conditions"]["layer3_corpus_band"]["band"][0]
            ),
            "session_p10": float(train["gate"]["conditions"]["layer3_session_p10"]["p10"]),
            "session_p10_min": float(train["gate"]["conditions"]["layer3_session_p10"]["required_min"]),
            "session_p10_margin": float(
                train["gate"]["conditions"]["layer3_session_p10"]["p10"]
                - train["gate"]["conditions"]["layer3_session_p10"]["required_min"]
            ),
        },
        "split_construction": {
            "official_ranges": _split_ranges(official_ids),
            "release_available_train_n": int(release_train_ids.size),
            "roster_id_summary": _summary(roster_ids.astype(np.float64)),
            "roster_vs_release_available_train_id_ks": {
                "statistic": float(roster_ks.statistic),
                "p_value": float(roster_ks.pvalue),
                "note": "只检验 roster 是否明显偏离 train 会话编号分布，不证明内容同分布",
            },
        },
        "session_distribution": {
            "val": _summary(val_values),
            "train_confirmation": _summary(train_values),
            "train_minus_val_mean_bootstrap": _bootstrap_difference(train_values, val_values, np.mean, rng),
            "train_minus_val_p10_bootstrap": _bootstrap_difference(
                train_values,
                val_values,
                lambda values: float(np.percentile(values, 10, method="linear")),
                rng,
            ),
            "ks_two_sample": {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)},
            "mann_whitney": {
                "u": float(mw.statistic),
                "p_value": float(mw.pvalue),
                "probability_train_greater_than_val": float(mw.statistic / (train_values.size * val_values.size)),
            },
        },
        "layer2_shift": {
            channel: {
                metric: {
                    "val": float(val["layer2_vad"][channel][metric]),
                    "train_confirmation": float(train["layer2_vad"][channel][metric]),
                    "delta_train_minus_val": float(
                        train["layer2_vad"][channel][metric] - val["layer2_vad"][channel][metric]
                    ),
                }
                for metric in ("precision", "recall", "f1")
            }
            for channel in ("ch0", "ch1")
        },
        "class_contributions": class_rows,
        "composition": {
            "val_gold": _aggregate_gold_rates(val_rows),
            "train_gold": _aggregate_gold_rates(train_rows),
            "duration_bands_val": _group_summary(val_rows, "duration_band"),
            "duration_bands_train": _group_summary(train_rows, "duration_band"),
            "id_bands_val": _group_summary(val_rows, "id_band"),
            "id_bands_train": _group_summary(train_rows, "id_band"),
            "short_duration_comparison": {
                "val": _summary(np.array([row["macro_f1"] for row in val_short])),
                "train_confirmation": _summary(np.array([row["macro_f1"] for row in train_short])),
            },
            "train_near_split_boundary_id_ge_3500": _summary(
                np.array([row["macro_f1"] for row in train_near_boundary])
            ),
            "decoded_audio": {
                "val_speech_dbfs": _summary(np.array([row["speech_dbfs_mean"] for row in val_rows])),
                "train_speech_dbfs": _summary(np.array([row["speech_dbfs_mean"] for row in train_rows])),
                "train_minus_val_speech_dbfs_bootstrap": _bootstrap_difference(
                    np.array([row["speech_dbfs_mean"] for row in train_rows]),
                    np.array([row["speech_dbfs_mean"] for row in val_rows]),
                    np.mean,
                    rng,
                ),
                "val_all_dbfs": _summary(np.array([row["all_dbfs_mean"] for row in val_rows])),
                "train_all_dbfs": _summary(np.array([row["all_dbfs_mean"] for row in train_rows])),
                "val_low_energy_tail": _aggregate_low_energy(val_rows),
                "train_low_energy_tail": _aggregate_low_energy(train_rows),
                "val_per_session_fraction_below_minus_50_dbfs": _summary(
                    np.array([row["speech_fraction_below_m50"] for row in val_rows])
                ),
                "train_per_session_fraction_below_minus_50_dbfs": _summary(
                    np.array([row["speech_fraction_below_m50"] for row in train_rows])
                ),
                "train_minus_val_fraction_below_minus_50_dbfs_bootstrap": (
                    _bootstrap_difference(
                        np.array([row["speech_fraction_below_m50"] for row in train_rows]),
                        np.array([row["speech_fraction_below_m50"] for row in val_rows]),
                        np.mean,
                        rng,
                    )
                ),
                "train_minus_val_speech_frame_p10_dbfs_bootstrap": _bootstrap_difference(
                    np.array([row["speech_frame_dbfs_p10_mean"] for row in train_rows]),
                    np.array([row["speech_frame_dbfs_p10_mean"] for row in val_rows]),
                    np.mean,
                    rng,
                ),
            },
        },
        "correlations": {
            "val": {
                field: _spearman(val_rows, field)
                for field in (
                    "session_id",
                    "duration_s",
                    "vad_rate_mean",
                    "speech_dbfs_mean",
                    "speech_fraction_below_m50",
                    "gold_eot_per_hour",
                    "gold_hold_per_hour",
                    "gold_bot_per_hour",
                    "gold_bc_per_hour",
                )
            },
            "train_confirmation": {
                field: _spearman(train_rows, field)
                for field in (
                    "session_id",
                    "duration_s",
                    "vad_rate_mean",
                    "speech_dbfs_mean",
                    "speech_fraction_below_m50",
                    "gold_eot_per_hour",
                    "gold_hold_per_hour",
                    "gold_bot_per_hour",
                    "gold_bc_per_hour",
                )
            },
        },
        "session_rows": val_rows + train_rows,
        "qa_checks": {
            "zero_processing_errors": bool(
                val["n_errors"] == 0
                and train["n_errors"] == 0
                and val["n_incomplete_skipped"] == 0
                and train["n_incomplete_skipped"] == 0
            ),
            "layer1_exact_both": bool(
                val["layer1_protocol"]["macro_f1"] == 1.0 and train["layer1_protocol"]["macro_f1"] == 1.0
            ),
            "unique_sessions_both": bool(
                len({row["session"] for row in val_rows}) == len(val_rows)
                and len({row["session"] for row in train_rows}) == len(train_rows)
            ),
            "train_rows_match_roster": bool([row["session"] for row in train_rows] == roster["sessions"]),
            "class_contributions_reconcile": bool(
                np.isclose(
                    sum(row["macro_gap_contribution"] for row in class_rows),
                    train["macro_f1"] - val["macro_f1"],
                    rtol=0,
                    atol=1e-12,
                )
            ),
            "gold_counts_reconcile": bool(
                all(
                    sum(int(row[f"gold_{event_class}"]) for row in rows) == report["per_class"][event_class]["n_gold"]
                    for rows, report in ((val_rows, val), (train_rows, train))
                    for event_class in G0_CLASSES
                )
            ),
            "audio_lengths_verified_during_energy_scan": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-repair-ref", default=DEFAULT_PRE_REPAIR_REF)
    parser.add_argument(
        "--audio-jobs",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) - 2)),
        help="解码音频能量诊断的并行进程数",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "g0_postmortem.json",
    )
    args = parser.parse_args()

    report = build_report(args.pre_repair_ref, args.audio_jobs, reuse_output=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写出事后诊断：{args.output}")


if __name__ == "__main__":
    main()
