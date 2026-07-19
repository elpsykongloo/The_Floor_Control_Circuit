"""WP1：G0 失败诊断（处置甲第一步，PREREG #12；**诊断非判据**）。

背景：G0 一次性确认（PREREG #9）裁决为 fail——层2 ch0 F1=0.8995（<0.90）、
层3 语料级 0.4769（低于等价带下界 0.48）。层2 分解显示精确率 0.98 而召回率仅
0.83/0.84：Silero 在 Mimi 解码音频上漏检约 1/6 语音帧，且逐类 F1 四类全降、
边界类 eot/bot 掉幅最大。本工具在**允许的诊断集合**（val 295 + test 探索 20）上：

  1) 默认模式：冻结参数单遍运行，分解层2 漏检（FN）——漏检段时长分布、
     漏检类型（整段丢失/起点侵蚀/终点侵蚀/段中断裂）、漏检 vs 命中段的响度差、
     漏检对 eot/bot 金标边界帧的波及比例；
  2) --sweep：修复候选实验（阈值 × 响度归一化网格）→ 层2 F1 与层3 语料级
     macro-F1 响应面。**任何修复参数必须先登记 PREREG 变更（#13 起）再于
     train 侧全新一次性确认集裁决，本工具输出不构成 Gate 依据**；
  3) --count-train-pool：清点官方 train 侧发布物可用、从未被 G0 读取的会话，
     为未来的全新确认集回填数量。

用法：
  uv run python scripts/wp1_g0_diagnose.py [--split val] [--limit N]
  uv run python scripts/wp1_g0_diagnose.py --sweep [--sweep-limit 60]
  uv run python scripts/wp1_g0_diagnose.py --count-train-pool
产出：reports/g0_diagnosis.json + reports/g0_诊断报告.md（sweep/清点并入同一报告）
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_config, load_paths
from floor_circuit.events.g0 import G0_CLASSES, accumulate_counts, finalize_counts
from floor_circuit.events.g0_official import (
    compute_official_labels,
    segments_to_frame_track,
    track_prf,
)
from floor_circuit.events.vad import SileroVad

REQUIRED_GOLD = ("gold_ch0.npz", "gold_ch1.npz")
REQUIRED_AUDIO = ("audio_ch0.wav", "audio_ch1.wav")
FN_DURATION_BUCKETS_MS = (80, 160, 320, 640)

_DIAGNOSE_WORKER_VAD: SileroVad | None = None
_DIAGNOSE_WORKER_HZ = 0.0
_DIAGNOSE_WORKER_TOL = 0
_SWEEP_WORKER_VADS: dict[float, SileroVad] | None = None
_SWEEP_WORKER_HZ = 0.0
_SWEEP_WORKER_TOL = 0
_SWEEP_WORKER_THRESHOLDS: list[float] = []
_SWEEP_WORKER_LOUDNORMS: list[float | None] = []


def _session_dirs(root: Path, split: str | None, limit: int | None) -> list[Path]:
    """与 wp1_g0_calibrate 同口径的目录扫描（诊断工具独立副本）。"""
    dirs = []
    for p in sorted(d for d in root.iterdir() if d.is_dir()):
        if not all((p / f).exists() for f in (*REQUIRED_GOLD, *REQUIRED_AUDIO)):
            continue
        if split is not None:
            try:
                meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
                if meta.get("split") != split:
                    continue
            except Exception:
                continue
        dirs.append(p)
    return dirs[:limit] if limit else dirs


def _load_gold(sdir: Path, ch: int) -> dict[str, np.ndarray]:
    z = np.load(sdir / f"gold_ch{ch}.npz", allow_pickle=False)
    return {k: z[k] for k in (*G0_CLASSES, "vad")}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """布尔轨 → [i0, i1) 连续段列表。"""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _classify_fn_run(fn_run: tuple[int, int], gold_runs: list[tuple[int, int]]) -> str:
    """漏检段与其所在金标语音段的关系：整段丢失/起点侵蚀/终点侵蚀/段中断裂。"""
    a, b = fn_run
    for g0, g1 in gold_runs:
        if g0 <= a and b <= g1:
            if a == g0 and b == g1:
                return "whole_segment_missed"
            if a == g0:
                return "onset_erosion"
            if b == g1:
                return "offset_erosion"
            return "mid_gap"
    return "outside_gold"  # 理论不可达（FN ⊆ 金标语音帧）


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """±radius 帧膨胀（用于容差波及判定）。"""
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones(2 * radius + 1, dtype=np.int8)
    return np.convolve(mask.astype(np.int8), kernel, mode="same") > 0


def _rms_dbfs(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return float("nan")
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(wav, dtype=np.float64))) + 1e-12))


def _span_dbfs(wav: np.ndarray, sr: int, runs: list[tuple[int, int]], hz: float) -> list[float]:
    values = []
    for a, b in runs:
        s0, s1 = int(a / hz * sr), int(b / hz * sr)
        if s1 > s0:
            values.append(_rms_dbfs(wav[s0 : min(s1, len(wav))]))
    return [v for v in values if np.isfinite(v)]


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _bucket_ms(duration_ms: float) -> str:
    for bound in FN_DURATION_BUCKETS_MS:
        if duration_ms <= bound:
            return f"<={bound}ms"
    return f">{FN_DURATION_BUCKETS_MS[-1]}ms"


def _loudness_normalize(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(np.asarray(wav, dtype=np.float64))
    if not np.isfinite(loudness):
        return wav
    out = pyln.normalize.loudness(np.asarray(wav, dtype=np.float64), loudness, target_lufs)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _resolve_jobs(requested: int, n_tasks: int) -> int:
    """把 0 解释为自动并发，并保证不会创建空闲工作进程。"""

    if requested < 0:
        raise ValueError("--jobs 必须为非负整数（0 表示自动）")
    available = max(1, (os.cpu_count() or 2) - 2)
    jobs = available if requested == 0 else requested
    return min(max(1, jobs), max(1, n_tasks))


def _configure_process_worker() -> None:
    """限制每个 VAD 子进程的内部线程，避免进程数与 torch 线程数相乘。"""

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _diagnose_session(sdir: Path, vad: SileroVad, hz: float, tol: int) -> dict:
    """计算单会话诊断充分统计量；主进程按目录序确定性归并。"""

    from floor_circuit.stimuli.qc import load_wav

    channels: dict[str, dict] = {}
    boundary = {cls: {"affected": 0, "total": 0} for cls in ("eot", "bot", "bc")}
    session_fn = 0
    for ch in (0, 1):
        key = f"ch{ch}"
        gold = _load_gold(sdir, ch)
        w, sr = load_wav(sdir / f"audio_ch{ch}.wav")
        n_frames = len(gold["vad"])
        pred = segments_to_frame_track(vad.segments(w, sr), n_frames, hz, rule="majority")
        gold_vad = gold["vad"].astype(bool)
        pred_vad = pred.astype(bool)
        fn = gold_vad & ~pred_vad
        fp = pred_vad & ~gold_vad
        tp = gold_vad & pred_vad
        counts = {
            "fn": int(fn.sum()),
            "fp": int(fp.sum()),
            "tp": int(tp.sum()),
            "gold": int(gold_vad.sum()),
        }
        session_fn += counts["fn"]

        gold_runs = _runs(gold_vad)
        fn_runs = _runs(fn)
        fn_types: dict[str, int] = {}
        fn_buckets: dict[str, int] = {}
        for run in fn_runs:
            kind = _classify_fn_run(run, gold_runs)
            fn_types[kind] = fn_types.get(kind, 0) + 1
            bucket = _bucket_ms((run[1] - run[0]) / hz * 1000.0)
            fn_buckets[bucket] = fn_buckets.get(bucket, 0) + 1
        channels[key] = {
            "vad_counts": counts,
            "fn_type_counts": fn_types,
            "fn_bucket_counts": fn_buckets,
            "fn_dbfs": _span_dbfs(w, sr, fn_runs, hz),
            "tp_dbfs": _span_dbfs(w, sr, _runs(tp), hz),
        }

        fn_dilated = _dilate(fn, tol)
        for cls in ("eot", "bot", "bc"):
            track = gold[cls].astype(bool)
            boundary[cls]["total"] += int(track.sum())
            boundary[cls]["affected"] += int((track & fn_dilated).sum())
    return {
        "session": sdir.name,
        "fn_frames": session_fn,
        "channels": channels,
        "boundary": boundary,
    }


def _init_diagnose_worker(cfg: dict, hz: float, tol: int) -> None:
    global _DIAGNOSE_WORKER_HZ, _DIAGNOSE_WORKER_TOL, _DIAGNOSE_WORKER_VAD

    _configure_process_worker()
    _DIAGNOSE_WORKER_VAD = SileroVad(cfg)
    _DIAGNOSE_WORKER_HZ = hz
    _DIAGNOSE_WORKER_TOL = tol


def _diagnose_worker(path: str) -> dict:
    if _DIAGNOSE_WORKER_VAD is None:
        raise RuntimeError("诊断工作进程尚未初始化")
    return _diagnose_session(
        Path(path),
        _DIAGNOSE_WORKER_VAD,
        _DIAGNOSE_WORKER_HZ,
        _DIAGNOSE_WORKER_TOL,
    )


def diagnose(root: Path, split: str | None, limit: int | None, jobs: int = 1) -> dict:
    """冻结参数单遍：层2 FN 分解 + 边界类波及（诊断，非判据）。"""

    cfg = load_config("events")
    hz = float(cfg["g0"]["frame_hz"])
    tol = int(cfg["g0"]["tolerance_frames"])
    dirs = _session_dirs(root, split, limit)
    if not dirs:
        raise SystemExit(f"{root} 下没有四文件齐全的 {split or '全部'} 会话")

    worker_count = _resolve_jobs(jobs, len(dirs))
    if worker_count == 1:
        vad = SileroVad(cfg)
        results = (_diagnose_session(sdir, vad, hz, tol) for sdir in dirs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_diagnose_worker,
            initargs=(cfg, hz, tol),
        )
        results = executor.map(_diagnose_worker, (str(sdir) for sdir in dirs), chunksize=1)

    fn_type_counts: dict[str, dict[str, int]] = {"ch0": {}, "ch1": {}}
    fn_bucket_counts: dict[str, dict[str, int]] = {"ch0": {}, "ch1": {}}
    fn_dbfs: dict[str, list[float]] = {"ch0": [], "ch1": []}
    tp_dbfs: dict[str, list[float]] = {"ch0": [], "ch1": []}
    vad_counts = {
        "ch0": {"fn": 0, "fp": 0, "tp": 0, "gold": 0},
        "ch1": {"fn": 0, "fp": 0, "tp": 0, "gold": 0},
    }
    boundary = {cls: {"affected": 0, "total": 0} for cls in ("eot", "bot", "bc")}
    per_session: list[dict] = []
    try:
        for result in results:
            for key, channel in result["channels"].items():
                for name, value in channel["vad_counts"].items():
                    vad_counts[key][name] += value
                for name, value in channel["fn_type_counts"].items():
                    fn_type_counts[key][name] = fn_type_counts[key].get(name, 0) + value
                for name, value in channel["fn_bucket_counts"].items():
                    fn_bucket_counts[key][name] = fn_bucket_counts[key].get(name, 0) + value
                fn_dbfs[key].extend(channel["fn_dbfs"])
                tp_dbfs[key].extend(channel["tp_dbfs"])
            for cls, cell in result["boundary"].items():
                boundary[cls]["affected"] += cell["affected"]
                boundary[cls]["total"] += cell["total"]
            per_session.append(
                {"session": result["session"], "fn_frames": result["fn_frames"]}
            )
            print(f"{result['session']}: FN {result['fn_frames']} 帧")
    finally:
        if worker_count > 1:
            executor.shutdown()

    layer2 = {}
    for key, counts in vad_counts.items():
        n_pred = counts["tp"] + counts["fp"]
        prec = counts["tp"] / n_pred if n_pred else 0.0
        rec = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
        layer2[key] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            **counts,
        }
    return {
        "n_sessions": len(dirs),
        "split": split,
        "frozen_vad": dict(cfg["vad"]),
        "layer2_decomposition": layer2,
        "fn_run_types": fn_type_counts,
        "fn_run_duration_buckets": fn_bucket_counts,
        "fn_span_dbfs": {k: _quantiles(v) for k, v in fn_dbfs.items()},
        "tp_span_dbfs": {k: _quantiles(v) for k, v in tp_dbfs.items()},
        "boundary_impact": {
            cls: {
                **cell,
                "fraction": cell["affected"] / cell["total"] if cell["total"] else 0.0,
            }
            for cls, cell in boundary.items()
        },
        "worst_sessions_by_fn": sorted(per_session, key=lambda r: -r["fn_frames"])[:10],
    }


def _new_sweep_cells(
    thresholds: list[float],
    loudnorms: list[float | None],
) -> dict[str, dict]:
    """创建顺序稳定的响应面累加器。"""

    cells: dict[str, dict] = {}
    for norm in loudnorms:
        for thr in thresholds:
            cells[f"thr{thr:g}_norm{'none' if norm is None else f'{norm:g}lufs'}"] = {
                "threshold": thr,
                "loudnorm_lufs": norm,
                "vad_stats": {
                    "ch0": {"tp": 0, "n_pred": 0, "n_gold": 0},
                    "ch1": {"tp": 0, "n_pred": 0, "n_gold": 0},
                },
                "totals": None,
            }
    return cells


def _sweep_session(
    sdir: Path,
    hz: float,
    tol: int,
    thresholds: list[float],
    loudnorms: list[float | None],
    vads: dict[float, SileroVad],
) -> dict:
    """计算单会话、全部响应面格的整数充分统计量。"""

    from floor_circuit.stimuli.qc import load_wav

    cells = _new_sweep_cells(thresholds, loudnorms)
    gold = {ch: _load_gold(sdir, ch) for ch in (0, 1)}
    raw = {ch: load_wav(sdir / f"audio_ch{ch}.wav") for ch in (0, 1)}
    n_frames = {ch: len(gold[ch]["vad"]) for ch in (0, 1)}
    for norm in loudnorms:
        wavs = {
            ch: (
                raw[ch][0]
                if norm is None
                else _loudness_normalize(raw[ch][0], raw[ch][1], norm),
                raw[ch][1],
            )
            for ch in (0, 1)
        }
        for thr in thresholds:
            cell = cells[f"thr{thr:g}_norm{'none' if norm is None else f'{norm:g}lufs'}"]
            pred_vad = {}
            for ch in (0, 1):
                w, sr = wavs[ch]
                pred_vad[ch] = segments_to_frame_track(
                    vads[thr].segments(w, sr),
                    n_frames[ch],
                    hz,
                    rule="majority",
                )
            e2e = compute_official_labels(pred_vad[0], pred_vad[1])
            for ch in (0, 1):
                prf = track_prf(pred_vad[ch], gold[ch]["vad"])
                st = cell["vad_stats"][f"ch{ch}"]
                st["tp"] += round(prf["precision"] * prf["n_pred"])
                st["n_pred"] += prf["n_pred"]
                st["n_gold"] += prf["n_gold"]
                pred_tracks = {cls: e2e[f"{cls}_ch{ch}"] for cls in G0_CLASSES}
                cell["totals"] = accumulate_counts(
                    cell["totals"],
                    pred_tracks,
                    gold[ch],
                    tol,
                )
    return {"session": sdir.name, "cells": cells}


def _init_sweep_worker(
    cfg: dict,
    hz: float,
    tol: int,
    thresholds: list[float],
    loudnorms: list[float | None],
) -> None:
    global _SWEEP_WORKER_HZ, _SWEEP_WORKER_LOUDNORMS, _SWEEP_WORKER_THRESHOLDS
    global _SWEEP_WORKER_TOL, _SWEEP_WORKER_VADS

    _configure_process_worker()
    _SWEEP_WORKER_HZ = hz
    _SWEEP_WORKER_TOL = tol
    _SWEEP_WORKER_THRESHOLDS = thresholds
    _SWEEP_WORKER_LOUDNORMS = loudnorms
    _SWEEP_WORKER_VADS = {
        thr: SileroVad({**cfg, "vad": {**cfg["vad"], "threshold": thr}})
        for thr in thresholds
    }


def _sweep_worker(path: str) -> dict:
    if _SWEEP_WORKER_VADS is None:
        raise RuntimeError("响应面工作进程尚未初始化")
    return _sweep_session(
        Path(path),
        _SWEEP_WORKER_HZ,
        _SWEEP_WORKER_TOL,
        _SWEEP_WORKER_THRESHOLDS,
        _SWEEP_WORKER_LOUDNORMS,
        _SWEEP_WORKER_VADS,
    )


def _merge_totals(
    total: dict[str, dict[str, int]] | None,
    partial: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """以整数加法归并语料级充分统计量。"""

    if total is None:
        return {
            cls: {name: int(value) for name, value in partial[cls].items()}
            for cls in G0_CLASSES
        }
    for cls in G0_CLASSES:
        for name, value in partial[cls].items():
            total[cls][name] += int(value)
    return total


def sweep(
    root: Path,
    split: str,
    sweep_limit: int,
    thresholds: list[float],
    loudnorms: list[float | None],
    jobs: int = 1,
) -> dict:
    """修复候选响应面（诊断集合内；输出不构成 Gate 依据）。"""

    cfg = load_config("events")
    hz = float(cfg["g0"]["frame_hz"])
    tol = int(cfg["g0"]["tolerance_frames"])
    dirs = _session_dirs(root, split, sweep_limit if sweep_limit > 0 else None)
    if not dirs:
        raise SystemExit(f"{root} 下没有四文件齐全的 {split} 会话")
    vads = {
        thr: SileroVad({**cfg, "vad": {**cfg["vad"], "threshold": thr}})
        for thr in thresholds
    }
    worker_count = _resolve_jobs(jobs, len(dirs))
    if worker_count == 1:
        results = (
            _sweep_session(sdir, hz, tol, thresholds, loudnorms, vads)
            for sdir in dirs
        )
    else:
        del vads
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_sweep_worker,
            initargs=(cfg, hz, tol, thresholds, loudnorms),
        )
        results = executor.map(_sweep_worker, (str(sdir) for sdir in dirs), chunksize=1)

    cells = _new_sweep_cells(thresholds, loudnorms)
    try:
        for result in results:
            for name, partial in result["cells"].items():
                cell = cells[name]
                for key, stats in partial["vad_stats"].items():
                    for field, value in stats.items():
                        cell["vad_stats"][key][field] += int(value)
                cell["totals"] = _merge_totals(cell["totals"], partial["totals"])
            print(f"{result['session']}: sweep 完成 {len(cells)} 格")
    finally:
        if worker_count > 1:
            executor.shutdown()

    out: dict[str, dict] = {}
    for name, cell in cells.items():
        vad_report = {}
        for key, st in cell["vad_stats"].items():
            prec = st["tp"] / st["n_pred"] if st["n_pred"] else 0.0
            rec = st["tp"] / st["n_gold"] if st["n_gold"] else 0.0
            vad_report[key] = {
                "precision": prec,
                "recall": rec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            }
        out[name] = {
            "threshold": cell["threshold"],
            "loudnorm_lufs": cell["loudnorm_lufs"],
            "layer2_vad": vad_report,
            "layer3_corpus_macro_f1": finalize_counts(cell["totals"])["macro_f1"],
        }
        print(
            f"{name}: 层2 F1 {vad_report['ch0']['f1']:.4f}/{vad_report['ch1']['f1']:.4f}，"
            f"层3 {out[name]['layer3_corpus_macro_f1']:.4f}"
        )
    return {
        "note": (
            "修复候选响应面（诊断集合内，非判据）：任何修复参数必须先登记 PREREG 变更，"
            "并在 train 侧全新一次性确认集上裁决；等价带须依修复后管线的 val 分布重推"
        ),
        "n_sessions": len(dirs),
        "split": split,
        "cells": out,
    }


def count_train_pool(prep_root: Path) -> dict:
    """清点官方 train 侧发布物可用、且从未进入 dualturn_prep 的会话（未来确认集池）。"""
    from floor_circuit.data.dualturn import load_splits, split_sessions

    dataset_root = Path(load_paths()["datasets"]["dualturn"])
    payload = load_splits(dataset_root)
    train_ids = set(split_sessions(dataset_root, "train"))
    if not train_ids:
        raise SystemExit("splits.json 的 train 划分解析到 0 个会话")

    import pyarrow.parquet as pq

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        data_dir = dataset_root
    shard_glob = "train-*.parquet" if any(data_dir.glob("train-*.parquet")) else "*.parquet"
    release_ids: set[str] = set()
    for shard in sorted(data_dir.glob(shard_glob)):
        table = pq.read_table(shard, columns=["session_id"])
        release_ids.update(str(v) for v in table.column("session_id").to_pylist())

    prepared = {d.name for d in prep_root.iterdir() if d.is_dir()} if prep_root.exists() else set()
    available = sorted(train_ids & release_ids)
    already_prepared = sorted(train_ids & prepared)
    declared_without_audio = {str(s) for s in payload.get("sessions_without_audio", [])}
    return {
        "note": "train 侧从未被 G0 读取过（already_prepared 必须为空，否则该会话不再新鲜）",
        "shard_glob": shard_glob,
        "n_train_split_ids": len(train_ids),
        "n_release_available": len(available),
        "n_already_prepared": len(already_prepared),
        "already_prepared": already_prepared,
        "n_missing_from_release": len(train_ids - release_ids),
        "n_declared_without_audio_in_train": len(train_ids & declared_without_audio),
    }


def _write_diagnosis_markdown(payload: dict) -> None:
    """把当前已有的分解、响应面与训练池清点统一写入同一报告。"""

    lines = ["# G0 失败诊断报告（处置甲，PREREG #12；诊断非判据）", ""]
    if "decomposition" in payload:
        dec = payload["decomposition"]
        lines += [
            f"- 诊断集合：{dec['split']}（{dec['n_sessions']} 会话）；冻结 VAD 参数 {dec['frozen_vad']}",
            "",
            "## 层2 漏检分解（Silero@Mimi 解码 vs 官方金标 VAD）",
            "",
            "| 通道 | precision | recall | F1 | FN 帧 | FP 帧 |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        for key, cell in dec["layer2_decomposition"].items():
            lines.append(
                f"| {key} | {cell['precision']:.4f} | {cell['recall']:.4f} "
                f"| {cell['f1']:.4f} | {cell['fn']} | {cell['fp']} |"
            )
        lines += ["", "## 漏检段类型与响度", ""]
        for key in ("ch0", "ch1"):
            lines.append(
                f"- {key} 漏检段类型：{dec['fn_run_types'][key]}；"
                f"时长分桶：{dec['fn_run_duration_buckets'][key]}"
            )
            lines.append(
                f"- {key} 响度（RMS dBFS 中位）：漏检段 "
                f"{dec['fn_span_dbfs'][key].get('p50', float('nan')):.1f} "
                f"vs 命中段 {dec['tp_span_dbfs'][key].get('p50', float('nan')):.1f}"
            )
        lines += ["", "## 边界类波及（金标事件帧落入 FN±tol 的比例）", ""]
        for cls, cell in dec["boundary_impact"].items():
            lines.append(
                f"- {cls}: {cell['affected']}/{cell['total']}（{cell['fraction']:.3f}）"
            )
    if "sweep" in payload:
        lines += [
            "",
            "## 修复候选响应面（非判据）",
            "",
            "| 格 | 层2 F1 ch0/ch1 | 层3 语料级 |",
            "| --- | --- | --- |",
        ]
        for name, cell in payload["sweep"]["cells"].items():
            lines.append(
                f"| {name} | {cell['layer2_vad']['ch0']['f1']:.4f}/"
                f"{cell['layer2_vad']['ch1']['f1']:.4f} "
                f"| {cell['layer3_corpus_macro_f1']:.4f} |"
            )
    if "train_pool" in payload:
        pool = payload["train_pool"]
        lines += [
            "",
            "## train 侧新确认集池清点",
            "",
            f"- split 会话：{pool['n_train_split_ids']}；发布可用：{pool['n_release_available']}",
            f"- 已被读取：{pool['n_already_prepared']}（必须为 0）；"
            f"发布缺失：{pool['n_missing_from_release']}；"
            f"声明无音频：{pool['n_declared_without_audio_in_train']}",
        ]
    lines += [
        "",
        "> 诊断非判据：修复参数须先登记 PREREG 变更，再于 train 侧全新一次性确认集裁决。",
        "",
    ]
    path = REPORTS_DIR / "g0_诊断报告.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="默认 <data_root>/dualturn_prep")
    ap.add_argument("--split", default="val", help="诊断集合（只允许 val 或探索用途）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="修复候选响应面（阈值×响度归一化）")
    ap.add_argument("--sweep-limit", type=int, default=60, help="sweep 会话数上限（0=全部）")
    ap.add_argument("--sweep-thresholds", default="0.4,0.45,0.5")
    ap.add_argument("--sweep-loudnorm", default="none,-23", help="逗号分隔：none 或目标 LUFS")
    ap.add_argument("--count-train-pool", action="store_true", help="清点 train 侧未来确认集池")
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="会话级工作进程数；0=自动使用逻辑处理器数减 2",
    )
    args = ap.parse_args()
    root = Path(args.root) if args.root else data_root() / "dualturn_prep"
    if args.split == "test" and not args.count_train_pool:
        raise SystemExit("test 确认集已一次性启封且不得用于诊断调参；诊断集合只允许 val（或显式探索目录）")

    existing: dict = {}
    report_path = REPORTS_DIR / "g0_diagnosis.json"
    if report_path.is_file():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if args.count_train_pool:
        existing["train_pool"] = count_train_pool(root)
        write_report_json("g0_diagnosis.json", existing)
        _write_diagnosis_markdown(existing)
        pool = existing["train_pool"]
        print(
            f"train 池：splits {pool['n_train_split_ids']}，发布可用 {pool['n_release_available']}，"
            f"已被读取 {pool['n_already_prepared']}（必须为 0）"
        )
        return

    if args.sweep:
        thresholds = [float(v) for v in args.sweep_thresholds.split(",")]
        loudnorms: list[float | None] = [
            None if v.strip().lower() == "none" else float(v) for v in args.sweep_loudnorm.split(",")
        ]
        existing["sweep"] = sweep(
            root,
            args.split,
            args.sweep_limit,
            thresholds,
            loudnorms,
            jobs=args.jobs,
        )
        write_report_json("g0_diagnosis.json", existing)
        _write_diagnosis_markdown(existing)
        return

    existing["decomposition"] = diagnose(root, args.split, args.limit, jobs=args.jobs)
    existing["note"] = "G0 失败诊断（处置甲，PREREG #12）：诊断非判据；修复登记与 train 侧新确认见 PREREG"
    write_report_json("g0_diagnosis.json", existing)
    _write_diagnosis_markdown(existing)


if __name__ == "__main__":
    main()
