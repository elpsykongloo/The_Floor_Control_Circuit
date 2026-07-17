"""WP7：MVE 探针 + 基线 + G1 裁决（缓存 ingest 与 wp1 标签生成完成后运行）。

用法：uv run python scripts/wp7_run_mve.py [--skip-gru]
产出：reports/mve_报告.md + reports/mve_summary.json；原始分数落 <data_root>/mve/。
说明：声学 GRU 与 Mimi 基线所需的特征/潜表征分别来自本脚本内特征提取与 runner 的
mimi_latent 导出；hazard 基线由标签表直接构建。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import REPO_ROOT, REPORTS_DIR

from floor_circuit.config import data_root, load_config
from floor_circuit.mve.dataset import build_session_data, eligible_rows
from floor_circuit.mve.preflight import (
    RunSpec,
    preflight_mve_inputs,
    sync_labels_atomic,
    validate_baseline_alignment,
)
from floor_circuit.mve.run import (
    average_over_seeds,
    evaluate_target,
    overall_g1,
    probe_grid,
    render_report,
)
from floor_circuit.probes.baselines import acoustic_frames, fit_hazard, hazard_features
from floor_circuit.probes.gru import make_windows, train_eval_gru
from floor_circuit.probes.linear import fit_probe, score_sessions
from floor_circuit.probes.stats import PerSession


def _split_sessions(mve_cfg: dict) -> tuple[list[str], list[str]]:
    split = json.loads((REPO_ROOT / "configs" / "splits" / "candor.json").read_text(encoding="utf-8"))
    n_train = int(mve_cfg["n_sessions_train"])
    n_eval = int(mve_cfg["n_sessions_eval"])
    train = split["splits"]["probe_train"][:n_train]
    evals = split["splits"]["probe_val"][:n_eval]
    if len(train) != n_train or len(evals) != n_eval:
        raise RuntimeError(f"冻结划分容量不足：probe_train={len(train)}/{n_train}，probe_val={len(evals)}/{n_eval}")
    if set(train) & set(evals):
        raise RuntimeError("冻结的 probe_train 与 probe_val 存在会话重叠")
    return train, evals


def _labels_root() -> Path:
    return data_root() / "events" / "candor_labels_flat"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_wav_prefix(path: Path, duration_s: float) -> tuple[np.ndarray, int]:
    """只读取缓存时间域覆盖的音频前缀，防止声学基线越过 600 秒。"""
    import soundfile as sf

    with sf.SoundFile(str(path), mode="r") as handle:
        sample_rate = int(handle.samplerate)
        n_frames = min(len(handle), int(np.ceil(duration_s * sample_rate)))
        wav = handle.read(frames=n_frames, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), sample_rate


def _invalidate_g1_outputs() -> None:
    """正式计算开始前移除旧裁决，失败时不保留可被误读的历史结论。"""
    for path in (REPORTS_DIR / "mve_报告.md", REPORTS_DIR / "mve_summary.json"):
        if path.exists():
            path.unlink()


def _write_report_json_atomic(name: str, payload: dict) -> Path:
    path = REPORTS_DIR / name
    body = {"generated_at": datetime.now(UTC).isoformat(), **payload}
    _write_text_atomic(path, json.dumps(body, ensure_ascii=False, indent=1, default=repr))
    print(f"[report] {path}")
    return path


def prepare_labels_flat(sessions: list[str]) -> dict[str, str]:
    """把 WP1 标签原子刷新到 MVE 平铺目录，禁止沿用旧副本。"""
    src_dir = data_root() / "events" / "candor"
    return sync_labels_atomic(src_dir, _labels_root(), sessions)


def hazard_baseline(
    sessions_train: list[str],
    sessions_eval: list[str],
    target: str,
    delta_ms,
    run_specs: dict[tuple[str, int], RunSpec],
) -> PerSession:
    """T5 状态序列 → hazard 特征 → logistic。评估集输出按会话组织。"""
    grids = load_config("grids")
    step_s = float(grids["clocks"]["moshi"]["step_ms"]) / 1000.0
    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sessions_train + sessions_eval:
        labels = pd.read_parquet(_labels_root() / f"{sid}.parquet")
        xs, ys = [], []
        for ch in (0, 1):
            n_steps = run_specs[(sid, ch)].n_steps
            t5 = labels[(labels["target"] == "T5") & (labels["agent_channel"] == ch)].sort_values("step")
            states = t5["label"].to_numpy()[:n_steps]
            X_all = hazard_features(states, step_s)
            rows = eligible_rows(labels, target, delta_ms, ch, max_steps=n_steps)
            steps = rows["step"].to_numpy(dtype=np.int64)
            xs.append(X_all[steps])
            ys.append(rows["label"].to_numpy(dtype=np.int64))
        feats[sid] = (np.concatenate(xs), np.concatenate(ys))
    X_tr = np.concatenate([feats[s][0] for s in sessions_train])
    y_tr = np.concatenate([feats[s][1] for s in sessions_train])
    clf = fit_hazard(X_tr, y_tr)
    return {sid: (feats[sid][1], clf.predict_proba(feats[sid][0])[:, 1]) for sid in sessions_eval}


def acoustic_gru_baseline(
    sessions_train: list[str],
    sessions_eval: list[str],
    target: str,
    delta_ms,
    seed: int,
    run_specs: dict[tuple[str, int], RunSpec],
) -> PerSession:
    """对方+自身双通道声学特征拼接 → 2 s 窗 GRU。特征缓存到 <data_root>/mve/acoustic/。"""
    cache = data_root() / "mve" / "acoustic"
    cache.mkdir(parents=True, exist_ok=True)

    def feats_for(sid: str, n_steps: int, clock_hz: float) -> np.ndarray:
        feature_path = cache / f"{sid}_steps{n_steps}.npy"
        meta_path = cache / f"{sid}_steps{n_steps}.json"
        expected_meta = {
            "schema_version": 2,
            "n_steps": n_steps,
            "clock_hz": clock_hz,
            "source_audio_hashes": list(run_specs[(sid, 0)].source_audio_hashes),
            "features": ["rms", "f0", "spectral_flux", "zcr"],
            "channels": [0, 1],
        }
        if feature_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cached = np.load(feature_path, allow_pickle=False)
                if (
                    meta == expected_meta
                    and cached.shape == (n_steps, 8)
                    and cached.dtype == np.float32
                    and np.isfinite(cached).all()
                ):
                    return cached
            except Exception:
                pass

        duration_s = n_steps / clock_hz
        w0, sr0 = _load_wav_prefix(
            data_root() / "candor_extracted" / sid / "audio_ch0.wav",
            duration_s,
        )
        w1, sr1 = _load_wav_prefix(
            data_root() / "candor_extracted" / sid / "audio_ch1.wav",
            duration_s,
        )
        a0, a1 = acoustic_frames(w0, sr0), acoustic_frames(w1, sr1)
        if len(a0) < n_steps or len(a1) < n_steps:
            raise RuntimeError(f"{sid} 声学特征不足：ch0={len(a0)}，ch1={len(a1)}，期望 {n_steps}")
        out = np.concatenate([a0[:n_steps], a1[:n_steps]], axis=1).astype(np.float32)
        _write_npy_atomic(feature_path, out)
        _write_text_atomic(meta_path, json.dumps(expected_meta, ensure_ascii=False, indent=1))
        return out

    def windows_xy(sid: str) -> tuple[np.ndarray, np.ndarray]:
        labels = pd.read_parquet(_labels_root() / f"{sid}.parquet")
        xs, ys = [], []
        for ch in (0, 1):
            spec = run_specs[(sid, ch)]
            feats = feats_for(sid, spec.n_steps, spec.clock_hz)
            f = feats if ch == 0 else feats[:, list(range(4, 8)) + list(range(0, 4))]  # 角色对称交换
            rows = eligible_rows(labels, target, delta_ms, ch, max_steps=spec.n_steps)
            steps = rows["step"].to_numpy(dtype=np.int64)
            xs.append(make_windows(f, steps))
            ys.append(rows["label"].to_numpy(dtype=np.int64))
        return np.concatenate(xs), np.concatenate(ys)

    tr = [windows_xy(s) for s in sessions_train]
    n_val = max(1, len(tr) // 8)
    return train_eval_gru(tr[n_val:], tr[:n_val], {s: windows_xy(s) for s in sessions_eval}, seed=seed)


def mimi_baseline(
    sessions_train: list[str],
    sessions_eval: list[str],
    target: str,
    delta_ms,
    mve_cfg: dict,
    runs_root: Path,
) -> PerSession:
    data = build_session_data(
        runs_root,
        _labels_root(),
        sessions_train + sessions_eval,
        layer=-1,
        target=target,
        delta_ms=delta_ms,
        feature="mimi",
    )
    fit = fit_probe(
        data,
        sessions_train,
        sessions_eval,
        list(mve_cfg["probe_c_grid"]),
        seed=0,
        neg_ratio=int(mve_cfg["neg_downsample_ratio"]),
    )
    return score_sessions(fit, data, sessions_eval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-gru",
        action="store_true",
        help="保留的诊断参数；正式 G1 禁止跳过声学 GRU",
    )
    ap.add_argument("--runs-root", default=None, help="ingest 后的 zarr run 根目录")
    args = ap.parse_args()
    _invalidate_g1_outputs()
    if args.skip_gru:
        raise SystemExit("--skip-gru 不满足冻结的三基线协议，不能生成 G1 裁决")

    grids = load_config("grids")
    mve_cfg = grids["mve"]
    train, evals = _split_sessions(mve_cfg)
    sessions = train + evals
    label_hashes = prepare_labels_flat(sessions)
    runs_root = Path(args.runs_root) if args.runs_root else data_root() / "activations" / "moshi" / "mve_r1_zarr"
    clock_hz = float(grids["clocks"]["moshi"]["hz"])
    expected_n_steps = round(float(mve_cfg["max_minutes_per_session"]) * 60.0 * clock_hz)
    run_specs, preflight = preflight_mve_inputs(
        runs_root,
        _labels_root(),
        sessions,
        [int(layer) for layer in mve_cfg["layers"]],
        expected_n_steps,
        clock_hz,
        int(mve_cfg["t1_delta_ms"]),
    )
    preflight["label_sha256"] = label_hashes
    _write_report_json_atomic("mve_preflight.json", preflight)

    per_target: dict[str, dict] = {}
    raw_dir = data_root() / "mve"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for target in mve_cfg["targets"]:
        delta = int(mve_cfg["t1_delta_ms"]) if target == "T1" else None
        data_by_layer = {
            layer: build_session_data(runs_root, _labels_root(), train + evals, layer, target, delta)
            for layer in mve_cfg["layers"]
        }
        cells = probe_grid(
            data_by_layer,
            train,
            evals,
            list(mve_cfg["seeds"]),
            list(mve_cfg["probe_c_grid"]),
            int(mve_cfg["neg_downsample_ratio"]),
            target,
        )
        summary = average_over_seeds(cells)
        baselines: dict[str, PerSession] = {
            "hazard": hazard_baseline(train, evals, target, delta, run_specs),
            "mimi": mimi_baseline(train, evals, target, delta, mve_cfg, runs_root),
            "acoustic_gru": acoustic_gru_baseline(
                train,
                evals,
                target,
                delta,
                seed=0,
                run_specs=run_specs,
            ),
        }
        required_baselines = {"hazard", "mimi", "acoustic_gru"}
        best_layer = max(summary, key=lambda layer: summary[layer]["auc_mean"])
        validate_baseline_alignment(
            summary[best_layer]["rep_per_session"],
            baselines,
            required_baselines,
            target,
        )
        per_target[target] = evaluate_target(
            summary,
            baselines,
            int(mve_cfg["bootstrap_n"]),
            float(mve_cfg["g1_full_threshold"]),
            float(mve_cfg["g1_backup_threshold"]),
        )
    overall = overall_g1(per_target, float(mve_cfg["g1_full_threshold"]), float(mve_cfg["g1_backup_threshold"]))
    meta = {
        "layers": mve_cfg["layers"],
        "seeds": mve_cfg["seeds"],
        "bootstrap_n": mve_cfg["bootstrap_n"],
        "n_train_sessions": len(train),
        "n_eval_sessions": len(evals),
    }
    _write_text_atomic(REPORTS_DIR / "mve_报告.md", render_report(per_target, overall, meta))
    _write_report_json_atomic(
        "mve_summary.json",
        {
            "overall": overall,
            "per_target": {
                t: {k: v for k, v in m.items() if k != "layer_summary"} | {"layer_summary": m["layer_summary"]}
                for t, m in per_target.items()
            },
        },
    )
    print(f"G1 裁决：{overall['verdict']}（优势 {overall['advantage_point']:+.4f}，CI 下界 {overall['ci_lo']:+.4f}）")


if __name__ == "__main__":
    main()
