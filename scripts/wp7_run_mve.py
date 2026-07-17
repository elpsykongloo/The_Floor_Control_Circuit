"""WP7：MVE 探针 + 基线 + G1 裁决（缓存 ingest 与 wp1 标签生成完成后运行）。

用法：uv run python scripts/wp7_run_mve.py [--skip-gru]
产出：reports/mve_报告.md + reports/mve_summary.json；原始分数落 <data_root>/mve/。
说明：声学 GRU 与 Mimi 基线所需的特征/潜表征分别来自本脚本内特征提取与 runner 的
mimi_latent 导出；hazard 基线由标签表直接构建。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import REPO_ROOT, REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.mve.dataset import build_session_data, eligible_rows, run_dir_for
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
from floor_circuit.stimuli.qc import load_wav


def _split_sessions(mve_cfg: dict) -> tuple[list[str], list[str]]:
    split = json.loads((REPO_ROOT / "configs" / "splits" / "candor.json").read_text(encoding="utf-8"))
    train = split["splits"]["probe_train"][: int(mve_cfg["n_sessions_train"])]
    evals = split["splits"]["probe_val"][: int(mve_cfg["n_sessions_eval"])]
    return train, evals


def _labels_root() -> Path:
    return data_root() / "events" / "candor_labels_flat"


def prepare_labels_flat(sessions: list[str]) -> None:
    """把 wp1 的 <sid>.labels.parquet 汇到 mve 数据集期望的目录名下（幂等）。"""
    src_dir = data_root() / "events" / "candor"
    dst = _labels_root()
    dst.mkdir(parents=True, exist_ok=True)
    for sid in sessions:
        s, d = src_dir / f"{sid}.labels.parquet", dst / f"{sid}.parquet"
        if s.exists() and not d.exists():
            d.write_bytes(s.read_bytes())


def hazard_baseline(sessions_train: list[str], sessions_eval: list[str], target: str, delta_ms) -> PerSession:
    """T5 状态序列 → hazard 特征 → logistic。评估集输出按会话组织。"""
    grids = load_config("grids")
    step_s = float(grids["clocks"]["moshi"]["step_ms"]) / 1000.0
    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sessions_train + sessions_eval:
        labels = pd.read_parquet(_labels_root() / f"{sid}.parquet")
        xs, ys = [], []
        for ch in (0, 1):
            t5 = labels[(labels["target"] == "T5") & (labels["agent_channel"] == ch)].sort_values("step")
            states = t5["label"].to_numpy()
            X_all = hazard_features(states, step_s)
            rows = eligible_rows(labels, target, delta_ms, ch)
            steps = rows["step"].to_numpy(dtype=np.int64)
            keep = steps < len(X_all)
            xs.append(X_all[steps[keep]])
            ys.append(rows["label"].to_numpy(dtype=np.int64)[keep])
        feats[sid] = (np.concatenate(xs), np.concatenate(ys))
    X_tr = np.concatenate([feats[s][0] for s in sessions_train])
    y_tr = np.concatenate([feats[s][1] for s in sessions_train])
    clf = fit_hazard(X_tr, y_tr)
    return {
        sid: (feats[sid][1], clf.predict_proba(feats[sid][0])[:, 1]) for sid in sessions_eval
    }


def acoustic_gru_baseline(
    sessions_train: list[str], sessions_eval: list[str], target: str, delta_ms, seed: int
) -> PerSession:
    """对方+自身双通道声学特征拼接 → 2 s 窗 GRU。特征缓存到 <data_root>/mve/acoustic/。"""
    cache = data_root() / "mve" / "acoustic"
    cache.mkdir(parents=True, exist_ok=True)

    def feats_for(sid: str) -> np.ndarray:
        f = cache / f"{sid}.npy"
        if f.exists():
            return np.load(f, allow_pickle=False)
        w0, sr0 = load_wav(data_root() / "candor_extracted" / sid / "audio_ch0.wav")
        w1, sr1 = load_wav(data_root() / "candor_extracted" / sid / "audio_ch1.wav")
        a0, a1 = acoustic_frames(w0, sr0), acoustic_frames(w1, sr1)
        n = min(len(a0), len(a1))
        out = np.concatenate([a0[:n], a1[:n]], axis=1)
        np.save(f, out)
        return out

    def windows_xy(sid: str) -> tuple[np.ndarray, np.ndarray]:
        labels = pd.read_parquet(_labels_root() / f"{sid}.parquet")
        feats = feats_for(sid)
        xs, ys = [], []
        for ch in (0, 1):
            f = feats if ch == 0 else feats[:, list(range(4, 8)) + list(range(0, 4))]  # 角色对称交换
            rows = eligible_rows(labels, target, delta_ms, ch)
            steps = rows["step"].to_numpy(dtype=np.int64)
            keep = steps < len(f)
            xs.append(make_windows(f, steps[keep]))
            ys.append(rows["label"].to_numpy(dtype=np.int64)[keep])
        return np.concatenate(xs), np.concatenate(ys)

    tr = [windows_xy(s) for s in sessions_train]
    n_val = max(1, len(tr) // 8)
    return train_eval_gru(
        tr[n_val:], tr[:n_val], {s: windows_xy(s) for s in sessions_eval}, seed=seed
    )


def mimi_baseline(
    sessions_train: list[str], sessions_eval: list[str], target: str, delta_ms, mve_cfg: dict
) -> PerSession:
    runs_root = data_root() / "activations" / "moshi" / "mve_r1_zarr"
    data = build_session_data(
        runs_root, _labels_root(), sessions_train + sessions_eval,
        layer=-1, target=target, delta_ms=delta_ms, feature="mimi",
    )
    fit = fit_probe(data, sessions_train, sessions_eval, list(mve_cfg["probe_c_grid"]), seed=0,
                    neg_ratio=int(mve_cfg["neg_downsample_ratio"]))
    return score_sessions(fit, data, sessions_eval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gru", action="store_true", help="跳过声学 GRU（快速首跑）")
    ap.add_argument("--runs-root", default=None, help="ingest 后的 zarr run 根目录")
    args = ap.parse_args()
    mve_cfg = load_config("grids")["mve"]
    train, evals = _split_sessions(mve_cfg)
    prepare_labels_flat(train + evals)
    runs_root = Path(args.runs_root) if args.runs_root else data_root() / "activations" / "moshi" / "mve_r1_zarr"
    missing = [
        s
        for s in train + evals
        for ch in (0, 1)
        if not (run_dir_for(runs_root, s, ch) / "manifest.json").exists()
    ]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 个 run（先 wp7_cache_mve + wp5_ingest --batch）。样例：{missing[:4]}")

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
            data_by_layer, train, evals, list(mve_cfg["seeds"]), list(mve_cfg["probe_c_grid"]),
            int(mve_cfg["neg_downsample_ratio"]), target,
        )
        summary = average_over_seeds(cells)
        baselines: dict[str, PerSession] = {"hazard": hazard_baseline(train, evals, target, delta)}
        try:
            baselines["mimi"] = mimi_baseline(train, evals, target, delta, mve_cfg)
        except Exception as e:
            print(f"警告：Mimi 基线失败（{e!r}），报告中将注明")
        if not args.skip_gru:
            baselines["acoustic_gru"] = acoustic_gru_baseline(train, evals, target, delta, seed=0)
        per_target[target] = evaluate_target(
            summary, baselines, int(mve_cfg["bootstrap_n"]),
            float(mve_cfg["g1_full_threshold"]), float(mve_cfg["g1_backup_threshold"]),
        )
    overall = overall_g1(per_target, float(mve_cfg["g1_full_threshold"]), float(mve_cfg["g1_backup_threshold"]))
    meta = {
        "layers": mve_cfg["layers"], "seeds": mve_cfg["seeds"], "bootstrap_n": mve_cfg["bootstrap_n"],
        "n_train_sessions": len(train), "n_eval_sessions": len(evals),
    }
    (REPORTS_DIR / "mve_报告.md").write_text(render_report(per_target, overall, meta), encoding="utf-8")
    write_report_json("mve_summary.json", {"overall": overall, "per_target": {
        t: {k: v for k, v in m.items() if k != "layer_summary"} | {"layer_summary": m["layer_summary"]}
        for t, m in per_target.items()
    }})
    print(f"G1 裁决：{overall['verdict']}（优势 {overall['advantage_point']:+.4f}，CI 下界 {overall['ci_lo']:+.4f}）")


if __name__ == "__main__":
    main()
