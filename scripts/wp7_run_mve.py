"""WP7：MVE 探针 + 基线 + G1 裁决（缓存 ingest 与 wp1 标签生成完成后运行）。

用法：uv run python scripts/wp7_run_mve.py [--skip-gru]
产出：reports/mve_报告.md + reports/mve_summary.json；原始分数落 <data_root>/mve/。
说明：声学 GRU 与 Mimi 基线所需的特征/潜表征分别来自本脚本内特征提取与 runner 的
mimi_latent 导出；hazard 基线由标签表直接构建。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import REPO_ROOT, REPORTS_DIR

from floor_circuit.config import data_root, load_config
from floor_circuit.mve.artifacts import (
    ANALYSIS_SOURCE_PATHS,
    FROZEN_G1_BOOTSTRAP_N,
    FROZEN_G1_BOOTSTRAP_SEED,
    ScoreBundleWriter,
)
from floor_circuit.mve.dataset import (
    TrainingSamplePlan,
    build_training_sample_plan,
    eligible_rows,
    load_session_feature,
    load_training_sample,
)
from floor_circuit.mve.preflight import (
    RunSpec,
    preflight_mve_inputs,
    sync_labels_atomic,
    validate_seeded_baseline_alignment,
)
from floor_circuit.mve.run import (
    ProbeCell,
    average_over_seeds,
    evaluate_target,
    overall_g1,
    render_report,
)
from floor_circuit.probes.baselines import acoustic_frames, fit_hazard, hazard_features
from floor_circuit.probes.gru import CONTEXT_STEPS, make_windows, train_eval_gru
from floor_circuit.probes.linear import fit_probe_streaming
from floor_circuit.probes.stats import (
    PerSession,
    ScoreCollection,
    SeededPerSession,
    pooled_metrics,
)

ACOUSTIC_FEATURE_DIM = 8
ACOUSTIC_TRAIN_MAX_BYTES = 4 * 1024**3
ACOUSTIC_EVAL_MAX_BYTES = 2 * 1024**3


def _runner_code_version() -> str:
    """按缓存计划同一算法绑定当前 Moshi runner 的提交与内容。"""

    sources = (
        ("shared", REPO_ROOT / "runners" / "_shared" / "moshi_family.py"),
        ("entry", REPO_ROOT / "runners" / "moshi" / "run.py"),
    )
    content = hashlib.sha256()
    for label, path in sources:
        content.update(label.encode("ascii") + b"\0")
        content.update(path.read_bytes())
        content.update(b"\0")
    try:
        source_commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                "-1",
                "--format=%H",
                "--",
                *(str(path.relative_to(REPO_ROOT)) for _, path in sources),
            ],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法读取 runner 最近提交，拒绝生成 G1 裁决") from exc
    return f"{source_commit[:7]}+runner.{content.hexdigest()}"


def _analysis_code_provenance() -> dict:
    """绑定本次 G1 统计、采样、数据读取与分数序列化的代码内容。"""

    content = hashlib.sha256()
    source_commits: dict[str, str | None] = {}
    for relative in ANALYSIS_SOURCE_PATHS:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        encoded_path = relative.encode("utf-8")
        content.update(len(encoded_path).to_bytes(8, "big"))
        content.update(encoded_path)
        content.update(len(payload).to_bytes(8, "big"))
        content.update(payload)
        try:
            commit = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(REPO_ROOT),
                    "log",
                    "-1",
                    "--format=%H",
                    "--",
                    relative,
                ],
                text=True,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"无法读取分析代码提交：{relative}") from exc
        source_commits[relative] = commit or None
    try:
        repository_head = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法读取分析代码的仓库提交") from exc
    content_sha256 = content.hexdigest()
    return {
        "version": f"{repository_head[:7]}+analysis.{content_sha256}",
        "repository_head": repository_head,
        "content_sha256": content_sha256,
        "sources": list(ANALYSIS_SOURCE_PATHS),
        "source_commits": source_commits,
    }


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


def _mve_summary_payload(
    overall: dict,
    per_target: dict[str, dict],
    score_bundle: dict,
) -> dict:
    """构造可审计的 G1 小结，并显式引用逐会话分数包。"""

    return {
        "overall": overall,
        "per_target": {
            target: {
                key: value
                for key, value in result.items()
                if key != "layer_summary"
            }
            | {"layer_summary": result["layer_summary"]}
            for target, result in per_target.items()
        },
        "score_bundle": score_bundle,
    }


def _publish_g1_outputs(
    *,
    score_writer: ScoreBundleWriter,
    runs_root: Path,
    runner_code_version: str,
    label_hashes: dict[str, str],
    preflight_report_path: Path,
    analysis_protocol: dict,
    per_target: dict[str, dict],
    overall: dict,
    meta: dict,
) -> dict:
    """发布最终分数清单与两份裁决报告；任一步失败都撤下全部完成标志。"""

    try:
        score_bundle = score_writer.finalize(
            runs_root=runs_root,
            runner_code_version=runner_code_version,
            label_hashes=label_hashes,
            preflight_report_path=preflight_report_path,
            analysis_protocol=analysis_protocol,
        )
        report_meta = {**meta, "score_bundle": score_bundle}
        report_text = render_report(per_target, overall, report_meta)
        summary_payload = _mve_summary_payload(overall, per_target, score_bundle)
        _write_text_atomic(REPORTS_DIR / "mve_报告.md", report_text)
        _write_report_json_atomic("mve_summary.json", summary_payload)
    except BaseException:
        try:
            score_writer.remove_manifest()
        finally:
            _invalidate_g1_outputs()
        raise
    return score_bundle


def prepare_labels_flat(sessions: list[str]) -> dict[str, str]:
    """把 WP1 标签原子刷新到 MVE 平铺目录，禁止沿用旧副本。"""
    src_dir = data_root() / "events" / "candor"
    return sync_labels_atomic(src_dir, _labels_root(), sessions)


def linear_feature_cells(
    sessions_eval: list[str],
    target: str,
    delta_ms: int | None,
    layer: int,
    feature: str,
    seeds: list[int],
    c_grid: list[float],
    runs_root: Path,
    run_specs: dict[tuple[str, int], RunSpec],
    plans: dict[int, TrainingSamplePlan],
) -> list[ProbeCell]:
    """单层、单特征串行拟合；训练先抽样，验证逐会话读取。"""

    cells: list[ProbeCell] = []
    for seed in seeds:
        plan = plans[seed]
        X_train, y_train = load_training_sample(
            runs_root,
            plan,
            layer,
            feature=feature,
        )

        def provide_eval(sid: str) -> tuple[np.ndarray, np.ndarray]:
            return load_session_feature(
                runs_root,
                _labels_root(),
                sid,
                run_specs,
                layer,
                target,
                delta_ms,
                feature=feature,
            )

        fit, per_session = fit_probe_streaming(
            X_train,
            y_train,
            sessions_eval,
            provide_eval,
            c_grid,
            seed,
        )
        metrics = pooled_metrics(per_session)
        metrics["best_c"] = fit.best_c
        cells.append(
            ProbeCell(
                layer=layer,
                target=target,
                seed=seed,
                metrics=metrics,
                per_session=per_session,
            )
        )
        del X_train, y_train, fit
    return cells


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
    out: PerSession = {}
    for sid in sessions_eval:
        X_eval, y_eval = feats[sid]
        scores = (
            clf.predict_proba(X_eval)[:, 1]
            if len(y_eval)
            else np.empty(0, dtype=np.float64)
        )
        out[sid] = (y_eval, scores)
    return out


def _acoustic_windows_peak_bytes(
    session_ids: list[str],
    target: str,
    delta_ms: int | None,
    run_specs: dict[tuple[str, int], RunSpec],
) -> int:
    """估计窗口列表加拼接矩阵同时存在时的保守峰值。"""

    n_rows = 0
    for sid in session_ids:
        labels = pd.read_parquet(_labels_root() / f"{sid}.parquet")
        for channel in (0, 1):
            spec = run_specs[(sid, channel)]
            n_rows += len(
                eligible_rows(
                    labels,
                    target,
                    delta_ms,
                    channel,
                    max_steps=spec.n_steps,
                )
            )
    one_copy = n_rows * CONTEXT_STEPS * ACOUSTIC_FEATURE_DIM * np.dtype(np.float32).itemsize
    return 2 * one_copy


def acoustic_gru_baseline(
    sessions_train: list[str],
    sessions_eval: list[str],
    target: str,
    delta_ms,
    seed: int,
    run_specs: dict[tuple[str, int], RunSpec],
) -> PerSession:
    """对方+自身双通道声学特征拼接 → 2 s 窗 GRU。特征缓存到 <data_root>/mve/acoustic/。"""
    train_peak = _acoustic_windows_peak_bytes(
        sessions_train,
        target,
        delta_ms,
        run_specs,
    )
    eval_peak = _acoustic_windows_peak_bytes(
        sessions_eval,
        target,
        delta_ms,
        run_specs,
    )
    if train_peak > ACOUSTIC_TRAIN_MAX_BYTES:
        raise MemoryError(
            f"{target} 声学 GRU 训练窗口预计峰值 {train_peak / 1024**3:.2f} GiB，"
            f"超过上限 {ACOUSTIC_TRAIN_MAX_BYTES / 1024**3:.2f} GiB"
        )
    if eval_peak > ACOUSTIC_EVAL_MAX_BYTES:
        raise MemoryError(
            f"{target} 声学 GRU 评估窗口预计峰值 {eval_peak / 1024**3:.2f} GiB，"
            f"超过上限 {ACOUSTIC_EVAL_MAX_BYTES / 1024**3:.2f} GiB"
        )
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
    sessions_eval: list[str],
    target: str,
    delta_ms: int | None,
    mve_cfg: dict,
    runs_root: Path,
    run_specs: dict[tuple[str, int], RunSpec],
    seeds: list[int],
    training_plans: dict[int, TrainingSamplePlan],
) -> tuple[SeededPerSession, dict[int, float]]:
    cells = linear_feature_cells(
        sessions_eval=sessions_eval,
        target=target,
        delta_ms=delta_ms,
        layer=-1,
        feature="mimi",
        seeds=seeds,
        c_grid=list(mve_cfg["probe_c_grid"]),
        runs_root=runs_root,
        run_specs=run_specs,
        plans=training_plans,
    )
    if sorted(cell.seed for cell in cells) != sorted(seeds):
        raise RuntimeError("Mimi 基线没有返回完整的探针种子集合")
    return (
        {cell.seed: cell.per_session for cell in cells},
        {cell.seed: float(cell.metrics["best_c"]) for cell in cells},
    )


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
    bootstrap_n = int(mve_cfg["bootstrap_n"])
    if bootstrap_n != FROZEN_G1_BOOTSTRAP_N:
        raise RuntimeError("G1 bootstrap 次数与冻结协议不一致")
    analysis_protocol = {
        "bootstrap_n": bootstrap_n,
        "bootstrap_seed": FROZEN_G1_BOOTSTRAP_SEED,
        "code": _analysis_code_provenance(),
    }
    train, evals = _split_sessions(mve_cfg)
    sessions = train + evals
    label_hashes = prepare_labels_flat(sessions)
    runs_root = Path(args.runs_root) if args.runs_root else data_root() / "activations" / "moshi" / "mve_r1_zarr"
    clock_hz = float(grids["clocks"]["moshi"]["hz"])
    expected_n_steps = round(float(mve_cfg["max_minutes_per_session"]) * 60.0 * clock_hz)
    runner_code_version = _runner_code_version()
    run_specs, preflight = preflight_mve_inputs(
        runs_root,
        _labels_root(),
        data_root() / "candor_extracted",
        sessions,
        {"train": train, "val": evals},
        [int(layer) for layer in mve_cfg["layers"]],
        expected_n_steps,
        clock_hz,
        int(mve_cfg["t1_delta_ms"]),
        runner_code_version,
        float(mve_cfg["max_minutes_per_session"]) * 60.0,
        float(mve_cfg["mimi_chunk_seconds"]),
        int(mve_cfg["forward_chunk_steps"]),
    )
    preflight["label_sha256"] = label_hashes
    preflight_report_path = _write_report_json_atomic("mve_preflight.json", preflight)

    per_target: dict[str, dict] = {}
    raw_dir = data_root() / "mve"
    raw_dir.mkdir(parents=True, exist_ok=True)
    probe_seeds = [int(seed) for seed in mve_cfg["seeds"]]
    score_writer = ScoreBundleWriter.create(
        raw_dir / "g1_scores",
        relative_base=data_root(),
        eval_session_order=evals,
        targets=[str(target) for target in mve_cfg["targets"]],
        layers=[int(layer) for layer in mve_cfg["layers"]],
        seeds=probe_seeds,
    )
    for target in mve_cfg["targets"]:
        delta = int(mve_cfg["t1_delta_ms"]) if target == "T1" else None
        plans = {
            seed: build_training_sample_plan(
                _labels_root(),
                train,
                run_specs,
                target,
                delta,
                int(mve_cfg["neg_downsample_ratio"]),
                seed,
            )
            for seed in probe_seeds
        }
        summary: dict[int, dict] = {}
        for layer_value in mve_cfg["layers"]:
            layer = int(layer_value)
            cells = linear_feature_cells(
                sessions_eval=evals,
                target=target,
                delta_ms=delta,
                layer=layer,
                feature="acts",
                seeds=probe_seeds,
                c_grid=list(mve_cfg["probe_c_grid"]),
                runs_root=runs_root,
                run_specs=run_specs,
                plans=plans,
            )
            for cell in cells:
                score_writer.add(
                    target=target,
                    kind="probe",
                    per_session=cell.per_session,
                    layer=layer,
                    seed=cell.seed,
                    best_c=float(cell.metrics["best_c"]),
                )
            summary[layer] = average_over_seeds(cells)[layer]
            del cells
        hazard_scores = hazard_baseline(train, evals, target, delta, run_specs)
        mimi_scores, mimi_best_c = mimi_baseline(
            evals,
            target,
            delta,
            mve_cfg,
            runs_root,
            run_specs,
            probe_seeds,
            plans,
        )
        acoustic_scores = acoustic_gru_baseline(
            train,
            evals,
            target,
            delta,
            seed=0,
            run_specs=run_specs,
        )
        baselines: dict[str, ScoreCollection] = {
            "hazard": hazard_scores,
            "mimi": mimi_scores,
            "acoustic_gru": acoustic_scores,
        }
        for seed, scores in mimi_scores.items():
            score_writer.add(
                target=target,
                kind="mimi",
                per_session=scores,
                layer=None,
                seed=seed,
                best_c=mimi_best_c[seed],
            )
        score_writer.add(
            target=target,
            kind="hazard",
            per_session=hazard_scores,
            layer=None,
            seed=0,
        )
        score_writer.add(
            target=target,
            kind="acoustic_gru",
            per_session=acoustic_scores,
            layer=None,
            seed=0,
        )
        required_baselines = {"hazard", "mimi", "acoustic_gru"}
        best_layer = max(summary, key=lambda layer: summary[layer]["auc_mean"])
        validate_seeded_baseline_alignment(
            summary[best_layer]["per_seed"],
            baselines,
            required_baselines,
            target,
        )
        per_target[target] = evaluate_target(
            summary,
            baselines,
            bootstrap_n,
            float(mve_cfg["g1_full_threshold"]),
            float(mve_cfg["g1_backup_threshold"]),
            boot_seed=FROZEN_G1_BOOTSTRAP_SEED,
        )
    overall = overall_g1(per_target, float(mve_cfg["g1_full_threshold"]), float(mve_cfg["g1_backup_threshold"]))
    meta = {
        "layers": mve_cfg["layers"],
        "seeds": mve_cfg["seeds"],
        "bootstrap_n": mve_cfg["bootstrap_n"],
        "n_train_sessions": len(train),
        "n_eval_sessions": len(evals),
    }
    _publish_g1_outputs(
        score_writer=score_writer,
        runs_root=runs_root,
        runner_code_version=runner_code_version,
        label_hashes=label_hashes,
        preflight_report_path=preflight_report_path,
        analysis_protocol=analysis_protocol,
        per_target=per_target,
        overall=overall,
        meta=meta,
    )
    print(f"G1 裁决：{overall['verdict']}（优势 {overall['advantage_point']:+.4f}，CI 下界 {overall['ci_lo']:+.4f}）")


if __name__ == "__main__":
    main()
