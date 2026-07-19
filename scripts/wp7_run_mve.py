"""WP7：MVE 探针 + 基线 + G1 裁决（缓存 ingest 与 wp1 标签生成完成后运行）。

用法：uv run python scripts/wp7_run_mve.py [--runs-root ...]
     uv run python scripts/wp7_run_mve.py --ablation-text-pad   # 旧 PAD 缓存消融复盘
产出：reports/mve_报告.md + reports/mve_summary.json；原始分数落 <data_root>/mve/。
消融模式输出隔离为 *_ablation_pad 文件，不构成正式 G1。

PREREG #7 协议要点（2026-07-18）：
- 正式缓存必须 text_mode=greedy（预检强制核验 manifest）；
- 时间对齐（#8 锚定）：标签步 s 观测截止 (s+1)·τ（在线刚接收完对方帧 s）；
  acts 读行 s+1，Mimi/hazard/声学读行 s；acts[0]（initial）弃用，末标签步丢弃；
- 嵌套选择：C 与最优层在 probe_train 内层划分（inner_train/inner_val）上选择，
  probe_val 仅用于最终一次性报告。
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
from floor_circuit.mve.alignment import (
    ANALYSIS_MAX_LABEL_STEP,
    ANALYSIS_TIME_ALIGNMENT,
    CONTEXT_TRUNCATION,
    MIN_ELIGIBLE_STEP,
    MODEL_CONTEXT_STEPS,
    feature_row_indices,
    min_eligible_step_for,
    usable_label_steps,
)
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
    paired_seed_mean_advantage_bootstrap,
    pooled_metrics,
    seed_mean_metrics,
)

ACOUSTIC_FEATURE_DIM = 8
ACOUSTIC_TRAIN_MAX_BYTES = 4 * 1024**3
ACOUSTIC_EVAL_MAX_BYTES = 2 * 1024**3


def _runner_code_version(entry: Path | None = None) -> str:
    """按缓存计划同一算法绑定当前 Moshi runner 的提交与内容。"""

    if entry is None:
        entry = REPO_ROOT / "runners" / "moshi" / "run.py"
    sources = (
        ("shared", REPO_ROOT / "runners" / "_shared" / "moshi_family.py"),
        ("entry", entry),
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


def _accepted_runner_code_versions() -> list[str]:
    """读取持久批处理计划登记的当前实现与可续跑历史实现。"""
    batch_entry = REPO_ROOT / "runners" / "moshi" / "run_batch.py"
    current = _runner_code_version(batch_entry)
    plan_path = REPORTS_DIR / "wp7_cache_plan.json"
    if not plan_path.is_file():
        return [current]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    planned = str(plan.get("batch_code_version", ""))
    planned_digest = planned.partition("+runner.")[2]
    current_digest = current.partition("+runner.")[2]
    if (
        len(planned_digest) != 64
        or len(current_digest) != 64
        or planned_digest != current_digest
    ):
        raise RuntimeError("缓存计划与当前持久 runner 不一致，请重新生成批处理计划")
    versions = {str(value) for value in plan.get("accepted_code_versions", [])}
    versions.add(current)
    return sorted(versions)


def _runner_code_version_set_id(versions: list[str]) -> str:
    """把多实现但同协议的缓存版本集合绑定为单一分数包标识。"""
    if len(versions) == 1:
        return versions[0]
    digest = hashlib.sha256(
        json.dumps(sorted(versions), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return f"runner-set.{digest}"


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


def _split_sessions(mve_cfg: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    """返回 (train, evals, inner_train, inner_val)。

    嵌套选择划分（PREREG #7）：按冻结划分文件内的既有顺序取前 inner_val_sessions 个
    为 inner_val，其余为 inner_train。冻结文件按会话 id（UUID）字典序排列——顺序与
    会话内容无关，等效于任意固定划分；会话本身在冻结时已随机分配到 probe_train。
    """
    split = json.loads((REPO_ROOT / "configs" / "splits" / "candor.json").read_text(encoding="utf-8"))
    n_train = int(mve_cfg["n_sessions_train"])
    n_eval = int(mve_cfg["n_sessions_eval"])
    train = split["splits"]["probe_train"][:n_train]
    evals = split["splits"]["probe_val"][:n_eval]
    if len(train) != n_train or len(evals) != n_eval:
        raise RuntimeError(f"冻结划分容量不足：probe_train={len(train)}/{n_train}，probe_val={len(evals)}/{n_eval}")
    if set(train) & set(evals):
        raise RuntimeError("冻结的 probe_train 与 probe_val 存在会话重叠")
    n_inner_val = int(mve_cfg["inner_val_sessions"])
    if not 0 < n_inner_val < n_train:
        raise RuntimeError(f"inner_val_sessions={n_inner_val} 必须在 (0, {n_train}) 内")
    inner_val = train[:n_inner_val]
    inner_train = train[n_inner_val:]
    return train, evals, inner_train, inner_val


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


def _output_names(ablation_tag: str | None) -> tuple[str, str]:
    if ablation_tag:
        return f"mve_报告_ablation_{ablation_tag}.md", f"mve_summary_ablation_{ablation_tag}.json"
    return "mve_报告.md", "mve_summary.json"


def _invalidate_g1_outputs(ablation_tag: str | None = None) -> None:
    """本次计算开始前移除**本模式自己的**旧结论，失败时不保留可被误读的历史结论。"""
    for name in _output_names(ablation_tag):
        path = REPORTS_DIR / name
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
    protocol: dict,
    descriptive: dict,
) -> dict:
    """构造可审计的 G1 小结，并显式引用逐会话分数包、协议声明与描述性附表。"""

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
        "protocol": protocol,
        "descriptive": descriptive,
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
    protocol: dict,
    descriptive: dict,
    ablation_tag: str | None = None,
) -> dict:
    """发布最终分数清单与两份裁决报告；任一步失败都撤下全部完成标志。"""

    report_name, summary_name = _output_names(ablation_tag)
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
        summary_payload = _mve_summary_payload(
            overall, per_target, score_bundle, protocol, descriptive
        )
        _write_text_atomic(REPORTS_DIR / report_name, report_text)
        _write_report_json_atomic(summary_name, summary_payload)
    except BaseException:
        try:
            score_writer.remove_manifest()
        finally:
            _invalidate_g1_outputs(ablation_tag)
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
    inner_plans: dict[int, TrainingSamplePlan],
    sessions_inner_val: list[str],
    min_step: int = MIN_ELIGIBLE_STEP,
) -> list[ProbeCell]:
    """两阶段嵌套拟合（PREREG #7）。

    阶段 1：inner_train 抽样上拟合全部 C，在 inner_val 上选 best_c（选择证据记入
    metrics.selection_*）；阶段 2：以 best_c 在全量 probe_train 抽样上重训，
    在 probe_val 一次性评估。probe_val 分数不参与任何选择。
    min_step 仅供 mimi_prev（#11 描述性变体）覆盖，须与传入 plans 的构建口径一致。
    """

    cells: list[ProbeCell] = []
    for seed in seeds:

        def provide_session(sid: str) -> tuple[np.ndarray, np.ndarray]:
            return load_session_feature(
                runs_root,
                _labels_root(),
                sid,
                run_specs,
                layer,
                target,
                delta_ms,
                feature=feature,
                min_step=min_step,
            )

        X_inner, y_inner = load_training_sample(
            runs_root,
            inner_plans[seed],
            layer,
            feature=feature,
        )
        selection_fit, _selection_scores = fit_probe_streaming(
            X_inner,
            y_inner,
            sessions_inner_val,
            provide_session,
            c_grid,
            seed,
        )
        del X_inner, y_inner
        best_c = float(selection_fit.best_c)
        selection_auc_by_c = {
            float(c): float(auc) for c, auc in selection_fit.val_auc_by_c.items()
        }
        del selection_fit

        X_train, y_train = load_training_sample(
            runs_root,
            plans[seed],
            layer,
            feature=feature,
        )
        fit, per_session = fit_probe_streaming(
            X_train,
            y_train,
            sessions_eval,
            provide_session,
            [best_c],
            seed,
        )
        if float(fit.best_c) != best_c:
            raise RuntimeError(f"阶段 2 的 C={fit.best_c} 偏离内层选定值 {best_c}")
        metrics = pooled_metrics(per_session)
        metrics["best_c"] = best_c
        metrics["selection_auc"] = selection_auc_by_c[best_c]
        metrics["selection_auc_by_c"] = selection_auc_by_c
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
    """T5 状态序列 → hazard 特征 → logistic。评估集输出按会话组织。

    时间对齐（PREREG #7/#8）：标签步 s 使用 states[0..s]（观测截止 (s+1)·τ），
    即读取 hazard 特征矩阵的第 s 行（feature_row_indices 统一映射）。
    """
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
            rows = eligible_rows(
                labels,
                target,
                delta_ms,
                ch,
                max_steps=usable_label_steps(n_steps),
                min_step=MIN_ELIGIBLE_STEP,
            )
            steps = rows["step"].to_numpy(dtype=np.int64)
            xs.append(X_all[feature_row_indices("hazard", steps)])
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
                    max_steps=usable_label_steps(spec.n_steps),
                    min_step=MIN_ELIGIBLE_STEP,
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

        def expected_meta(f0_backends: list[str]) -> dict:
            # schema 3（PREREG #7 审查修复）：登记 F0 后端并使 yin 旧足迹（4·hop）缓存失效
            return {
                "schema_version": 3,
                "n_steps": n_steps,
                "clock_hz": clock_hz,
                "source_audio_hashes": list(run_specs[(sid, 0)].source_audio_hashes),
                "features": ["rms", "f0", "spectral_flux", "zcr"],
                "channels": [0, 1],
                "f0_backends": f0_backends,
            }

        if feature_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cached = np.load(feature_path, allow_pickle=False)
                if (
                    meta == expected_meta(meta.get("f0_backends", []))
                    and meta.get("f0_backends")
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
        a0, meta0 = acoustic_frames(w0, sr0, return_meta=True)
        a1, meta1 = acoustic_frames(w1, sr1, return_meta=True)
        if len(a0) < n_steps or len(a1) < n_steps:
            raise RuntimeError(f"{sid} 声学特征不足：ch0={len(a0)}，ch1={len(a1)}，期望 {n_steps}")
        out = np.concatenate([a0[:n_steps], a1[:n_steps]], axis=1).astype(np.float32)
        _write_npy_atomic(feature_path, out)
        _write_text_atomic(
            meta_path,
            json.dumps(
                expected_meta([meta0["f0_backend"], meta1["f0_backend"]]),
                ensure_ascii=False,
                indent=1,
            ),
        )
        return out

    def windows_xy(sid: str) -> tuple[np.ndarray, np.ndarray]:
        labels = pd.read_parquet(_labels_root() / f"{sid}.parquet")
        xs, ys = [], []
        for ch in (0, 1):
            spec = run_specs[(sid, ch)]
            feats = feats_for(sid, spec.n_steps, spec.clock_hz)
            f = feats if ch == 0 else feats[:, list(range(4, 8)) + list(range(0, 4))]  # 角色对称交换
            rows = eligible_rows(
                labels,
                target,
                delta_ms,
                ch,
                max_steps=usable_label_steps(spec.n_steps),
                min_step=MIN_ELIGIBLE_STEP,
            )
            steps = rows["step"].to_numpy(dtype=np.int64)
            # 时间对齐（PREREG #7/#8）：窗尾 = s，观测截止 (s+1)·τ
            xs.append(make_windows(f, feature_row_indices("acoustic", steps)))
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
    inner_plans: dict[int, TrainingSamplePlan],
    sessions_inner_val: list[str],
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
        inner_plans=inner_plans,
        sessions_inner_val=sessions_inner_val,
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
    ap.add_argument(
        "--ablation-text-pad",
        action="store_true",
        help="对旧 text_mode=pad 缓存跑消融复盘：输出 *_ablation_pad 文件，不构成正式 G1",
    )
    args = ap.parse_args()
    ablation_tag = "pad" if args.ablation_text_pad else None
    _invalidate_g1_outputs(ablation_tag)
    if args.skip_gru:
        raise SystemExit("--skip-gru 不满足冻结的三基线协议，不能生成 G1 裁决")

    grids = load_config("grids")
    mve_cfg = grids["mve"]
    bootstrap_n = int(mve_cfg["bootstrap_n"])
    if bootstrap_n != FROZEN_G1_BOOTSTRAP_N:
        raise RuntimeError("G1 bootstrap 次数与冻结协议不一致")
    if (
        int(mve_cfg["context_steps"]) != MODEL_CONTEXT_STEPS
        or int(mve_cfg["analysis_max_label_step"]) != ANALYSIS_MAX_LABEL_STEP
    ):
        raise RuntimeError(
            "grids.yaml 的上下文截断镜像与 mve/alignment.py 权威值不一致（PREREG #11）"
        )
    frozen_text_mode = str(mve_cfg["text_mode"])
    if frozen_text_mode != "greedy":
        raise RuntimeError(f"冻结的正式文本流协议必须为 greedy，配置为 {frozen_text_mode!r}")
    expected_text_mode = "pad" if args.ablation_text_pad else frozen_text_mode
    analysis_protocol = {
        "bootstrap_n": bootstrap_n,
        "bootstrap_seed": FROZEN_G1_BOOTSTRAP_SEED,
        "code": _analysis_code_provenance(),
    }
    train, evals, inner_train, inner_val = _split_sessions(mve_cfg)
    sessions = train + evals
    label_hashes = prepare_labels_flat(sessions)
    if args.runs_root:
        runs_root = Path(args.runs_root)
    else:
        cache_name = "mve_r1" if expected_text_mode == "pad" else f"mve_r1_{expected_text_mode}"
        runs_root = data_root() / "activations" / "moshi" / f"{cache_name}_zarr"
    clock_hz = float(grids["clocks"]["moshi"]["hz"])
    expected_n_steps = round(float(mve_cfg["max_minutes_per_session"]) * 60.0 * clock_hz)
    accepted_runner_versions = (
        [_runner_code_version()]
        if args.ablation_text_pad
        else _accepted_runner_code_versions()
    )
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
        accepted_runner_versions,
        float(mve_cfg["max_minutes_per_session"]) * 60.0,
        float(mve_cfg["mimi_chunk_seconds"]),
        int(mve_cfg["forward_chunk_steps"]),
        expected_text_mode=expected_text_mode,
        enforce_code_version=not args.ablation_text_pad,
        require_time_alignment=not args.ablation_text_pad,
    )
    observed_versions = preflight["observed_code_versions"]
    if args.ablation_text_pad and len(observed_versions) != 1:
        raise RuntimeError(f"PAD 消融缓存含多个 runner 版本：{observed_versions}")
    # 允许集合可以含尚未产生缓存的当前提交版本；分数包只绑定实际观测集合。
    runner_code_version = _runner_code_version_set_id(observed_versions)
    preflight["label_sha256"] = label_hashes
    preflight["runner_code_version_set_id"] = runner_code_version
    preflight_name = (
        "mve_preflight.json" if not ablation_tag else f"mve_preflight_ablation_{ablation_tag}.json"
    )
    preflight_report_path = _write_report_json_atomic(preflight_name, preflight)
    analysis_protocol["runner_code_versions"] = observed_versions
    analysis_protocol["runner_code_version_set_id"] = runner_code_version

    per_target: dict[str, dict] = {}
    raw_dir = data_root() / "mve"
    raw_dir.mkdir(parents=True, exist_ok=True)
    probe_seeds = [int(seed) for seed in mve_cfg["seeds"]]
    bundle_name = "g1_scores" if not ablation_tag else f"g1_scores_ablation_{ablation_tag}"
    score_writer = ScoreBundleWriter.create(
        raw_dir / bundle_name,
        relative_base=data_root(),
        eval_session_order=evals,
        targets=[str(target) for target in mve_cfg["targets"]],
        layers=[int(layer) for layer in mve_cfg["layers"]],
        seeds=probe_seeds,
    )

    def _target_analysis(
        target: str,
        delta: int | None,
        writer: ScoreBundleWriter | None,
        *,
        with_matched_mimi: bool = False,
    ) -> dict:
        """单目标全协议分析（嵌套选择 + 三基线 + bootstrap）。

        writer 非空 = 正式路径（分数落冻结 34 项分数包）；writer=None = 描述性路径
        （PREREG #8 附表：同协议计算，无分数包条目、无裁决效力）。
        with_matched_mimi = 附带计算 #11 信息下括号 Mimi 变体（读行 s−1，仅描述性；
        其条目由调用方从结果中弹出并放入 descriptive，不进入 per_target）。
        """
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
        inner_plans = {
            seed: build_training_sample_plan(
                _labels_root(),
                inner_train,
                run_specs,
                target,
                delta,
                int(mve_cfg["neg_downsample_ratio"]),
                seed,
            )
            for seed in probe_seeds
        }
        for seed, plan in inner_plans.items():
            n_positive = plan.n_selected_positive
            if n_positive == 0 or plan.n_selected == n_positive:
                raise RuntimeError(
                    f"{target}/seed{seed}: inner_train 抽样缺少正类或负类，无法做嵌套选择"
                )
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
                inner_plans=inner_plans,
                sessions_inner_val=inner_val,
            )
            if writer is not None:
                for cell in cells:
                    writer.add(
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
            inner_plans,
            inner_val,
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
        if writer is not None:
            for seed, scores in mimi_scores.items():
                writer.add(
                    target=target,
                    kind="mimi",
                    per_session=scores,
                    layer=None,
                    seed=seed,
                    best_c=mimi_best_c[seed],
                )
            writer.add(
                target=target,
                kind="hazard",
                per_session=hazard_scores,
                layer=None,
                seed=0,
            )
            writer.add(
                target=target,
                kind="acoustic_gru",
                per_session=acoustic_scores,
                layer=None,
                seed=0,
            )
        required_baselines = {"hazard", "mimi", "acoustic_gru"}
        # PREREG #7：最优层按 inner_val 选择 AUC 的种子均值选定，与 probe_val 无关；
        # 并列时取层号较小者（max 的首个命中，summary 按层升序构建）。
        best_layer = max(summary, key=lambda ell: summary[ell]["selection_auc_mean"])
        validate_seeded_baseline_alignment(
            summary[best_layer]["per_seed"],
            baselines,
            required_baselines,
            target,
        )
        result = evaluate_target(
            summary,
            baselines,
            bootstrap_n,
            float(mve_cfg["g1_full_threshold"]),
            float(mve_cfg["g1_backup_threshold"]),
            boot_seed=FROZEN_G1_BOOTSTRAP_SEED,
            best_layer=best_layer,
        )
        result["selection"] = {
            "rule": "best_layer = argmax_layer mean_seed(inner_val AUC at best_c)，并列取较小层号",
            "best_layer": int(best_layer),
            "selection_auc_mean_by_layer": {
                int(layer): summary[layer].get("selection_auc_mean")
                for layer in sorted(summary)
            },
        }
        if with_matched_mimi:
            # PREREG #11 描述性"信息下括号"变体：Mimi 双通道潜表征读行 s−1（少看当前帧），
            # 嵌套选择协议与官方 Mimi 基线完全相同；行集少每角色 step 0 一行（占比 <0.04%）。
            prev_min_step = min_eligible_step_for("mimi_prev")
            prev_plans = {
                seed: build_training_sample_plan(
                    _labels_root(),
                    train,
                    run_specs,
                    target,
                    delta,
                    int(mve_cfg["neg_downsample_ratio"]),
                    seed,
                    min_step=prev_min_step,
                )
                for seed in probe_seeds
            }
            prev_inner_plans = {
                seed: build_training_sample_plan(
                    _labels_root(),
                    inner_train,
                    run_specs,
                    target,
                    delta,
                    int(mve_cfg["neg_downsample_ratio"]),
                    seed,
                    min_step=prev_min_step,
                )
                for seed in probe_seeds
            }
            prev_cells = linear_feature_cells(
                sessions_eval=evals,
                target=target,
                delta_ms=delta,
                layer=-1,
                feature="mimi_prev",
                seeds=probe_seeds,
                c_grid=list(mve_cfg["probe_c_grid"]),
                runs_root=runs_root,
                run_specs=run_specs,
                plans=prev_plans,
                inner_plans=prev_inner_plans,
                sessions_inner_val=inner_val,
                min_step=prev_min_step,
            )
            prev_scores = {cell.seed: cell.per_session for cell in prev_cells}
            prev_metrics = seed_mean_metrics(prev_scores)
            probe_vs_prev = paired_seed_mean_advantage_bootstrap(
                summary[best_layer]["per_seed"],
                {"matched_mimi": prev_scores},
                n_boot=bootstrap_n,
                seed=FROZEN_G1_BOOTSTRAP_SEED,
            )
            result["matched_mimi_descriptive"] = {
                "note": (
                    "信息下括号变体（PREREG #11，非判据）：Mimi 双通道潜表征读行 s−1"
                    "（观测截止 s·τ）；若探针明显高于本变体而不高于官方 Mimi，"
                    "则官方基线的优势主要来自同帧连续潜表征的输入特权"
                ),
                "feature": "mimi_prev",
                "min_eligible_step": prev_min_step,
                "n_seeds": prev_metrics["n_seeds"],
                "auc_mean": prev_metrics["auc_mean"],
                "auc_sd": prev_metrics["auc_sd"],
                "best_c_by_seed": {
                    cell.seed: float(cell.metrics["best_c"]) for cell in prev_cells
                },
                "probe_auc_mean": result["advantage"]["probe_auc"],
                "official_mimi_auc_mean": result["baseline_metrics"]["mimi"]["auc_mean"],
                "probe_minus_matched": {
                    "advantage_point": probe_vs_prev["advantage_point"],
                    "ci_lo": probe_vs_prev["ci_lo"],
                    "ci_hi": probe_vs_prev["ci_hi"],
                },
            }
        return result

    matched_mimi_by_target: dict[str, dict] = {}
    for target in mve_cfg["targets"]:
        delta = int(mve_cfg["t1_delta_ms"]) if target == "T1" else None
        result = _target_analysis(
            str(target),
            delta,
            score_writer,
            with_matched_mimi=not ablation_tag,
        )
        matched = result.pop("matched_mimi_descriptive", None)
        if matched is not None:
            matched_mimi_by_target[str(target)] = matched
        per_target[target] = result

    # 描述性附表（PREREG #8/#11）：同协议计算，剥离裁决字段，无分数包条目
    descriptive: dict = {
        "note": "描述性附表（PREREG #8/#11）：非 G1 判据，无分数包条目，独立审计不复算",
        "T1": {},
    }
    if matched_mimi_by_target:
        descriptive["matched_mimi"] = matched_mimi_by_target
    if not ablation_tag:
        for delta_value in mve_cfg.get("t1_descriptive_deltas_ms", []):
            delta = int(delta_value)
            entry = _target_analysis("T1", delta, None)
            entry.pop("verdict", None)
            entry["delta_ms"] = delta
            entry["net_lead_ms"] = [delta - 80, delta]
            entry["layer_summary"] = {
                int(layer): values for layer, values in entry["layer_summary"].items()
            }
            descriptive["T1"][str(delta)] = entry
            print(
                f"描述性 T1 δ{delta}：优势 {entry['advantage']['advantage_point']:+.4f}"
                f"（净前瞻 [{delta - 80},{delta}) ms，非判据）"
            )

    overall = overall_g1(per_target, float(mve_cfg["g1_full_threshold"]), float(mve_cfg["g1_backup_threshold"]))
    meta = {
        "layers": mve_cfg["layers"],
        "seeds": mve_cfg["seeds"],
        "bootstrap_n": mve_cfg["bootstrap_n"],
        "n_train_sessions": len(train),
        "n_eval_sessions": len(evals),
        "n_inner_train_sessions": len(inner_train),
        "n_inner_val_sessions": len(inner_val),
        "text_mode": expected_text_mode,
        "ablation": ablation_tag,
        "t1_delta_ms": int(mve_cfg["t1_delta_ms"]),
        "context_truncation": dict(CONTEXT_TRUNCATION),
        "descriptive": descriptive,
    }
    protocol = {
        "text_mode": expected_text_mode,
        "ablation": ablation_tag,
        "time_alignment": dict(ANALYSIS_TIME_ALIGNMENT),
        "context_truncation": dict(CONTEXT_TRUNCATION),
        "nested_selection": {
            "inner_val_sessions": inner_val,
            "n_inner_train": len(inner_train),
            "n_inner_val": len(inner_val),
        },
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
        protocol=protocol,
        descriptive=descriptive,
        ablation_tag=ablation_tag,
    )
    prefix = "消融结论（非正式 G1）" if ablation_tag else "G1 裁决"
    print(f"{prefix}：{overall['verdict']}（优势 {overall['advantage_point']:+.4f}，CI 下界 {overall['ci_lo']:+.4f}）")


if __name__ == "__main__":
    main()
