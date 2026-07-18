"""MVE 最终裁决前的输入同步与硬预检。

本模块只检查冻结协议所需的磁盘契约，不加载完整激活矩阵。任何输入不完整都应在训练探针前失败，
避免耗时计算结束后才发现缓存、标签或 Mimi 基线缺失。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from floor_circuit.cachelib.manifest import load_manifest, sha256_file
from floor_circuit.mve.alignment import (
    MIN_ELIGIBLE_STEP,
    RUNNER_TIME_ALIGNMENT,
    usable_label_steps,
)
from floor_circuit.mve.dataset import eligible_rows, run_dir_for
from floor_circuit.probes.stats import PerSession, ScoreCollection, SeededPerSession


@dataclass(frozen=True)
class RunSpec:
    """单个角色缓存经过预检后的时间域信息。"""

    session_id: str
    agent_channel: int
    run_dir: Path
    n_steps: int
    clock_hz: float
    source_audio_hashes: tuple[str, ...]
    code_version: str | None = None  # manifest 实测值（消融模式的溯源依据）


class MvePreflightError(RuntimeError):
    """MVE 输入无法支持正式 G1 裁决。"""


def validate_baseline_alignment(
    reference: PerSession,
    baselines: dict[str, PerSession],
    required_names: set[str],
    target: str,
) -> None:
    """确认三基线与探针在同一会话、同一标签行上逐项对齐。"""

    issues: list[str] = []
    if set(baselines) != required_names:
        issues.append(f"基线集合 {sorted(baselines)}，期望 {sorted(required_names)}")
    reference_sessions = sorted(reference)
    for name, scores_by_session in baselines.items():
        if sorted(scores_by_session) != reference_sessions:
            issues.append(f"{name}: 会话集合与探针不一致")
            continue
        for sid in reference_sessions:
            y_reference = np.asarray(reference[sid][0])
            y_baseline, scores = (np.asarray(value) for value in scores_by_session[sid])
            if not np.array_equal(y_baseline, y_reference):
                issues.append(f"{name}/{sid}: 标签行与探针不一致")
            if len(scores) != len(y_reference):
                issues.append(f"{name}/{sid}: 分数长度 {len(scores)}，标签长度 {len(y_reference)}")
            elif not np.isfinite(scores).all():
                issues.append(f"{name}/{sid}: 分数含非有限值")
    if issues:
        sample = "；".join(issues[:10])
        suffix = f"；另有 {len(issues) - 10} 项" if len(issues) > 10 else ""
        raise MvePreflightError(f"{target} 三基线时间域对齐失败：{sample}{suffix}")


def validate_seeded_baseline_alignment(
    probes: SeededPerSession,
    baselines: dict[str, ScoreCollection],
    required_names: set[str],
    target: str,
) -> None:
    """逐种子确认探针与单/多种子基线使用完全相同的标签行。"""

    issues: list[str] = []
    probe_seeds = sorted(probes)
    if not probe_seeds:
        issues.append("探针种子集合为空")
    if set(baselines) != required_names:
        issues.append(f"基线集合 {sorted(baselines)}，期望 {sorted(required_names)}")
    if issues:
        raise MvePreflightError(f"{target} 多种子时间域对齐失败：" + "；".join(issues))

    reference_seed = probe_seeds[0]
    reference = probes[reference_seed]
    reference_sessions = sorted(reference)

    def check(candidate: PerSession, label: str, expected: PerSession) -> None:
        if sorted(candidate) != sorted(expected):
            issues.append(f"{label}: 会话集合不一致")
            return
        for sid in sorted(expected):
            y_expected = np.asarray(expected[sid][0])
            y_candidate, scores = (np.asarray(value) for value in candidate[sid])
            if not np.array_equal(y_candidate, y_expected):
                issues.append(f"{label}/{sid}: 标签行与探针不一致")
            if len(scores) != len(y_expected):
                issues.append(
                    f"{label}/{sid}: 分数长度 {len(scores)}，标签长度 {len(y_expected)}"
                )
            elif not np.isfinite(scores).all():
                issues.append(f"{label}/{sid}: 分数含非有限值")

    for seed in probe_seeds:
        check(probes[seed], f"probe/seed{seed}", reference)

    for name, scores in baselines.items():
        first = next(iter(scores.values()), None)
        if isinstance(first, dict):
            seeded = scores
            if sorted(seeded) != probe_seeds:
                issues.append(
                    f"{name}: 种子集合 {sorted(seeded)}，期望 {probe_seeds}"
                )
                continue
            for seed in probe_seeds:
                check(seeded[seed], f"{name}/seed{seed}", probes[seed])
        else:
            for seed in probe_seeds:
                check(scores, f"{name}/seed{seed}", probes[seed])

    if not reference_sessions:
        issues.append("探针会话集合为空")
    if issues:
        sample = "；".join(issues[:10])
        suffix = f"；另有 {len(issues) - 10} 项" if len(issues) > 10 else ""
        raise MvePreflightError(f"{target} 多种子时间域对齐失败：{sample}{suffix}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sync_labels_atomic(
    source_dir: str | Path,
    dest_dir: str | Path,
    session_ids: list[str],
) -> dict[str, str]:
    """从 WP1 权威目录原子刷新标签平铺目录，并回传逐文件哈希。

    每次运行都覆盖目标文件，确保 WP1 重跑后不会继续使用旧的平铺副本。写入采用同目录临时文件
    加原子替换；替换后再次读取并核对哈希。
    """

    source_dir, dest_dir = Path(source_dir), Path(dest_dir)
    missing = []
    for sid in session_ids:
        for path in (
            source_dir / f"{sid}.labels.parquet",
            source_dir / f"{sid}.complete.json",
        ):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        sample = "；".join(missing[:5])
        raise MvePreflightError(f"缺少 {len(missing)} 份 WP1 标签或完成标记。样例：{sample}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for sid in session_ids:
        source = source_dir / f"{sid}.labels.parquet"
        marker_path = source_dir / f"{sid}.complete.json"
        dest = dest_dir / f"{sid}.parquet"
        payload = source.read_bytes()
        digest = _sha256_bytes(payload)
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            declared = marker["outputs"]["labels"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise MvePreflightError(f"WP1 完成标记无法读取：{sid}（{exc!r}）") from exc
        if (
            marker.get("schema_version") != 1
            or marker.get("session") != sid
            or declared.get("name") != source.name
            or declared.get("size") != len(payload)
            or declared.get("sha256") != digest
        ):
            raise MvePreflightError(f"WP1 完成标记与标签内容不一致：{sid}")
        _write_bytes_atomic(dest, payload)
        if _sha256_bytes(dest.read_bytes()) != digest:
            raise MvePreflightError(f"标签原子刷新后哈希不一致：{sid}")
        hashes[sid] = digest
    return hashes


def _validate_label(
    path: Path,
    sid: str,
    run_specs: dict[tuple[str, int], RunSpec],
    t1_delta_ms: int,
) -> list[str]:
    issues: list[str] = []
    try:
        labels = pd.read_parquet(path)
    except Exception as exc:
        return [f"{sid}: 标签无法读取（{exc!r}）"]

    required_columns = {"target", "agent_channel", "step", "label", "delta_ms"}
    missing_columns = sorted(required_columns - set(labels.columns))
    if missing_columns:
        return [f"{sid}: 标签缺列 {missing_columns}"]
    if labels.empty:
        return [f"{sid}: 标签为空"]

    steps = pd.to_numeric(labels["step"], errors="coerce")
    if steps.isna().any() or (steps < 0).any():
        issues.append(f"{sid}: 标签 step 含非法值")
    if labels.duplicated(["target", "agent_channel", "step", "delta_ms"]).any():
        issues.append(f"{sid}: 标签存在重复键")

    for channel in (0, 1):
        spec = run_specs.get((sid, channel))
        if spec is None:
            continue
        # T1 按特征装配的真实行域（步 0..n_steps−2）检查；T5 逐步覆盖必须查满 0..n_steps−1
        for target, delta, max_steps in (
            ("T1", t1_delta_ms, usable_label_steps(spec.n_steps)),
            ("T5", None, spec.n_steps),
        ):
            rows = eligible_rows(
                labels,
                target,
                delta,
                channel,
                max_steps=max_steps,
                min_step=MIN_ELIGIBLE_STEP,
            )
            if rows.empty:
                issues.append(f"{sid}/agent{channel}: 前 {spec.n_steps} 步缺少 {target} 标签")
        for target, delta in (("T1", t1_delta_ms), ("T4", None)):
            rows = eligible_rows(
                labels,
                target,
                delta,
                channel,
                max_steps=usable_label_steps(spec.n_steps),
                min_step=MIN_ELIGIBLE_STEP,
            )
            if not rows.empty and not np.isin(rows["label"].to_numpy(), [0, 1]).all():
                issues.append(f"{sid}/agent{channel}: {target} 含非二值标签")
        t5 = eligible_rows(labels, "T5", None, channel, max_steps=spec.n_steps)
        if len(t5) != spec.n_steps or not np.array_equal(
            t5["step"].to_numpy(dtype=np.int64),
            np.arange(spec.n_steps, dtype=np.int64),
        ):
            issues.append(f"{sid}/agent{channel}: T5 必须逐步覆盖 0..{spec.n_steps - 1}，实际 {len(t5)} 行")
    return issues


def _validate_target_pools(
    labels_root: Path,
    session_pools: dict[str, list[str]],
    run_specs: dict[tuple[str, int], RunSpec],
    t1_delta_ms: int,
) -> tuple[list[str], dict]:
    """允许 T4 单角色或单会话无事件，但训练池与验证池都必须具备二分类监督。

    计数口径与特征装配一致（min_step=MIN_ELIGIBLE_STEP）：快照里的 n_rows/正负类
    统计对应真实可用行集，而非含 step 0 的原始标签行。
    """

    issues: list[str] = []
    summary: dict[str, dict] = {}
    for pool_name in ("train", "val"):
        target_values: dict[str, list[np.ndarray]] = {"T1": [], "T4": []}
        for sid in session_pools[pool_name]:
            label_path = labels_root / f"{sid}.parquet"
            if not label_path.is_file():
                continue
            try:
                labels = pd.read_parquet(label_path)
            except Exception:
                continue
            for channel in (0, 1):
                spec = run_specs.get((sid, channel))
                if spec is None:
                    continue
                for target, delta in (("T1", t1_delta_ms), ("T4", None)):
                    rows = eligible_rows(
                        labels,
                        target,
                        delta,
                        channel,
                        max_steps=usable_label_steps(spec.n_steps),
                        min_step=MIN_ELIGIBLE_STEP,
                    )
                    target_values[target].append(rows["label"].to_numpy(dtype=np.int64))
        pool_summary: dict[str, dict] = {}
        for target, parts in target_values.items():
            values = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            n_positive = int(np.sum(values == 1))
            n_negative = int(np.sum(values == 0))
            pool_summary[target] = {
                "n_rows": len(values),
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
            if n_positive == 0 or n_negative == 0:
                issues.append(
                    f"{pool_name} 池 {target} 必须同时含正负类，"
                    f"当前正类={n_positive}、负类={n_negative}"
                )
        summary[pool_name] = pool_summary
    return issues, summary


def _validate_run(
    runs_root: Path,
    sid: str,
    channel: int,
    layers: list[int],
    expected_n_steps: int,
    expected_clock_hz: float,
    expected_code_version: str,
    expected_max_seconds: float,
    expected_mimi_chunk_seconds: float,
    expected_forward_chunk_steps: int,
    current_audio_hashes: dict[str, str] | None,
    expected_text_mode: str,
    enforce_code_version: bool,
    require_time_alignment: bool,
) -> tuple[RunSpec | None, list[str]]:
    run_dir = run_dir_for(runs_root, sid, channel)
    prefix = f"{sid}/agent{channel}"
    issues: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, [f"{prefix}: 缺少 manifest.json"]
    try:
        manifest = load_manifest(run_dir)
    except Exception as exc:
        return None, [f"{prefix}: manifest 无法校验（{exc!r}）"]

    if manifest.model != "moshi" or manifest.mode != "R1":
        issues.append(f"{prefix}: 模型/模式应为 moshi/R1")
    if manifest.session_id != sid:
        issues.append(f"{prefix}: manifest.session_id={manifest.session_id!r}")
    if manifest.layers != layers:
        issues.append(f"{prefix}: 层列表 {manifest.layers}，期望 {layers}")
    if manifest.n_steps != expected_n_steps:
        issues.append(f"{prefix}: n_steps={manifest.n_steps}，期望 {expected_n_steps}")
    if manifest.clock_hz is None or not np.isclose(manifest.clock_hz, expected_clock_hz):
        issues.append(f"{prefix}: clock_hz={manifest.clock_hz}，期望 {expected_clock_hz}")
    if manifest.text_mode != expected_text_mode:
        issues.append(
            f"{prefix}: text_mode={manifest.text_mode!r}，冻结协议要求 {expected_text_mode!r}"
        )
    if enforce_code_version and manifest.code_version != expected_code_version:
        issues.append(
            f"{prefix}: code_version={manifest.code_version!r}，期望 {expected_code_version!r}"
        )
    if len(manifest.source_audio) != 2 or any(len(str(digest)) != 64 for digest in manifest.source_audio.values()):
        issues.append(f"{prefix}: source_audio 应含两条 SHA-256")
    source_by_name = {Path(path).name: str(digest) for path, digest in manifest.source_audio.items()}
    if current_audio_hashes is None:
        issues.append(f"{prefix}: 当前源音频哈希不可用")
    elif source_by_name != current_audio_hashes:
        issues.append(f"{prefix}: manifest 源音频哈希与当前 CANDOR WAV 不一致")
    if not manifest.mimi_latent:
        issues.append(f"{prefix}: manifest 未声明 mimi_latent")
    execution = manifest.extra.get("execution", {})

    def numeric_matches(value, expected: float) -> bool:
        try:
            return bool(np.isclose(float(value), expected))
        except (TypeError, ValueError):
            return False

    if execution.get("forward_mode") != "streaming_teacher_forced_backbone":
        issues.append(f"{prefix}: forward_mode 不是有状态 teacher-forced backbone")
    if not numeric_matches(execution.get("max_seconds"), expected_max_seconds):
        issues.append(f"{prefix}: max_seconds={execution.get('max_seconds')!r}，期望 {expected_max_seconds}")
    if not numeric_matches(execution.get("mimi_chunk_seconds"), expected_mimi_chunk_seconds):
        issues.append(
            f"{prefix}: mimi_chunk_seconds={execution.get('mimi_chunk_seconds')!r}，"
            f"期望 {expected_mimi_chunk_seconds}"
        )
    if execution.get("mimi_chunk_frames") != 1:
        issues.append(f"{prefix}: Mimi 流式块必须恰好为 1 帧")
    if execution.get("forward_chunk_steps") != expected_forward_chunk_steps:
        issues.append(
            f"{prefix}: forward_chunk_steps={execution.get('forward_chunk_steps')!r}，"
            f"期望 {expected_forward_chunk_steps}"
        )
    for field in ("transformer_state_preserved", "mimi_state_preserved", "depformer_skipped"):
        if execution.get(field) is not True:
            issues.append(f"{prefix}: execution.{field} 必须为 true")
    if manifest.extra.get("delay_application") != "global_once_before_streaming_forward":
        issues.append(f"{prefix}: delay_application 不是全时间轴只施加一次")
    if execution.get("latent_kind") != "pre_quantization_continuous":
        issues.append(f"{prefix}: Mimi latent 不是量化前连续表征")
    if require_time_alignment:
        declared = execution.get("time_alignment")
        if declared != RUNNER_TIME_ALIGNMENT:
            issues.append(
                f"{prefix}: execution.time_alignment={declared!r}，"
                f"期望 {RUNNER_TIME_ALIGNMENT!r}（PREREG #7）"
            )

    n_steps = int(manifest.n_steps or 0)
    clock_hz = float(manifest.clock_hz or 0.0)
    if n_steps <= 0 or clock_hz <= 0:
        return None, issues

    try:
        group = zarr.open_group(str(run_dir), mode="r")
    except Exception as exc:
        return None, [*issues, f"{prefix}: zarr 组无法打开（{exc!r}）"]

    required_arrays = [f"acts_L{layer}" for layer in layers] + ["mimi_latent"]
    for name in required_arrays:
        try:
            item = group[name]
        except Exception:
            issues.append(f"{prefix}: 缺少数组 {name}")
            continue
        if not isinstance(item, zarr.Array):
            issues.append(f"{prefix}: {name} 不是 zarr 数组")
            continue
        array = item
        shape = tuple(int(value) for value in array.shape)
        if len(shape) != 2 or shape[0] != n_steps or shape[1] <= 0:
            issues.append(f"{prefix}: {name} 形状 {shape} 与 n_steps={n_steps} 不一致")
        if np.dtype(array.dtype) != np.dtype(np.float16):
            issues.append(f"{prefix}: {name} dtype={array.dtype}，期望 float16")
        if name.startswith("acts_L") and len(shape) == 2 and manifest.hidden_dim not in (None, shape[1]):
            issues.append(f"{prefix}: {name} 隐层维度 {shape[1]} 与 manifest {manifest.hidden_dim} 不一致")

    return RunSpec(
        sid,
        channel,
        run_dir,
        n_steps,
        clock_hz,
        tuple(sorted(str(value) for value in manifest.source_audio.values())),
        manifest.code_version,
    ), issues


def preflight_mve_inputs(
    runs_root: str | Path,
    labels_root: str | Path,
    audio_root: str | Path,
    session_ids: list[str],
    session_pools: dict[str, list[str]],
    layers: list[int],
    expected_n_steps: int,
    expected_clock_hz: float,
    t1_delta_ms: int,
    expected_code_version: str,
    expected_max_seconds: float,
    expected_mimi_chunk_seconds: float,
    expected_forward_chunk_steps: int,
    *,
    expected_text_mode: str,
    enforce_code_version: bool = True,
    require_time_alignment: bool = True,
) -> tuple[dict[tuple[str, int], RunSpec], dict]:
    """硬校验正式 G1 所需的 400 个 run、五类数组与 200 份标签。

    expected_text_mode：冻结的 R1 文本流协议（正式 = greedy；PAD 消融显式传 pad）。
    enforce_code_version / require_time_alignment：仅 PAD 消融复盘允许放宽
    （旧缓存由撤回前的 runner 生成，无 time_alignment 声明）。
    """

    runs_root, labels_root, audio_root = Path(runs_root), Path(labels_root), Path(audio_root)
    if len(session_ids) != len(set(session_ids)):
        raise MvePreflightError("MVE 会话列表含重复项")
    if set(session_pools) != {"train", "val"}:
        raise MvePreflightError("session_pools 必须且只能包含 train/val")
    train_ids, val_ids = session_pools["train"], session_pools["val"]
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise MvePreflightError("MVE train/val 池含重复会话")
    if set(train_ids) & set(val_ids):
        raise MvePreflightError("MVE train/val 池存在会话重叠")
    if set(train_ids) | set(val_ids) != set(session_ids):
        raise MvePreflightError("MVE train/val 池与总会话列表不一致")

    issues: list[str] = []
    specs: dict[tuple[str, int], RunSpec] = {}
    audio_hashes: dict[str, dict[str, str] | None] = {}
    for sid in session_ids:
        current: dict[str, str] = {}
        for channel in (0, 1):
            path = audio_root / sid / f"audio_ch{channel}.wav"
            if not path.is_file():
                issues.append(f"{sid}: 缺少当前源音频 {path}")
                continue
            current[path.name] = sha256_file(path)
        audio_hashes[sid] = current if len(current) == 2 else None
    for sid in session_ids:
        for channel in (0, 1):
            spec, run_issues = _validate_run(
                runs_root,
                sid,
                channel,
                layers,
                expected_n_steps,
                expected_clock_hz,
                expected_code_version,
                expected_max_seconds,
                expected_mimi_chunk_seconds,
                expected_forward_chunk_steps,
                audio_hashes[sid],
                expected_text_mode,
                enforce_code_version,
                require_time_alignment,
            )
            issues.extend(run_issues)
            if spec is not None:
                specs[(sid, channel)] = spec
        spec0, spec1 = specs.get((sid, 0)), specs.get((sid, 1))
        if spec0 is not None and spec1 is not None:
            if spec0.source_audio_hashes != spec1.source_audio_hashes:
                issues.append(f"{sid}: 两个角色 run 的源音频哈希集合不一致")
            if spec0.n_steps != spec1.n_steps:
                issues.append(f"{sid}: 两个角色 run 的 n_steps 不一致")

    for sid in session_ids:
        label_path = labels_root / f"{sid}.parquet"
        if not label_path.is_file():
            issues.append(f"{sid}: 缺少平铺标签")
            continue
        issues.extend(_validate_label(label_path, sid, specs, t1_delta_ms))

    pool_issues, pool_summary = _validate_target_pools(
        labels_root,
        session_pools,
        specs,
        t1_delta_ms,
    )
    issues.extend(pool_issues)

    expected_runs = len(session_ids) * 2
    if len(specs) != expected_runs:
        issues.append(f"通过 manifest 基础读取的 run 为 {len(specs)}/{expected_runs}")
    if issues:
        sample = "\n".join(f"- {issue}" for issue in issues[:40])
        suffix = f"\n……另有 {len(issues) - 40} 项" if len(issues) > 40 else ""
        raise MvePreflightError(f"MVE 输入预检失败，共 {len(issues)} 项：\n{sample}{suffix}")

    report = {
        "status": "passed",
        "n_sessions": len(session_ids),
        "n_runs": len(specs),
        "layers": layers,
        "required_arrays_per_run": [f"acts_L{layer}" for layer in layers] + ["mimi_latent"],
        "expected_n_steps": expected_n_steps,
        "expected_clock_hz": expected_clock_hz,
        "expected_text_mode": expected_text_mode,
        "enforce_code_version": enforce_code_version,
        "require_time_alignment": require_time_alignment,
        "observed_code_versions": sorted(
            {str(spec.code_version) for spec in specs.values()}
        ),
        "expected_code_version": expected_code_version,
        "expected_max_seconds": expected_max_seconds,
        "expected_mimi_chunk_seconds": expected_mimi_chunk_seconds,
        "expected_forward_chunk_steps": expected_forward_chunk_steps,
        "source_audio_hashes_verified": len(session_ids) * 2,
        "latent_kind": "pre_quantization_continuous",
        "n_labels": len(session_ids),
        "target_pools": pool_summary,
    }
    return specs, report
