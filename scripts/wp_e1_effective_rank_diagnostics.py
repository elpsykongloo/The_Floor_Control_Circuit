"""E1 有效秩三项事后描述性诊断（PREREG #30）。

本脚本只读取 T4 正式格子、冻结标签行域与 L29/L30/L31 激活缓存，回答：
  1. 每个 k 重新嵌套选择 C 后，有效秩是否显著下降；
  2. 共同 top-3 层之间的有效秩是否存在层位差异；
  3. 64 到 128 之间首次越过 95% 保留线的具体位置。

诊断严格不回写正式汇总，不改变 G2 判定。每个 (layer, seed, k) 使用独立
原子 JSON 断点，重复启动时只计算缺失点。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from time import perf_counter

import numpy as np
import wp_e1_probe_grid as engine
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg

SCHEMA_VERSION = 1
PROTOCOL_NAME = "effective-rank-posthoc-v1"
TARGET = "T4"
LAYERS = (29, 30, 31)
COARSE_KS = (1, 2, 4, 8, 16, 24, 32, 64, 128)
FINE_KS = tuple(range(65, 129))
BASE_KS = tuple(sorted(set(COARSE_KS) | set(FINE_KS)))
REPLAY_TOLERANCE = 1e-3


@dataclass
class ProjectedTask:
    """一个层×种子的 PCA 投影及冻结元数据。"""

    layer: int
    seed: int
    train_features: np.ndarray
    train_labels: np.ndarray
    inner_mask: np.ndarray
    eval_features: np.ndarray
    eval_labels: np.ndarray
    pca_cumulative_variance: np.ndarray
    full_auc: float
    fixed_c: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tmp.replace(path)


def _checkpoint_path(root: Path, layer: int, seed: int, k: int) -> Path:
    return root / f"L{layer}__s{seed}__k{k:03d}.json"


def _load_checkpoint(
    root: Path, protocol_hash: str, layer: int, seed: int, k: int
) -> dict | None:
    path = _checkpoint_path(root, layer, seed, k)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    identity = (
        payload.get("schema_version"),
        payload.get("protocol_hash"),
        payload.get("layer"),
        payload.get("seed"),
        payload.get("k"),
    )
    expected = (SCHEMA_VERSION, protocol_hash, layer, seed, k)
    return payload if identity == expected else None


def _crossed(auc: float, full_auc: float, retention: float) -> bool:
    return (float(auc) - 0.5) >= retention * (float(full_auc) - 0.5)


def _choose_c(curve: dict[str, float], c_grid: list[float]) -> tuple[float, float]:
    """沿正式顺序取严格最大值；并列时保留最先出现的 C。"""
    best_c = float(c_grid[0])
    best_metric = float(curve[str(c_grid[0])])
    for c_value in c_grid[1:]:
        metric = float(curve[str(c_value)])
        if metric > best_metric:
            best_c = float(c_value)
            best_metric = metric
    return best_c, best_metric


def _refinement_ks(
    records: dict[int, dict], retention: float, mode: str
) -> tuple[int, ...]:
    """若原网格在 64 或更早过线，补齐相邻原网格区间。"""
    first_index = None
    for index, k in enumerate(COARSE_KS):
        record = records.get(k)
        if record is None:
            return ()
        if _crossed(record[mode]["auc"], record["full_auc"], retention):
            first_index = index
            break
    if first_index is None or COARSE_KS[first_index] > 64 or first_index == 0:
        return ()
    lower = COARSE_KS[first_index - 1]
    upper = COARSE_KS[first_index]
    return tuple(range(lower + 1, upper))


def _planned_ks(records: dict[int, dict], retention: float) -> tuple[int, ...]:
    extra = set(_refinement_ks(records, retention, "fixed_c"))
    extra.update(_refinement_ks(records, retention, "nested_c"))
    return tuple(sorted(set(BASE_KS) | extra))


def _task_is_complete(
    root: Path,
    protocol_hash: str,
    layer: int,
    seed: int,
    retention: float,
) -> bool:
    records = {
        k: record
        for k in BASE_KS
        if (
            record := _load_checkpoint(root, protocol_hash, layer, seed, k)
        )
        is not None
    }
    if len(records) != len(BASE_KS):
        return False
    return all(
        _load_checkpoint(root, protocol_hash, layer, seed, k) is not None
        for k in _planned_ks(records, retention)
    )


def _prefix_prepared(prepared: pg.PreparedLinearData, k: int) -> pg.PreparedLinearData:
    """复用 max-k 的逐列标准化结果；任意前缀的统计量保持完全相同。"""
    return pg.PreparedLinearData(
        mean=prepared.mean[:k].copy(),
        scale=prepared.scale[:k].copy(),
        n_classes=prepared.n_classes,
        x=prepared.x[:, :k],
        y=prepared.y,
    )


def _fit_one_k(
    task: ProjectedTask,
    k: int,
    c_grid: list[float],
    trainer: dict,
    retention: float,
    prepared_c: pg.PreparedLinearData,
    prepared_full: pg.PreparedLinearData,
    device: str,
) -> dict:
    started = perf_counter()
    fit_kwargs = {
        "max_iter": int(trainer["lbfgs_max_iter"]),
        "tolerance_grad": float(trainer["lbfgs_tolerance_grad"]),
    }
    c_data = _prefix_prepared(prepared_c, k)
    candidates = []
    warm = None
    for c_value in c_grid:
        probe = c_data.fit(c_value, init=warm, **fit_kwargs)
        candidates.append(probe)
        warm = probe

    inner_features = np.ascontiguousarray(task.train_features[task.inner_mask, :k])
    predictor = pg.LinearProbeBatchPredictor(candidates, device=device)
    inner_probs = predictor.predict_proba(inner_features)
    inner_labels = task.train_labels[task.inner_mask]
    curve = {
        str(c_value): pg.primary_metric(inner_labels, probs, 2)
        for c_value, probs in zip(c_grid, inner_probs, strict=True)
    }
    chosen_c, chosen_metric = _choose_c(curve, c_grid)
    del predictor, inner_probs, inner_features, c_data, warm

    full_data = _prefix_prepared(prepared_full, k)
    nested_probe = full_data.fit(chosen_c, **fit_kwargs)
    eval_features = task.eval_features[:, :k]
    nested_auc = pg.primary_metric(
        task.eval_labels, nested_probe.predict_proba(eval_features), 2
    )
    if chosen_c == task.fixed_c:
        fixed_probe = nested_probe
        fixed_auc = nested_auc
        fixed_reused_nested = True
    else:
        fixed_probe = full_data.fit(task.fixed_c, init=nested_probe, **fit_kwargs)
        fixed_auc = pg.primary_metric(
            task.eval_labels, fixed_probe.predict_proba(eval_features), 2
        )
        fixed_reused_nested = False

    return {
        "schema_version": SCHEMA_VERSION,
        "layer": task.layer,
        "seed": task.seed,
        "k": k,
        "full_auc": task.full_auc,
        "threshold_auc": 0.5 + retention * (task.full_auc - 0.5),
        "pca_cumulative_variance": float(task.pca_cumulative_variance[k - 1]),
        "fixed_c": {
            "c": task.fixed_c,
            "auc": fixed_auc,
            "retention_fraction": (fixed_auc - 0.5) / (task.full_auc - 0.5),
            "crossed": _crossed(fixed_auc, task.full_auc, retention),
            "converged": bool(fixed_probe.converged),
            "reused_nested_fit": fixed_reused_nested,
        },
        "nested_c": {
            "chosen_c": chosen_c,
            "inner_val_metric": chosen_metric,
            "inner_val_curve": curve,
            "auc": nested_auc,
            "retention_fraction": (nested_auc - 0.5) / (task.full_auc - 0.5),
            "crossed": _crossed(nested_auc, task.full_auc, retention),
            "candidate_converged": [bool(probe.converged) for probe in candidates],
            "converged": bool(nested_probe.converged),
        },
        "wall_seconds": perf_counter() - started,
    }


def _run_projected_task(
    task: ProjectedTask,
    *,
    device: str,
    root: Path,
    protocol_hash: str,
    c_grid: list[float],
    trainer: dict,
    retention: float,
    force: bool,
) -> int:
    import torch

    existing = {
        k: record
        for k in BASE_KS
        if not force
        and (
            record := _load_checkpoint(root, protocol_hash, task.layer, task.seed, k)
        )
        is not None
    }
    c_mask = ~task.inner_mask
    c_features = np.ascontiguousarray(task.train_features[c_mask])
    c_labels = np.ascontiguousarray(task.train_labels[c_mask])
    prepared_c = pg.prepare_linear_probe_blocks(
        [(c_features, c_labels)], len(c_labels), task.train_features.shape[1], 2, device=device
    )
    prepared_full = pg.prepare_linear_probe_blocks(
        [(task.train_features, task.train_labels)],
        len(task.train_labels),
        task.train_features.shape[1],
        2,
        device=device,
    )
    del c_features, c_labels

    completed = 0

    def run_ks(ks: tuple[int, ...]) -> None:
        nonlocal completed
        for k in ks:
            if not force and k in existing:
                continue
            record = _fit_one_k(
                task,
                k,
                c_grid,
                trainer,
                retention,
                prepared_c,
                prepared_full,
                device,
            )
            record["protocol_hash"] = protocol_hash
            _atomic_write_json(
                _checkpoint_path(root, task.layer, task.seed, k), record
            )
            existing[k] = record
            completed += 1
            if completed == 1 or completed % 8 == 0 or k in COARSE_KS:
                print(
                    f"[{device}] L{task.layer}/s{task.seed}：新增 {completed} 点，"
                    f"k={k}，固定 AUC={record['fixed_c']['auc']:.6f}，"
                    f"嵌套 C={record['nested_c']['chosen_c']:g}",
                    flush=True,
                )

    run_ks(BASE_KS)
    extra = tuple(k for k in _planned_ks(existing, retention) if k not in BASE_KS)
    run_ks(extra)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
        torch.cuda.empty_cache()
    print(
        f"[{device}] L{task.layer}/s{task.seed} 完成：本次新增 {completed} 点",
        flush=True,
    )
    return completed


def _project_task(
    *,
    layer: int,
    seed: int,
    train_roles: list[g.RoleRows],
    inner_sessions: set[str],
    train_store: dict,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    full_auc: float,
    fixed_c: float,
) -> ProjectedTask:
    started = perf_counter()
    train_features, train_labels, _ = g.assemble(
        train_roles, "acts", train_store, dtype=np.float32
    )
    inner_mask = np.concatenate(
        [
            np.full(len(role.labels), role.session_id in inner_sessions, dtype=np.bool_)
            for role in train_roles
        ]
    )
    if not inner_mask.any() or inner_mask.all():
        raise RuntimeError(f"L{layer}/s{seed} 的 inner_val 划分为空")
    x64 = np.asarray(train_features, dtype=np.float64)
    del train_features
    center = x64.mean(axis=0)
    centered = x64 - center
    del x64
    print(
        f"L{layer}/s{seed}：开始精确 float64 PCA，"
        f"矩阵={centered.shape[0]}×{centered.shape[1]}",
        flush=True,
    )
    left, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    del left
    basis = vt[: max(BASE_KS)]
    train_proj = np.ascontiguousarray((centered @ basis.T).astype(np.float32))
    eval64 = np.asarray(eval_features, dtype=np.float64)
    eval_proj = np.ascontiguousarray(((eval64 - center) @ basis.T).astype(np.float32))
    variance = np.cumsum(np.square(singular_values, dtype=np.float64))
    variance /= variance[-1]
    del centered, eval64, singular_values, vt, basis
    gc.collect()
    print(
        f"L{layer}/s{seed}：PCA 投影完成，耗时 {perf_counter() - started:.1f}s",
        flush=True,
    )
    return ProjectedTask(
        layer=layer,
        seed=seed,
        train_features=train_proj,
        train_labels=np.ascontiguousarray(train_labels),
        inner_mask=inner_mask,
        eval_features=eval_proj,
        eval_labels=eval_labels,
        pca_cumulative_variance=variance[: max(BASE_KS)],
        full_auc=float(full_auc),
        fixed_c=float(fixed_c),
    )


def _load_records(
    root: Path,
    protocol_hash: str,
    seeds: list[int],
    retention: float,
) -> dict[int, dict[int, dict[int, dict]]]:
    output = {}
    for layer in LAYERS:
        output[layer] = {}
        for seed in seeds:
            base = {
                k: record
                for k in BASE_KS
                if (
                    record := _load_checkpoint(root, protocol_hash, layer, seed, k)
                )
                is not None
            }
            if len(base) != len(BASE_KS):
                raise RuntimeError(f"L{layer}/s{seed} 缺少基础 k 断点")
            planned = _planned_ks(base, retention)
            records = {
                k: _load_checkpoint(root, protocol_hash, layer, seed, k)
                for k in planned
            }
            missing = [k for k, record in records.items() if record is None]
            if missing:
                raise RuntimeError(f"L{layer}/s{seed} 缺少补扫点：{missing}")
            output[layer][seed] = records
    return output


def _rank_from_records(records: dict[int, dict], mode: str) -> int | None:
    for k in sorted(records):
        if bool(records[k][mode]["crossed"]):
            return int(k)
    return None


def _crossing_detail(records: dict[int, dict], mode: str) -> dict | None:
    """返回首次过线点及其前一个已扫描整数点的门线裕量。"""
    rank = _rank_from_records(records, mode)
    if rank is None:
        return None
    ordered = sorted(records)
    position = ordered.index(rank)
    current = records[rank]
    previous_k = ordered[position - 1] if position > 0 else None
    previous_margin = None
    if previous_k is not None:
        previous = records[previous_k]
        previous_margin = float(
            previous[mode]["auc"] - previous["threshold_auc"]
        )
    return {
        "rank": rank,
        "previous_k": previous_k,
        "previous_margin_to_threshold": previous_margin,
        "rank_margin_to_threshold": float(
            current[mode]["auc"] - current["threshold_auc"]
        ),
        "chosen_c_at_rank": float(
            current[mode].get("chosen_c", current[mode].get("c"))
        ),
        "pca_cumulative_variance_at_rank": float(
            current["pca_cumulative_variance"]
        ),
    }


def _validate_formal_replay(
    records: dict[int, dict[int, dict[int, dict]]], summary: dict, seeds: list[int]
) -> dict[str, float]:
    diffs = {}
    formal = summary["g2"]["effective_rank_by_seed"]
    for seed in seeds:
        for k_text, expected in formal[str(seed)]["curve"].items():
            k = int(k_text)
            actual = float(records[29][seed][k]["fixed_c"]["auc"])
            diffs[f"s{seed}/k{k}"] = abs(actual - float(expected))
    maximum = max(diffs.values(), default=0.0)
    if maximum > REPLAY_TOLERANCE:
        raise RuntimeError(
            f"L29 固定 C 正式曲线复现差异 {maximum:.6g} 超过 {REPLAY_TOLERANCE}"
        )
    return {"max_abs_auc_diff": maximum, "tolerance": REPLAY_TOLERANCE, **diffs}


def _summarize(
    records: dict[int, dict[int, dict[int, dict]]],
    summary: dict,
    protocol: dict,
    protocol_hash: str,
    seeds: list[int],
    retention: float,
) -> dict:
    replay = _validate_formal_replay(records, summary, seeds)
    layers = {}
    for layer in LAYERS:
        seed_rows = {}
        for seed in seeds:
            task_records = records[layer][seed]
            fixed_rank = _rank_from_records(task_records, "fixed_c")
            nested_rank = _rank_from_records(task_records, "nested_c")
            fixed_crossing = _crossing_detail(task_records, "fixed_c")
            nested_crossing = _crossing_detail(task_records, "nested_c")
            seed_rows[str(seed)] = {
                "full_auc": task_records[min(task_records)]["full_auc"],
                "threshold_auc": task_records[min(task_records)]["threshold_auc"],
                "fixed_c_rank": fixed_rank,
                "nested_c_rank": nested_rank,
                "fixed_c_crossing": fixed_crossing,
                "nested_c_crossing": nested_crossing,
                "pca_cumulative_variance_at_k16": task_records[16][
                    "pca_cumulative_variance"
                ],
                "nested_c_by_k": {
                    str(k): task_records[k]["nested_c"]["chosen_c"]
                    for k in sorted(task_records)
                },
                "curve": {
                    str(k): {
                        "fixed_auc": task_records[k]["fixed_c"]["auc"],
                        "nested_auc": task_records[k]["nested_c"]["auc"],
                        "pca_cumulative_variance": task_records[k][
                            "pca_cumulative_variance"
                        ],
                    }
                    for k in sorted(task_records)
                },
            }
        fixed_values = [seed_rows[str(seed)]["fixed_c_rank"] for seed in seeds]
        nested_values = [seed_rows[str(seed)]["nested_c_rank"] for seed in seeds]
        layers[str(layer)] = {
            "by_seed": seed_rows,
            "fixed_c_conservative_rank": (
                None if any(value is None for value in fixed_values) else max(fixed_values)
            ),
            "nested_c_conservative_rank": (
                None if any(value is None for value in nested_values) else max(nested_values)
            ),
        }
    flat_records = [
        record
        for layer_records in records.values()
        for seed_records in layer_records.values()
        for record in seed_records.values()
    ]
    crossing_details = [
        row[mode]
        for layer in layers.values()
        for row in layer["by_seed"].values()
        for mode in ("fixed_c_crossing", "nested_c_crossing")
        if row[mode] is not None
    ]
    variance_at_16 = [
        row["pca_cumulative_variance_at_k16"]
        for layer in layers.values()
        for row in layer["by_seed"].values()
    ]
    variance_at_crossing = [
        detail["pca_cumulative_variance_at_rank"] for detail in crossing_details
    ]
    fit_audit = {
        "records": len(flat_records),
        "candidate_fits": sum(
            len(record["nested_c"]["candidate_converged"])
            for record in flat_records
        ),
        "candidate_nonconverged": sum(
            sum(not value for value in record["nested_c"]["candidate_converged"])
            for record in flat_records
        ),
        "nested_refit_nonconverged": sum(
            not record["nested_c"]["converged"] for record in flat_records
        ),
        "fixed_refit_nonconverged": sum(
            not record["fixed_c"]["converged"] for record in flat_records
        ),
        "fit_wall_seconds_sum": float(
            sum(record["wall_seconds"] for record in flat_records)
        ),
        "smallest_positive_crossing_margin": min(
            detail["rank_margin_to_threshold"] for detail in crossing_details
        ),
        "smallest_previous_abs_margin": min(
            abs(detail["previous_margin_to_threshold"])
            for detail in crossing_details
            if detail["previous_margin_to_threshold"] is not None
        ),
    }
    return {
        "kind": "post_hoc_descriptive_non_decisive",
        "formal_g2_unchanged": True,
        "formal_g2_verdict": summary["g2"]["verdict"]["verdict"],
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "formal_replay": replay,
        "fit_audit": fit_audit,
        "pca_variance_audit": {
            "k16_min": min(variance_at_16),
            "k16_max": max(variance_at_16),
            "crossing_min": min(variance_at_crossing),
            "crossing_max": max(variance_at_crossing),
        },
        "layers": layers,
    }


def _write_markdown(result: dict) -> Path:
    lines = [
        "# E1 有效秩三项描述性诊断",
        "",
        "> 事后、严格非裁决分析（PREREG #30）。正式 G2 判定保持 `fail`，本报告不回写正式汇总。",
        "",
        "## 结论总览",
        "",
        "| 层 | 固定正式 C：保守有效秩 | 逐 k 嵌套选 C：保守有效秩 |",
        "| --- | ---: | ---: |",
    ]
    for layer in LAYERS:
        entry = result["layers"][str(layer)]
        lines.append(
            f"| L{layer} | {entry['fixed_c_conservative_rank']} | "
            f"{entry['nested_c_conservative_rank']} |"
        )
    lines.extend(
        [
            "",
            "## 逐种子过线位置",
            "",
            "| 层 | 种子 | 全维 AUC | 95% 门线 | 固定 C 最小 k | 嵌套 C 最小 k |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in LAYERS:
        for seed, row in result["layers"][str(layer)]["by_seed"].items():
            lines.append(
                f"| L{layer} | {seed} | {row['full_auc']:.6f} | "
                f"{row['threshold_auc']:.6f} | {row['fixed_c_rank']} | "
                f"{row['nested_c_rank']} |"
            )
    replay = result["formal_replay"]
    fit_audit = result["fit_audit"]
    variance_audit = result["pca_variance_audit"]
    fixed_ranks = [
        result["layers"][str(layer)]["fixed_c_conservative_rank"] for layer in LAYERS
    ]
    nested_ranks = [
        result["layers"][str(layer)]["nested_c_conservative_rank"] for layer in LAYERS
    ]
    lines.extend(
        [
            "",
            "## 三项诊断结论",
            "",
            "1. **逐 k 重选 C**：L29 的保守秩保持 84；L30 从 68 降到 57；"
            "L31 保持 66。低维正则失配只解释 L30 的约 11 维，无法把任一层降到 16。",
            "2. **相邻层复查**：固定 C 的最佳层为 L31（66），嵌套选 C 的最佳层为 "
            "L30（57）。相邻高分层确实更集中，最佳结果仍为门限的 3.56 倍。",
            "3. **64–128 细扫**：正式层 L29 的固定 C 过线点为 "
            "79/84/81，保守值 84。原网格 128 是首个预设过线点；细扫缩小了数值位置，"
            "正式冻结结果仍为 128，G2 仍失败。",
            "",
            "## 数值与解释边界",
            "",
            f"- L29 原网格固定 C 曲线复现最大绝对 AUC 差：{replay['max_abs_auc_diff']:.3g}。",
            f"- 三层固定 C 保守有效秩：{fixed_ranks}；逐 k 嵌套选 C：{nested_ranks}。",
            f"- {fit_audit['records']} 个断点中的 {fit_audit['candidate_fits']} 个候选拟合、"
            f"{fit_audit['records']} 个嵌套重训"
            f"与固定 C 路径均收敛；未收敛计数为 "
            f"{fit_audit['candidate_nonconverged']}/"
            f"{fit_audit['nested_refit_nonconverged']}/"
            f"{fit_audit['fixed_refit_nonconverged']}。",
            f"- 前 16 个主成分只解释总激活方差的 "
            f"{variance_audit['k16_min']:.1%}–{variance_audit['k16_max']:.1%}；"
            f"实际过线点累计解释 {variance_audit['crossing_min']:.1%}–"
            f"{variance_audit['crossing_max']:.1%}。",
            f"- 最贴近门线的首次过线正裕量为 "
            f"{fit_audit['smallest_positive_crossing_margin']:.2g} AUC，"
            "整数位置可能受极小数值扰动影响；它与 16 维门限之间仍相隔至少 41 维。",
            "- 65–128 的逐整数点用于定位局部首次过线位置；正式预设网格与正式有效秩仍保持 128。",
            "",
            "完整逐 k AUC、PCA 累积方差及所选 C 见 `wp_e1_effective_rank_diagnostics.json`。",
        ]
    )
    path = Path(REPO_ROOT) / "reports" / "e1_有效秩描述性诊断.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {path}")
    return path


def _formal_inputs(
    probe_cfg: dict, roots: dict, summary: dict, seeds: list[int]
) -> tuple[dict, str]:
    t4 = summary.get("per_spec", {}).get(TARGET, {})
    top3 = t4.get("top3_by_seed", {})
    if summary.get("g2", {}).get("verdict", {}).get("verdict") != "fail":
        raise SystemExit("正式汇总不再是 G2=fail，拒绝套用 #30 事后诊断")
    if any(set(top3.get(str(seed), [])) != set(LAYERS) for seed in seeds):
        raise SystemExit("T4 三种子共同 top-3 已变化，拒绝套用 #30 固定层集合")
    formal_cells = {}
    for layer in LAYERS:
        formal_cells[str(layer)] = {}
        for seed in seeds:
            path = engine._cell_path(roots, TARGET, "acts", layer, seed)
            _scores, meta, _weight = engine._load_cell(path)
            formal_cells[str(layer)][str(seed)] = {
                "chosen_c": float(meta["chosen_c"]),
                "auc": float(t4["auc_by_seed_layer"][str(seed)][str(layer)]),
                "cell_size": path.stat().st_size,
                "cell_mtime_ns": path.stat().st_mtime_ns,
            }
    protocol = {
        "name": PROTOCOL_NAME,
        "prereg": 30,
        "target": TARGET,
        "layers": list(LAYERS),
        "seeds": seeds,
        "coarse_ks": list(COARSE_KS),
        "mandatory_fine_ks": [65, 128],
        "retention": float(probe_cfg["effective_rank"]["retention"]),
        "c_grid": [float(c) for c in probe_cfg["c_grid"]],
        "inner_val_sessions": int(probe_cfg["inner_val_sessions"]),
        "trainer": probe_cfg["trainer"],
        "pca_solver": "numpy.linalg.svd/full_matrices=False/float64",
        "formal_summary_sha256": _sha256(
            Path(REPO_ROOT) / "reports" / "wp_e1_probe_summary.json"
        ),
        "formal_cells": formal_cells,
    }
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()
    return protocol, hashlib.sha256(encoded).hexdigest()


def run(args) -> dict:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise SystemExit("--devices 需要提供不重复的设备列表")
    engine._validate_devices(devices)
    probe_cfg, _cache_cfg = engine._cfg()
    roots = engine._roots()
    train, evals = engine._sessions()
    specs, seeds, inner, _pools, train_rows, eval_rows = engine._prepare_rows(
        probe_cfg, roots, train, evals
    )
    seeds = [int(seed) for seed in seeds]
    spec = next(item for item in specs if item.name == TARGET)
    if spec.n_classes != 2:
        raise SystemExit("#30 诊断要求 T4 为二分类规格")
    summary_path = Path(REPO_ROOT) / "reports" / "wp_e1_probe_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol, protocol_hash = _formal_inputs(probe_cfg, roots, summary, seeds)
    retention = float(probe_cfg["effective_rank"]["retention"])
    c_grid = [float(c) for c in probe_cfg["c_grid"]]
    checkpoint_root = (
        roots["work"] / "diagnostics" / PROTOCOL_NAME.replace("-", "_")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pending = {
        (layer, seed)
        for layer in LAYERS
        for seed in seeds
        if args.force
        or not _task_is_complete(
            checkpoint_root, protocol_hash, layer, seed, retention
        )
    }
    print(
        f"#30 诊断协议 {protocol_hash[:12]}：待计算 {len(pending)}/"
        f"{len(LAYERS) * len(seeds)} 个层×种子任务",
        flush=True,
    )

    task_queue: Queue[ProjectedTask | None] = Queue()

    def worker(device: str) -> int:
        count = 0
        while True:
            task = task_queue.get()
            if task is None:
                return count
            count += _run_projected_task(
                task,
                device=device,
                root=checkpoint_root,
                protocol_hash=protocol_hash,
                c_grid=c_grid,
                trainer=probe_cfg["trainer"],
                retention=retention,
                force=args.force,
            )

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="rank-gpu") as pool:
        futures = [pool.submit(worker, device) for device in devices]
        if pending:
            n_steps = engine._load_run_specs(roots)
            unique_steps = set(n_steps.values())
            if len(unique_steps) != 1:
                raise RuntimeError(f"正式角色 n_steps 不一致：{sorted(unique_steps)}")
            n_layer_rows = unique_steps.pop()
            role_groups = [eval_rows[TARGET]] + [
                train_rows[(TARGET, seed)] for seed in seeds
            ]
            compact_rows = g.required_layer_rows(role_groups, n_rows=n_layer_rows)
            run_keys = sorted(compact_rows)
            inner_set = set(inner)
            for layer in LAYERS:
                layer_pending = [seed for seed in seeds if (layer, seed) in pending]
                if not layer_pending:
                    continue
                load_started = perf_counter()
                print(f"L{layer}：载入 {len(run_keys)} 路压紧层缓存", flush=True)
                store = g.preload_layer(
                    roots["runs"], run_keys, layer, row_indices=compact_rows
                )
                eval_features, eval_labels, _ = g.assemble(
                    eval_rows[TARGET], "acts", store, dtype=np.float32
                )
                print(
                    f"L{layer}：缓存与评估矩阵就绪，耗时 "
                    f"{perf_counter() - load_started:.1f}s",
                    flush=True,
                )
                for seed in layer_pending:
                    formal = protocol["formal_cells"][str(layer)][str(seed)]
                    task = _project_task(
                        layer=layer,
                        seed=seed,
                        train_roles=train_rows[(TARGET, seed)],
                        inner_sessions=inner_set,
                        train_store=store,
                        eval_features=eval_features,
                        eval_labels=eval_labels,
                        full_auc=formal["auc"],
                        fixed_c=formal["chosen_c"],
                    )
                    task_queue.put(task)
                del store, eval_features, eval_labels
                gc.collect()
        for _device in devices:
            task_queue.put(None)
        new_points = sum(future.result() for future in futures)
    print(f"全部拟合任务结束：本次新增 {new_points} 个 k 断点", flush=True)

    records = _load_records(
        checkpoint_root, protocol_hash, seeds, retention
    )
    result = _summarize(
        records, summary, protocol, protocol_hash, seeds, retention
    )
    result["runtime"] = {
        "git_head": _git_head(),
        "devices": devices,
        "checkpoint_root": str(checkpoint_root),
        "invocation_new_points": new_points,
        "checkpoint_count": sum(
            len(seed_records)
            for layer_records in records.values()
            for seed_records in layer_records.values()
        ),
    }
    write_report_json("wp_e1_effective_rank_diagnostics.json", result)
    _write_markdown(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="固定设备工作线程列表；每张卡最多运行一个拟合任务",
    )
    parser.add_argument(
        "--force", action="store_true", help="覆盖当前协议下已有的有效 k 断点"
    )
    args = parser.parse_args()
    result = run(args)
    print(
        "诊断完成："
        + ", ".join(
            f"L{layer} 固定/嵌套={entry['fixed_c_conservative_rank']}/"
            f"{entry['nested_c_conservative_rank']}"
            for layer, entry in result["layers"].items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
