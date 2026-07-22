"""WP-E1-2/3：E1 探针全网格引擎（PREREG #18）。

阶段（按序）：
  labels   校验 500 会话标签齐备；缺失时生成精确清单，交给
           `wp_e1_run_missing_events.py --session-list <清单>` 增算，再平铺同步并登记 n_steps。
  parity   torch-GPU 训练器 vs sklearn 奇偶校验门（#18(d)）：T4 与 T1_d400 × 层 20，
           前 20 训练会话。未过则禁止 run。
  run      层主序全网格：单进程共享层缓存，--devices cuda:0,cuda:1 按规格双 GPU
           并行；每层训练侧 800 路与评估侧 200 路分阶段载入。附带层无关基线
           （Mimi / hazard / GRU）。
  finalize 汇总全部格子 → 选层 → ℓ* 处 MLP/打乱标签/有效秩 → G2 判定 →
           bootstrap 优势 → reports/wp_e1_probe_summary.json + e1_探针网格报告.md。

产物根：<data_root>/e1_probe/（cells/*.npz 每格逐会话分数 + 元数据）。
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg
from floor_circuit.e1.g2 import evaluate_g2, pairwise_abs_cosines, top_layers_by_auc
from floor_circuit.e1.sets import e1_sessions
from floor_circuit.mve.alignment import feature_row_indices

FRAME_HZ = 12.5
LEGACY_E1_LABEL_FINGERPRINTS = {
    # 79e3bf2：后续差异仅为 labels.py 文档字符串与 G0 专用模块/配置，不改变 T1–T5 产物。
    (
        "ba5ee88469f9ffc1273ca13a6475d917b5f53895d4d1a51a92e320d4fd006695",
        "9a7bfe1a18b74e5957bcb457d263fb30948cbae1d696f695c19f28f3ae26410a",
    )
}


def _cfg() -> tuple[dict, dict]:
    grids = load_config("grids")
    return grids["e1"]["probe"], grids["e1"]["cache"]


def _validate_devices(devices: list[str]) -> None:
    """在读取大缓存前验证 torch 后端，给 CPU wheel 提供明确诊断。"""
    import torch

    for name in devices:
        device = torch.device(name)
        if device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise SystemExit(
                f"请求 {name}，当前 torch={torch.__version__} 无可用 CUDA。"
                "请先执行 uv sync，确认安装的是项目锁定的 CUDA wheel。"
            )
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise SystemExit(
                f"请求 {name}，但当前只发现 {torch.cuda.device_count()} 张 CUDA 设备"
            )


def _run_devices(args) -> list[str]:
    if not args.devices:
        return [str(args.device)]
    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices 至少需要一个设备")
    if args.num_shards != 1:
        raise SystemExit("单进程多设备模式要求 --num-shards 1")
    if len(set(devices)) != len(devices):
        raise SystemExit("--devices 不得包含重复设备")
    return devices


def _physical_memory_bytes() -> int | None:
    """读取物理内存总量；仅用于在大缓存分配前阻止必然换页的启动方式。"""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _guard_layer_cache_memory(args, cache_cfg, train_rows) -> None:
    physical = _physical_memory_bytes()
    if physical is None:
        return
    train_keys = {
        (role.session_id, role.agent_channel)
        for roles in train_rows.values()
        for role in roles
    }
    bytes_per_process = (
        len(train_keys)
        * 3000
        * int(cache_cfg["expected_hidden_dim"])
        * np.dtype(str(cache_cfg["dtype"])).itemsize
    )
    # 为系统、标签行域、角色临时块与 Python 运行时保留 6 GiB。
    required = bytes_per_process * int(args.num_shards) + 6 * (1 << 30)
    if int(args.num_shards) > 1 and required > physical:
        raise SystemExit(
            f"当前启动方式预计需要至少 {required / (1 << 30):.1f} GiB 物理内存，"
            f"本机仅 {physical / (1 << 30):.1f} GiB。请改用单进程双卡："
            "--num-shards 1 --devices cuda:0,cuda:1"
        )


def _roots() -> dict[str, Path]:
    _, cache_cfg = _cfg()
    base = data_root()
    runs_root = base / "activations" / str(cache_cfg["model"]) / (str(cache_cfg["out_group"]) + "_zarr")
    return {
        "runs": runs_root,
        "work": base / "e1_probe",
        "labels": base / "e1_probe" / "labels",
        "cells": base / "e1_probe" / "cells",
        "events": base / "events" / "candor",
        "audio": base / "candor_extracted",
    }


def _sessions() -> tuple[list[str], list[str]]:
    payload = json.loads((REPO_ROOT / "configs" / "splits" / "candor.json").read_text(encoding="utf-8"))
    sets = e1_sessions(payload)
    return list(sets.train), list(sets.eval)


def _run_specs_path(roots: dict) -> Path:
    return roots["work"] / "run_specs.json"


def _event_labels_path(roots: dict, sid: str) -> Path:
    """事件管线的冻结标签产物命名契约。"""
    return roots["events"] / f"{sid}.labels.parquet"


def _validated_label_record(
    roots: dict, sid: str, accepted_fingerprints: set[tuple[str, str]]
) -> tuple[list | None, str | None]:
    """核对标签文件、完成标记、源音频状态及已登记的代码/设置指纹。"""
    label_path = _event_labels_path(roots, sid)
    marker_path = roots["events"] / f"{sid}.complete.json"
    if not label_path.is_file() or not marker_path.is_file():
        return None, "missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        declared = marker["outputs"]["labels"]
        source_audio = marker["input"]["source_audio"]
        fingerprint = (
            str(marker["input"]["event_pipeline_code_sha256"]),
            str(marker["input"]["settings_sha256"]),
        )
        if (
            marker.get("schema_version") != 1
            or marker.get("session") != sid
            or fingerprint not in accepted_fingerprints
        ):
            return None, "fingerprint"
        if declared.get("name") != label_path.name:
            return None, "marker_name"
        stat = label_path.stat()
        if int(declared.get("size", -1)) != stat.st_size:
            return None, "size"
        digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
        if declared.get("sha256") != digest:
            return None, "sha256"
        for channel in (0, 1):
            audio = roots["audio"] / sid / f"audio_ch{channel}.wav"
            audio_stat = audio.stat()
            expected = source_audio[audio.name]
            if (
                int(expected["size"]) != audio_stat.st_size
                or int(expected["mtime_ns"]) != audio_stat.st_mtime_ns
            ):
                return None, "source_audio"
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None, "invalid_marker"
    return [sid, stat.st_size, stat.st_mtime_ns, *fingerprint], None


def _accepted_label_fingerprints() -> set[tuple[str, str]]:
    """返回当前生产指纹与已登记的 T1–T5 等价历史指纹。"""
    import wp1_run_events as wp1
    from wp_e1_run_missing_events import settings_sha256

    grids = load_config("grids")
    current_code = wp1._event_pipeline_code_sha256()
    _step_s, _deltas, current_settings = settings_sha256(
        load_config("events"), grids, "moshi", "en", current_code
    )
    return {
        (current_code, current_settings),
        *LEGACY_E1_LABEL_FINGERPRINTS,
    }


def _copy_label_if_needed(source: Path, target: Path) -> None:
    """按大小与时间跳过有效副本；需要更新时用同目录临时文件原子替换。"""
    import shutil

    if target.is_file():
        source_stat = source.stat()
        target_stat = target.stat()
        if (
            target_stat.st_size == source_stat.st_size
            and target_stat.st_mtime_ns >= source_stat.st_mtime_ns
        ):
            return
    tmp = target.with_suffix(".tmp.parquet")
    shutil.copy2(source, tmp)
    tmp.replace(target)


def _load_run_specs(roots: dict) -> dict[tuple[str, int], int]:
    payload = json.loads(_run_specs_path(roots).read_text(encoding="utf-8"))
    return {(sid, int(ch)): int(n) for sid, ch, n in payload["roles"]}


def _frozen_window_and_steps() -> tuple[float, int]:
    """从冻结配置推导分析窗与步数，供标签清单尚未齐备的独立阶段使用。"""
    grids = load_config("grids")
    cache_cfg = grids["e1"]["cache"]
    window_s = float(grids["e1"]["windows_s"][str(cache_cfg["model"])])
    n_steps = round(window_s * FRAME_HZ)
    return window_s, n_steps


def _role_steps_or_frozen(
    roots: dict, sessions: list[str]
) -> dict[tuple[str, int], int]:
    """优先读取 labels 阶段清单；缺失时沿冻结窗口生成同值步数。"""
    if _run_specs_path(roots).is_file():
        return _load_run_specs(roots)
    _window_s, n_steps = _frozen_window_and_steps()
    return {(sid, channel): n_steps for sid in sessions for channel in (0, 1)}


def stage_labels(args) -> None:
    roots = _roots()
    train, evals = _sessions()
    sessions = train + evals
    accepted_fingerprints = _accepted_label_fingerprints()
    label_files = []
    invalid = {}
    for sid in sessions:
        record, reason = _validated_label_record(
            roots, sid, accepted_fingerprints
        )
        if record is None:
            invalid[sid] = reason
        else:
            label_files.append(record)
    if invalid:
        missing = list(invalid)
        listing = roots["work"] / "missing_label_sessions.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("\n".join(missing), encoding="utf-8")
        (roots["work"] / "invalid_label_sessions.json").write_text(
            json.dumps(invalid, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        raise SystemExit(
            f"{len(missing)} 个会话缺失或标签指纹无效（清单 {listing}）。先跑：\n"
            f"  uv run python scripts/wp_e1_run_missing_events.py --session-list \"{listing}\"\n"
            "（仅处理清单内会话，已有有效缓存仍会复用），完成后重跑本阶段。"
        )
    roots["labels"].mkdir(parents=True, exist_ok=True)

    for sid in sessions:
        target = roots["labels"] / f"{sid}.parquet"
        source = _event_labels_path(roots, sid)
        _copy_label_if_needed(source, target)
    roles = []
    for sid in sessions:
        for channel in (0, 1):
            manifest = json.loads(
                (roots["runs"] / f"{sid}_agent{channel}" / "manifest.json").read_text(encoding="utf-8")
            )
            if int(manifest["n_steps"]) != 3000:
                raise SystemExit(f"{sid}_agent{channel} n_steps={manifest['n_steps']} ≠ 3000")
            roles.append([sid, channel, int(manifest["n_steps"])])
    roots["work"].mkdir(parents=True, exist_ok=True)
    _run_specs_path(roots).write_text(
        json.dumps(
            {"n_sessions": len(sessions), "roles": roles, "labels": label_files},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report_json(
        "wp_e1_probe_labels.json",
        {"n_sessions": len(sessions), "n_roles": len(roles), "labels_root": str(roots["labels"])},
    )
    print(f"标签阶段就绪：{len(sessions)} 会话 / {len(roles)} 路")


def _cell_path(roots: dict, spec_name: str, feature: str, layer: int | None, seed: int) -> Path:
    tag = "none" if layer is None else str(layer)
    return roots["cells"] / f"{spec_name}__{feature}__L{tag}__s{seed}.npz"


def _fit_path(roots: dict, spec_name: str, layer: int, seed: int) -> Path:
    return roots["work"] / "fits" / f"{spec_name}__acts__L{layer}__s{seed}.npz"


def _save_fit(path: Path, probe: pg.LinearProbe, meta: dict) -> None:
    """训练/评估两阶段之间的小型断点；不含逐会话分数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "__meta__": np.frombuffer(
            json.dumps({**meta, "fit_format": 1}, ensure_ascii=False).encode(),
            dtype=np.uint8,
        ),
        "mean": probe.mean.astype(np.float32),
        "scale": probe.scale.astype(np.float32),
        "weight": probe.weight.astype(np.float32),
        "bias": probe.bias.astype(np.float32),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(path)


def _load_fit(path: Path) -> tuple[pg.LinearProbe, dict]:
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(bytes(payload["__meta__"]).decode())
        if int(meta.pop("fit_format", -1)) != 1:
            raise ValueError(f"未知拟合断点格式：{path}")
        probe = pg.LinearProbe(
            mean=payload["mean"].astype(np.float32),
            scale=payload["scale"].astype(np.float32),
            weight=payload["weight"].astype(np.float32),
            bias=payload["bias"].astype(np.float32),
            n_classes=int(meta["n_classes"]),
            c_value=float(meta["chosen_c"]),
            converged=bool(meta["converged"]),
        )
    return probe, meta


def _save_cell(path: Path, scores: dict, meta: dict, weight: np.ndarray | None) -> None:
    """原子写入格子；保留未压缩数组，避免对概率流执行低收益单线程压缩。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_bytes = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode(), dtype=np.uint8)
    arrays: dict[str, np.ndarray] = {"__meta__": meta_bytes}
    if weight is not None:
        arrays["__weight__"] = weight.astype(np.float32)
    for sid, (y, p) in scores.items():
        arrays[f"y::{sid}"] = np.asarray(y, dtype=np.int16)
        arrays[f"p::{sid}"] = np.asarray(p, dtype=np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _load_cell(path: Path) -> tuple[dict, dict, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(bytes(payload["__meta__"]).decode())
        weight = payload["__weight__"] if "__weight__" in payload.files else None
        scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in payload.files:
            if key.startswith("y::"):
                sid = key[3:]
                scores[sid] = (payload[key].astype(np.int64), payload[f"p::{sid}"].astype(np.float64))
    return scores, meta, weight


def _row_plan_path(roots: dict) -> Path:
    return roots["work"] / "row_plan_v1.npz"


def _row_plan_signature(
    probe_cfg: dict, roots: dict, train: list[str], evals: list[str]
) -> str:
    digest = hashlib.sha256()
    digest.update(_run_specs_path(roots).read_bytes())
    digest.update(
        json.dumps(
            {"version": 1, "probe_cfg": probe_cfg, "train": train, "eval": evals},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def _pack_role_rows(arrays: dict[str, np.ndarray], prefix: str, roles) -> None:
    offsets = np.zeros(len(roles) + 1, dtype=np.int64)
    if roles:
        offsets[1:] = np.cumsum([len(role.labels) for role in roles])
    arrays[f"{prefix}__sessions"] = np.asarray(
        [role.session_id for role in roles], dtype=str
    )
    arrays[f"{prefix}__channels"] = np.asarray(
        [role.agent_channel for role in roles], dtype=np.int8
    )
    arrays[f"{prefix}__offsets"] = offsets
    arrays[f"{prefix}__steps"] = (
        np.concatenate([role.steps for role in roles]).astype(np.int32)
        if roles
        else np.empty(0, dtype=np.int32)
    )
    arrays[f"{prefix}__labels"] = (
        np.concatenate([role.labels for role in roles]).astype(np.int8)
        if roles
        else np.empty(0, dtype=np.int8)
    )


def _unpack_role_rows(payload, prefix: str) -> list[g.RoleRows]:
    sessions = payload[f"{prefix}__sessions"]
    channels = payload[f"{prefix}__channels"]
    offsets = payload[f"{prefix}__offsets"]
    steps = payload[f"{prefix}__steps"]
    labels = payload[f"{prefix}__labels"]
    return [
        g.RoleRows(
            str(sessions[index]),
            int(channels[index]),
            steps[int(offsets[index]) : int(offsets[index + 1])].astype(np.int64),
            labels[int(offsets[index]) : int(offsets[index + 1])].astype(np.int64),
        )
        for index in range(len(sessions))
    ]


def _save_row_plan(path, signature, specs, seeds, train_rows, eval_rows) -> None:
    arrays: dict[str, np.ndarray] = {
        "__meta__": np.frombuffer(
            json.dumps(
                {"version": 1, "signature": signature}, ensure_ascii=False
            ).encode(),
            dtype=np.uint8,
        )
    }
    for spec in specs:
        _pack_role_rows(arrays, f"eval__{spec.name}", eval_rows[spec.name])
        for seed in seeds:
            _pack_role_rows(
                arrays,
                f"train__{spec.name}__s{seed}",
                train_rows[(spec.name, seed)],
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _load_row_plan(path, signature, specs, seeds):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(bytes(payload["__meta__"]).decode())
            if meta != {"version": 1, "signature": signature}:
                return None
            train_rows = {
                (spec.name, seed): _unpack_role_rows(
                    payload, f"train__{spec.name}__s{seed}"
                )
                for spec in specs
                for seed in seeds
            }
            eval_rows = {
                spec.name: _unpack_role_rows(payload, f"eval__{spec.name}")
                for spec in specs
            }
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return train_rows, eval_rows


def _prepare_rows(probe_cfg: dict, roots: dict, train: list[str], evals: list[str]):
    """预构建每 (spec, seed) 的训练行与每 spec 的评估行（层无关，仅标签域）。"""
    specs = g.expand_specs(probe_cfg)
    n_steps = _load_run_specs(roots)
    seeds = [int(s) for s in probe_cfg["seeds"]]
    inner = train[: int(probe_cfg["inner_val_sessions"])]
    train_rows: dict[tuple[str, int], list[g.RoleRows]] = {}
    pool_by_seed: dict[int, list[str]] = {}
    for seed in seeds:
        pool_by_seed[seed] = g.seed_train_sessions(train, probe_cfg, seed)
    signature = _row_plan_signature(probe_cfg, roots, train, evals)
    cached = _load_row_plan(_row_plan_path(roots), signature, specs, seeds)
    if cached is not None:
        train_rows, eval_rows = cached
        print(f"复用标签行域缓存：{_row_plan_path(roots)}")
        return specs, seeds, inner, pool_by_seed, train_rows, eval_rows
    raw_train = g.build_rows_multi(
        roots["labels"], train, n_steps, specs, probe_cfg
    )
    for seed in seeds:
        pool = pool_by_seed[seed]
        pool_set = set(pool)
        for spec in specs:
            selected = [
                role for role in raw_train[spec.name] if role.session_id in pool_set
            ]
            train_rows[(spec.name, seed)] = g.sample_role_rows(
                selected, spec, probe_cfg, seed, downsample=True
            )
    eval_rows = g.build_rows_multi(
        roots["labels"], evals, n_steps, specs, probe_cfg
    )
    _save_row_plan(
        _row_plan_path(roots), signature, specs, seeds, train_rows, eval_rows
    )
    print(f"写入标签行域缓存：{_row_plan_path(roots)}")
    return specs, seeds, inner, pool_by_seed, train_rows, eval_rows


def _fit_cell(
    spec, probe_cfg, train_roles, inner_sessions, feature, store, device
) -> tuple[object, dict]:
    """C 路径（inner_val 选择）→ 整池正式重训；返回 (探针, 元数据)。"""
    trainer = probe_cfg["trainer"]
    inner_set = set(inner_sessions)
    c_roles = [r for r in train_roles if r.session_id not in inner_set]
    inner_roles = [r for r in train_roles if r.session_id in inner_set]
    n_c, n_dim = g.feature_layout(c_roles, feature, store)
    prepared = pg.prepare_linear_probe_blocks(
        ((block, role.labels) for role, block in g.feature_blocks(c_roles, feature, store)),
        n_c,
        n_dim,
        spec.n_classes,
        device=device,
    )
    candidates = []
    warm = None
    for c_value in [float(c) for c in probe_cfg["c_grid"]]:
        probe = prepared.fit(
            c_value,
            max_iter=int(trainer["lbfgs_max_iter"]),
            tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
            init=warm,
        )
        warm = probe
        candidates.append(probe)
    inner_scores = g.eval_cell_scores_many(
        candidates, inner_roles, feature, store, device=device
    )
    best = None
    curve = {}
    for c_value, scores in zip(probe_cfg["c_grid"], inner_scores, strict=True):
        metric = g.pooled_primary_metric(scores, spec.n_classes)
        curve[str(c_value)] = metric
        if best is None or metric > best[1]:
            best = (float(c_value), metric)
    del prepared, candidates, inner_scores
    n_full, n_dim_full = g.feature_layout(train_roles, feature, store)
    prepared_full = pg.prepare_linear_probe_blocks(
        ((block, role.labels) for role, block in g.feature_blocks(train_roles, feature, store)),
        n_full,
        n_dim_full,
        spec.n_classes,
        device=device,
    )
    final = prepared_full.fit(
        best[0],
        max_iter=int(trainer["lbfgs_max_iter"]),
        tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
    )
    del prepared_full
    meta = {
        "chosen_c": best[0],
        "inner_val_curve": curve,
        "inner_val_metric": best[1],
        "n_train_rows": n_full,
        "converged": final.converged,
    }
    return final, meta


def _prepare_parity_inputs(
    roots: dict, sessions: list[str]
) -> dict[tuple[str, int], int]:
    """让 parity 只依赖冻结的前 20 会话，可与其余 Zarr 摄取和标签增算重叠。"""
    if _run_specs_path(roots).is_file() and all(
        (roots["labels"] / f"{sid}.parquet").is_file() for sid in sessions
    ):
        return _load_run_specs(roots)

    accepted = _accepted_label_fingerprints()
    roots["labels"].mkdir(parents=True, exist_ok=True)
    invalid_labels = []
    for sid in sessions:
        record, reason = _validated_label_record(roots, sid, accepted)
        if record is None:
            invalid_labels.append(f"{sid}:{reason}")
            continue
        source = _event_labels_path(roots, sid)
        target = roots["labels"] / f"{sid}.parquet"
        _copy_label_if_needed(source, target)
    if invalid_labels:
        raise SystemExit(
            "parity 前 20 会话存在无效标签：" + ", ".join(invalid_labels[:5])
        )

    _window_s, expected_steps = _frozen_window_and_steps()
    missing_roles = []
    for sid in sessions:
        for channel in (0, 1):
            manifest_path = roots["runs"] / f"{sid}_agent{channel}" / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if int(manifest["n_steps"]) != expected_steps:
                    raise ValueError("n_steps 不符")
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                missing_roles.append(f"{sid}_agent{channel}")
    if missing_roles:
        raise SystemExit(
            f"parity 尚缺 {len(missing_roles)} 路完整 Zarr：{missing_roles[:5]}"
        )
    return {
        (sid, channel): expected_steps
        for sid in sessions
        for channel in (0, 1)
    }


def stage_parity(args) -> None:
    _validate_devices([str(args.device)])
    probe_cfg, _ = _cfg()
    roots = _roots()
    train, _evals = _sessions()
    layer = 20
    sessions = train[:20]
    n_steps = _prepare_parity_inputs(roots, sessions)
    store = g.preload_layer(roots["runs"], [(sid, ch) for sid in sessions for ch in (0, 1)], layer)
    trainer = probe_cfg["trainer"]
    results = {}
    for spec in g.expand_specs(probe_cfg):
        if spec.name not in ("T4", "T1_d400"):
            continue
        roles = g.build_rows(roots["labels"], sessions, n_steps, spec, probe_cfg, 0, downsample=True)
        split = max(1, int(len(sessions) * 0.7))
        fit_set = set(sessions[:split])
        fit_roles = [r for r in roles if r.session_id in fit_set]
        val_roles = [r for r in roles if r.session_id not in fit_set]
        n_fit, n_dim = g.feature_layout(fit_roles, "acts", store)
        prepared = pg.prepare_linear_probe_blocks(
            (
                (block, role.labels)
                for role, block in g.feature_blocks(fit_roles, "acts", store)
            ),
            n_fit,
            n_dim,
            2,
            device=args.device,
        )
        torch_probe = prepared.fit(0.001)
        del prepared
        torch_scores = g.eval_cell_scores_many(
            [torch_probe], val_roles, "acts", store, device=args.device
        )[0]
        auc_torch = g.pooled_primary_metric(torch_scores, 2)
        x_fit, y_fit, _ = g.assemble(fit_roles, "acts", store)
        x_val, y_val, _ = g.assemble(val_roles, "acts", store)
        x_fit = np.asarray(x_fit, dtype=np.float32)
        x_val = np.asarray(x_val, dtype=np.float32)
        weight_ref, prob_ref = pg.sklearn_reference_fit(x_fit, y_fit, 0.001, seed=0)
        auc_ref = pg.primary_metric(y_val, prob_ref(x_val), 2)
        direction = torch_probe.direction()
        ref_dir = weight_ref / np.linalg.norm(weight_ref)
        # torch 权重作用于标准化域，与 sklearn 同域，可直接比方向
        cosine = float(abs(np.dot(direction, ref_dir)))
        results[spec.name] = {
            "auc_torch": auc_torch,
            "auc_sklearn": auc_ref,
            "abs_auc_diff": abs(auc_torch - auc_ref),
            "direction_abs_cos": cosine,
        }
    ok = all(
        r["abs_auc_diff"] <= float(trainer["parity_max_auc_diff"])
        and r["direction_abs_cos"] >= float(trainer["parity_min_direction_cos"])
        for r in results.values()
    )
    report = {"verdict": "passed" if ok else "failed", "layer": layer, "cells": results}
    write_report_json("wp_e1_probe_parity.json", report)
    (roots["work"] / "parity_ok").write_text("passed" if ok else "failed", encoding="utf-8")
    print(f"奇偶校验门 {report['verdict']}：{json.dumps(results, ensure_ascii=False)[:400]}")
    if not ok:
        raise SystemExit(1)


def _assign_spec_groups(groups, train_rows, devices):
    """按估计乘加量做贪心均衡；每张卡由一个固定线程独占。"""
    assignments = [[] for _ in devices]
    loads = [0 for _ in devices]
    weighted = []
    for spec, pending_seeds in groups:
        rows = sum(
            sum(len(role.labels) for role in train_rows[(spec.name, seed)])
            for seed in pending_seeds
        )
        weighted.append((rows * spec.n_classes, spec, pending_seeds))
    for cost, spec, pending_seeds in sorted(weighted, key=lambda item: item[0], reverse=True):
        target = min(range(len(devices)), key=loads.__getitem__)
        assignments[target].append((spec, pending_seeds))
        loads[target] += cost
    return assignments


def _layer_pass(
    layer,
    specs,
    seeds,
    inner,
    train_rows,
    eval_rows,
    probe_cfg,
    roots,
    devices,
    force,
):
    pending = [
        (spec, seed)
        for spec in specs
        for seed in seeds
        if force or not _cell_path(roots, spec.name, "acts", layer, seed).is_file()
    ]
    if not pending:
        print(f"层 {layer}：全部格子已存在，跳过")
        return
    pending_by_spec = [
        (spec, [seed for candidate, seed in pending if candidate.name == spec.name])
        for spec in specs
    ]
    pending_by_spec = [(spec, values) for spec, values in pending_by_spec if values]
    assignments = _assign_spec_groups(pending_by_spec, train_rows, devices)
    cached_fits = {}
    if not force:
        for spec, pending_seeds in pending_by_spec:
            for seed in pending_seeds:
                path = _fit_path(roots, spec.name, layer, seed)
                if path.is_file():
                    cached_fits[(spec.name, seed)] = _load_fit(path)
    train_keys = sorted(
        {
            (role.session_id, role.agent_channel)
            for spec, pending_seeds in pending_by_spec
            for seed in pending_seeds
            if (spec.name, seed) not in cached_fits
            for role in train_rows[(spec.name, seed)]
        }
    )
    if train_keys:
        print(f"层 {layer}：载入训练侧 {len(train_keys)} 路")
        train_store = g.preload_layer(roots["runs"], train_keys, layer)
    else:
        print(f"层 {layer}：全部复用拟合断点，跳过训练侧层缓存")
        train_store = {}

    def fit_groups(device, assigned):
        fitted = []
        for spec, pending_seeds in assigned:
            for seed in pending_seeds:
                fit_path = _fit_path(roots, spec.name, layer, seed)
                if (spec.name, seed) in cached_fits:
                    probe, meta = cached_fits[(spec.name, seed)]
                    print(f"L{layer} {spec.name} s{seed}：复用拟合断点")
                else:
                    probe, meta = _fit_cell(
                        spec,
                        probe_cfg,
                        train_rows[(spec.name, seed)],
                        inner,
                        "acts",
                        train_store,
                        device,
                    )
                    meta["n_classes"] = spec.n_classes
                    _save_fit(fit_path, probe, meta)
                fitted.append((spec, seed, probe, meta, device))
                print(
                    f"L{layer} {spec.name} s{seed} 拟合："
                    f"inner={meta['inner_val_metric']:.4f} C={meta['chosen_c']}"
                )
        return fitted

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="e1-gpu") as pool:
        futures = [
            pool.submit(fit_groups, device, assigned)
            for device, assigned in zip(devices, assignments, strict=True)
            if assigned
        ]
        fitted = [item for future in futures for item in future.result()]
    train_store.clear()
    gc.collect()

    eval_keys = sorted(
        {
            (role.session_id, role.agent_channel)
            for spec, _pending_seeds in pending_by_spec
            for role in eval_rows[spec.name]
        }
    )
    print(f"层 {layer}：释放训练缓存，载入评估侧 {len(eval_keys)} 路")
    eval_store = g.preload_layer(roots["runs"], eval_keys, layer)
    fitted_by_spec = {
        spec.name: [item for item in fitted if item[0].name == spec.name]
        for spec, _ in pending_by_spec
    }

    def evaluate_groups(device, assigned):
        for spec, _pending_seeds in assigned:
            items = fitted_by_spec[spec.name]
            probes = [item[2] for item in items]
            score_sets = g.eval_cell_scores_many(
                probes, eval_rows[spec.name], "acts", eval_store, device=device
            )
            for (item_spec, seed, probe, meta, _), scores in zip(
                items, score_sets, strict=True
            ):
                meta.update(
                    {
                        "spec": item_spec.name,
                        "feature": "acts",
                        "layer": layer,
                        "seed": seed,
                    }
                )
                weight = probe.direction() if item_spec.n_classes == 2 else None
                _save_cell(
                    _cell_path(roots, item_spec.name, "acts", layer, seed),
                    scores,
                    meta,
                    weight,
                )
                print(f"L{layer} {item_spec.name} s{seed}：评估与断点写入完成")

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="e1-eval") as pool:
        futures = [
            pool.submit(evaluate_groups, device, assigned)
            for device, assigned in zip(devices, assignments, strict=True)
            if assigned
        ]
        for future in futures:
            future.result()
    eval_store.clear()
    fitted.clear()
    gc.collect()


def _baseline_pass(specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, devices, force):
    """层无关基线：Mimi（全规格）、hazard（T1/T2/T4）、GRU（二分类规格）。"""
    keys = sorted(
        {(r.session_id, r.agent_channel) for rows in train_rows.values() for r in rows}
        | {(r.session_id, r.agent_channel) for rows in eval_rows.values() for r in rows}
    )
    mimi = g.preload_mimi(roots["runs"], keys)
    groups = [
        (
            spec,
            [
                seed
                for seed in seeds
                if force or not _cell_path(roots, spec.name, "mimi", None, seed).is_file()
            ],
        )
        for spec in specs
    ]
    groups = [(spec, pending_seeds) for spec, pending_seeds in groups if pending_seeds]
    assignments = _assign_spec_groups(groups, train_rows, devices)

    def run_groups(device, assigned):
        for spec, pending_seeds in assigned:
            fitted = [
                _fit_cell(
                    spec,
                    probe_cfg,
                    train_rows[(spec.name, seed)],
                    inner,
                    "mimi",
                    mimi,
                    device,
                )
                for seed in pending_seeds
            ]
            score_sets = g.eval_cell_scores_many(
                [probe for probe, _meta in fitted],
                eval_rows[spec.name],
                "mimi",
                mimi,
                device=device,
            )
            for seed, (_probe, meta), scores in zip(
                pending_seeds, fitted, score_sets, strict=True
            ):
                meta.update(
                    {"spec": spec.name, "feature": "mimi", "layer": None, "seed": seed}
                )
                _save_cell(
                    _cell_path(roots, spec.name, "mimi", None, seed),
                    scores,
                    meta,
                    None,
                )
                print(f"mimi {spec.name} s{seed}：inner={meta['inner_val_metric']:.4f}")

    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="e1-mimi-gpu") as pool:
        futures = [
            pool.submit(run_groups, device, assigned)
            for device, assigned in zip(devices, assignments, strict=True)
            if assigned
        ]
        for future in futures:
            future.result()
    mimi.clear()
    gc.collect()
    _hazard_and_gru(specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, force)


def _hazard_and_gru(specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, force):
    from floor_circuit.probes.baselines import fit_hazard, hazard_features
    from floor_circuit.probes.gru import make_windows, train_eval_gru

    n_steps = _load_run_specs(roots)
    step_s = 1.0 / FRAME_HZ
    hazard_store: dict[tuple[str, int], np.ndarray] = {}
    acoustic_dir = roots["work"] / "acoustic"

    def hazard_of(key: tuple[str, int]) -> np.ndarray:
        if key not in hazard_store:
            frame = pd.read_parquet(roots["labels"] / f"{key[0]}.parquet")
            states = g.t5_state_array(frame, key[1], n_steps[key])
            hazard_store[key] = hazard_features(states, step_s).astype(np.float32)
        return hazard_store[key]

    def acoustic_of(key: tuple[str, int]) -> np.ndarray:
        path = acoustic_dir / f"{key[0]}_ch{key[1]}.npy"
        if not path.is_file():
            raise SystemExit(
                f"缺声学特征缓存 {path}；先跑 --stage acoustic（CPU，可并行）"
            )
        return np.load(path, allow_pickle=False)

    def rows_to_xy(roles, provider, windowed: bool):
        per_session: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role in roles:
            feats = provider((role.session_id, role.agent_channel))
            rows = feature_row_indices("hazard", role.steps)
            x = make_windows(feats, rows) if windowed else feats[rows]
            y = role.labels
            if role.session_id in per_session:
                x0, y0 = per_session[role.session_id]
                per_session[role.session_id] = (np.concatenate([x0, x]), np.concatenate([y0, y]))
            else:
                per_session[role.session_id] = (x, y)
        return per_session

    for spec in specs:
        for seed in seeds:
            hazard_path = _cell_path(roots, spec.name, "hazard", None, seed)
            if spec.target in ("T1", "T2", "T4") and (force or not hazard_path.is_file()):
                train_map = rows_to_xy(train_rows[(spec.name, seed)], hazard_of, windowed=False)
                eval_map = rows_to_xy(eval_rows[spec.name], hazard_of, windowed=False)
                x_tr = np.concatenate([x for x, _ in train_map.values()])
                y_tr = np.concatenate([y for _, y in train_map.values()])
                model = fit_hazard(x_tr, y_tr, seed=seed)
                scores = {
                    sid: (y, model.predict_proba(x)) for sid, (x, y) in eval_map.items()
                }
                _save_cell(
                    hazard_path,
                    scores,
                    {"spec": spec.name, "feature": "hazard", "layer": None, "seed": seed},
                    None,
                )
                print(f"hazard {spec.name} s{seed} 完成")
            gru_path = _cell_path(roots, spec.name, "gru", None, seed)
            if spec.n_classes == 2 and (force or not gru_path.is_file()):
                gru_cfg = probe_cfg["gru"]
                inner_set = set(inner)
                train_map = rows_to_xy(
                    [r for r in train_rows[(spec.name, seed)] if r.session_id not in inner_set],
                    acoustic_of,
                    windowed=True,
                )
                val_map = rows_to_xy(
                    [r for r in train_rows[(spec.name, seed)] if r.session_id in inner_set],
                    acoustic_of,
                    windowed=True,
                )
                eval_map = rows_to_xy(eval_rows[spec.name], acoustic_of, windowed=True)
                result = train_eval_gru(
                    list(train_map.values()),
                    list(val_map.values()),
                    eval_map,
                    seed=seed,
                    hidden=int(gru_cfg["hidden"]),
                    max_epochs=int(gru_cfg["max_epochs"]),
                    batch_size=int(gru_cfg["batch_size"]),
                    lr=float(gru_cfg["lr"]),
                    patience=int(gru_cfg["patience"]),
                )
                scores = {
                    sid: (y, np.stack([1.0 - p, p], axis=1)) for sid, (y, p) in result.items()
                }
                _save_cell(
                    gru_path,
                    scores,
                    {"spec": spec.name, "feature": "gru", "layer": None, "seed": seed},
                    None,
                )
                print(f"gru {spec.name} s{seed} 完成")


def _valid_acoustic_output(path: Path, n_steps: int) -> bool:
    """只读取 NPY 头验证断点，损坏或形状不符时重新计算。"""
    if not path.is_file():
        return False
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        return values.shape == (n_steps, 4) and values.dtype == np.float32
    except (OSError, ValueError):
        return False


def _extract_acoustic_role(task: tuple[str, str, int, float]) -> dict:
    """有界读取单路前缀并原子写入声学特征；函数保持顶层以支持 Windows 多进程。"""
    import time

    import soundfile as sf
    from threadpoolctl import threadpool_limits

    from floor_circuit.probes.baselines import acoustic_frames

    audio_s, output_s, n_steps, window_s = task
    audio_path = Path(audio_s)
    output_path = Path(output_s)
    started = time.perf_counter()
    with sf.SoundFile(str(audio_path)) as source:
        if int(source.channels) != 1:
            raise ValueError(f"声学输入必须为单通道：{audio_path}")
        sample_rate = int(source.samplerate)
        wav = source.read(
            frames=int(window_s * sample_rate),
            dtype="float32",
            always_2d=False,
        )
    read_s = time.perf_counter() - started
    feature_started = time.perf_counter()
    with threadpool_limits(limits=1):
        feats = acoustic_frames(wav, sample_rate)[:n_steps]
    if len(feats) < n_steps:
        feats = np.pad(feats, ((0, n_steps - len(feats)), (0, 0)))
    values = feats.astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.npy")
    np.save(tmp, values, allow_pickle=False)
    tmp.replace(output_path)
    return {
        "output": output_s,
        "read_s": read_s,
        "feature_s": time.perf_counter() - feature_started,
    }


def _pending_acoustic_tasks(
    tasks: list[tuple[str, str, int, float]],
    num_shards: int,
    shard_id: int,
    *,
    force: bool,
) -> list[tuple[str, str, int, float]]:
    """先按完整任务表固定分片，再过滤断点，避免错峰启动导致分片漂移。"""
    sharded = tasks[shard_id::num_shards]
    return [
        task
        for task in sharded
        if force or not _valid_acoustic_output(Path(task[1]), int(task[2]))
    ]


def stage_acoustic(args) -> None:
    """预计算双通道声学帧特征；支持有界读取、原子断点和单进程受控工作池。"""
    roots = _roots()
    train, evals = _sessions()
    sessions = train + evals
    out_dir = roots["work"] / "acoustic"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = _role_steps_or_frozen(roots, sessions)
    window_s, _expected_steps = _frozen_window_and_steps()
    all_tasks = [
        (
            str(roots["audio"] / sid / f"audio_ch{channel}.wav"),
            str(out_dir / f"{sid}_ch{channel}.npy"),
            n_steps[(sid, channel)],
            window_s,
        )
        for sid in sessions
        for channel in (0, 1)
    ]
    tasks = _pending_acoustic_tasks(
        all_tasks,
        int(args.num_shards),
        int(args.shard_id),
        force=bool(args.force),
    )
    if args.acoustic_limit is not None:
        tasks = tasks[: int(args.acoustic_limit)]
    workers = min(int(args.acoustic_workers), max(1, len(tasks)))
    results = []
    if workers == 1:
        iterator = map(_extract_acoustic_role, tasks)
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index % 50 == 0:
                print(f"声学特征 {index}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            for index, result in enumerate(
                pool.map(_extract_acoustic_role, tasks, chunksize=1), start=1
            ):
                results.append(result)
                if index % 50 == 0:
                    print(f"声学特征 {index}/{len(tasks)}")
    read_s = sum(float(result["read_s"]) for result in results)
    feature_s = sum(float(result["feature_s"]) for result in results)
    print(
        f"声学特征完成 {len(tasks)} 路（shard {args.shard_id}/{args.num_shards}，"
        f"workers={workers}，累计读取 {read_s:.1f}s，特征 {feature_s:.1f}s）"
    )


def stage_run(args) -> None:
    probe_cfg, cache_cfg = _cfg()
    roots = _roots()
    devices = _run_devices(args)
    _validate_devices(devices)
    if (roots["work"] / "parity_ok").read_text(encoding="utf-8").strip() != "passed":
        raise SystemExit("奇偶校验门未通过，禁止正式网格（PREREG #18(d)）")
    train, evals = _sessions()
    specs, seeds, inner, _pools, train_rows, eval_rows = _prepare_rows(probe_cfg, roots, train, evals)
    _guard_layer_cache_memory(args, cache_cfg, train_rows)
    layers = list(range(int(cache_cfg["n_layers"])))[args.shard_id :: args.num_shards]
    print(
        f"shard {args.shard_id}/{args.num_shards}：层 {layers}；"
        f"设备 {devices}（单进程共享层缓存）"
    )
    for layer in layers:
        _layer_pass(
            layer,
            specs,
            seeds,
            inner,
            train_rows,
            eval_rows,
            probe_cfg,
            roots,
            devices,
            args.force,
        )
    if args.shard_id == 0:
        _baseline_pass(
            specs,
            seeds,
            inner,
            train_rows,
            eval_rows,
            probe_cfg,
            roots,
            devices,
            args.force,
        )
    print("run 阶段完成")


def _cluster_auc_plan(cell, sids, n_classes):
    """把逐行 AUC 压缩成会话×会话的正负样本有序对充分统计量。"""
    classes = [1] if n_classes == 2 else list(range(n_classes))
    plans = []
    for cls in classes:
        positives = []
        negatives = []
        for sid in sids:
            y, probs = cell[sid]
            scores = np.asarray(probs)[:, cls]
            mask = np.asarray(y) == cls
            positives.append(scores[mask])
            negatives.append(np.sort(scores[~mask], kind="mergesort"))
        pos_counts = np.asarray([len(values) for values in positives], dtype=np.float64)
        neg_counts = np.asarray([len(values) for values in negatives], dtype=np.float64)
        pair_scores = np.zeros((len(sids), len(sids)), dtype=np.float64)
        for pos_index, pos_scores in enumerate(positives):
            if not len(pos_scores):
                continue
            for neg_index, neg_scores in enumerate(negatives):
                if not len(neg_scores):
                    continue
                lower = np.searchsorted(neg_scores, pos_scores, side="left")
                upper = np.searchsorted(neg_scores, pos_scores, side="right")
                pair_scores[pos_index, neg_index] = float(
                    np.sum(lower + 0.5 * (upper - lower), dtype=np.float64)
                )
        plans.append((pos_counts, neg_counts, pair_scores))
    return plans


def _cluster_auc_values(plan, counts):
    """对一批会话重采样计数同时计算 binary 或 macro-OVR AUC。"""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts[None, :]
    per_class = []
    for pos_counts, neg_counts, pair_scores in plan:
        n_pos = counts @ pos_counts
        n_neg = counts @ neg_counts
        numerator = np.einsum(
            "bi,ij,bj->b", counts, pair_scores, counts, optimize=True
        )
        values = np.full(len(counts), np.nan, dtype=np.float64)
        valid = (n_pos > 0) & (n_neg > 0)
        values[valid] = numerator[valid] / (n_pos[valid] * n_neg[valid])
        per_class.append(values)
    matrix = np.stack(per_class, axis=1)
    valid_counts = np.sum(np.isfinite(matrix), axis=1)
    totals = np.nansum(matrix, axis=1)
    return np.divide(
        totals,
        valid_counts,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )


def _bootstrap_adv(probe_cells, base_cells, n_classes, n_boot, seed=20260717):
    """会话重采样 × 种子均值；排序只做一次，重采样仅计算 100×100 二次型。"""
    sids = sorted(set.intersection(*[set(c.keys()) for c in probe_cells + base_cells]))
    if not sids:
        raise ValueError("探针与基线没有共同评估会话")
    probe_plans = [_cluster_auc_plan(cell, sids, n_classes) for cell in probe_cells]
    base_plans = [_cluster_auc_plan(cell, sids, n_classes) for cell in base_cells]
    point_counts = np.ones(len(sids), dtype=np.float64)
    point = float(
        np.mean([_cluster_auc_values(plan, point_counts)[0] for plan in probe_plans])
        - np.mean([_cluster_auc_values(plan, point_counts)[0] for plan in base_plans])
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sids), size=(n_boot, len(sids)))
    counts = np.stack(
        [np.bincount(row, minlength=len(sids)) for row in draws]
    ).astype(np.float64)
    probe_values = np.stack(
        [_cluster_auc_values(plan, counts) for plan in probe_plans]
    )
    base_values = np.stack(
        [_cluster_auc_values(plan, counts) for plan in base_plans]
    )
    valid = np.isfinite(probe_values).all(axis=0) & np.isfinite(base_values).all(axis=0)
    samples = probe_values[:, valid].mean(axis=0) - base_values[:, valid].mean(axis=0)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "advantage": point,
        "ci95": [float(lo), float(hi)],
        "n_boot_effective": len(samples),
    }


def stage_finalize(args) -> None:
    _validate_devices([str(args.device)])
    probe_cfg, cache_cfg = _cfg()
    roots = _roots()
    train, evals = _sessions()
    specs, seeds, inner, _pools, train_rows, eval_rows = _prepare_rows(probe_cfg, roots, train, evals)
    n_layers = int(cache_cfg["n_layers"])
    summary: dict = {"per_spec": {}, "prereg": "#18/#19"}
    primary = str(probe_cfg["g2_primary_target"])
    for spec in specs:
        auc = {}
        for seed in seeds:
            per_layer = {}
            for layer in range(n_layers):
                scores, _meta, _ = _load_cell(_cell_path(roots, spec.name, "acts", layer, seed))
                per_layer[layer] = g.pooled_primary_metric(scores, spec.n_classes)
            auc[seed] = per_layer
        top3 = {seed: top_layers_by_auc(auc[seed], 3) for seed in seeds}
        counts: dict[int, int] = {}
        for layers_ in top3.values():
            for layer in layers_:
                counts[layer] = counts.get(layer, 0) + 1
        shared = [layer for layer, n in counts.items() if n == len(seeds)]
        best_layer = (
            min(shared)
            if shared
            else min(max(auc[s], key=auc[s].get) for s in seeds)
        )
        baselines = {}
        for feature in ("mimi", "hazard", "gru"):
            cells = []
            for seed in seeds:
                path = _cell_path(roots, spec.name, feature, None, seed)
                if path.is_file():
                    cells.append(_load_cell(path)[0])
            if cells:
                sids = sorted(set.intersection(*[set(c.keys()) for c in cells]))
                pooled = float(
                    np.mean(
                        [
                            pg.primary_metric(
                                np.concatenate([c[sid][0] for sid in sids]),
                                np.concatenate([c[sid][1] for sid in sids]),
                                spec.n_classes,
                            )
                            for c in cells
                        ]
                    )
                )
                baselines[feature] = pooled
        probe_cells = [
            _load_cell(_cell_path(roots, spec.name, "acts", best_layer, seed))[0] for seed in seeds
        ]
        strongest = max(baselines, key=baselines.get) if baselines else None
        adv = None
        if strongest:
            base_cells = [
                _load_cell(_cell_path(roots, spec.name, strongest, None, seed))[0]
                for seed in seeds
                if _cell_path(roots, spec.name, strongest, None, seed).is_file()
            ]
            adv = _bootstrap_adv(
                probe_cells, base_cells, spec.n_classes, int(probe_cfg["bootstrap_n"])
            )
            adv["strongest_baseline"] = strongest
        summary["per_spec"][spec.name] = {
            "n_classes": spec.n_classes,
            "auc_by_seed_layer": {str(s): {str(k): v for k, v in auc[s].items()} for s in seeds},
            "top3_by_seed": {str(s): top3[s] for s in seeds},
            "selected_layer": best_layer,
            "probe_auc_seed_mean_at_selected": float(
                np.mean([auc[s][best_layer] for s in seeds])
            ),
            "baseline_pooled_auc": baselines,
            "advantage_vs_strongest": adv,
        }
    # G2（主目标 T4，#18(g)）：ℓ* 处方向余弦 + 有效秩 + MLP/打乱标签
    p_entry = summary["per_spec"][primary]
    layer_star = int(p_entry["selected_layer"])
    spec_primary = next(s for s in specs if s.name == primary)
    train_keys = sorted(
        {
            (r.session_id, r.agent_channel)
            for seed in seeds
            for r in train_rows[(primary, seed)]
        }
    )
    train_store = g.preload_layer(roots["runs"], train_keys, layer_star)
    inner_set = set(inner)
    train_payload = {}
    for seed in seeds:
        roles = train_rows[(primary, seed)]
        inner_roles = [r for r in roles if r.session_id in inner_set]
        x_tr, y_tr, _ = g.assemble(roles, "acts", train_store)
        x_in, y_in, _ = g.assemble(inner_roles, "acts", train_store)
        train_payload[seed] = (x_tr, y_tr, x_in, y_in)
    del train_store
    gc.collect()
    eval_keys = sorted(
        {(r.session_id, r.agent_channel) for r in eval_rows[primary]}
    )
    eval_store = g.preload_layer(roots["runs"], eval_keys, layer_star)
    x_ev, y_ev, _ = g.assemble(eval_rows[primary], "acts", eval_store)
    x_ev = np.asarray(x_ev, dtype=np.float32)
    del eval_store
    gc.collect()
    directions: dict[int, np.ndarray] = {}
    er_by_seed = {}
    mlp_gap = {}
    shuffled = {}
    for seed in seeds:
        eval_scores_full, cell_meta, weight = _load_cell(
            _cell_path(roots, primary, "acts", layer_star, seed)
        )
        directions[seed] = np.asarray(weight, dtype=np.float64)
        x_tr, y_tr, x_in, y_in = train_payload.pop(seed)
        x_tr = np.asarray(x_tr, dtype=np.float32)
        x_in = np.asarray(x_in, dtype=np.float32)
        chosen_c = float(cell_meta["chosen_c"])
        er = pg.effective_rank(
            x_tr, y_tr, x_ev, y_ev, spec_primary.n_classes, chosen_c,
            [int(k) for k in probe_cfg["effective_rank"]["ks"]],
            float(probe_cfg["effective_rank"]["retention"]),
            device=args.device,
        )
        er_by_seed[seed] = er
        predict, _best, _epochs = pg.fit_mlp_probe(
            x_tr, y_tr, spec_primary.n_classes,
            x_in, y_in,
            probe_cfg["mlp"], seed, device=args.device,
        )
        mlp_auc = pg.primary_metric(y_ev, predict(x_ev), spec_primary.n_classes)
        linear_auc = g.pooled_primary_metric(eval_scores_full, spec_primary.n_classes)
        mlp_gap[seed] = {"mlp_auc": mlp_auc, "linear_auc": linear_auc, "gap": mlp_auc - linear_auc}
        rng = np.random.default_rng(1000 + seed)
        y_shuffled = y_tr.copy()
        rng.shuffle(y_shuffled)
        probe_shuffled = pg.fit_linear_probe(x_tr, y_shuffled, spec_primary.n_classes, chosen_c, device=args.device)
        shuffled[seed] = pg.primary_metric(y_ev, probe_shuffled.predict_proba(x_ev), spec_primary.n_classes)
        del x_tr, x_in
        gc.collect()
    cosines = pairwise_abs_cosines(directions)
    er_max = max(
        (er["effective_rank"] if er["effective_rank"] is not None else 10**9)
        for er in er_by_seed.values()
    )
    g2 = evaluate_g2(
        auc_by_seed_layer={
            s: {int(k): v for k, v in p_entry["auc_by_seed_layer"][str(s)].items()} for s in seeds
        },
        effective_rank=float(er_max),
        direction_cosines=cosines,
        g2_cfg=load_config("grids")["e1"]["g2"],
    )
    summary["g2"] = {
        "primary_target": primary,
        "selected_layer": layer_star,
        "direction_abs_cosines": {k: float(v) for k, v in cosines.items()},
        "effective_rank_by_seed": {str(s): er_by_seed[s] for s in seeds},
        "effective_rank_conservative": None if er_max >= 10**9 else int(er_max),
        "mlp_contrast": {str(s): mlp_gap[s] for s in seeds},
        "shuffled_auc_by_seed": {str(s): shuffled[s] for s in seeds},
        "verdict": g2,
    }
    write_report_json("wp_e1_probe_summary.json", summary)
    lines = [
        "# E1 探针网格报告（Moshi，PREREG #18/#19）",
        "",
        f"- G2 主目标 {primary}：ℓ* = L{layer_star}，判定 = **{g2['verdict']}**",
        f"- 方向 |cos| 最小值 {min(cosines.values()):.4f}；"
        f"有效秩（保守）{summary['g2']['effective_rank_conservative']}",
        "",
        "| 规格 | ℓ* | 探针(种子均值) | 最强基线 | 优势 [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        entry = summary["per_spec"][spec.name]
        adv = entry["advantage_vs_strongest"]
        if adv:
            adv_text = (
                f"{adv['advantage']:+.4f} [{adv['ci95'][0]:+.4f},"
                f"{adv['ci95'][1]:+.4f}] vs {adv['strongest_baseline']}"
            )
        else:
            adv_text = "（无基线格）"
        lines.append(
            f"| {spec.name} | L{entry['selected_layer']} | "
            f"{entry['probe_auc_seed_mean_at_selected']:.4f} | "
            f"{max(entry['baseline_pooled_auc'].values(), default=float('nan')):.4f} | "
            f"{adv_text} |"
        )
    (Path(REPO_ROOT) / "reports" / "e1_探针网格报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"finalize 完成：G2 = {g2['verdict']}（报告已写 reports/）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["labels", "parity", "acoustic", "run", "finalize"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--devices",
        default=None,
        help="run 阶段单进程多卡列表，例如 cuda:0,cuda:1；会共享一份层缓存",
    )
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument(
        "--acoustic-workers",
        type=int,
        default=1,
        help="acoustic 阶段内部工作进程数；与外部分片二选一控制总并发",
    )
    ap.add_argument(
        "--acoustic-limit",
        type=int,
        default=None,
        help="仅处理当前 acoustic 分片的前 N 个未完成任务，用于并发压力探针",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("要求 num-shards > 0 且 0 <= shard-id < num-shards")
    if args.acoustic_workers <= 0:
        raise SystemExit("--acoustic-workers 必须为正整数")
    if args.acoustic_limit is not None and args.acoustic_limit <= 0:
        raise SystemExit("--acoustic-limit 必须为正整数")
    {
        "labels": stage_labels,
        "parity": stage_parity,
        "acoustic": stage_acoustic,
        "run": stage_run,
        "finalize": stage_finalize,
    }[args.stage](args)


if __name__ == "__main__":
    main()
