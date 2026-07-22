"""WP-E1-2/3：E1 探针全网格引擎（PREREG #18）。

阶段（按序）：
  labels   校验 500 会话标签齐备（缺失则先跑 `uv run python scripts/wp1_run_events.py`
           增算——事件管线带指纹缓存，只会计算缺失会话）并平铺同步 + 登记 n_steps。
  parity   torch-GPU 训练器 vs sklearn 奇偶校验门（#18(d)）：T4 与 T1_d400 × 层 20，
           前 20 训练会话。未过则禁止 run。
  run      层主序全网格：--num-shards 2 --shard-id {0,1} 双 GPU 各扫奇偶层；
           shard 0 附带层无关基线（Mimi / hazard / GRU）。
  finalize 汇总全部格子 → 选层 → ℓ* 处 MLP/打乱标签/有效秩 → G2 判定 →
           bootstrap 优势 → reports/wp_e1_probe_summary.json + e1_探针网格报告.md。

产物根：<data_root>/e1_probe/（cells/*.npz 每格逐会话分数 + 元数据）。
"""

from __future__ import annotations

import argparse
import json
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


def _cfg() -> tuple[dict, dict]:
    grids = load_config("grids")
    return grids["e1"]["probe"], grids["e1"]["cache"]


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


def _load_run_specs(roots: dict) -> dict[tuple[str, int], int]:
    payload = json.loads(_run_specs_path(roots).read_text(encoding="utf-8"))
    return {(sid, int(ch)): int(n) for sid, ch, n in payload["roles"]}


def stage_labels(args) -> None:
    roots = _roots()
    train, evals = _sessions()
    sessions = train + evals
    missing = [sid for sid in sessions if not (roots["events"] / f"{sid}.parquet").is_file()]
    if missing:
        listing = roots["work"] / "missing_label_sessions.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("\n".join(missing), encoding="utf-8")
        raise SystemExit(
            f"{len(missing)} 个会话缺标签（清单 {listing}）。先跑：\n"
            "  uv run python scripts/wp1_run_events.py\n"
            "（事件管线指纹缓存只增算缺失会话），完成后重跑本阶段。"
        )
    roots["labels"].mkdir(parents=True, exist_ok=True)
    import shutil

    for sid in sessions:
        target = roots["labels"] / f"{sid}.parquet"
        source = roots["events"] / f"{sid}.parquet"
        if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            shutil.copy2(source, target)
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
        json.dumps({"n_sessions": len(sessions), "roles": roles}, ensure_ascii=False), encoding="utf-8"
    )
    write_report_json(
        "wp_e1_probe_labels.json",
        {"n_sessions": len(sessions), "n_roles": len(roles), "labels_root": str(roots["labels"])},
    )
    print(f"标签阶段就绪：{len(sessions)} 会话 / {len(roles)} 路")


def _cell_path(roots: dict, spec_name: str, feature: str, layer: int | None, seed: int) -> Path:
    tag = "none" if layer is None else str(layer)
    return roots["cells"] / f"{spec_name}__{feature}__L{tag}__s{seed}.npz"


def _save_cell(path: Path, scores: dict, meta: dict, weight: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_bytes = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode(), dtype=np.uint8)
    arrays: dict[str, np.ndarray] = {"__meta__": meta_bytes}
    if weight is not None:
        arrays["__weight__"] = weight.astype(np.float32)
    for sid, (y, p) in scores.items():
        arrays[f"y::{sid}"] = np.asarray(y, dtype=np.int16)
        arrays[f"p::{sid}"] = np.asarray(p, dtype=np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
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


def _prepare_rows(probe_cfg: dict, roots: dict, train: list[str], evals: list[str]):
    """预构建每 (spec, seed) 的训练行与每 spec 的评估行（层无关，仅标签域）。"""
    specs = g.expand_specs(probe_cfg)
    n_steps = _load_run_specs(roots)
    seeds = [int(s) for s in probe_cfg["seeds"]]
    inner = train[: int(probe_cfg["inner_val_sessions"])]
    train_rows: dict[tuple[str, int], list[g.RoleRows]] = {}
    pool_by_seed: dict[int, list[str]] = {}
    for seed in seeds:
        pool = g.seed_train_sessions(train, probe_cfg, seed)
        pool_by_seed[seed] = pool
        for spec in specs:
            train_rows[(spec.name, seed)] = g.build_rows(
                roots["labels"], pool, n_steps, spec, probe_cfg, seed, downsample=True
            )
    eval_rows = {
        spec.name: g.build_rows(roots["labels"], evals, n_steps, spec, probe_cfg, 0, downsample=False)
        for spec in specs
    }
    return specs, seeds, inner, pool_by_seed, train_rows, eval_rows


def _fit_cell(
    spec, probe_cfg, train_roles, inner_sessions, feature, store, device
) -> tuple[object, dict]:
    """C 路径（inner_val 选择）→ 整池正式重训；返回 (探针, 元数据)。"""
    trainer = probe_cfg["trainer"]
    inner_set = set(inner_sessions)
    c_roles = [r for r in train_roles if r.session_id not in inner_set]
    inner_roles = [r for r in train_roles if r.session_id in inner_set]
    x_c, y_c, _ = g.assemble(c_roles, feature, store)
    x_inner, y_inner, _ = g.assemble(inner_roles, feature, store)
    x_c32 = np.asarray(x_c, dtype=np.float32)
    x_inner32 = np.asarray(x_inner, dtype=np.float32)
    best = None
    curve = {}
    warm = None
    for c_value in [float(c) for c in probe_cfg["c_grid"]]:
        probe = pg.fit_linear_probe(
            x_c32, y_c, spec.n_classes, c_value,
            device=device,
            max_iter=int(trainer["lbfgs_max_iter"]),
            tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
            init=warm,
        )
        warm = probe
        metric = pg.primary_metric(y_inner, probe.predict_proba(x_inner32), spec.n_classes)
        curve[str(c_value)] = metric
        if best is None or metric > best[1]:
            best = (c_value, metric)
    del x_c, x_c32, x_inner, x_inner32
    x_full, y_full, _ = g.assemble(train_roles, feature, store)
    final = pg.fit_linear_probe(
        np.asarray(x_full, dtype=np.float32), y_full, spec.n_classes, best[0],
        device=device,
        max_iter=int(trainer["lbfgs_max_iter"]),
        tolerance_grad=float(trainer["lbfgs_tolerance_grad"]),
    )
    meta = {
        "chosen_c": best[0],
        "inner_val_curve": curve,
        "inner_val_metric": best[1],
        "n_train_rows": len(y_full),
        "converged": final.converged,
    }
    return final, meta


def stage_parity(args) -> None:
    probe_cfg, _ = _cfg()
    roots = _roots()
    train, _evals = _sessions()
    n_steps = _load_run_specs(roots)
    layer = 20
    sessions = train[:20]
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
        x_fit, y_fit, _ = g.assemble(fit_roles, "acts", store)
        x_val, y_val, _ = g.assemble(val_roles, "acts", store)
        x_fit = np.asarray(x_fit, dtype=np.float32)
        x_val = np.asarray(x_val, dtype=np.float32)
        torch_probe = pg.fit_linear_probe(x_fit, y_fit, 2, 0.001, device=args.device)
        auc_torch = pg.primary_metric(y_val, torch_probe.predict_proba(x_val), 2)
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


def _layer_pass(layer, specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, device, force):
    keys = sorted(
        {(r.session_id, r.agent_channel) for rows in train_rows.values() for r in rows}
        | {(r.session_id, r.agent_channel) for rows in eval_rows.values() for r in rows}
    )
    pending = [
        (spec, seed)
        for spec in specs
        for seed in seeds
        if force or not _cell_path(roots, spec.name, "acts", layer, seed).is_file()
    ]
    if not pending:
        print(f"层 {layer}：全部格子已存在，跳过")
        return
    store = g.preload_layer(roots["runs"], keys, layer)
    for spec, seed in pending:
        probe, meta = _fit_cell(
            spec, probe_cfg, train_rows[(spec.name, seed)], inner, "acts", store, device
        )
        scores = g.eval_cell_scores(probe, eval_rows[spec.name], "acts", store)
        meta.update({"spec": spec.name, "feature": "acts", "layer": layer, "seed": seed})
        weight = probe.direction() if spec.n_classes == 2 else None
        _save_cell(_cell_path(roots, spec.name, "acts", layer, seed), scores, meta, weight)
        print(f"L{layer} {spec.name} s{seed}：inner={meta['inner_val_metric']:.4f} C={meta['chosen_c']}")
    del store


def _baseline_pass(specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, device, force):
    """层无关基线：Mimi（全规格）、hazard（T1/T2/T4）、GRU（二分类规格）。"""
    keys = sorted(
        {(r.session_id, r.agent_channel) for rows in train_rows.values() for r in rows}
        | {(r.session_id, r.agent_channel) for rows in eval_rows.values() for r in rows}
    )
    mimi = g.preload_mimi(roots["runs"], keys)
    for spec in specs:
        for seed in seeds:
            path = _cell_path(roots, spec.name, "mimi", None, seed)
            if not force and path.is_file():
                continue
            probe, meta = _fit_cell(
                spec, probe_cfg, train_rows[(spec.name, seed)], inner, "mimi", mimi, device
            )
            scores = g.eval_cell_scores(probe, eval_rows[spec.name], "mimi", mimi)
            meta.update({"spec": spec.name, "feature": "mimi", "layer": None, "seed": seed})
            _save_cell(path, scores, meta, None)
            print(f"mimi {spec.name} s{seed}：inner={meta['inner_val_metric']:.4f}")
    del mimi
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


def stage_acoustic(args) -> None:
    """预计算 500 会话双通道声学帧特征（RMS/F0/谱通量/ZCR，可多进程分片）。"""
    from floor_circuit.probes.baselines import acoustic_frames

    roots = _roots()
    train, evals = _sessions()
    out_dir = roots["work"] / "acoustic"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = _load_run_specs(roots)
    tasks = [
        (sid, ch)
        for sid in (train + evals)
        for ch in (0, 1)
        if not (out_dir / f"{sid}_ch{ch}.npy").is_file()
    ]
    tasks = tasks[args.shard_id :: args.num_shards]
    import soundfile as sf

    for index, (sid, channel) in enumerate(tasks, start=1):
        wav, sr = sf.read(str(roots["audio"] / sid / f"audio_ch{channel}.wav"), dtype="float32")
        wav = wav[: int(240.0 * sr)]
        feats = acoustic_frames(wav, sr)[: n_steps[(sid, channel)]]
        if len(feats) < n_steps[(sid, channel)]:
            feats = np.pad(feats, ((0, n_steps[(sid, channel)] - len(feats)), (0, 0)))
        np.save(out_dir / f"{sid}_ch{channel}.npy", feats.astype(np.float32), allow_pickle=False)
        if index % 50 == 0:
            print(f"声学特征 {index}/{len(tasks)}")
    print(f"声学特征完成 {len(tasks)} 路（shard {args.shard_id}/{args.num_shards}）")


def stage_run(args) -> None:
    probe_cfg, cache_cfg = _cfg()
    roots = _roots()
    if (roots["work"] / "parity_ok").read_text(encoding="utf-8").strip() != "passed":
        raise SystemExit("奇偶校验门未通过，禁止正式网格（PREREG #18(d)）")
    train, evals = _sessions()
    specs, seeds, inner, _pools, train_rows, eval_rows = _prepare_rows(probe_cfg, roots, train, evals)
    layers = list(range(int(cache_cfg["n_layers"])))[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}：层 {layers}")
    for layer in layers:
        _layer_pass(
            layer, specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, args.device, args.force
        )
    if args.shard_id == 0:
        _baseline_pass(specs, seeds, inner, train_rows, eval_rows, probe_cfg, roots, args.device, args.force)
    print("run 阶段完成")


def _bootstrap_adv(probe_cells, base_cells, n_classes, n_boot, seed=20260717):
    """会话重采样 × 种子均值的优势 bootstrap（冻结统计规范）。"""
    sids = sorted(set.intersection(*[set(c.keys()) for c in probe_cells + base_cells]))
    rng = np.random.default_rng(seed)

    def stat(take: list[str]) -> float:
        def seed_mean(cells):
            values = []
            for cell in cells:
                ys = np.concatenate([cell[sid][0] for sid in take])
                ps = np.concatenate([cell[sid][1] for sid in take])
                values.append(pg.primary_metric(ys, ps, n_classes))
            return float(np.mean(values))

        return seed_mean(probe_cells) - seed_mean(base_cells)

    point = stat(sids)
    samples = []
    for _ in range(n_boot):
        take = [sids[i] for i in rng.integers(0, len(sids), len(sids))]
        try:
            samples.append(stat(take))
        except ValueError:
            continue
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"advantage": point, "ci95": [float(lo), float(hi)], "n_boot_effective": len(samples)}


def stage_finalize(args) -> None:
    probe_cfg, cache_cfg = _cfg()
    roots = _roots()
    train, evals = _sessions()
    specs, seeds, inner, _pools, train_rows, eval_rows = _prepare_rows(probe_cfg, roots, train, evals)
    n_layers = int(cache_cfg["n_layers"])
    summary: dict = {"per_spec": {}, "prereg": "#17"}
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
    keys = sorted(
        {(r.session_id, r.agent_channel) for r in train_rows[(primary, seeds[0])]}
        | {(r.session_id, r.agent_channel) for r in eval_rows[primary]}
        | {
            (r.session_id, r.agent_channel)
            for seed in seeds
            for r in train_rows[(primary, seed)]
        }
    )
    store = g.preload_layer(roots["runs"], keys, layer_star)
    directions: dict[int, np.ndarray] = {}
    er_by_seed = {}
    mlp_gap = {}
    shuffled = {}
    for seed in seeds:
        _scores_w, _meta_w, weight = _load_cell(_cell_path(roots, primary, "acts", layer_star, seed))
        directions[seed] = np.asarray(weight, dtype=np.float64)
        x_tr, y_tr, _ = g.assemble(train_rows[(primary, seed)], "acts", store)
        x_tr = np.asarray(x_tr, dtype=np.float32)
        eval_scores_full = _load_cell(_cell_path(roots, primary, "acts", layer_star, seed))[0]
        x_ev, y_ev, _ = g.assemble(eval_rows[primary], "acts", store)
        x_ev = np.asarray(x_ev, dtype=np.float32)
        chosen_c = float(
            _load_cell(_cell_path(roots, primary, "acts", layer_star, seed))[1]["chosen_c"]
        )
        er = pg.effective_rank(
            x_tr, y_tr, x_ev, y_ev, spec_primary.n_classes, chosen_c,
            [int(k) for k in probe_cfg["effective_rank"]["ks"]],
            float(probe_cfg["effective_rank"]["retention"]),
            device=args.device,
        )
        er_by_seed[seed] = er
        inner_set = set(inner)
        inner_roles = [r for r in train_rows[(primary, seed)] if r.session_id in inner_set]
        x_in, y_in, _ = g.assemble(inner_roles, "acts", store)
        predict, _best, _epochs = pg.fit_mlp_probe(
            x_tr, y_tr, spec_primary.n_classes,
            np.asarray(x_in, dtype=np.float32), y_in,
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
        "# E1 探针网格报告（Moshi，PREREG #18）",
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
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {
        "labels": stage_labels,
        "parity": stage_parity,
        "acoustic": stage_acoustic,
        "run": stage_run,
        "finalize": stage_finalize,
    }[args.stage](args)


if __name__ == "__main__":
    main()
