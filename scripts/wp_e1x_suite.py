"""WP-E1X：E1-X 探索性套件引擎（PREREG #33；严格非裁决）。

阶段（相互独立，均支持断点续跑；建议顺序 geometry → leadtime → decompose →
trajectory → t2h → anatomy → finalize）：
  geometry   X4 方向几何：跨目标/跨 δ/跨层余弦、零样本转移矩阵、PC 质量谱、
             差分均值方向与随机对照方向 + 投影尺度 → 导出 E2-lite 方向文件。
  leadtime   X1 先知曲线：T4 特征行向锚点前平移 0..25 步逐档重训（探针 vs Mimi）。
  decompose  X3 声学分解：Mimi-GRU 序列基线、acts←Mimi 岭回归残差探针、
             重建探针、残差 rank-k 曲线。
  trajectory X2 决策变量轨迹：全步投影 → 事件对齐均值曲线 → 分叉时刻与
             领先量 → 接话延迟回归。
  t2h        X5a T2 视野扫描（步栅变体，h ∈ {3,5,10,20}；h=5 与正式标签对账）。
  anatomy    X5b T4 错误解剖：F0 末端斜率 / 对方 IPU 长度 / 接话速度分桶优势。
  finalize   汇总 → reports/wp_e1x_summary.json + reports/e1x_探索套件报告.md。

硬边界：只读正式产物与冻结行域；不回写 wp_e1_probe_summary.json；不触
causal_eval。产物根 <data_root>/e1x/。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
from pathlib import Path

import numpy as np
import wp_e1_probe_grid as engine
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1 import grid as g
from floor_circuit.e1 import probe_gpu as pg
from floor_circuit.e1x import anatomy as ax
from floor_circuit.e1x import core as cx
from floor_circuit.e1x import trajectory as tx
from floor_circuit.mve.alignment import usable_label_steps
from floor_circuit.probes.gru import make_windows, train_eval_gru
from floor_circuit.schemas import State

SCHEMA = "e1x-suite-v1"


# ---------------------------------------------------------------------------
# 公共上下文
# ---------------------------------------------------------------------------


def _ctx():
    probe_cfg, cache_cfg = engine._cfg()
    e1x_cfg = load_config("grids")["e1"]["e1x"]
    roots = engine._roots()
    train, evals = engine._sessions()
    specs, seeds, inner, _pools, train_rows, eval_rows = engine._prepare_rows(
        probe_cfg, roots, train, evals
    )
    x_root = data_root() / "e1x"
    x_root.mkdir(parents=True, exist_ok=True)
    return {
        "probe_cfg": probe_cfg,
        "cache_cfg": cache_cfg,
        "e1x": e1x_cfg,
        "roots": roots,
        "train": train,
        "evals": evals,
        "specs": {spec.name: spec for spec in specs},
        "seeds": seeds,
        "inner": inner,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "x_root": x_root,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _inner_mask(roles: list[g.RoleRows], inner: list[str]) -> np.ndarray:
    inner_set = set(inner)
    parts = [np.full(len(r.labels), r.session_id in inner_set, dtype=bool) for r in roles]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def _pooled_eval(roles: list[g.RoleRows], feature: str, store) -> tuple[np.ndarray, np.ndarray]:
    x, y, _sid = g.assemble(roles, feature, store)
    return np.asarray(x), np.asarray(y)


def _required_rows(groups: list[list[g.RoleRows]], n_rows: int) -> dict:
    return g.required_layer_rows(groups, n_rows=n_rows)


def _chance_cells(cells: dict) -> dict:
    """常数 0.5 概率的对照格：并列平均秩使 AUC 恰为 0.5，供 |AUC−0.5| 的 CI。"""
    out = {}
    for sid, (y, _p) in cells.items():
        probs = np.full((len(y), 2), 0.5, dtype=np.float64)
        out[sid] = (np.asarray(y), probs)
    return out


def _states_for_role(roots: dict, sid: str, channel: int, n_steps: int) -> np.ndarray:
    import pandas as pd

    frame = pd.read_parquet(roots["labels"] / f"{sid}.parquet")
    return g.t5_state_array(frame, channel, n_steps)


# ---------------------------------------------------------------------------
# geometry（X4）
# ---------------------------------------------------------------------------

_T5_CLASS_NAMES = {0: "SPEAK", 1: "LISTEN", 2: "OV_YIELD", 3: "OV_HOLD", 4: "GAP"}
_T3_CLASS_NAMES = {0: "bc", 1: "grab", 2: "other"}


def _load_probe(roots: dict, spec_name: str, layer: int, seed: int) -> pg.LinearProbe | None:
    path = engine._fit_path(roots, spec_name, layer, seed)
    if not path.is_file():
        return None
    probe, _meta = engine._load_fit(path)
    return probe


def _direction_set(ctx, layer: int, seed: int) -> dict[str, np.ndarray]:
    """L{layer} 处全部规格的原始空间单位方向（缺拟合断点的规格跳过并登记）。"""
    roots = ctx["roots"]
    out: dict[str, np.ndarray] = {}
    for name, spec in ctx["specs"].items():
        probe = _load_probe(roots, name, layer, seed)
        if probe is None:
            continue
        if spec.n_classes == 2:
            out[name] = cx.raw_direction(probe)
        else:
            class_names = _T3_CLASS_NAMES if name == "T3" else _T5_CLASS_NAMES
            for cls, cls_name in class_names.items():
                out[f"{name}:{cls_name}"] = cx.raw_direction(probe, class_index=cls)
    return out


def stage_geometry(args) -> None:
    ctx = _ctx()
    engine._validate_devices([str(args.device)])
    e1x_cfg = ctx["e1x"]
    roots = ctx["roots"]
    layer = int(e1x_cfg["layer_primary"])
    seeds = ctx["seeds"]
    out_path = ctx["x_root"] / "geometry.json"
    if _load_json(out_path) is not None and not args.force:
        print(f"geometry 已存在：{out_path}（--force 重跑）")
        return

    payload: dict = {"schema": SCHEMA, "layer": layer, "missing_fits": []}

    # 1) 跨目标余弦矩阵（逐种子）与 T1 δ 旋转
    per_seed_cos = {}
    for seed in seeds:
        dirs = _direction_set(ctx, layer, seed)
        if not dirs:
            raise SystemExit(f"L{layer} 无任何拟合断点（fits/*.npz）；确认正式网格已完成")
        per_seed_cos[str(seed)] = cx.cosine_matrix(dirs)
    payload["cosine_by_seed"] = per_seed_cos
    names0 = list(per_seed_cos[str(seeds[0])])
    payload["cosine_names"] = names0
    payload["cosine_mean_abs"] = {
        a: {
            b: float(np.mean([abs(per_seed_cos[str(s)][a][b]) for s in seeds]))
            for b in names0
        }
        for a in names0
    }

    # 2) T4 跨层方向（L28–L31）
    cross_layer = {}
    for seed in seeds:
        dirs = {}
        for cand in (28, 29, 30, 31):
            probe = _load_probe(roots, "T4", cand, seed)
            if probe is not None:
                dirs[f"L{cand}"] = cx.raw_direction(probe)
        if len(dirs) >= 2:
            cross_layer[str(seed)] = cx.cosine_matrix(dirs)
    payload["t4_cross_layer_cosine"] = cross_layer

    # 3) 二分类零样本转移矩阵（统一在 L{layer}，评估行 = 各规格冻结评估行域）
    binary_specs = [n for n, s in ctx["specs"].items() if s.n_classes == 2]
    eval_groups = [ctx["eval_rows"][n] for n in binary_specs]
    n_steps_map = engine._load_run_specs(roots)
    n_rows = next(iter(n_steps_map.values()))
    eval_required = _required_rows(eval_groups, n_rows)
    eval_store = g.preload_layer(
        roots["runs"], sorted(eval_required), layer, row_indices=eval_required
    )
    transfer: dict[str, dict[str, list[float]]] = {a: {b: [] for b in binary_specs} for a in binary_specs}
    for seed in seeds:
        probes = {n: _load_probe(roots, n, layer, seed) for n in binary_specs}
        missing = [n for n, p in probes.items() if p is None]
        payload["missing_fits"].extend(f"{n}@L{layer}/s{seed}" for n in missing)
        for b_name in binary_specs:
            x_ev, y_ev = _pooled_eval(ctx["eval_rows"][b_name], "acts", eval_store)
            if not ((y_ev == 1).any() and (y_ev == 0).any()):
                continue
            x_ev32 = np.asarray(x_ev, dtype=np.float32)
            for a_name in binary_specs:
                probe = probes[a_name]
                if probe is None:
                    continue
                auc = pg.primary_metric(y_ev, probe.predict_proba(x_ev32), 2)
                transfer[a_name][b_name].append(float(auc))
            del x_ev, x_ev32
    payload["transfer_auc_mean"] = {
        a: {b: (float(np.mean(v)) if v else None) for b, v in row.items()}
        for a, row in transfer.items()
    }
    del eval_store
    gc.collect()

    # 4) T4 训练行：差分均值方向、投影尺度、PC 质量谱、随机对照方向
    train_groups = [ctx["train_rows"][("T4", seed)] for seed in seeds]
    train_required = _required_rows(train_groups, n_rows)
    train_store = g.preload_layer(
        roots["runs"], sorted(train_required), layer, row_indices=train_required
    )
    union_roles: list[g.RoleRows] = []
    seen: set[tuple[str, int, int]] = set()
    for roles in train_groups:
        for role in roles:
            for pos, step in enumerate(role.steps):
                key = (role.session_id, role.agent_channel, int(step))
                if key in seen:
                    continue
                seen.add(key)
                union_roles.append(
                    g.RoleRows(
                        role.session_id,
                        role.agent_channel,
                        np.asarray([step]),
                        np.asarray([role.labels[pos]]),
                    )
                )
    x_all, y_all, _ = g.assemble(union_roles, "acts", train_store)
    x_all = np.asarray(x_all, dtype=np.float32)
    y_all = np.asarray(y_all)
    diffmeans = cx.diff_means_direction(x_all, y_all)

    directions: dict[str, np.ndarray] = {"diffmeans": diffmeans}
    for seed in seeds:
        probe = _load_probe(roots, "T4", layer, seed)
        if probe is None:
            raise SystemExit(f"缺 T4@L{layer}/s{seed} 拟合断点，无法导出主方向")
        directions[f"probe_s{seed}"] = cx.raw_direction(probe)
    stack = np.stack([directions[f"probe_s{s}"] for s in seeds])
    mean_dir = stack.mean(axis=0)
    directions["probe_meanseed"] = mean_dir / float(np.linalg.norm(mean_dir))
    rng = np.random.default_rng(int(e1x_cfg["direction_seed"]))
    for i in range(int(e1x_cfg["n_random_directions"])):
        vec = rng.standard_normal(x_all.shape[1])
        directions[f"random_r{i}"] = vec / float(np.linalg.norm(vec))

    proj_stats = {
        name: cx.projection_std(x_all, vec) for name, vec in directions.items()
    }
    payload["direction_cosines"] = cx.cosine_matrix(
        {k: v for k, v in directions.items() if not k.startswith("random")}
    )
    payload["projection_std"] = {k: float(v) for k, v in proj_stats.items()}

    # PC 质量谱（逐种子基；报告累计质量与参与率）
    mass_summary = {}
    for seed in seeds:
        seed_roles = ctx["train_rows"][("T4", seed)]
        x_seed, _y_seed, _ = g.assemble(seed_roles, "acts", train_store)
        x64 = np.asarray(x_seed, dtype=np.float64)
        centered = x64 - x64.mean(axis=0)
        _, svals, vt = np.linalg.svd(centered, full_matrices=False)
        var_share = (svals**2) / float((svals**2).sum())
        spectrum = cx.pc_mass_spectrum(vt, directions[f"probe_s{seed}"])
        checkpoints = [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, min(1024, len(vt))]
        mass_summary[str(seed)] = {
            "participation_ratio": spectrum["participation_ratio"],
            "cumulative_mass": {
                str(k): float(spectrum["cumulative"][k - 1]) for k in checkpoints if k <= len(vt)
            },
            "cumulative_variance": {
                str(k): float(np.cumsum(var_share)[k - 1]) for k in checkpoints if k <= len(vt)
            },
        }
        del x_seed, x64, centered, svals, vt
        gc.collect()
    payload["t4_pc_mass"] = mass_summary

    dir_path = ctx["x_root"] / "directions"
    dir_path.mkdir(parents=True, exist_ok=True)
    npz_payload = {name: vec.astype(np.float32) for name, vec in directions.items()}
    npz_payload["__meta__"] = np.frombuffer(
        json.dumps(
            {
                "schema": "e1x-directions-v1",
                "layer": layer,
                "sign": "+v 指向 T4 label=1（对方话轮 complete，可接话感知）",
                "scale": "steer = alpha * proj_std[name] * unit(v)",
                "proj_std": {k: float(v) for k, v in proj_stats.items()},
            },
            ensure_ascii=False,
        ).encode(),
        dtype=np.uint8,
    )
    np.savez(dir_path / "T4_L29.npz", **npz_payload)
    payload["directions_npz"] = str(dir_path / "T4_L29.npz")

    _atomic_json(out_path, payload)
    write_report_json("wp_e1x_geometry.json", payload)
    print(f"geometry 完成：{out_path}")


# ---------------------------------------------------------------------------
# leadtime（X1）
# ---------------------------------------------------------------------------


def stage_leadtime(args) -> None:
    ctx = _ctx()
    engine._validate_devices([str(args.device)])
    e1x_cfg = ctx["e1x"]
    roots = ctx["roots"]
    layer = int(e1x_cfg["layer_primary"])
    shifts = [int(k) for k in e1x_cfg["t4_shift_steps"]]
    max_shift = max(shifts)
    seeds = ctx["seeds"]
    cells_dir = ctx["x_root"] / "leadtime" / "cells"
    stage_dir = ctx["x_root"] / "leadtime"

    train_roles = {
        seed: cx.restrict_min_step(ctx["train_rows"][("T4", seed)], max_shift) for seed in seeds
    }
    eval_roles = cx.restrict_min_step(ctx["eval_rows"]["T4"], max_shift)
    n_steps_map = engine._load_run_specs(roots)
    n_rows = next(iter(n_steps_map.values()))

    train_groups = [cx.shift_roles(train_roles[s], k) for s in seeds for k in shifts]
    eval_groups = [cx.shift_roles(eval_roles, k) for k in shifts]
    train_required = _required_rows(train_groups, n_rows)
    train_store = g.preload_layer(
        roots["runs"], sorted(train_required), layer, row_indices=train_required
    )
    eval_required = _required_rows(eval_groups, n_rows)
    eval_store = g.preload_layer(
        roots["runs"], sorted(eval_required), layer, row_indices=eval_required
    )
    mimi_keys = sorted(
        {(r.session_id, ch) for roles in train_roles.values() for r in roles for ch in (0, 1)}
        | {(r.session_id, ch) for r in eval_roles for ch in (0, 1)}
    )
    mimi_store = g.preload_mimi(roots["runs"], mimi_keys)

    results = []
    for shift in shifts:
        shift_path = stage_dir / f"shift_{shift:02d}.json"
        cached = _load_json(shift_path)
        if cached is not None and not args.force:
            results.append(cached)
            print(f"[leadtime] shift={shift} 复用断点")
            continue
        probe_cells, mimi_cells = [], []
        seed_stats = {}
        for seed in seeds:
            shifted_train = cx.shift_roles(train_roles[seed], shift)
            shifted_eval = cx.shift_roles(eval_roles, shift)
            inner_mask = _inner_mask(shifted_train, ctx["inner"])
            x_tr, y_tr, _ = g.assemble(shifted_train, "acts", train_store)
            fit = cx.nested_probe_fit(
                np.asarray(x_tr), y_tr, inner_mask, 2,
                ctx["probe_cfg"]["c_grid"], device=str(args.device),
            )
            cells_p = g.eval_cell_scores(fit.probe, shifted_eval, "acts", eval_store)
            m_tr, my_tr, _ = g.assemble(shifted_train, "mimi", mimi_store)
            fit_m = cx.nested_probe_fit(
                np.asarray(m_tr), my_tr, inner_mask, 2,
                ctx["probe_cfg"]["c_grid"], device=str(args.device),
            )
            cells_m = g.eval_cell_scores(fit_m.probe, shifted_eval, "mimi", mimi_store)
            engine._save_cell(
                cells_dir / f"T4_shift{shift:02d}__acts__L{layer}__s{seed}.npz",
                cells_p,
                {"shift": shift, "seed": seed, "chosen_c": fit.chosen_c, "n_classes": 2},
                fit.probe.weight,
            )
            engine._save_cell(
                cells_dir / f"T4_shift{shift:02d}__mimi__Lnone__s{seed}.npz",
                cells_m,
                {"shift": shift, "seed": seed, "chosen_c": fit_m.chosen_c, "n_classes": 2},
                fit_m.probe.weight,
            )
            probe_cells.append(cells_p)
            mimi_cells.append(cells_m)
            seed_stats[str(seed)] = {
                "probe_auc": g.pooled_primary_metric(cells_p, 2),
                "mimi_auc": g.pooled_primary_metric(cells_m, 2),
                "chosen_c": fit.chosen_c,
                "n_train_rows": len(y_tr),
            }
            del x_tr, m_tr
            gc.collect()
        adv = engine._bootstrap_adv(probe_cells, mimi_cells, 2, int(e1x_cfg["bootstrap_n"]))
        entry = {
            "shift_steps": shift,
            "lead_ms": shift * 80,
            "per_seed": seed_stats,
            "probe_auc_mean": float(np.mean([v["probe_auc"] for v in seed_stats.values()])),
            "mimi_auc_mean": float(np.mean([v["mimi_auc"] for v in seed_stats.values()])),
            "advantage": adv,
        }
        _atomic_json(shift_path, entry)
        results.append(entry)
        print(
            f"[leadtime] shift={shift}（{shift * 80} ms）probe={entry['probe_auc_mean']:.4f} "
            f"mimi={entry['mimi_auc_mean']:.4f} adv={adv['advantage']:+.4f}"
        )
    payload = {
        "schema": SCHEMA,
        "layer": layer,
        "n_anchor_train_events_per_seed": {
            str(s): int(sum(len(r.labels) for r in train_roles[s])) for s in seeds
        },
        "n_anchor_eval_events": int(sum(len(r.labels) for r in eval_roles)),
        "curve": results,
    }
    _atomic_json(stage_dir / "leadtime.json", payload)
    write_report_json("wp_e1x_leadtime.json", payload)
    print("leadtime 完成")


# ---------------------------------------------------------------------------
# decompose（X3）
# ---------------------------------------------------------------------------


def _mimi_sequence(store: dict, sid: str, channel: int) -> np.ndarray:
    own = np.asarray(store[(sid, channel)], dtype=np.float32)
    other = np.asarray(store[(sid, 1 - channel)], dtype=np.float32)
    return np.concatenate([own, other], axis=1)


def _gru_sets(
    roles: list[g.RoleRows], mimi_store: dict, context: int
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    out = []
    for role in roles:
        sequence = _mimi_sequence(mimi_store, role.session_id, role.agent_channel)
        rows = np.asarray(role.steps, dtype=np.int64)  # mimi 行 = 标签步 s（#8）
        windows = make_windows(sequence, rows, context)
        out.append((role.session_id, role.agent_channel, windows, role.labels))
    return out


def stage_decompose(args) -> None:
    ctx = _ctx()
    engine._validate_devices([str(args.device)])
    e1x_cfg = ctx["e1x"]
    roots = ctx["roots"]
    layer = int(e1x_cfg["layer_primary"])
    seeds = ctx["seeds"]
    stage_dir = ctx["x_root"] / "decompose"
    n_steps_map = engine._load_run_specs(roots)
    n_rows = next(iter(n_steps_map.values()))

    train_groups = [ctx["train_rows"][("T4", s)] for s in seeds]
    eval_roles = ctx["eval_rows"]["T4"]
    required = _required_rows([*train_groups, eval_roles], n_rows)
    acts_store = g.preload_layer(roots["runs"], sorted(required), layer, row_indices=required)
    mimi_keys = sorted(
        {(r.session_id, ch) for roles in [*train_groups, eval_roles] for r in roles for ch in (0, 1)}
    )
    mimi_store = g.preload_mimi(roots["runs"], mimi_keys)

    entries = {}
    for seed in seeds:
        seed_path = stage_dir / f"seed_{seed}.json"
        cached = _load_json(seed_path)
        if cached is not None and not args.force:
            entries[str(seed)] = cached
            print(f"[decompose] seed={seed} 复用断点")
            continue
        roles_tr = ctx["train_rows"][("T4", seed)]
        inner_mask = _inner_mask(roles_tr, ctx["inner"])
        x_tr, y_tr, _ = g.assemble(roles_tr, "acts", acts_store)
        x_tr = np.asarray(x_tr, dtype=np.float32)
        m_tr, _my, _ = g.assemble(roles_tr, "mimi", mimi_store)
        m_tr = np.asarray(m_tr, dtype=np.float32)
        x_ev, y_ev = _pooled_eval(eval_roles, "acts", acts_store)
        x_ev = np.asarray(x_ev, dtype=np.float32)
        m_ev, _mye = _pooled_eval(eval_roles, "mimi", mimi_store)
        m_ev = np.asarray(m_ev, dtype=np.float32)

        fit_acts = cx.nested_probe_fit(
            x_tr, y_tr, inner_mask, 2, ctx["probe_cfg"]["c_grid"], device=str(args.device)
        )
        auc_acts = pg.primary_metric(y_ev, fit_acts.probe.predict_proba(x_ev), 2)
        fit_mimi = cx.nested_probe_fit(
            m_tr, y_tr, inner_mask, 2, ctx["probe_cfg"]["c_grid"], device=str(args.device)
        )
        auc_mimi = pg.primary_metric(y_ev, fit_mimi.probe.predict_proba(m_ev), 2)

        # 岭回归：λ 按 inner 重建 R² 选择（保守：最大化被声学解释的份额）
        lam_curve = {}
        best_lam = None
        for lam in [float(v) for v in e1x_cfg["ridge_lambdas"]]:
            model_c = cx.fit_ridge(m_tr[~inner_mask], x_tr[~inner_mask], lam)
            r2 = model_c.r_squared(m_tr[inner_mask], x_tr[inner_mask])
            lam_curve[str(lam)] = float(r2)
            if best_lam is None or r2 > lam_curve[str(best_lam)]:
                best_lam = str(lam)
        ridge = cx.fit_ridge(m_tr, x_tr, float(best_lam))
        r_tr = ridge.residual_std(m_tr, x_tr)
        r_ev = ridge.residual_std(m_ev, x_ev)
        recon_tr = ridge.predict_std(m_tr).astype(np.float32)
        recon_ev = ridge.predict_std(m_ev).astype(np.float32)

        fit_resid = cx.nested_probe_fit(
            r_tr, y_tr, inner_mask, 2, ctx["probe_cfg"]["c_grid"], device=str(args.device)
        )
        auc_resid = pg.primary_metric(y_ev, fit_resid.probe.predict_proba(r_ev), 2)
        fit_recon = cx.nested_probe_fit(
            recon_tr, y_tr, inner_mask, 2, ctx["probe_cfg"]["c_grid"], device=str(args.device)
        )
        auc_recon = pg.primary_metric(y_ev, fit_recon.probe.predict_proba(recon_ev), 2)

        er = pg.effective_rank(
            r_tr, y_tr, r_ev, y_ev, 2, fit_resid.chosen_c,
            [int(k) for k in e1x_cfg["residual_rank_ks"]], 0.95, device=str(args.device),
        )
        rank_vs_acts_full = None
        for k in sorted(int(k) for k in er["curve"]):
            if (er["curve"][str(k)] - 0.5) >= 0.95 * (auc_acts - 0.5):
                rank_vs_acts_full = int(k)
                break
        # 残差特征非 store 装配，逐会话残差格子按 eval_roles 顺序手工切分：
        resid_cells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        probs_all = fit_resid.probe.predict_proba(r_ev)
        offset = 0
        for role in eval_roles:
            stop = offset + len(role.labels)
            y_slice = y_ev[offset:stop]
            p_slice = probs_all[offset:stop]
            if role.session_id in resid_cells:
                y0, p0 = resid_cells[role.session_id]
                resid_cells[role.session_id] = (
                    np.concatenate([y0, y_slice]),
                    np.concatenate([p0, p_slice]),
                )
            else:
                resid_cells[role.session_id] = (y_slice.copy(), p_slice.copy())
            offset = stop
        adv_resid = engine._bootstrap_adv(
            [resid_cells], [_chance_cells(resid_cells)], 2, int(e1x_cfg["bootstrap_n"])
        )

        del x_tr, x_ev, r_tr, r_ev, recon_tr, recon_ev
        gc.collect()

        # Mimi-GRU（最强声学序列基线）
        gru_cfg = e1x_cfg["mimi_gru"]
        inner_set = set(ctx["inner"])
        train_sets = _gru_sets(
            [r for r in roles_tr if r.session_id not in inner_set], mimi_store,
            int(gru_cfg["context_steps"]),
        )
        val_sets = _gru_sets(
            [r for r in roles_tr if r.session_id in inner_set], mimi_store,
            int(gru_cfg["context_steps"]),
        )
        eval_sets_list = _gru_sets(eval_roles, mimi_store, int(gru_cfg["context_steps"]))
        eval_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sid, _ch, windows, labels in eval_sets_list:
            if sid in eval_sets:
                w0, y0 = eval_sets[sid]
                eval_sets[sid] = (np.concatenate([w0, windows]), np.concatenate([y0, labels]))
            else:
                eval_sets[sid] = (windows, labels)
        gru_cells = train_eval_gru(
            [(w, y) for _sid, _ch, w, y in train_sets],
            [(w, y) for _sid, _ch, w, y in val_sets],
            eval_sets,
            seed=seed,
            hidden=int(gru_cfg["hidden"]),
            max_epochs=int(gru_cfg["max_epochs"]),
            batch_size=int(gru_cfg["batch_size"]),
            lr=float(gru_cfg["lr"]),
            patience=int(gru_cfg["patience"]),
            device=str(args.device),
        )
        gru_cells = {
            sid: (y, np.stack([1.0 - p, p], axis=1)) for sid, (y, p) in gru_cells.items()
        }
        auc_gru = g.pooled_primary_metric(gru_cells, 2)
        engine._save_cell(
            stage_dir / "cells" / f"T4_mimigru__s{seed}.npz",
            gru_cells,
            {"seed": seed, "context": int(gru_cfg["context_steps"])},
            None,
        )
        del train_sets, val_sets, eval_sets_list, eval_sets
        gc.collect()

        entry = {
            "seed": seed,
            "auc_acts": float(auc_acts),
            "auc_mimi_frame": float(auc_mimi),
            "auc_mimi_gru": float(auc_gru),
            "auc_recon": float(auc_recon),
            "auc_resid": float(auc_resid),
            "resid_auc_ci_vs_chance": adv_resid,
            "ridge_lambda_curve_r2": lam_curve,
            "ridge_lambda": float(best_lam),
            "ridge_inner_r2": lam_curve[best_lam],
            "resid_effective_rank": er,
            "resid_rank_vs_acts_full": rank_vs_acts_full,
        }
        _atomic_json(seed_path, entry)
        entries[str(seed)] = entry
        print(
            f"[decompose] seed={seed} acts={auc_acts:.4f} mimi={auc_mimi:.4f} "
            f"gru={auc_gru:.4f} recon={auc_recon:.4f} resid={auc_resid:.4f} "
            f"resid_rank={er['effective_rank']}"
        )

    # acts vs Mimi-GRU 的会话级优势 CI（用正式 acts 格子 + 新 GRU 格子）
    probe_cells = []
    gru_cells_all = []
    for seed in seeds:
        scores, _meta, _w = engine._load_cell(engine._cell_path(roots, "T4", "acts", layer, seed))
        probe_cells.append(scores)
        scores_g, _m2, _w2 = engine._load_cell(stage_dir / "cells" / f"T4_mimigru__s{seed}.npz")
        gru_cells_all.append(scores_g)
    adv_vs_gru = engine._bootstrap_adv(probe_cells, gru_cells_all, 2, int(e1x_cfg["bootstrap_n"]))
    payload = {
        "schema": SCHEMA,
        "layer": layer,
        "per_seed": entries,
        "acts_vs_mimi_gru_advantage": adv_vs_gru,
    }
    _atomic_json(stage_dir / "decompose.json", payload)
    write_report_json("wp_e1x_decompose.json", payload)
    print(f"decompose 完成：acts−MimiGRU 优势 {adv_vs_gru['advantage']:+.4f}")


# ---------------------------------------------------------------------------
# trajectory（X2）
# ---------------------------------------------------------------------------


def _logit_series(probe: pg.LinearProbe, matrix: np.ndarray, rows: slice) -> np.ndarray:
    block = np.asarray(matrix[rows], dtype=np.float32)
    z = (block - probe.mean) / probe.scale
    return (z @ probe.weight.T + probe.bias)[:, 0].astype(np.float64)


def stage_trajectory(args) -> None:
    ctx = _ctx()
    engine._validate_devices([str(args.device)])
    e1x_cfg = ctx["e1x"]
    roots = ctx["roots"]
    layer = int(e1x_cfg["layer_primary"])
    seeds = ctx["seeds"]
    traj_cfg = e1x_cfg["trajectory"]
    window = int(traj_cfg["window_steps"])
    stage_dir = ctx["x_root"] / "trajectory"
    out_path = stage_dir / "trajectory.json"
    if _load_json(out_path) is not None and not args.force:
        print(f"trajectory 已存在：{out_path}（--force 重跑）")
        return
    n_steps_map = engine._load_run_specs(roots)

    probes_t4 = {s: _load_probe(roots, "T4", layer, s) for s in seeds}
    probes_t1 = {s: _load_probe(roots, "T1_d800", layer, s) for s in seeds}
    if any(p is None for p in probes_t4.values()):
        raise SystemExit(f"缺 T4@L{layer} 拟合断点")

    eval_keys = sorted({(r.session_id, r.agent_channel) for r in ctx["eval_rows"]["T4"]})
    eval_key_set = {(sid, ch) for sid, ch in eval_keys} | {
        (sid, 1 - ch) for sid, ch in eval_keys
    }
    acts_store = g.preload_layer(roots["runs"], sorted(eval_key_set), layer)
    mimi_store = g.preload_mimi(roots["runs"], sorted(eval_key_set))

    # Mimi 决策变量：T4 标准行域上的嵌套拟合（逐种子）
    mimi_probes = {}
    train_mimi_keys = sorted(
        {
            (r.session_id, ch)
            for s in seeds
            for r in ctx["train_rows"][("T4", s)]
            for ch in (0, 1)
        }
    )
    train_mimi_store = g.preload_mimi(roots["runs"], train_mimi_keys)
    for seed in seeds:
        roles_tr = ctx["train_rows"][("T4", seed)]
        m_tr, y_tr, _ = g.assemble(roles_tr, "mimi", train_mimi_store)
        mimi_probes[seed] = cx.nested_probe_fit(
            np.asarray(m_tr), y_tr, _inner_mask(roles_tr, ctx["inner"]), 2,
            ctx["probe_cfg"]["c_grid"], device=str(args.device),
        ).probe
    del train_mimi_store
    gc.collect()

    # 全步决策变量（种子平均 logit）；索引 = 标签步 s
    series_t4: dict[tuple[str, int], np.ndarray] = {}
    series_t1: dict[tuple[str, int], np.ndarray] = {}
    series_mimi: dict[tuple[str, int], np.ndarray] = {}
    states_cache: dict[tuple[str, int], np.ndarray] = {}
    for sid, channel in eval_keys:
        n_steps = n_steps_map[(sid, channel)]
        usable = usable_label_steps(n_steps)
        acts = acts_store[(sid, channel)]
        z_t4 = np.mean(
            [_logit_series(probes_t4[s], acts, slice(1, usable + 1)) for s in seeds], axis=0
        )
        series_t4[(sid, channel)] = z_t4
        if all(p is not None for p in probes_t1.values()):
            series_t1[(sid, channel)] = np.mean(
                [_logit_series(probes_t1[s], acts, slice(1, usable + 1)) for s in seeds], axis=0
            )
        own = np.asarray(mimi_store[(sid, channel)][:usable], dtype=np.float32)
        other = np.asarray(mimi_store[(sid, 1 - channel)][:usable], dtype=np.float32)
        mimi_features = np.concatenate([own, other], axis=1)
        z_m = np.mean(
            [
                (
                    (mimi_features - mimi_probes[s].mean) / mimi_probes[s].scale
                    @ mimi_probes[s].weight.T
                    + mimi_probes[s].bias
                )[:, 0]
                for s in seeds
            ],
            axis=0,
        )
        series_mimi[(sid, channel)] = z_m.astype(np.float64)
        states_cache[(sid, channel)] = _states_for_role(roots, sid, channel, n_steps)
    del acts_store
    gc.collect()

    anchors = []
    labels = []
    for role in ctx["eval_rows"]["T4"]:
        for pos, step in enumerate(role.steps):
            anchors.append((role.session_id, role.agent_channel, int(step)))
            labels.append(int(role.labels[pos]))
    labels = np.asarray(labels)

    result: dict = {"schema": SCHEMA, "layer": layer, "window_steps": window}
    curves = {}
    for name, series in (("t4_logit", series_t4), ("mimi_logit", series_mimi)):
        matrix, session_index = tx.aligned_matrix(series, anchors, window)
        div = tx.divergence_step(
            matrix, labels, session_index,
            n_boot=int(e1x_cfg["bootstrap_n"]),
            min_consecutive=int(traj_cfg["min_consecutive_sig"]),
        )
        group = tx.group_mean_curves(matrix, labels)
        curves[name] = {
            "offsets_steps": list(range(-window, window + 1)),
            "mean_complete": [float(v) for v in group["mean_label1"]],
            "mean_incomplete": [float(v) for v in group["mean_label0"]],
            "diff": [float(v) for v in div["diff"]],
            "ci_lo": [float(v) for v in div["ci_lo"]],
            "ci_hi": [float(v) for v in div["ci_hi"]],
            "divergence_offset_steps": div["divergence_offset"],
            "divergence_offset_ms": (
                None if div["divergence_offset"] is None else div["divergence_offset"] * 80
            ),
            "n_complete": group["n_label1"],
            "n_incomplete": group["n_label0"],
        }
    lead = None
    if (
        curves["t4_logit"]["divergence_offset_steps"] is not None
        and curves["mimi_logit"]["divergence_offset_steps"] is not None
    ):
        lead = (
            curves["mimi_logit"]["divergence_offset_steps"]
            - curves["t4_logit"]["divergence_offset_steps"]
        ) * 80
    result["event_aligned"] = curves
    result["internal_lead_ms_vs_mimi"] = lead

    # 自方 onset 对齐（描述性单组曲线）
    onset_anchors = []
    for (sid, channel), states in states_cache.items():
        usable = len(series_t4[(sid, channel)])
        for step in tx.onset_steps_from_states(states):
            if window <= step < usable - window:
                onset_anchors.append((sid, channel, int(step)))
    onset_curves = {}
    for name, series in (("t4_logit", series_t4), ("t1d800_logit", series_t1)):
        if not series:
            continue
        matrix, _idx = tx.aligned_matrix(series, onset_anchors, window)
        onset_curves[name] = [float(v) for v in np.nanmean(matrix, axis=0)]
    result["own_onset_aligned"] = {
        "offsets_steps": list(range(-window, window + 1)),
        "n_onsets": len(onset_anchors),
        "curves": onset_curves,
    }

    # 接话延迟：complete 锚点的 logit 幅值 vs 后续 agent onset 步距
    censor = int(traj_cfg["gap_censor_steps"])
    gap_pairs = []
    fast_labels = []
    fast_scores = []
    for (sid, channel, step), label in zip(anchors, labels, strict=True):
        if label != 1:
            continue
        series = series_t4[(sid, channel)]
        if step >= len(series):
            continue
        states = states_cache[(sid, channel)]
        gap = tx.next_state_step(states, step, State.SPEAK.value, censor)
        score = float(series[step])
        if gap is not None:
            gap_pairs.append((score, gap))
        fast_scores.append(score)
        fast_labels.append(int(gap is not None and gap <= 5))
    spearman = None
    if len(gap_pairs) >= 10:
        from scipy.stats import spearmanr

        rho, pval = spearmanr([a for a, _ in gap_pairs], [b for _, b in gap_pairs])
        spearman = {"rho": float(rho), "p": float(pval), "n": len(gap_pairs)}
    fast_auc = None
    fast_arr = np.asarray(fast_labels)
    if (fast_arr == 1).any() and (fast_arr == 0).any():
        fast_auc = float(pg._binary_auc(fast_arr, np.asarray(fast_scores)))
    result["gap_regression"] = {
        "spearman_logit_vs_gap": spearman,
        "auc_logit_predicts_fast_take_400ms": fast_auc,
        "n_complete_anchors": int((labels == 1).sum()),
        "n_uncensored_gaps": len(gap_pairs),
        "censor_steps": censor,
    }

    _atomic_json(out_path, result)
    write_report_json("wp_e1x_trajectory.json", result)
    print(
        f"trajectory 完成：T4 分叉 {curves['t4_logit']['divergence_offset_ms']} ms，"
        f"Mimi 分叉 {curves['mimi_logit']['divergence_offset_ms']} ms，领先 {lead} ms"
    )


# ---------------------------------------------------------------------------
# t2h（X5a）
# ---------------------------------------------------------------------------


def stage_t2h(args) -> None:
    ctx = _ctx()
    engine._validate_devices([str(args.device)])
    e1x_cfg = ctx["e1x"]
    roots = ctx["roots"]
    seeds = ctx["seeds"]
    stage_dir = ctx["x_root"] / "t2h"
    out_path = stage_dir / "t2h.json"
    if _load_json(out_path) is not None and not args.force:
        print(f"t2h 已存在：{out_path}（--force 重跑）")
        return
    horizons = [int(h) for h in e1x_cfg["t2_horizon_steps"]]
    max_h = max(horizons)
    n_steps_map = engine._load_run_specs(roots)

    summary = json.loads(
        (REPO_ROOT / "reports" / "wp_e1_probe_summary.json").read_text(encoding="utf-8")
    )
    layer = int(summary["per_spec"]["T2"]["selected_layer"])

    def usable_anchor_roles(roles: list[g.RoleRows]) -> list[g.RoleRows]:
        out = []
        for role in roles:
            limit = n_steps_map[(role.session_id, role.agent_channel)] - max_h - 1
            keep = role.steps <= limit
            if keep.any():
                out.append(
                    g.RoleRows(role.session_id, role.agent_channel, role.steps[keep], role.labels[keep])
                )
        return out

    train_roles = {s: usable_anchor_roles(ctx["train_rows"][("T2", s)]) for s in seeds}
    eval_roles = usable_anchor_roles(ctx["eval_rows"]["T2"])
    n_rows = next(iter(n_steps_map.values()))
    required = _required_rows([*train_roles.values(), eval_roles], n_rows)
    acts_store = g.preload_layer(roots["runs"], sorted(required), layer, row_indices=required)
    mimi_keys = sorted(
        {
            (r.session_id, ch)
            for roles in [*train_roles.values(), eval_roles]
            for r in roles
            for ch in (0, 1)
        }
    )
    mimi_store = g.preload_mimi(roots["runs"], mimi_keys)

    states_cache: dict[tuple[str, int], np.ndarray] = {}

    def relabel(roles: list[g.RoleRows], horizon: int) -> list[g.RoleRows]:
        out = []
        for role in roles:
            key = (role.session_id, role.agent_channel)
            if key not in states_cache:
                states_cache[key] = _states_for_role(roots, *key, n_steps_map[key])
            labels_h = ax.t2_horizon_labels(states_cache[key], role.steps, horizon)
            out.append(g.RoleRows(role.session_id, role.agent_channel, role.steps, labels_h))
        return out

    # h=5（400 ms）与正式标签一致率
    agree_num = 0
    agree_den = 0
    for role in eval_roles:
        labels_h5 = ax.t2_horizon_labels(
            states_cache.setdefault(
                (role.session_id, role.agent_channel),
                _states_for_role(
                    roots, role.session_id, role.agent_channel,
                    n_steps_map[(role.session_id, role.agent_channel)],
                ),
            ),
            role.steps,
            5,
        )
        agree_num += int((labels_h5 == role.labels).sum())
        agree_den += len(role.labels)
    agreement_h5 = agree_num / max(agree_den, 1)

    results = []
    for horizon in horizons:
        probe_cells, mimi_cells = [], []
        seed_stats = {}
        skip = False
        for seed in seeds:
            roles_h = relabel(train_roles[seed], horizon)
            eval_h = relabel(eval_roles, horizon)
            y_all = np.concatenate([r.labels for r in roles_h])
            if len(np.unique(y_all)) < 2:
                skip = True
                break
            inner_mask = _inner_mask(roles_h, ctx["inner"])
            x_tr, y_tr, _ = g.assemble(roles_h, "acts", acts_store)
            fit = cx.nested_probe_fit(
                np.asarray(x_tr), y_tr, inner_mask, 2,
                ctx["probe_cfg"]["c_grid"], device=str(args.device),
            )
            cells_p = g.eval_cell_scores(fit.probe, eval_h, "acts", acts_store)
            m_tr, my_tr, _ = g.assemble(roles_h, "mimi", mimi_store)
            fit_m = cx.nested_probe_fit(
                np.asarray(m_tr), my_tr, inner_mask, 2,
                ctx["probe_cfg"]["c_grid"], device=str(args.device),
            )
            cells_m = g.eval_cell_scores(fit_m.probe, eval_h, "mimi", mimi_store)
            probe_cells.append(cells_p)
            mimi_cells.append(cells_m)
            seed_stats[str(seed)] = {
                "probe_auc": g.pooled_primary_metric(cells_p, 2),
                "mimi_auc": g.pooled_primary_metric(cells_m, 2),
                "positive_rate_train": float(np.mean(y_tr)),
            }
        if skip:
            results.append({"horizon_steps": horizon, "skipped": "训练标签单类"})
            continue
        adv = engine._bootstrap_adv(probe_cells, mimi_cells, 2, int(e1x_cfg["bootstrap_n"]))
        results.append(
            {
                "horizon_steps": horizon,
                "horizon_ms": horizon * 80,
                "per_seed": seed_stats,
                "probe_auc_mean": float(np.mean([v["probe_auc"] for v in seed_stats.values()])),
                "mimi_auc_mean": float(np.mean([v["mimi_auc"] for v in seed_stats.values()])),
                "advantage": adv,
            }
        )
        print(f"[t2h] h={horizon} 步 adv={adv['advantage']:+.4f}")
    payload = {
        "schema": SCHEMA,
        "layer": layer,
        "grid_variant_note": "步栅近似（80 ms 状态离开发声态），非正式 dt=10 ms 合格 OFFSET",
        "agreement_h5_vs_formal": float(agreement_h5),
        "curve": results,
    }
    _atomic_json(out_path, payload)
    write_report_json("wp_e1x_t2h.json", payload)
    print(f"t2h 完成：h=5 与正式标签一致率 {agreement_h5:.4f}")


# ---------------------------------------------------------------------------
# anatomy（X5b）
# ---------------------------------------------------------------------------


def stage_anatomy(args) -> None:
    ctx = _ctx()
    e1x_cfg = ctx["e1x"]
    roots = ctx["roots"]
    layer = int(e1x_cfg["layer_primary"])
    seeds = ctx["seeds"]
    stage_dir = ctx["x_root"] / "anatomy"
    out_path = stage_dir / "anatomy.json"
    if _load_json(out_path) is not None and not args.force:
        print(f"anatomy 已存在：{out_path}（--force 重跑）")
        return
    n_steps_map = engine._load_run_specs(roots)
    eval_roles = ctx["eval_rows"]["T4"]

    probe_cells = [
        engine._load_cell(engine._cell_path(roots, "T4", "acts", layer, s))[0] for s in seeds
    ]
    mimi_cells = [
        engine._load_cell(engine._cell_path(roots, "T4", "mimi", None, s))[0] for s in seeds
    ]

    # 行级特征（顺序契约 = eval_roles 构建序；与格子逐会话拼接序一致，长度硬校验）
    acoustic_dir = roots["work"] / "acoustic"
    f0_values: list[float | None] = []
    ipu_lengths: list[float | None] = []
    take_fast: list[int | None] = []
    per_role_meta = []
    states_cache: dict[tuple[str, int], np.ndarray] = {}
    for role in eval_roles:
        other_key = (role.session_id, 1 - role.agent_channel)
        acoustic = np.load(
            acoustic_dir / f"{role.session_id}_ch{1 - role.agent_channel}.npy",
            allow_pickle=False,
        )
        if other_key not in states_cache:
            states_cache[other_key] = _states_for_role(roots, *other_key, n_steps_map[other_key])
        agent_key = (role.session_id, role.agent_channel)
        if agent_key not in states_cache:
            states_cache[agent_key] = _states_for_role(roots, *agent_key, n_steps_map[agent_key])
        other_states = states_cache[other_key]
        agent_states = states_cache[agent_key]
        speaking_other = np.isin(
            other_states,
            [
                State.SPEAK.value,
                State.OVERLAP_YIELD.value,
                State.OVERLAP_HOLD.value,
                State.OVERLAP_UNRESOLVED.value,
            ],
        )
        for pos, step in enumerate(role.steps):
            f0_values.append(ax.f0_slope(acoustic, int(step), tail_steps=6))
            run = 0
            cursor = int(step)
            while cursor >= 0 and speaking_other[cursor]:
                run += 1
                cursor -= 1
            ipu_lengths.append(float(run * 0.08))
            if int(role.labels[pos]) == 1:
                gap = tx.next_state_step(agent_states, int(step), State.SPEAK.value, 25)
                take_fast.append(int(gap is not None and gap <= 5))
            else:
                take_fast.append(None)
        per_role_meta.append((role.session_id, len(role.steps)))

    def masks_from_bucket(bucket: np.ndarray, value: int) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        offset = 0
        for role in eval_roles:
            stop = offset + len(role.steps)
            piece = bucket[offset:stop] == value
            if role.session_id in masks:
                masks[role.session_id] = np.concatenate([masks[role.session_id], piece])
            else:
                masks[role.session_id] = piece
            offset = stop
        return masks

    def bucket_advantage(bucket: np.ndarray, value: int) -> dict | None:
        masks = masks_from_bucket(bucket, value)
        filtered_probe = [ax.filter_cells_by_mask(c, masks) for c in probe_cells]
        filtered_mimi = [ax.filter_cells_by_mask(c, masks) for c in mimi_cells]
        common = set.intersection(*[set(c) for c in filtered_probe + filtered_mimi]) if filtered_probe[0] else set()
        usable = []
        for sid in common:
            y = filtered_probe[0][sid][0]
            if (y == 1).any() and (y == 0).any():
                usable.append(sid)
        if len(usable) < 8:
            return None
        filtered_probe = [{sid: c[sid] for sid in usable} for c in filtered_probe]
        filtered_mimi = [{sid: c[sid] for sid in usable} for c in filtered_mimi]
        adv = engine._bootstrap_adv(filtered_probe, filtered_mimi, 2, int(e1x_cfg["bootstrap_n"]))
        return {
            "n_sessions": len(usable),
            "n_rows": sum(len(c[0]) for c in filtered_probe[0].values()),
            "probe_auc": float(np.mean([g.pooled_primary_metric(c, 2) for c in filtered_probe])),
            "mimi_auc": float(np.mean([g.pooled_primary_metric(c, 2) for c in filtered_mimi])),
            "advantage": adv,
        }

    families: dict[str, dict] = {}
    f0_bucket, f0_edges = ax.tercile_buckets(f0_values)
    families["f0_final_slope"] = {
        "edges": f0_edges,
        "buckets": {
            {0: "falling", 1: "middle", 2: "rising"}[v]: bucket_advantage(f0_bucket, v)
            for v in (0, 1, 2)
        },
    }
    ipu_bucket, ipu_edges = ax.tercile_buckets(ipu_lengths)
    families["prior_ipu_length"] = {
        "edges": ipu_edges,
        "buckets": {
            {0: "short", 1: "middle", 2: "long"}[v]: bucket_advantage(ipu_bucket, v)
            for v in (0, 1, 2)
        },
    }
    take_arr = np.array([-1 if v is None else v for v in take_fast], dtype=np.int64)
    families["take_speed_complete_only"] = {
        "note": "仅 label=1（complete）行；fast = agent 于 ≤400 ms 内 SPEAK",
        "buckets": {
            "fast_take": bucket_advantage(take_arr, 1),
            "slow_or_no_take": bucket_advantage(take_arr, 0),
        },
    }

    payload = {"schema": SCHEMA, "layer": layer, "families": families}
    _atomic_json(out_path, payload)
    write_report_json("wp_e1x_anatomy.json", payload)
    print("anatomy 完成")


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def stage_finalize(args) -> None:
    ctx = _ctx()
    x_root = ctx["x_root"]
    parts = {
        "geometry": _load_json(x_root / "geometry.json"),
        "leadtime": _load_json(x_root / "leadtime" / "leadtime.json"),
        "decompose": _load_json(x_root / "decompose" / "decompose.json"),
        "trajectory": _load_json(x_root / "trajectory" / "trajectory.json"),
        "t2h": _load_json(x_root / "t2h" / "t2h.json"),
        "anatomy": _load_json(x_root / "anatomy" / "anatomy.json"),
    }
    payload = {"schema": SCHEMA, "stages_present": [k for k, v in parts.items() if v], **{
        k: v for k, v in parts.items() if v is not None
    }}
    write_report_json("wp_e1x_summary.json", payload)

    lines = ["# E1-X 探索套件报告（PREREG #33；严格非裁决）", ""]
    if parts["leadtime"]:
        lines += ["## X1 先知曲线（T4，特征行前移）", "",
                  "| 前移 (ms) | 探针 AUC | Mimi AUC | 优势 [95% CI] |", "| --- | --- | --- | --- |"]
        for entry in parts["leadtime"]["curve"]:
            adv = entry["advantage"]
            lines.append(
                f"| {entry['lead_ms']} | {entry['probe_auc_mean']:.4f} | {entry['mimi_auc_mean']:.4f} "
                f"| {adv['advantage']:+.4f} [{adv['ci95'][0]:+.4f},{adv['ci95'][1]:+.4f}] |"
            )
        lines.append("")
    if parts["trajectory"]:
        ea = parts["trajectory"]["event_aligned"]
        lines += [
            "## X2 决策变量轨迹",
            "",
            f"- T4 logit 分叉时刻：{ea['t4_logit']['divergence_offset_ms']} ms（相对对方 IPU 末端）",
            f"- Mimi logit 分叉时刻：{ea['mimi_logit']['divergence_offset_ms']} ms",
            f"- 内部证据领先量：{parts['trajectory']['internal_lead_ms_vs_mimi']} ms",
            f"- 接话延迟 Spearman：{parts['trajectory']['gap_regression']['spearman_logit_vs_gap']}",
            "",
        ]
    if parts["decompose"]:
        lines += ["## X3 声学分解（T4@L29）", "",
                  "| 种子 | acts | Mimi 帧 | Mimi-GRU | 重建 | 残差 | 残差有效秩 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for seed, entry in parts["decompose"]["per_seed"].items():
            lines.append(
                f"| {seed} | {entry['auc_acts']:.4f} | {entry['auc_mimi_frame']:.4f} | "
                f"{entry['auc_mimi_gru']:.4f} | {entry['auc_recon']:.4f} | {entry['auc_resid']:.4f} | "
                f"{entry['resid_effective_rank']['effective_rank']} |"
            )
        adv = parts["decompose"]["acts_vs_mimi_gru_advantage"]
        lines += ["", f"- acts − Mimi-GRU 优势：{adv['advantage']:+.4f} "
                  f"[{adv['ci95'][0]:+.4f},{adv['ci95'][1]:+.4f}]", ""]
    if parts["geometry"]:
        mass = parts["geometry"].get("t4_pc_mass", {})
        lines += ["## X4 方向几何", ""]
        cos = parts["geometry"].get("direction_cosines", {})
        if "probe_meanseed" in cos and "diffmeans" in cos:
            lines.append(f"- cos(探针方向, 差分均值方向) = {cos['probe_meanseed']['diffmeans']:+.4f}")
        for seed, entry in mass.items():
            lines.append(
                f"- 种子 {seed}：方向参与率 {entry['participation_ratio']:.1f}，"
                f"前 16 PC 质量 {entry['cumulative_mass'].get('16', float('nan')):.3f}"
                f"（方差 {entry['cumulative_variance'].get('16', float('nan')):.3f}）"
            )
        lines.append("")
    if parts["t2h"]:
        lines += ["## X5a T2 视野扫描（步栅变体）", "",
                  f"- h=5 与正式标签一致率：{parts['t2h']['agreement_h5_vs_formal']:.4f}", ""]
        for entry in parts["t2h"]["curve"]:
            if "skipped" in entry:
                lines.append(f"- h={entry['horizon_steps']}：跳过（{entry['skipped']}）")
            else:
                adv = entry["advantage"]
                lines.append(
                    f"- h={entry['horizon_steps']} 步（{entry['horizon_ms']} ms）："
                    f"探针 {entry['probe_auc_mean']:.4f} vs Mimi {entry['mimi_auc_mean']:.4f}，"
                    f"优势 {adv['advantage']:+.4f} [{adv['ci95'][0]:+.4f},{adv['ci95'][1]:+.4f}]"
                )
        lines.append("")
    if parts["anatomy"]:
        lines += ["## X5b T4 错误解剖（分桶优势）", ""]
        for family, entry in parts["anatomy"]["families"].items():
            lines.append(f"### {family}")
            for bucket, stats in entry["buckets"].items():
                if stats is None:
                    lines.append(f"- {bucket}：样本不足，跳过")
                else:
                    adv = stats["advantage"]
                    lines.append(
                        f"- {bucket}：探针 {stats['probe_auc']:.4f} vs Mimi {stats['mimi_auc']:.4f}，"
                        f"优势 {adv['advantage']:+.4f} [{adv['ci95'][0]:+.4f},{adv['ci95'][1]:+.4f}]"
                        f"（{stats['n_rows']} 行 / {stats['n_sessions']} 会话）"
                    )
            lines.append("")
    report_path = Path(REPO_ROOT) / "reports" / "e1x_探索套件报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"finalize 完成：{report_path}")


STAGES = {
    "geometry": stage_geometry,
    "leadtime": stage_leadtime,
    "decompose": stage_decompose,
    "trajectory": stage_trajectory,
    "t2h": stage_t2h,
    "anatomy": stage_anatomy,
    "finalize": stage_finalize,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="E1-X 探索性套件（PREREG #33）")
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "all":
        for name in ("geometry", "leadtime", "decompose", "trajectory", "t2h", "anatomy", "finalize"):
            STAGES[name](args)
    else:
        STAGES[args.stage](args)


if __name__ == "__main__":
    main()
