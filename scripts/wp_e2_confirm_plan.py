"""WP-E2C：E2 确认臂计划生成器（PREREG #40(b)；探索性）。

在 E2-lite 充分性阳性之后，用最小成本回答四个确认问题：
  必要性     —— clamp_L29：生成时把 v̂ 分量钳到训练行投影均值 μ_v（信息移除）；
  时间特异性 —— probe 方向 ±4 只在 respond 门（用户合格 offset 后 1.2 s）或
               user_speech 门（用户发声帧）内注入；
  轴特异性   —— T1_d800 / T5:SPEAK 方向（floor 家族内其他轴）同剂量连续注入；
  层特异性   —— 同一 L29 方向在 L20 / L31 注入（尺度按该层投影 σ 重标定）。

基线不重跑：复用 E2-lite baseline 运行（同会话同种子，公共随机数配对）。
前置：E2-lite 计划与其 baseline 运行齐备；正式 fits（T1_d800 / T5@L29）在盘。
产出：<data_root>/e2_confirm/{e2_confirm.plan.json, directions_confirm.npz, gates/*.npz}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import wp_e1_probe_grid as engine
from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1 import grid as g
from floor_circuit.e1x import confirm as cf
from floor_circuit.e1x import core as cx
from floor_circuit.e1x.mask_cache import cached_mask
from floor_circuit.events.vad import SileroVad

PLAN_SCHEMA = "e2_confirm_plan_v1"
DT = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _t4_union_roles(train_rows: dict, seeds: list[int]) -> list[g.RoleRows]:
    """三种子 T4 训练行的去重并集（与 geometry 投影统计同口径）。"""
    union: list[g.RoleRows] = []
    seen: set[tuple[str, int, int]] = set()
    for seed in seeds:
        for role in train_rows[("T4", seed)]:
            for position, step in enumerate(role.steps):
                key = (role.session_id, role.agent_channel, int(step))
                if key in seen:
                    continue
                seen.add(key)
                union.append(
                    g.RoleRows(
                        role.session_id,
                        role.agent_channel,
                        np.asarray([step]),
                        np.asarray([role.labels[position]]),
                    )
                )
    return union


def _axis_direction(roots: dict, name: str, layer: int, seeds: list[int]) -> np.ndarray:
    """轴特异性对照方向：跨种子原始空间单位方向的单位化均值。"""
    vectors = []
    for seed in seeds:
        if name == "T1_d800":
            probe, _meta = engine._load_fit(engine._fit_path(roots, "T1_d800", layer, seed))
            vectors.append(cx.raw_direction(probe))
        elif name == "T5_SPEAK":
            probe, _meta = engine._load_fit(engine._fit_path(roots, "T5", layer, seed))
            vectors.append(cx.raw_direction(probe, class_index=0))
        else:
            raise SystemExit(f"未知轴对照方向：{name}")
    aligned = [vectors[0]]
    for vector in vectors[1:]:
        aligned.append(vector if float(vector @ vectors[0]) >= 0 else -vector)
    mean = np.mean(aligned, axis=0)
    return mean / float(np.linalg.norm(mean))


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 确认臂计划生成（PREREG #40(b)）")
    parser.add_argument("--lite-plan", default=None, help="默认 <data_root>/e2_lite/e2_lite.plan.json")
    parser.add_argument(
        "--baseline-runs-root",
        default=None,
        help="E2-lite baseline 运行根（默认自动优先 <out_root>_optimized/runs）",
    )
    parser.add_argument("--n-sessions", type=int, default=None)
    args = parser.parse_args()

    grids = load_config("grids")
    cfg = grids["e1"]["e2_confirm"]
    events_cfg = load_config("events")
    base = data_root()
    lite_plan_path = (
        Path(args.lite_plan)
        if args.lite_plan
        else base / str(grids["e1"]["e2_lite"]["out_group"]) / "e2_lite.plan.json"
    )
    lite_plan = json.loads(lite_plan_path.read_text(encoding="utf-8"))

    n_sessions = int(args.n_sessions or cfg["n_sessions"])
    sessions = lite_plan["sessions"][:n_sessions]
    if len(sessions) < n_sessions:
        raise SystemExit("E2-lite 计划中的会话不足所需数量")

    # 基线运行根：优先已验收 optimized，全部 baseline 必须已完成
    if args.baseline_runs_root:
        baseline_root = Path(args.baseline_runs_root)
    else:
        optimized = Path(str(lite_plan["out_root"]) + "_optimized") / "runs"
        baseline_root = optimized if optimized.is_dir() else Path(lite_plan["out_root"]) / "runs"
    incomplete = []
    for session in sessions:
        manifest = baseline_root / f"{session['session_id']}__baseline" / "manifest.json"
        try:
            if not json.loads(manifest.read_text(encoding="utf-8")).get("completed"):
                incomplete.append(session["session_id"])
        except (OSError, json.JSONDecodeError):
            incomplete.append(session["session_id"])
    if incomplete:
        raise SystemExit(f"E2-lite baseline 未闭合的会话：{incomplete[:5]}（根 {baseline_root}）")

    main_layer = int(cfg["layer"])
    if main_layer != int(lite_plan["layer"]):
        raise SystemExit(f"确认臂主层 {main_layer} ≠ E2-lite 层 {lite_plan['layer']}")
    all_layers = sorted({main_layer, *[int(v) for v in cfg["wrong_layers"]]})

    # 方向集合：probe_meanseed（复用 E2-lite 方向包）+ 轴对照（自正式 fits）
    with np.load(Path(lite_plan["directions_npz"]), allow_pickle=False) as payload:
        probe_meanseed = np.asarray(payload["probe_meanseed"], dtype=np.float64)
    probe_cfg, _cache_cfg = engine._cfg()
    roots = engine._roots()
    train, evals = engine._sessions()
    _specs, seeds, _inner, _pools, train_rows, _eval_rows = engine._prepare_rows(
        probe_cfg, roots, train, evals
    )
    directions: dict[str, np.ndarray] = {"probe_meanseed": probe_meanseed}
    for name in [str(v) for v in cfg["axis_directions"]]:
        directions[name] = _axis_direction(roots, name, main_layer, seeds)

    # 投影统计：每 (方向, 层) 在 T4 训练并集行上的均值与标准差
    union_roles = _t4_union_roles(train_rows, seeds)
    n_steps_map = engine._load_run_specs(roots)
    n_rows = next(iter(n_steps_map.values()))
    required = g.required_layer_rows([union_roles], n_rows=n_rows)
    proj_mean: dict[str, dict[str, float]] = {name: {} for name in directions}
    proj_std: dict[str, dict[str, float]] = {name: {} for name in directions}
    for layer in all_layers:
        store = g.preload_layer(roots["runs"], sorted(required), layer, row_indices=required)
        x_all, _y, _sid = g.assemble(union_roles, "acts", store)
        for name, vector in directions.items():
            projections = np.asarray(x_all, dtype=np.float64) @ vector
            proj_mean[name][str(layer)] = float(projections.mean())
            proj_std[name][str(layer)] = float(projections.std())
        del store, x_all
    lite_std = float(lite_plan["proj_std"]["probe_meanseed"])
    replicated = proj_std["probe_meanseed"][str(main_layer)]
    if abs(replicated / lite_std - 1.0) > 0.02:
        print(f"[警告] probe_meanseed@L{main_layer} 投影 σ={replicated:.4f} 与 E2-lite 计划 {lite_std:.4f} 偏差 >2%")

    # 事件锁定门：由用户通道 VAD 预计算（与生成无循环依赖）
    out_root = base / str(cfg["out_group"])
    gates_dir = out_root / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVad(events_cfg)
    n_frames = round(float(lite_plan["window_s"]) * float(lite_plan["frame_hz"]))
    frame_s = 1.0 / float(lite_plan["frame_hz"])
    gates_index: dict[str, dict] = {}
    for session in sessions:
        mask_user = cached_mask(
            out_root / "deep_masks", vad, Path(session["user_wav"]), events_cfg,
            total_dur=float(lite_plan["window_s"]), dt=DT, sample_rate=int(lite_plan["sample_rate"]),
        )
        respond = cf.respond_gate_frames(
            mask_user, DT, events_cfg["events"],
            window_s=float(cfg["gate_window_s"]), n_frames=n_frames, frame_s=frame_s,
        )
        user_speech = cf.user_speech_gate_frames(mask_user, DT, n_frames=n_frames, frame_s=frame_s)
        gate_path = gates_dir / f"{session['session_id']}.npz"
        np.savez(gate_path, respond=respond, user_speech=user_speech)
        gates_index[session["session_id"]] = {
            "path": str(gate_path),
            "sha256": _sha256(gate_path),
            "respond_active_frames": int(respond.sum()),
            "user_speech_active_frames": int(user_speech.sum()),
        }

    directions_path = out_root / "directions_confirm.npz"
    npz_payload = {name: vector.astype(np.float32) for name, vector in directions.items()}
    npz_payload["__meta__"] = np.frombuffer(
        json.dumps(
            {
                "schema": "e2-confirm-directions-v1",
                "layers": all_layers,
                "proj_mean": proj_mean,
                "proj_std": proj_std,
                "sign": "+probe_meanseed 指向 T4 label=1（complete）；轴对照方向指向各自类 1/类 SPEAK",
                "row_population": "T4 训练三种子去重并集（与 geometry proj_std 同口径）",
            },
            ensure_ascii=False,
        ).encode(),
        dtype=np.uint8,
    )
    np.savez(directions_path, **npz_payload)

    conditions = cf.build_confirm_conditions(cfg)
    plan = {
        "schema": PLAN_SCHEMA,
        "model": "moshi",
        "layer": main_layer,
        "layers_used": all_layers,
        "window_s": float(lite_plan["window_s"]),
        "sample_rate": int(lite_plan["sample_rate"]),
        "frame_hz": float(lite_plan["frame_hz"]),
        "temperature": float(lite_plan["temperature"]),
        "text_temperature": float(lite_plan["text_temperature"]),
        "top_k": int(lite_plan["top_k"]),
        "top_k_text": int(lite_plan["top_k_text"]),
        "scale_rule": "proj_std_per_layer",
        "directions_npz": str(directions_path),
        "directions_sha256": _sha256(directions_path),
        "proj_mean": proj_mean,
        "proj_std": proj_std,
        "gate_window_s": float(cfg["gate_window_s"]),
        "gates": gates_index,
        "baseline": {
            "lite_plan": str(lite_plan_path),
            "runs_root": str(baseline_root),
            "condition": "baseline",
        },
        "out_root": str(out_root),
        "conditions": conditions,
        "sessions": sessions,
    }
    plan_path = out_root / "e2_confirm.plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report_json(
        "wp_e2_confirm_plan.json",
        {
            "plan_path": str(plan_path),
            "n_sessions": len(sessions),
            "n_conditions": len(conditions),
            "n_runs": len(sessions) * len(conditions),
            "conditions": [c["name"] for c in conditions],
            "proj_std": proj_std,
            "proj_mean": proj_mean,
            "baseline_runs_root": str(baseline_root),
        },
    )
    print(
        f"确认臂计划已写 {plan_path}：{len(sessions)} 会话 × {len(conditions)} 条件 = "
        f"{len(sessions) * len(conditions)} 运行（基线复用 {baseline_root}）"
    )


if __name__ == "__main__":
    main()
