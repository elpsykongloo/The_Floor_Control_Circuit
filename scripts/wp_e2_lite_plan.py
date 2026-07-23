"""WP-E2L：E2-lite 方向级因果试点的计划生成器（PREREG #34；探索性）。

前置：wp_e1x_suite.py --stage geometry 已导出 <data_root>/e1x/directions/T4_L29.npz。
产出：<data_root>/e2_lite/e2_lite.plan.json —— 会话 × 条件的全部运行清单，
供 runners/moshi/run_steer.py（moshi venv）按分片执行。

会话来源 = E1 主评估集 probe_val[40:140] 前缀（探索性；causal_eval 25% 侧绝不读取）。
条件网格：主方向（T4 探针 probe_meanseed）α∈alphas_primary（含 0 = 基线，顺带缓存
R2 观察激活）；差分均值方向与随机对照方向各按配置档位。同会话跨条件共享采样种子
（公共随机数配对，缩小剂量对比方差）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1.sets import e1_sessions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_conditions(cfg: dict, direction_names: list[str]) -> list[dict]:
    """确定性条件网格；baseline = 主方向 α=0（注入向量为零，方向仅作记账）。"""
    conditions: list[dict] = []
    for alpha in [float(a) for a in cfg["alphas_primary"]]:
        name = "baseline" if alpha == 0.0 else f"probe_a{alpha:+g}"
        conditions.append(
            {
                "name": name,
                "direction": "probe_meanseed",
                "alpha": alpha,
                "cache_acts": alpha == 0.0,
            }
        )
    for alpha in [float(a) for a in cfg["alphas_diffmeans"]]:
        conditions.append(
            {"name": f"diffmeans_a{alpha:+g}", "direction": "diffmeans", "alpha": alpha, "cache_acts": False}
        )
    random_names = sorted(n for n in direction_names if n.startswith("random_r"))
    for rand in random_names:
        for alpha in [float(a) for a in cfg["alphas_random"]]:
            conditions.append(
                {"name": f"{rand}_a{alpha:+g}", "direction": rand, "alpha": alpha, "cache_acts": False}
            )
    names = [c["name"] for c in conditions]
    if len(names) != len(set(names)):
        raise ValueError("条件名重复")
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-lite 计划生成（PREREG #34）")
    parser.add_argument("--audio-root", default=None, help="覆盖音频根（默认 <data_root>/candor_extracted）")
    parser.add_argument("--n-sessions", type=int, default=None, help="覆盖会话数（默认取配置）")
    parser.add_argument(
        "--directions-npz",
        default=None,
        help=(
            "覆盖方向文件（e1x-directions-v1 schema）。默认 <data_root>/e1x/directions/"
            "T4_L29.npz（E1-X geometry 产物）；几何解剖支线的等价来源为 "
            "<data_root>/e1_probe/geometry/steering_L29.npz（#35，两条支线互为备援）"
        ),
    )
    args = parser.parse_args()

    grids = load_config("grids")
    cfg = grids["e1"]["e2_lite"]
    base = data_root()
    directions_npz = (
        Path(args.directions_npz)
        if args.directions_npz
        else base / "e1x" / "directions" / "T4_L29.npz"
    )
    if not directions_npz.is_file():
        raise SystemExit(
            f"缺方向文件 {directions_npz}：先跑 wp_e1x_suite.py --stage geometry，"
            "或用 --directions-npz 指向 wp_e1_geometry_autopsy.py spectrum 的 steering_L29.npz"
        )
    with np.load(directions_npz, allow_pickle=False) as payload:
        meta = json.loads(bytes(payload["__meta__"]).decode())
        direction_names = [k for k in payload.files if k != "__meta__"]
        for name in direction_names:
            vec = payload[name]
            if vec.ndim != 1 or not np.isfinite(vec).all():
                raise SystemExit(f"方向 {name} 非法")
    proj_std = meta["proj_std"]
    missing_std = [n for n in direction_names if n not in proj_std]
    if missing_std:
        raise SystemExit(f"方向缺投影尺度：{missing_std}")

    splits = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "splits" / "candor.json").read_text(encoding="utf-8")
    )
    sets = e1_sessions(splits)
    n_sessions = int(args.n_sessions or cfg["n_sessions"])
    session_ids = list(sets.eval)[:n_sessions]
    if len(session_ids) < n_sessions:
        raise SystemExit("E1 主评估集不足所需会话数")

    audio_root = Path(args.audio_root) if args.audio_root else base / "candor_extracted"
    user_channel = int(cfg["user_channel"])
    sessions = []
    for index, sid in enumerate(session_ids):
        wav = audio_root / sid / f"audio_ch{user_channel}.wav"
        if not wav.is_file():
            raise SystemExit(f"缺用户通道音频：{wav}")
        sessions.append(
            {
                "session_id": sid,
                "user_wav": str(wav),
                "user_channel": user_channel,
                "user_wav_sha256": _sha256(wav),
                "seed": int(cfg["base_seed"]) + index,
            }
        )

    conditions = build_conditions(cfg, direction_names)
    plan = {
        "schema": "e2_lite_plan_v1",
        "model": "moshi",
        "layer": int(cfg["layer"]),
        "window_s": float(cfg["window_s"]),
        "sample_rate": 24000,
        "frame_hz": 12.5,
        "temperature": float(cfg["temperature"]),
        "text_temperature": float(cfg["text_temperature"]),
        "top_k": int(cfg["top_k"]),
        "top_k_text": int(cfg["top_k_text"]),
        "scale_rule": str(cfg["scale"]),
        "directions_npz": str(directions_npz),
        "directions_sha256": _sha256(directions_npz),
        "proj_std": {k: float(v) for k, v in proj_std.items()},
        "cache_layers_baseline": [int(v) for v in cfg["cache_layers_baseline"]],
        "out_root": str(base / str(cfg["out_group"])),
        "conditions": conditions,
        "sessions": sessions,
    }
    plan_dir = base / str(cfg["out_group"])
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "e2_lite.plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    n_runs = len(sessions) * len(conditions)
    write_report_json(
        "wp_e2_lite_plan.json",
        {
            "plan_path": str(plan_path),
            "n_sessions": len(sessions),
            "n_conditions": len(conditions),
            "n_runs": n_runs,
            "conditions": [c["name"] for c in conditions],
            "estimated_generation_steps": n_runs * round(float(cfg["window_s"]) * 12.5),
        },
    )
    print(f"计划已写 {plan_path}：{len(sessions)} 会话 × {len(conditions)} 条件 = {n_runs} 运行")


if __name__ == "__main__":
    main()
