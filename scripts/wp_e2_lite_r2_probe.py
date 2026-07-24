"""WP-E2L-R2：R2 观察性探针（PREREG #40(d)；零新 GPU 前向，仓库 uv 环境）。

材料 = E2-lite baseline（α=0）运行顺带缓存的 L28–31 逐步激活 + 生成音频。
问题：R1 观察制式学到的 T4/T1 方向，在模型**自主生成**（R2）时是否同样可读，
且内部决策变量是否预测模型自己的接话行为——R1→R2 的闭环证据。

对齐声明：缓存行 i 在消费完输入帧 i 之后采集，观测截止 (i+1)·τ，
故行 i ≙ 决策步 i（与 R1 的 acts 行 s+1 同语义）；agent.wav 起点相对输入
时间轴平移 first_emitted_frame·τ（掩码侧已校正）。

三项分析：
  (i)  用户合格 offset 处的 T4 决策变量 → 预测 agent 是否 1.2 s 内合格 onset
       （AUC）+ 与接话延迟的 Spearman；
  (ii) agent 非发声步的 T1_d800 决策变量 → 预测自身 800 ms 内 onset（AUC）；
  (iii) 自身 onset 对齐的决策变量平均轨迹（±2 s）。

产出 reports/wp_e2_lite_r2_probe.json + reports/e2_lite_r2_观察探针报告.md。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wp_e1_probe_grid as engine
from _bootstrap import REPO_ROOT, write_report_json
from wp_e2_lite_deep import _resolve_runs_root

from floor_circuit.config import data_root, load_config
from floor_circuit.e1 import probe_gpu as pg
from floor_circuit.e1x import trajectory as tx
from floor_circuit.e1x.mask_cache import cached_mask, shifted_agent_mask
from floor_circuit.events.detect import qualified_offsets, qualified_onsets
from floor_circuit.events.vad import SileroVad

SCHEMA = "e2_lite_r2_probe_v1"
DT = 0.01
FRAME_S = 0.08
T1_HORIZON_STEPS = 10  # 800 ms
WINDOW_STEPS = 25


def _mask_to_frames(mask_dt: np.ndarray, n_frames: int) -> np.ndarray:
    per = FRAME_S / DT
    frames = np.zeros(n_frames, dtype=bool)
    for frame in range(n_frames):
        lo = round(frame * per)
        hi = round((frame + 1) * per)
        if lo >= len(mask_dt):
            break
        frames[frame] = bool(mask_dt[lo : max(hi, lo + 1)].any())
    return frames


def _logit_series(probes: list[pg.LinearProbe], acts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(acts, dtype=np.float32)
    series = []
    for probe in probes:
        z = (matrix - probe.mean) / probe.scale
        series.append((z @ probe.weight.T + probe.bias)[:, 0].astype(np.float64))
    return np.mean(series, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 观察性探针（PREREG #40(d)）")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--runs-root", default=None, help="baseline 运行根（默认自动优先 *_optimized）")
    parser.add_argument("--layer", type=int, default=29)
    args = parser.parse_args()

    grids = load_config("grids")
    lite_cfg = grids["e1"]["e2_lite"]
    analysis_cfg = lite_cfg["analysis"]
    events_cfg = load_config("events")
    plan_path = Path(args.plan) if args.plan else data_root() / str(lite_cfg["out_group"]) / "e2_lite.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs_root = _resolve_runs_root(plan, args.runs_root)
    layer = int(args.layer)
    if layer not in [int(v) for v in plan["cache_layers_baseline"]]:
        raise SystemExit(f"层 {layer} 不在 baseline 缓存层 {plan['cache_layers_baseline']} 中")
    roots = engine._roots()
    seeds = [int(s) for s in grids["e1"]["probe"]["seeds"]]
    probes_t4 = [engine._load_fit(engine._fit_path(roots, "T4", layer, s))[0] for s in seeds]
    probes_t1 = [engine._load_fit(engine._fit_path(roots, "T1_d800", layer, s))[0] for s in seeds]
    vad = SileroVad(events_cfg)
    mask_root = Path(plan["out_root"]) / "deep_masks"
    ev = events_cfg["events"]
    respond_window = float(analysis_cfg["response_window_s"])
    latency_max = float(analysis_cfg["latency_max_s"])

    offset_scores: list[float] = []
    offset_labels: list[int] = []
    offset_latencies: list[tuple[float, float]] = []
    offset_sessions: list[str] = []
    t1_scores: list[float] = []
    t1_labels: list[int] = []
    series_t4_by_role: dict[tuple[str, int], np.ndarray] = {}
    series_t1_by_role: dict[tuple[str, int], np.ndarray] = {}
    onset_anchors: list[tuple[str, int, int]] = []
    per_session_stats: dict[str, dict] = {}
    skipped: list[str] = []

    for session in plan["sessions"]:
        sid = session["session_id"]
        run_dir = runs_root / f"{sid}__baseline"
        manifest_path = run_dir / "manifest.json"
        acts_path = run_dir / f"acts_L{layer}.npy"
        if not (manifest_path.is_file() and acts_path.is_file()):
            skipped.append(sid)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("completed"):
            skipped.append(sid)
            continue
        acts = np.load(acts_path, allow_pickle=False)
        n_frames = int(manifest["n_frames_in"])
        if acts.shape != (n_frames, probes_t4[0].mean.shape[0]):
            raise SystemExit(f"{sid} 激活形状 {acts.shape} 与帧数 {n_frames} 不符——先核对缓存布局")
        z_t4 = _logit_series(probes_t4, acts)
        z_t1 = _logit_series(probes_t1, acts)
        series_t4_by_role[(sid, 0)] = z_t4
        series_t1_by_role[(sid, 0)] = z_t1

        mask_user = cached_mask(
            mask_root, vad, Path(session["user_wav"]), events_cfg,
            total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
        )
        agent_local = cached_mask(
            mask_root, vad, run_dir / "agent.wav", events_cfg,
            total_dur=float(plan["window_s"]), dt=DT, sample_rate=int(plan["sample_rate"]),
        )
        first_emitted = int(manifest.get("first_emitted_frame") or 0)
        mask_agent = shifted_agent_mask(agent_local, first_emitted, FRAME_S, DT)
        agent_onsets = qualified_onsets(mask_agent, DT, ev["onset_pre_silence_s"])
        user_offsets = qualified_offsets(mask_user, DT, ev["offset_post_silence_s"])

        hits = 0
        for t_off in user_offsets:
            step = min(n_frames - 1, int(t_off / FRAME_S))
            candidates = [t for t in agent_onsets if t > t_off]
            latency = (candidates[0] - t_off) if candidates else None
            label = int(latency is not None and latency <= respond_window)
            hits += label
            offset_scores.append(float(z_t4[step]))
            offset_labels.append(label)
            offset_sessions.append(sid)
            if latency is not None and latency <= latency_max:
                offset_latencies.append((float(z_t4[step]), float(latency)))

        agent_frames = _mask_to_frames(mask_agent, n_frames)
        for step in range(n_frames - T1_HORIZON_STEPS):
            if agent_frames[step]:
                continue
            t1_scores.append(float(z_t1[step]))
            t1_labels.append(int(agent_frames[step + 1 : step + 1 + T1_HORIZON_STEPS].any()))
        for step in np.flatnonzero(agent_frames & ~np.concatenate([[False], agent_frames[:-1]])):
            if WINDOW_STEPS <= int(step) < n_frames - WINDOW_STEPS:
                onset_anchors.append((sid, 0, int(step)))
        per_session_stats[sid] = {
            "n_user_offsets": len(user_offsets),
            "response_rate": (hits / len(user_offsets)) if user_offsets else None,
            "n_agent_onsets": len(agent_onsets),
        }

    if not offset_labels:
        raise SystemExit(f"没有可用 baseline 运行（缺 {len(skipped)} 个 / 无激活缓存）")

    labels_array = np.asarray(offset_labels)
    auc_offset = None
    if (labels_array == 1).any() and (labels_array == 0).any():
        auc_offset = float(pg._binary_auc(labels_array, np.asarray(offset_scores)))
    spearman = None
    if len(offset_latencies) >= 10:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(
            [a for a, _ in offset_latencies], [b for _, b in offset_latencies]
        )
        spearman = {"rho": float(rho), "p": float(pval), "n": len(offset_latencies)}
    t1_array = np.asarray(t1_labels)
    auc_t1 = None
    if (t1_array == 1).any() and (t1_array == 0).any():
        auc_t1 = float(pg._binary_auc(t1_array, np.asarray(t1_scores)))

    onset_curves = {}
    for name, series in (("t4_logit", series_t4_by_role), ("t1d800_logit", series_t1_by_role)):
        if series and onset_anchors:
            matrix, _idx = tx.aligned_matrix(series, onset_anchors, WINDOW_STEPS)
            onset_curves[name] = [float(v) for v in np.nanmean(matrix, axis=0)]

    payload = {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "layer": layer,
        "alignment": "缓存行 i ≙ 决策步 i（观测截止 (i+1)·τ）；agent 掩码已平移 first_emitted·τ",
        "n_sessions_used": len(per_session_stats),
        "skipped_sessions": skipped,
        "user_offset_prediction": {
            "n_offsets": len(offset_labels),
            "positive_rate": float(labels_array.mean()),
            "auc_t4_logit_predicts_response": auc_offset,
            "spearman_t4_logit_vs_latency": spearman,
        },
        "t1_self_onset_prediction": {
            "n_listening_steps": len(t1_labels),
            "positive_rate": float(t1_array.mean()) if len(t1_array) else None,
            "horizon_steps": T1_HORIZON_STEPS,
            "auc_t1d800_logit": auc_t1,
        },
        "own_onset_aligned": {
            "offsets_steps": list(range(-WINDOW_STEPS, WINDOW_STEPS + 1)),
            "n_onsets": len(onset_anchors),
            "curves": onset_curves,
        },
        "per_session": per_session_stats,
    }
    write_report_json("wp_e2_lite_r2_probe.json", payload)

    lines = [
        "# R2 观察性探针报告（PREREG #40(d)；探索性）",
        "",
        f"- 材料：{len(per_session_stats)} 个 baseline 运行的 L{layer} 逐步激活（R1 训练探针零样本读出）",
        f"- (i) 用户 offset 处 T4 决策变量预测 agent 1.2 s 内接话：AUC = {auc_offset}"
        f"（{len(offset_labels)} 个 offset，阳性率 {labels_array.mean():.3f}）",
        f"- (i) T4 决策变量 vs 接话延迟 Spearman：{spearman}",
        f"- (ii) 非发声步 T1_d800 决策变量预测自身 800 ms 内 onset：AUC = {auc_t1}"
        f"（{len(t1_labels)} 步）",
        f"- (iii) 自身 onset 对齐轨迹：{len(onset_anchors)} 个 onset，曲线见 JSON",
        "",
        "对齐声明：缓存行 i ≙ 决策步 i；agent 掩码已按 first_emitted_frame 平移。",
    ]
    report_path = Path(REPO_ROOT) / "reports" / "e2_lite_r2_观察探针报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"R2 观察探针完成：{report_path}")


if __name__ == "__main__":
    main()
