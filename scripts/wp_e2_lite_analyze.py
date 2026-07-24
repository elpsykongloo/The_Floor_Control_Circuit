"""WP-E2L：E2-lite 行为分析（PREREG #34；仓库 uv 环境，依赖 silero VAD）。

对每个已完成运行：Silero VAD（冻结参数）→ IPU 合并 → dt 栅格掩码 →
floor 行为指标（e1x/behavior.py）。逐运行 VAD 结果缓存为 JSON，重复执行只聚合。

聚合：同会话跨条件配对差（条件 − baseline）→ 会话级 bootstrap CI + 符号计数；
主方向剂量-反应（α 单调性）；随机方向对照的效应量参照（自由度有限，仅作
数量级参照）。产出 reports/wp_e2_lite_summary.json + reports/e2_lite_行为报告.md。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from pathlib import Path

import numpy as np
from _bootstrap import REPO_ROOT, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.e1x.behavior import METRIC_KEYS, bootstrap_mean_ci, floor_metrics, paired_deltas
from floor_circuit.events.ipu import build_ipus
from floor_circuit.events.vad import SileroVad, rasterize

VAD_CACHE_SCHEMA = "e2_lite_vad_v2"
USER_MASK_SCHEMA = "e2_lite_user_mask_v1"
VAD_LOOKAHEAD_S = 1.0
_WORKER_VAD: SileroVad | None = None
_WORKER_EVENTS_CFG: dict | None = None


def _mask_from_wav(
    vad: SileroVad, wav_path: Path, sample_rate_expected: int, total_dur: float, dt: float, ipu_gap: float
) -> np.ndarray:
    import soundfile as sf

    # 只读分析窗及边界判定所需的前视；原始会话常长达数十分钟。
    max_frames = round((float(total_dur) + VAD_LOOKAHEAD_S) * sample_rate_expected)
    wav, sr = sf.read(
        str(wav_path),
        dtype="float32",
        always_2d=False,
        frames=max_frames,
    )
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != sample_rate_expected:
        raise ValueError(f"{wav_path} 采样率 {sr} ≠ {sample_rate_expected}")
    segs = vad.segments(wav, sr)
    ipus = build_ipus(segs, ipu_gap)
    return rasterize(ipus, dt, total_dur)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _analysis_fingerprint(
    plan: dict,
    events_cfg: dict,
    analysis_cfg: dict,
) -> str:
    return _canonical_sha256(
        {
            "sample_rate": int(plan["sample_rate"]),
            "window_s": float(plan["window_s"]),
            "dt": 0.01,
            "vad_lookahead_s": VAD_LOOKAHEAD_S,
            "vad": events_cfg["vad"],
            "ipu": events_cfg["ipu"],
            "events": events_cfg["events"],
            "analysis": analysis_cfg,
        }
    )


def _agent_source_identity(run_dir: Path, manifest: dict) -> dict:
    stat = (run_dir / "agent.wav").stat()
    identity = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    digest = manifest.get("agent_wav_sha256")
    if digest:
        identity["sha256"] = str(digest)
    return identity


def _user_mask_cache_paths(
    cache_root: Path,
    manifest: dict,
    analysis_sha256: str,
) -> tuple[Path, Path, dict]:
    contract = {
        "schema": USER_MASK_SCHEMA,
        "session_id": manifest["session_id"],
        "user_wav_sha256": manifest["user_wav_sha256"],
        "analysis_sha256": analysis_sha256,
    }
    key = _canonical_sha256(contract)[:16]
    stem = f"{manifest['session_id']}__{key}"
    return cache_root / f"{stem}.npy", cache_root / f"{stem}.json", contract


def _load_user_mask(
    cache_root: Path,
    manifest: dict,
    analysis_sha256: str,
    expected_steps: int,
) -> np.ndarray | None:
    npy_path, meta_path, expected = _user_mask_cache_paths(
        cache_root,
        manifest,
        analysis_sha256,
    )
    if not npy_path.is_file() or not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        mask = np.load(npy_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    if mask.shape != (expected_steps,) or mask.dtype != np.bool_:
        return None
    if metadata.get("mask_sha256") != hashlib.sha256(mask.tobytes()).hexdigest():
        return None
    return mask


def _save_user_mask(
    cache_root: Path,
    manifest: dict,
    analysis_sha256: str,
    mask: np.ndarray,
) -> None:
    npy_path, meta_path, metadata = _user_mask_cache_paths(
        cache_root,
        manifest,
        analysis_sha256,
    )
    values = np.asarray(mask, dtype=bool)
    metadata["shape"] = list(values.shape)
    metadata["mask_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    _write_npy_atomic(npy_path, values)
    _write_json_atomic(meta_path, metadata)


def _compute_payload(
    run_dir: Path,
    manifest: dict,
    plan: dict,
    events_cfg: dict,
    analysis_cfg: dict,
    vad: SileroVad,
    mask_user: np.ndarray,
    analysis_sha256: str,
    agent_source: dict,
) -> dict:
    dt = 0.01
    total_dur = float(plan["window_s"])
    ipu_gap = float(events_cfg["ipu"]["merge_gap_s"])
    mask_agent = _mask_from_wav(
        vad,
        run_dir / "agent.wav",
        int(plan["sample_rate"]),
        total_dur,
        dt,
        ipu_gap,
    )
    n = min(len(mask_agent), len(mask_user))
    metrics = floor_metrics(
        mask_user[:n],
        mask_agent[:n],
        dt,
        events_cfg["events"],
        response_window_s=float(analysis_cfg["response_window_s"]),
        latency_max_s=float(analysis_cfg["latency_max_s"]),
        yield_windows_s=[float(v) for v in analysis_cfg["yield_windows_s"]],
    )
    payload = {
        "schema": VAD_CACHE_SCHEMA,
        "run_id": manifest["run_id"],
        "session_id": manifest["session_id"],
        "condition": manifest["condition"]["name"],
        "direction": manifest["condition"]["direction"],
        "alpha": manifest["condition"]["alpha"],
        "execution_backend": manifest.get("execution_backend", "eager_reference"),
        "analysis_sha256": analysis_sha256,
        "agent_source": agent_source,
        "metrics": metrics,
    }
    _write_json_atomic(run_dir / "behavior.json", payload)
    return payload


def _init_vad_worker(events_cfg: dict) -> None:
    global _WORKER_EVENTS_CFG, _WORKER_VAD
    import torch

    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    _WORKER_EVENTS_CFG = events_cfg
    _WORKER_VAD = SileroVad(events_cfg)


def _analyze_worker_task(task: dict) -> dict:
    if _WORKER_VAD is None or _WORKER_EVENTS_CFG is None:
        raise RuntimeError("VAD worker 尚未初始化")
    return _compute_payload(
        Path(task["run_dir"]),
        task["manifest"],
        task["plan"],
        _WORKER_EVENTS_CFG,
        task["analysis_cfg"],
        _WORKER_VAD,
        task["mask_user"],
        task["analysis_sha256"],
        task["agent_source"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-lite 行为分析（PREREG #34）")
    parser.add_argument("--plan", default=None, help="默认 <data_root>/e2_lite/e2_lite.plan.json")
    parser.add_argument("--runs-root", default=None, help="覆盖 runs/ 根目录")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="agent 语音活动检测进程数",
    )
    parser.add_argument(
        "--report-tag",
        default="",
        help="报告文件名后缀，例如 optimized；用于隔离候选与参考报告",
    )
    parser.add_argument("--force", action="store_true", help="忽略逐运行 VAD 缓存重算")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers 必须至少为 1")
    report_tag = str(args.report_tag).strip()
    if report_tag and not all(char.isalnum() or char in {"-", "_"} for char in report_tag):
        raise SystemExit("--report-tag 只允许字母、数字、连字符和下划线")
    report_suffix = f"_{report_tag}" if report_tag else ""

    grids = load_config("grids")
    cfg = grids["e1"]["e2_lite"]
    events_cfg = load_config("events")
    plan_path = Path(args.plan) if args.plan else data_root() / str(cfg["out_group"]) / "e2_lite.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs_root = Path(args.runs_root) if args.runs_root else Path(plan["out_root"]) / "runs"
    vad = SileroVad(events_cfg)
    analysis_sha256 = _analysis_fingerprint(plan, events_cfg, cfg["analysis"])
    user_mask_root = Path(plan["out_root"]) / "vad_user_masks"
    expected_mask_steps = round(float(plan["window_s"]) / 0.01)

    records: list[dict] = []
    missing: list[str] = []
    user_mask_cache: dict[str, np.ndarray] = {}
    user_mask_cache_hits = 0
    user_mask_cache_misses = 0
    tasks: list[dict] = []
    for session in plan["sessions"]:
        for condition in plan["conditions"]:
            run_id = f"{session['session_id']}__{condition['name']}"
            run_dir = runs_root / run_id
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                missing.append(run_id)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("completed"):
                missing.append(run_id)
                continue
            if not (run_dir / "agent.wav").is_file():
                missing.append(run_id)
                continue
            agent_source = _agent_source_identity(run_dir, manifest)
            cache_path = run_dir / "behavior.json"
            if cache_path.is_file() and not args.force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cached = {}
                if (
                    cached.get("schema") == VAD_CACHE_SCHEMA
                    and cached.get("analysis_sha256") == analysis_sha256
                    and cached.get("agent_source") == agent_source
                ):
                    records.append(cached)
                    continue
            user_path = Path(manifest["user_wav"])
            user_key = f"{manifest['user_wav_sha256']}:{analysis_sha256}:{user_path.resolve()}"
            if user_key not in user_mask_cache:
                mask = _load_user_mask(
                    user_mask_root,
                    manifest,
                    analysis_sha256,
                    expected_mask_steps,
                )
                if mask is None:
                    user_mask_cache_misses += 1
                    mask = _mask_from_wav(
                        vad,
                        user_path,
                        int(plan["sample_rate"]),
                        float(plan["window_s"]),
                        0.01,
                        float(events_cfg["ipu"]["merge_gap_s"]),
                    )
                    _save_user_mask(
                        user_mask_root,
                        manifest,
                        analysis_sha256,
                        mask,
                    )
                else:
                    user_mask_cache_hits += 1
                user_mask_cache[user_key] = mask
            tasks.append(
                {
                    "run_dir": str(run_dir),
                    "manifest": manifest,
                    "plan": {
                        "window_s": plan["window_s"],
                        "sample_rate": plan["sample_rate"],
                    },
                    "analysis_cfg": cfg["analysis"],
                    "mask_user": user_mask_cache[user_key],
                    "analysis_sha256": analysis_sha256,
                    "agent_source": agent_source,
                }
            )
    if tasks:
        if args.workers == 1:
            for index, task in enumerate(tasks):
                records.append(
                    _compute_payload(
                        Path(task["run_dir"]),
                        task["manifest"],
                        task["plan"],
                        events_cfg,
                        task["analysis_cfg"],
                        vad,
                        task["mask_user"],
                        task["analysis_sha256"],
                        task["agent_source"],
                    )
                )
                if (index + 1) % 10 == 0 or index + 1 == len(tasks):
                    print(f"语音活动检测 {index + 1}/{len(tasks)}")
        else:
            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=_init_vad_worker,
                initargs=(events_cfg,),
            ) as executor:
                for index, record in enumerate(executor.map(_analyze_worker_task, tasks, chunksize=1)):
                    records.append(record)
                    if (index + 1) % 10 == 0 or index + 1 == len(tasks):
                        print(f"并行语音活动检测 {index + 1}/{len(tasks)}")
    if not records:
        raise SystemExit(f"没有已完成的运行（缺 {len(missing)} 个）；先执行 run_steer.py")

    per_session: dict[str, dict[str, dict]] = {}
    for record in records:
        per_session.setdefault(record["session_id"], {})[record["condition"]] = record["metrics"]

    conditions = [c["name"] for c in plan["conditions"]]
    baseline = "baseline"
    if baseline not in conditions:
        raise SystemExit("计划缺 baseline（α=0）条件")
    aggregate: dict[str, dict] = {}
    for condition in conditions:
        if condition == baseline:
            continue
        aggregate[condition] = {}
        for metric in METRIC_KEYS:
            deltas = paired_deltas(per_session, condition, baseline, metric)
            aggregate[condition][metric] = bootstrap_mean_ci(deltas)

    # 主方向剂量-反应：α 与配对差的 Spearman（跨会话拼接）
    dose: dict[str, dict] = {}
    primary_conditions = [c for c in plan["conditions"] if c["direction"] == "probe_meanseed" and c["alpha"] != 0.0]
    for metric in METRIC_KEYS:
        pairs: list[tuple[float, float]] = []
        for condition in primary_conditions:
            for delta in paired_deltas(per_session, condition["name"], baseline, metric):
                pairs.append((float(condition["alpha"]), delta))
        if len(pairs) >= 10:
            from scipy.stats import spearmanr

            rho, pval = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            dose[metric] = {"spearman_rho": float(rho), "p": float(pval), "n": len(pairs)}
        else:
            dose[metric] = {"n": len(pairs)}

    # 随机方向对照：|Δ| 参照（同 |α|）
    control: dict[str, dict] = {}
    max_alpha = max(abs(float(c["alpha"])) for c in primary_conditions) if primary_conditions else None
    if max_alpha is not None:
        for metric in METRIC_KEYS:
            primary_abs = [
                abs(aggregate[c["name"]][metric]["mean"])
                for c in primary_conditions
                if abs(float(c["alpha"])) == max_alpha and aggregate[c["name"]][metric]["mean"] is not None
            ]
            random_abs = [
                abs(aggregate[c["name"]][metric]["mean"])
                for c in plan["conditions"]
                if c["direction"].startswith("random_r")
                and abs(float(c["alpha"])) == max_alpha
                and c["name"] in aggregate
                and aggregate[c["name"]][metric]["mean"] is not None
            ]
            control[metric] = {
                "primary_abs_mean_delta": primary_abs,
                "random_abs_mean_deltas": random_abs,
                "note": "随机对照自由度有限，仅作数量级参照",
            }

    payload = {
        "schema": "e2_lite_summary_v1",
        "plan": str(plan_path),
        "runs_root": str(runs_root),
        "workers": int(args.workers),
        "analysis_sha256": analysis_sha256,
        "user_mask_cache_hits": user_mask_cache_hits,
        "user_mask_cache_misses": user_mask_cache_misses,
        "n_runs_analyzed": len(records),
        "n_runs_missing": len(missing),
        "missing_runs": missing[:20],
        "paired_deltas_vs_baseline": aggregate,
        "dose_response_primary": dose,
        "random_control_reference": control,
    }
    write_report_json(f"wp_e2_lite_summary{report_suffix}.json", payload)

    lines = [
        "# E2-lite 行为报告（PREREG #34；探索性）",
        "",
        f"- 已分析运行：{len(records)}；缺失：{len(missing)}",
        f"- 注入：L{plan['layer']}，h ← h + α·s_v·v̂（s_v = 训练行投影标准差）",
        "",
        "## 主方向剂量-反应（配对差 vs baseline）",
        "",
    ]
    for condition in [c["name"] for c in primary_conditions]:
        lines.append(f"### {condition}")
        for metric in METRIC_KEYS:
            stats = aggregate[condition][metric]
            if stats["mean"] is None:
                lines.append(f"- {metric}：无有效配对")
            else:
                lines.append(
                    f"- {metric}：Δ={stats['mean']:+.4f} "
                    f"[{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}]"
                    f"（n={stats['n']}，+{stats['n_pos']}/−{stats['n_neg']}）"
                )
        lines.append("")
    lines += ["## α 单调性（Spearman）", ""]
    for metric, stats in dose.items():
        if "spearman_rho" in stats:
            lines.append(f"- {metric}：ρ={stats['spearman_rho']:+.3f}（p={stats['p']:.2e}，n={stats['n']}）")
        else:
            lines.append(f"- {metric}：样本不足（n={stats['n']}）")
    lines += ["", "## 差分均值方向与随机对照", ""]
    for condition in conditions:
        if condition == baseline or condition in {c["name"] for c in primary_conditions}:
            continue
        stats = aggregate.get(condition, {}).get("agent_speech_frac")
        if stats and stats["mean"] is not None:
            lines.append(
                f"- {condition}：agent_speech_frac Δ={stats['mean']:+.4f} "
                f"[{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}]（n={stats['n']}）"
            )
    report_path = Path(REPO_ROOT) / "reports" / f"e2_lite_行为报告{report_suffix}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"分析完成：{report_path}（缺失 {len(missing)} 个运行）")


if __name__ == "__main__":
    main()
