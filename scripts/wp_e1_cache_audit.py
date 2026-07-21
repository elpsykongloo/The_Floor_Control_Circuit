"""WP-E1-1：E1 缓存产物审计（独立字面量，不 import runner；PREREG #16(d)）。

对照计划 v2 全量核验缓存目录：
  uv run python scripts/wp_e1_cache_audit.py --plan <data_root>/e1_cache_plan/e1_r1_moshi.plan.json
逐路检查 manifest 契约、计划绑定、输入前缀指纹、分片头形状（mmap 零拷贝）、文件尺寸，
可选抽样有限值检查（--sample-finite，默认每路末分片 1 个）。输出
reports/wp_e1_cache_audit.json 与聚合遥测；任何失败以非零码退出。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

# 独立字面量：与 runner 声明逐字一致，但不从 runner import（审计独立性原则）
EXPECTED_TIME_ALIGNMENT = {
    "initial_token_position": 0,
    "acts_observed_through_offset_steps": 0,
}
EXPECTED_LATENT_KIND = "pre_quantization_continuous"
EXPECTED_LAYOUT = "stacked_tlh_v2"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_role(
    out_dir: Path,
    plan: dict,
    session: dict,
    channel: int,
    sample_finite: int,
) -> list[str]:
    """返回该路的全部问题（空列表 = 通过）。"""
    problems: list[str] = []
    settings = plan["settings"]
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return [f"缺少 manifest：{manifest_path}"]
    try:
        payload = _load_json(manifest_path)
    except json.JSONDecodeError as exc:
        return [f"manifest 不可解析：{manifest_path}（{exc}）"]
    execution = payload.get("extra", {}).get("execution", {})
    output_files = payload.get("extra", {}).get("output_files", {})
    e1 = payload.get("extra", {}).get("e1", {})

    if int(payload.get("schema_version", 0)) != 2:
        problems.append(f"schema_version={payload.get('schema_version')} ≠ 2")
    if payload.get("code_version") not in set(plan.get("accepted_code_versions", [])):
        problems.append(f"code_version 不在计划接受集：{payload.get('code_version')}")
    if payload.get("text_mode") != "greedy":
        problems.append(f"text_mode={payload.get('text_mode')} ≠ greedy")
    if payload.get("mimi_latent") is not True:
        problems.append("mimi_latent ≠ true")
    if payload.get("layers") != [int(v) for v in settings["layers"]]:
        problems.append("layers 与计划不一致")
    if int(payload.get("n_steps", -1)) != int(settings["expected_steps"]):
        problems.append(f"n_steps={payload.get('n_steps')} ≠ {settings['expected_steps']}")
    if execution.get("time_alignment") != EXPECTED_TIME_ALIGNMENT:
        problems.append(f"time_alignment 异常：{execution.get('time_alignment')}")
    if execution.get("latent_kind") != EXPECTED_LATENT_KIND:
        problems.append(f"latent_kind 异常：{execution.get('latent_kind')}")
    if float(execution.get("max_seconds", -1)) != float(settings["window_seconds"]):
        problems.append(f"max_seconds={execution.get('max_seconds')} ≠ {settings['window_seconds']}")
    if str(e1.get("plan_id")) != str(plan["plan_id"]):
        problems.append(f"plan_id 不符：{e1.get('plan_id')}")
    if str(e1.get("cohort")) != str(session["cohort"]):
        problems.append(f"cohort 不符：{e1.get('cohort')} ≠ {session['cohort']}")
    if int(e1.get("agent_channel", -1)) != channel:
        problems.append(f"agent_channel 不符：{e1.get('agent_channel')} ≠ {channel}")
    if str(e1.get("activation_layout")) != EXPECTED_LAYOUT:
        problems.append(f"activation_layout={e1.get('activation_layout')} ≠ {EXPECTED_LAYOUT}")
    expected_prefix = {"ch0": dict(session["prefix_ch0"]), "ch1": dict(session["prefix_ch1"])}
    if e1.get("input_prefix") != expected_prefix:
        problems.append("input_prefix 指纹与计划不符")

    if not isinstance(output_files, dict) or not output_files:
        problems.append("output_files 缺失")
        return problems
    for name, expected_size in output_files.items():
        path = out_dir / name
        if not path.is_file():
            problems.append(f"缺文件：{name}")
        elif path.stat().st_size != int(expected_size):
            problems.append(f"文件尺寸不符：{name}")
    part_names = sorted(
        name for name in output_files if name.startswith("acts_part") and name.endswith(".npy")
    )
    if len(part_names) != int(settings["expected_parts"]):
        problems.append(f"堆叠分片数 {len(part_names)} ≠ {settings['expected_parts']}")
        return problems

    rows_total = 0
    n_layers = len(settings["layers"])
    hidden = int(settings["expected_hidden_dim"])
    for name in part_names:
        path = out_dir / name
        if not path.is_file():
            continue
        header = np.load(path, allow_pickle=False, mmap_mode="r")
        if header.ndim != 3 or header.shape[1] != n_layers or header.shape[2] != hidden:
            problems.append(f"{name} 形状 {tuple(header.shape)} ≠ [*, {n_layers}, {hidden}]")
        if header.dtype != np.float16:
            problems.append(f"{name} dtype {header.dtype} ≠ float16")
        rows_total += int(header.shape[0])
        del header
    if rows_total != int(settings["expected_steps"]):
        problems.append(f"分片行合计 {rows_total} ≠ {settings['expected_steps']}")
    for name in part_names[-max(0, int(sample_finite)):]:
        block = np.load(out_dir / name, allow_pickle=False)
        if not np.isfinite(block.astype(np.float32)).all():
            problems.append(f"{name} 含非有限值")
    return problems


def summarize_telemetry(manifests: list[dict]) -> dict:
    steps_per_second = []
    temps = []
    peaks = []
    bytes_total = 0
    unavailable = 0
    for payload in manifests:
        telemetry = payload.get("extra", {}).get("e1", {}).get("telemetry", {})
        if telemetry.get("steps_per_second") is not None:
            steps_per_second.append(float(telemetry["steps_per_second"]))
        if telemetry.get("temperature_max_c") is not None:
            temps.append(float(telemetry["temperature_max_c"]))
        elif telemetry:
            unavailable += 1
        if telemetry.get("peak_memory_allocated_bytes") is not None:
            peaks.append(int(telemetry["peak_memory_allocated_bytes"]))
        bytes_total += int(telemetry.get("output_bytes", 0))
    def _stats(values: list[float]) -> dict | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "max": float(array.max()),
        }
    return {
        "steps_per_second": _stats(steps_per_second),
        "temperature_max_c": _stats(temps),
        "temperature_unavailable_roles": unavailable,
        "peak_memory_allocated_bytes_max": max(peaks) if peaks else None,
        "output_bytes_total": bytes_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="主计划或分片计划 JSON 路径")
    ap.add_argument("--sample-finite", type=int, default=1, help="每路抽样做有限值检查的末尾分片数")
    ap.add_argument("--limit", type=int, default=None, help="只审计前 N 个会话（联调用）")
    args = ap.parse_args()
    plan = _load_json(Path(args.plan))
    if int(plan.get("schema_version", 0)) != 2:
        raise SystemExit(f"计划 schema_version={plan.get('schema_version')} ≠ 2")
    sessions = plan["sessions"][: args.limit] if args.limit else plan["sessions"]
    failures: dict[str, list[str]] = {}
    manifests: list[dict] = []
    n_roles = 0
    for session in sessions:
        for channel in (0, 1):
            n_roles += 1
            out_dir = Path(session[f"out_agent{channel}"])
            problems = audit_role(out_dir, plan, session, channel, args.sample_finite)
            if problems:
                failures[out_dir.name] = problems
            else:
                manifests.append(_load_json(out_dir / "manifest.json"))
    report = {
        "plan_id": plan["plan_id"],
        "plan_path": str(args.plan),
        "n_sessions": len(sessions),
        "n_roles": n_roles,
        "n_passed": n_roles - len(failures),
        "n_failed": len(failures),
        "verdict": "passed" if not failures else "failed",
        "telemetry": summarize_telemetry(manifests),
        "failures": {name: problems[:8] for name, problems in sorted(failures.items())[:50]},
    }
    write_report_json("wp_e1_cache_audit.json", report)
    print(f"审计 {report['verdict']}：{report['n_passed']}/{n_roles} 路通过")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
