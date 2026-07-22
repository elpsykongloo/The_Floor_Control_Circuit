"""WP-E1-2 前置：E1 正式缓存（stacked_tlh_v2）计划驱动并行 Zarr 摄取。

  uv run python scripts/wp_e1_ingest.py --plan <data_root>/e1_cache_plan/e1_r1_moshi.plan.json
逐路 <out>_zarr 目标：已存在且步数/层齐全则跳过（断点续跑）；FLOOR_CIRCUIT_IO_JOBS
控制并行度（机械盘建议 2–4）。摄取后抽样回环校验（每路 1 层逐位对比堆叠源）。
wp5_ingest 的逐层回环对堆叠布局不兼容（#18(i)），E1 全量一律走本入口。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.e1.grid import io_jobs


def _dest_for(run_dir: Path) -> Path:
    return run_dir.parent.with_name(run_dir.parent.name + "_zarr") / run_dir.name


def _already_ingested(dest: Path, layers: list[int], n_steps: int) -> bool:
    import zarr

    if not (dest / "manifest.json").is_file():
        return False
    try:
        group = zarr.open_group(str(dest), mode="r")
        for layer in layers:
            if int(group[f"acts_L{layer}"].shape[0]) != n_steps:
                return False
        _ = group["mimi_latent"]
    except Exception:  # 任意损坏都判定为需重摄取
        return False
    return True


def ingest_role(task: tuple[str, list[int], int, int]) -> dict:
    from floor_circuit.cachelib.zarr_io import ingest_npy_run, read_acts

    run_dir_s, layers, n_steps, check_layer = task
    run_dir = Path(run_dir_s)
    dest = _dest_for(run_dir)
    if _already_ingested(dest, layers, n_steps):
        return {"run": run_dir.name, "status": "skipped"}
    retry_delays = (0, 2, 4, 8, 16)
    for attempt, delay_s in enumerate(retry_delays, start=1):
        if delay_s:
            time.sleep(delay_s)
        try:
            for partial in (dest.rglob("*.partial") if dest.is_dir() else ()):
                partial.unlink(missing_ok=True)
            manifest = ingest_npy_run(run_dir, dest)
            if manifest.n_steps != n_steps:
                return {
                    "run": run_dir.name,
                    "status": "failed",
                    "error": f"n_steps={manifest.n_steps}",
                }
            position = manifest.layers.index(check_layer)
            stored = read_acts(dest, check_layer)
            offset = 0
            ok = True
            for path in sorted(run_dir.glob("acts_part*.npy")):
                block = np.load(path, allow_pickle=False)
                rows = len(block)
                if not np.array_equal(stored[offset : offset + rows], block[:, position]):
                    ok = False
                    break
                offset += rows
            ok = ok and offset == n_steps
            return {
                "run": run_dir.name,
                "status": "ok" if ok else "failed",
                "roundtrip_layer": check_layer,
                "roundtrip_ok": ok,
                "attempts": attempt,
            }
        except PermissionError as exc:
            if attempt == len(retry_delays):
                return {
                    "run": run_dir.name,
                    "status": "failed",
                    "error": f"PermissionError: {exc}",
                    "attempts": attempt,
                }
        except Exception as exc:
            return {
                "run": run_dir.name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempt,
            }
    raise AssertionError("不可达的摄取重试分支")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if int(plan.get("schema_version", 0)) != 2:
        raise SystemExit("需要计划 v2")
    layers = [int(v) for v in plan["settings"]["layers"]]
    n_steps = int(plan["settings"]["expected_steps"])
    tasks = []
    for index, session in enumerate(plan["sessions"][: args.limit] if args.limit else plan["sessions"]):
        for channel in (0, 1):
            check_layer = layers[(index * 2 + channel) % len(layers)]
            tasks.append((session[f"out_agent{channel}"], layers, n_steps, check_layer))
    jobs = io_jobs(len(tasks))
    results: list[dict] = []
    if jobs == 1:
        results = [ingest_role(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(ingest_role, tasks, chunksize=4):
                results.append(result)
                if len(results) % 50 == 0:
                    print(f"摄取进度 {len(results)}/{len(tasks)}")
    counts = {status: sum(1 for r in results if r["status"] == status) for status in ("ok", "skipped", "failed")}
    report = {
        "plan_id": plan["plan_id"],
        "n_roles": len(tasks),
        **counts,
        "verdict": "passed" if counts["failed"] == 0 else "failed",
        "failures": [r for r in results if r["status"] == "failed"][:20],
    }
    write_report_json("wp_e1_ingest.json", report)
    print(f"摄取 {report['verdict']}：ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
