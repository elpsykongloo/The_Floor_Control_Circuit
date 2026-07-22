"""WP5：runner npy 分片 → zarr ingest + manifest 校验 + 回环抽查。

用法：
  uv run python scripts/wp5_ingest.py --src <runner 输出目录> [--dest <zarr 目录>]
  uv run python scripts/wp5_ingest.py --batch <父目录> [--dest-root <zarr 父目录>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.cachelib.zarr_io import ingest_npy_run, read_acts, read_array


def ingest_one(src: Path, dest: Path | None) -> dict:
    dest = dest or src.with_name(src.name + "_zarr")
    manifest = ingest_npy_run(src, dest)
    checks: dict[str, bool] = {}
    stacked_parts = sorted(src.glob("acts_part*.npy"))
    stacked_reference = (
        np.concatenate([np.load(path, allow_pickle=False) for path in stacked_parts], axis=0)
        if stacked_parts
        else None
    )
    for layer in manifest.layers:
        stored = read_acts(dest, layer)
        if stacked_reference is not None:  # stacked_tlh_v2（#16(d)）：逐层切片对比
            position = manifest.layers.index(layer)
            reference = stacked_reference[:, position]
        else:
            parts = sorted(src.glob(f"acts_L{layer}_part*.npy"))
            reference = np.concatenate(
                [np.load(path, allow_pickle=False) for path in parts], axis=0
            )
        checks[f"acts_L{layer}"] = bool(np.array_equal(stored, reference.astype(np.float16)))
    if manifest.mimi_latent:
        stored = read_array(dest, "mimi_latent")
        parts = sorted(src.glob("mimi_latent_part*.npy"))
        reference = np.concatenate([np.load(path, allow_pickle=False) for path in parts], axis=0)
        checks["mimi_latent"] = bool(np.array_equal(stored, reference.astype(np.float16)))
    ok = bool(checks) and all(checks.values())
    return {
        "src": str(src),
        "dest": str(dest),
        "session_id": manifest.session_id,
        "layers": manifest.layers,
        "n_steps": manifest.n_steps,
        "hidden_dim": manifest.hidden_dim,
        "roundtrip_ok": ok,
        "roundtrip_arrays": checks,
    }


def ingest_batch(batch: Path, dest_root: Path | None = None) -> list[dict]:
    """逐目录摄取；默认目标为同级 ``<批次名>_zarr/<run>``。"""
    dest_root = dest_root or batch.with_name(batch.name + "_zarr")
    results: list[dict] = []
    for sub in sorted(path for path in batch.iterdir() if path.is_dir() and not path.name.endswith("_zarr")):
        if not (sub / "manifest.json").exists():
            results.append({"src": str(sub), "error": "缺少 manifest.json，runner 输出未完成"})
            continue
        try:
            results.append(ingest_one(sub, dest_root / sub.name))
        except Exception as exc:
            results.append({"src": str(sub), "error": repr(exc)})
    if not results:
        results.append({"src": str(batch), "error": "批次目录中没有可摄取的 run"})
    return results


def _failed(result: dict) -> bool:
    return "error" in result or not result.get("roundtrip_ok", False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--src")
    source.add_argument("--batch", help="父目录：逐个摄取其下的 runner 输出目录")
    ap.add_argument("--dest")
    ap.add_argument("--dest-root", help="批量模式的 zarr 父目录")
    args = ap.parse_args(argv)
    if args.src and args.dest_root:
        ap.error("--dest-root 仅用于 --batch")
    if args.batch and args.dest:
        ap.error("--dest 仅用于 --src")

    if args.batch:
        results = ingest_batch(Path(args.batch), Path(args.dest_root) if args.dest_root else None)
    else:
        try:
            results = [ingest_one(Path(args.src), Path(args.dest) if args.dest else None)]
        except Exception as exc:
            results = [{"src": str(args.src), "error": repr(exc)}]
    n_ok = sum(1 for result in results if not _failed(result))
    n_failed = len(results) - n_ok
    write_report_json(
        "wp5_ingest_summary.json",
        {"n_runs": len(results), "n_ok": n_ok, "n_failed": n_failed, "runs": results},
    )
    print(f"ingest 完成 {n_ok}/{len(results)} 回环通过")
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
