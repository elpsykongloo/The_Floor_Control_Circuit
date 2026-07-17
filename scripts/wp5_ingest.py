"""WP5：runner npy 分片 → zarr ingest + manifest 校验 + 回环抽查。

用法：
  uv run python scripts/wp5_ingest.py --src <runner 输出目录> [--dest <zarr 目录>]
  uv run python scripts/wp5_ingest.py --batch <父目录>   # 逐个 ingest 子目录
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.cachelib.zarr_io import ingest_npy_run, read_acts


def ingest_one(src: Path, dest: Path | None) -> dict:
    dest = dest or src.with_name(src.name + "_zarr")
    manifest = ingest_npy_run(src, dest)
    layer = manifest.layers[0]
    stored = read_acts(dest, layer)
    part0 = np.load(sorted(src.glob(f"acts_L{layer}_part*.npy"))[0], allow_pickle=False)
    ok = bool(np.array_equal(stored[: len(part0)], part0.astype(np.float16)))
    return {
        "src": str(src),
        "dest": str(dest),
        "session_id": manifest.session_id,
        "layers": manifest.layers,
        "n_steps": manifest.n_steps,
        "hidden_dim": manifest.hidden_dim,
        "roundtrip_ok": ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dest")
    ap.add_argument("--batch", help="父目录：逐个 ingest 其下含 manifest.json 的子目录")
    args = ap.parse_args()
    results = []
    if args.batch:
        for sub in sorted(Path(args.batch).iterdir()):
            if (sub / "manifest.json").exists() and not sub.name.endswith("_zarr"):
                try:
                    results.append(ingest_one(sub, None))
                except Exception as e:
                    results.append({"src": str(sub), "error": repr(e)})
    elif args.src:
        results.append(ingest_one(Path(args.src), Path(args.dest) if args.dest else None))
    else:
        ap.error("需要 --src 或 --batch")
    n_ok = sum(1 for r in results if r.get("roundtrip_ok"))
    write_report_json("wp5_ingest_summary.json", {"n_runs": len(results), "n_ok": n_ok, "runs": results})
    print(f"ingest 完成 {n_ok}/{len(results)} 回环通过")


if __name__ == "__main__":
    main()
