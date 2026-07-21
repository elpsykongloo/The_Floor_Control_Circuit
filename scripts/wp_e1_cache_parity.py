"""WP-E1-1：E1 全层缓存 vs 历史 MVE 四层缓存的前缀一致性核验（冒烟阶梯第 3 级）。

流式因果协议下，240 s 新缓存的前 3000 步应与 600 s 历史缓存的前 3000 步逐元素一致
（Mimi 流式编码、transformer 流式前向、贪心文本自预测均为因果前缀性质）。用法：
  uv run python scripts/wp_e1_cache_parity.py --limit 2
比较对象：重叠 run 目录（<sid>_agent{ch}）的 acts（历史层 4/12/20/28）、text_tokens、
mimi_latent 的前 3000 行。默认要求逐位相等；--report-only 只报告不判失败。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_config

PREFIX_STEPS = 3000
HIST_LAYERS = [4, 12, 20, 28]


def _part_files(run_dir: Path, stem: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(stem)}_part(\d+)\.npy$")
    parts = []
    for path in run_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            parts.append((int(match.group(1)), path))
    return [path for _, path in sorted(parts)]


def _load_prefix_parts(run_dir: Path, stem: str, n_rows: int) -> np.ndarray:
    blocks = []
    remaining = n_rows
    for path in _part_files(run_dir, stem):
        if remaining <= 0:
            break
        block = np.load(path, allow_pickle=False)
        take = min(remaining, block.shape[0])
        blocks.append(block[:take])
        remaining -= take
    if remaining > 0:
        raise SystemExit(f"{run_dir} 的 {stem} 行数不足 {n_rows}")
    return np.concatenate(blocks, axis=0)


def _load_stacked_layer(run_dir: Path, layers: list[int], layer: int, n_rows: int) -> np.ndarray:
    position = layers.index(layer)
    stacked = _load_prefix_parts(run_dir, "acts", n_rows)  # [T, L, H]
    return np.ascontiguousarray(stacked[:, position])


def compare_run(new_dir: Path, old_dir: Path) -> dict:
    new_manifest = json.loads((new_dir / "manifest.json").read_text(encoding="utf-8"))
    old_manifest = json.loads((old_dir / "manifest.json").read_text(encoding="utf-8"))
    new_layers = [int(v) for v in new_manifest["layers"]]
    result: dict = {"run": new_dir.name, "comparisons": {}, "all_equal": True}

    def register(name: str, new_array: np.ndarray, old_array: np.ndarray) -> None:
        if new_array.shape != old_array.shape:
            result["comparisons"][name] = {
                "equal": False,
                "reason": f"形状 {new_array.shape} vs {old_array.shape}",
            }
            result["all_equal"] = False
            return
        equal = bool(np.array_equal(new_array, old_array))
        entry: dict = {"equal": equal}
        if not equal:
            diff = np.abs(
                new_array.astype(np.float64) - old_array.astype(np.float64)
            )
            entry["max_abs_diff"] = float(diff.max())
            entry["n_mismatch"] = int((diff > 0).sum())
        result["comparisons"][name] = entry
        result["all_equal"] &= equal

    for layer in HIST_LAYERS:
        if layer not in new_layers:
            raise SystemExit(f"新缓存层列表缺少历史层 {layer}")
        new_acts = _load_stacked_layer(new_dir, new_layers, layer, PREFIX_STEPS)
        old_acts = _load_prefix_parts(old_dir, f"acts_L{layer}", PREFIX_STEPS)
        register(f"acts_L{layer}", new_acts, old_acts)
    new_text = np.load(new_dir / "text_tokens.npy", allow_pickle=False)[:PREFIX_STEPS]
    old_text = np.load(old_dir / "text_tokens.npy", allow_pickle=False)[:PREFIX_STEPS]
    register("text_tokens", new_text, old_text)
    if new_manifest.get("mimi_latent") and old_manifest.get("mimi_latent"):
        register(
            "mimi_latent",
            _load_prefix_parts(new_dir, "mimi_latent", PREFIX_STEPS),
            _load_prefix_parts(old_dir, "mimi_latent", PREFIX_STEPS),
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-root", default=None, help="新 E1 缓存根（默认按 grids e1.cache 推导）")
    ap.add_argument("--mve-root", default=None, help="历史 MVE 缓存根（默认 mve_r1_greedy）")
    ap.add_argument("--limit", type=int, default=2, help="最多比较的重叠 run 数（0 = 全部）")
    ap.add_argument("--report-only", action="store_true", help="不一致时也不以失败码退出")
    args = ap.parse_args()
    cache_cfg = load_config("grids")["e1"]["cache"]
    model = str(cache_cfg["model"])
    e1_root = (
        Path(args.e1_root)
        if args.e1_root
        else data_root() / "activations" / model / str(cache_cfg["out_group"])
    )
    mve_root = (
        Path(args.mve_root)
        if args.mve_root
        else data_root() / "activations" / model / "mve_r1_greedy"
    )
    if not e1_root.is_dir():
        raise SystemExit(f"E1 缓存根不存在：{e1_root}")
    overlaps = sorted(
        run.name
        for run in e1_root.iterdir()
        if run.is_dir() and (run / "manifest.json").is_file()
        and (mve_root / run.name / "manifest.json").is_file()
    )
    if args.limit:
        overlaps = overlaps[: args.limit]
    if not overlaps:
        raise SystemExit(f"{e1_root} 与 {mve_root} 没有可比较的重叠 run")
    results = []
    for name in overlaps:
        print(f"比较 {name} …")
        results.append(compare_run(e1_root / name, mve_root / name))
    n_equal = sum(1 for item in results if item["all_equal"])
    report = {
        "e1_root": str(e1_root),
        "mve_root": str(mve_root),
        "prefix_steps": PREFIX_STEPS,
        "layers": HIST_LAYERS,
        "n_compared": len(results),
        "n_all_equal": n_equal,
        "verdict": "equal" if n_equal == len(results) else "mismatch",
        "runs": results,
    }
    write_report_json("wp_e1_cache_parity.json", report)
    print(f"前缀一致性：{n_equal}/{len(results)} run 逐位相等 → {report['verdict']}")
    if report["verdict"] != "equal" and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
