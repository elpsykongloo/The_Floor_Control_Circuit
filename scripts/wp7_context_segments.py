"""WP7：上下文分段诊断（PREREG #11 描述性登记，非判据）。

对**被 #11 取代的全窗（7500 步）G1 分数包**做上下文分段复算，把撤回复盘中的
临时分段分析固化为确定性工具：按 Moshi 官方 context=3000 步把标签步切为
[0,2998]（规格内）、[2999,5998]（首次淘汰后 / 替代 sink 驻留）、[5999,7498]
（二次淘汰后 / 公共吸引态），逐段报告探针与三基线的 AUC、优势与会话级
bootstrap CI。分段结果使用全窗训练的探针分数，**只作诊断，不构成任何裁决**。

自校验：
  - summary.score_bundle.manifest_sha256 必须与 --bundle 的 manifest 一致
    （防止在重跑后的新 summary 上误用；旧 summary 可用
    `git show 9386933:reports/mve_summary.json` 提取）；
  - 从权威标签 parquet 重建每会话行→(通道, 步) 映射，并与分数包内标签逐行全等；
  - 全窗段的优势必须与 summary 登记值零差异（重建正确性的硬校验）。

用法：
  uv run python scripts/wp7_context_segments.py --bundle <旧全窗分数包目录> \
      [--summary reports/mve_summary_旧.json]
产出：reports/mve_上下文分段诊断.json + reports/mve_上下文分段诊断.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _bootstrap import REPORTS_DIR, write_report_json

from floor_circuit.config import data_root, load_config
from floor_circuit.mve.artifacts import (
    read_per_session_npz,
    sha256_file,
    validate_score_bundle,
)
from floor_circuit.probes.stats import paired_seed_mean_advantage_bootstrap

# 与 mve/alignment.py 的 PREREG #11 常数一致（本工具面向全窗旧分数包，独立声明）
MODEL_CONTEXT_STEPS = 3000
FULL_WINDOW_N_STEPS = 7500  # 被取代协议的每路步数（预检当时逐路强制）
FLOAT_ATOL = 1e-12

SEGMENTS: list[tuple[str, int, int, str]] = [
    ("in_context", 0, 2998, "规格内窗口（acts 行 1..2999；#11 主判据窗）"),
    ("post_first_eviction", 2999, 5998, "初始 token 淘汰后（acts 行 3000..5999，含尖峰行与替代 sink 驻留期）"),
    ("post_second_eviction", 5999, 7498, "替代 sink 淘汰后（acts 行 6000..7499，公共吸引态）"),
    ("full_window", 0, 7498, "全窗（被取代协议的完整行集；用于与 summary 登记值对拍）"),
]


def _reconstruct_rows(
    labels_root: Path,
    session_id: str,
    target: str,
    delta_ms: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """按被取代的全窗协议重建 (labels, steps)：ch0→ch1 拼接、步升序、步 < 7499。"""

    frame = pd.read_parquet(labels_root / f"{session_id}.parquet")
    labels_parts: list[np.ndarray] = []
    steps_parts: list[np.ndarray] = []
    for channel in (0, 1):
        mask = (frame["target"] == target) & (frame["agent_channel"] == channel)
        if target == "T1":
            if delta_ms is None:
                raise SystemExit("T1 重建需要 delta_ms")
            mask &= frame["delta_ms"] == delta_ms
        rows = frame.loc[mask, ["step", "label"]].sort_values("step", kind="stable")
        steps = rows["step"].to_numpy(dtype=np.int64)
        keep = (steps >= 0) & (steps < FULL_WINDOW_N_STEPS - 1)
        steps_parts.append(steps[keep])
        labels_parts.append(rows["label"].to_numpy(dtype=np.int64)[keep])
    return np.concatenate(labels_parts), np.concatenate(steps_parts)


def _mask_collection(
    per_session_by_seed: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    steps_by_session: dict[str, np.ndarray],
    lo: int,
    hi: int,
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    out: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for seed, per_session in per_session_by_seed.items():
        masked: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sid, (y, p) in per_session.items():
            keep = (steps_by_session[sid] >= lo) & (steps_by_session[sid] <= hi)
            masked[sid] = (y[keep], p[keep])
        out[seed] = masked
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="被 #11 取代的全窗 G1 分数包目录")
    ap.add_argument(
        "--summary",
        default=str(REPORTS_DIR / "mve_summary.json"),
        help="与该分数包对应的旧 mve_summary.json（哈希绑定校验）",
    )
    ap.add_argument("--labels-root", default=None, help="默认 <data_root>/events/candor_labels_flat")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    manifest_path = bundle_dir / "manifest.json"
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    declared_sha = summary.get("score_bundle", {}).get("manifest_sha256")
    actual_sha = sha256_file(manifest_path)
    if declared_sha != actual_sha:
        raise SystemExit(
            "summary 与分数包不匹配：summary.score_bundle.manifest_sha256="
            f"{declared_sha}，实际 {actual_sha}。若正式结果已被 #11 重跑覆盖，"
            "请用 `git show <旧提交>:reports/mve_summary.json > reports/mve_summary_旧.json` "
            "提取旧 summary 后以 --summary 指定"
        )
    manifest = validate_score_bundle(manifest_path)
    labels_root = Path(args.labels_root) if args.labels_root else data_root() / "events" / "candor_labels_flat"
    mve_cfg = load_config("grids")["mve"]
    t1_delta_ms = int(mve_cfg["t1_delta_ms"])

    eval_sessions = list(manifest["eval_session_order"])
    items = {(
        str(item["target"]),
        str(item["kind"]),
        None if item["layer"] is None else int(item["layer"]),
        int(item["seed"]),
    ): item for item in manifest["items"]}

    per_target_out: dict[str, dict] = {}
    for target in ("T1", "T4"):
        declared = summary["per_target"][target]
        best_layer = int(declared["best_layer"])
        delta = t1_delta_ms if target == "T1" else None

        authoritative: dict[str, tuple[np.ndarray, np.ndarray]] = {
            sid: _reconstruct_rows(labels_root, sid, target, delta) for sid in eval_sessions
        }
        steps_by_session = {sid: steps for sid, (_labels, steps) in authoritative.items()}

        def load_item(
            kind: str,
            layer: int | None,
            seed: int,
            *,
            target: str = target,
            authoritative: dict[str, tuple[np.ndarray, np.ndarray]] = authoritative,
        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            item = items[(target, kind, layer, seed)]
            per_session = read_per_session_npz(bundle_dir / item["path"])
            for sid, (y, _p) in per_session.items():
                expected_labels = authoritative[sid][0]
                if not np.array_equal(np.asarray(y, dtype=np.int64), expected_labels):
                    raise SystemExit(
                        f"{target}/{kind}/L{layer}/seed{seed}/{sid}: 分数包标签与权威 parquet 重建不一致——"
                        "行→步映射失效，中止"
                    )
            return per_session

        probes = {seed: load_item("probe", best_layer, seed) for seed in (0, 1, 2)}
        baselines = {
            "mimi": {seed: load_item("mimi", None, seed) for seed in (0, 1, 2)},
            "hazard": {0: load_item("hazard", None, 0)},
            "acoustic_gru": {0: load_item("acoustic_gru", None, 0)},
        }

        segments_out: dict[str, dict] = {}
        for name, lo, hi, note in SEGMENTS:
            probe_masked = _mask_collection(probes, steps_by_session, lo, hi)
            baselines_masked = {
                base_name: _mask_collection(scores, steps_by_session, lo, hi)
                for base_name, scores in baselines.items()
            }
            n_rows = sum(len(y) for y, _p in probe_masked[0].values())
            n_positive = sum(int(np.sum(y)) for y, _p in probe_masked[0].values())
            adv_all = paired_seed_mean_advantage_bootstrap(
                probe_masked, baselines_masked, n_boot=args.n_boot, seed=args.boot_seed
            )
            adv_mimi = paired_seed_mean_advantage_bootstrap(
                probe_masked,
                {"mimi": baselines_masked["mimi"]},
                n_boot=args.n_boot,
                seed=args.boot_seed,
            )
            segments_out[name] = {
                "note": note,
                "label_step_range": [lo, hi],
                "acts_row_range": [lo + 1, hi + 1],
                "n_rows": n_rows,
                "n_positive": n_positive,
                "probe_auc_seed_mean": adv_all["probe_auc"],
                "baseline_auc_seed_mean": adv_all["baseline_aucs"],
                "advantage_vs_max_baseline": {
                    "point": adv_all["advantage_point"],
                    "ci_lo": adv_all["ci_lo"],
                    "ci_hi": adv_all["ci_hi"],
                    "n_boot_effective": adv_all["n_boot_effective"],
                },
                "advantage_vs_mimi": {
                    "point": adv_mimi["advantage_point"],
                    "ci_lo": adv_mimi["ci_lo"],
                    "ci_hi": adv_mimi["ci_hi"],
                    "n_boot_effective": adv_mimi["n_boot_effective"],
                },
            }
            print(
                f"{target}/{name}: 行 {n_rows}（正类 {n_positive}），"
                f"优势(vs max) {adv_all['advantage_point']:+.5f} "
                f"[{adv_all['ci_lo']:+.5f}, {adv_all['ci_hi']:+.5f}]"
            )

        declared_adv = declared["advantage"]
        full = segments_out["full_window"]["advantage_vs_max_baseline"]
        for mine, theirs, field_name in (
            (full["point"], declared_adv["advantage_point"], "advantage_point"),
            (full["ci_lo"], declared_adv["ci_lo"], "ci_lo"),
            (full["ci_hi"], declared_adv["ci_hi"], "ci_hi"),
        ):
            if not np.isclose(mine, theirs, rtol=0.0, atol=FLOAT_ATOL):
                raise SystemExit(
                    f"{target}: 全窗段 {field_name} 复算 {mine} ≠ summary 登记 {theirs}——重建失效，中止"
                )
        per_target_out[target] = {
            "best_layer": best_layer,
            "delta_ms": delta,
            "full_window_matches_summary": True,
            "segments": segments_out,
        }

    report = {
        "note": (
            "上下文分段诊断（PREREG #11 描述性登记）：非判据、无裁决字段；"
            "输入为被 #11 取代的全窗分数包，分段使用全窗训练的探针分数——"
            "正式裁决以截断窗重训重选的 G1 重跑为准"
        ),
        "context_steps": MODEL_CONTEXT_STEPS,
        "full_window_n_steps": FULL_WINDOW_N_STEPS,
        "bundle_manifest_sha256": actual_sha,
        "summary_path": str(Path(args.summary).resolve()),
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "per_target": per_target_out,
    }
    write_report_json("mve_上下文分段诊断.json", report)

    lines = [
        "# 上下文分段诊断（PREREG #11 描述性登记，非判据）",
        "",
        f"- 输入：被取代的全窗分数包（manifest SHA-256 `{actual_sha}`）；"
        f"context={MODEL_CONTEXT_STEPS} 步；分段用全窗训练的探针分数，仅作诊断",
        "- 全窗段与旧 summary 登记值零差异（重建正确性硬校验通过）",
        "",
    ]
    for target, block in per_target_out.items():
        lines += [
            f"## 目标 {target}（最优层 L{block['best_layer']}）",
            "",
            "| 段（标签步） | 行数 | 正类 | 探针 AUC | Mimi AUC | 优势 vs max（95% CI） | 优势 vs Mimi（95% CI） |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for name, _lo, _hi, _note in SEGMENTS:
            seg = block["segments"][name]
            adv = seg["advantage_vs_max_baseline"]
            advm = seg["advantage_vs_mimi"]
            lines.append(
                f"| {name} [{seg['label_step_range'][0]},{seg['label_step_range'][1]}] "
                f"| {seg['n_rows']} | {seg['n_positive']} "
                f"| {seg['probe_auc_seed_mean']:.4f} | {seg['baseline_auc_seed_mean']['mimi']:.4f} "
                f"| {adv['point']:+.4f} [{adv['ci_lo']:+.4f}, {adv['ci_hi']:+.4f}] "
                f"| {advm['point']:+.4f} [{advm['ci_lo']:+.4f}, {advm['ci_hi']:+.4f}] |"
            )
        lines.append("")
    lines += [f"> {report['note']}", ""]
    (REPORTS_DIR / "mve_上下文分段诊断.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {REPORTS_DIR / 'mve_上下文分段诊断.md'}")


if __name__ == "__main__":
    main()
