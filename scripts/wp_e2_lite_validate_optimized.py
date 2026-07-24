"""WP-E2L：优化后端相对既有参考样本的逐运行数值等价验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.config import data_root, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise ValueError(f"{path} 不是 16-bit 单声道 WAV")
        sample_rate = reader.getframerate()
        values = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return values, sample_rate


def _compare_array(
    reference: Path,
    candidate: Path,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    chunk_values: int = 1 << 20,
) -> dict:
    if not reference.is_file() or not candidate.is_file():
        return {
            "equal": False,
            "reason": "missing",
            "reference_exists": reference.is_file(),
            "candidate_exists": candidate.is_file(),
        }
    ref = np.load(reference, mmap_mode="r", allow_pickle=False)
    cand = np.load(candidate, mmap_mode="r", allow_pickle=False)
    if ref.shape != cand.shape or ref.dtype != cand.dtype:
        return {
            "equal": False,
            "reason": "shape_or_dtype",
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
            "reference_dtype": str(ref.dtype),
            "candidate_dtype": str(cand.dtype),
        }
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    different_values = 0
    outside_tolerance = 0
    max_abs_difference = 0.0
    max_rel_difference = 0.0
    for start in range(0, int(ref_flat.size), chunk_values):
        stop = min(start + chunk_values, int(ref_flat.size))
        ref_chunk = np.asarray(ref_flat[start:stop])
        cand_chunk = np.asarray(cand_flat[start:stop])
        equal_mask = np.equal(ref_chunk, cand_chunk)
        if np.issubdtype(ref.dtype, np.floating):
            equal_mask |= np.isnan(ref_chunk) & np.isnan(cand_chunk)
        different_values += int(np.count_nonzero(~equal_mask))
        if np.issubdtype(ref.dtype, np.number):
            ref_float = ref_chunk.astype(np.float64, copy=False)
            cand_float = cand_chunk.astype(np.float64, copy=False)
            difference = np.abs(ref_float - cand_float)
            finite_difference = difference[np.isfinite(difference)]
            if finite_difference.size:
                max_abs_difference = max(
                    max_abs_difference,
                    float(np.max(finite_difference)),
                )
            denominator = np.maximum(np.abs(ref_float), np.finfo(np.float64).tiny)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                relative = difference / denominator
            finite_relative = relative[np.isfinite(relative)]
            if finite_relative.size:
                max_rel_difference = max(
                    max_rel_difference,
                    float(np.max(finite_relative)),
                )
            close_mask = np.isclose(
                ref_float,
                cand_float,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
            outside_tolerance += int(np.count_nonzero(~close_mask))
        else:
            outside_tolerance += int(np.count_nonzero(~equal_mask))
    equal = different_values == 0
    equivalent = outside_tolerance == 0
    result = {
        "equal": equal,
        "equivalent": equivalent,
        "shape": list(ref.shape),
        "dtype": str(ref.dtype),
        "different_values": different_values,
        "outside_tolerance": outside_tolerance,
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if not equal:
        result["max_abs_difference"] = max_abs_difference
        result["max_rel_difference"] = max_rel_difference
    return result


def _compare_wav(reference: Path, candidate: Path) -> dict:
    if not reference.is_file() or not candidate.is_file():
        return {
            "equal": False,
            "reason": "missing",
            "reference_exists": reference.is_file(),
            "candidate_exists": candidate.is_file(),
        }
    ref_sha = _sha256(reference)
    cand_sha = _sha256(candidate)
    ref, ref_sr = _read_pcm(reference)
    cand, cand_sr = _read_pcm(candidate)
    pcm_equal = ref_sr == cand_sr and ref.shape == cand.shape and bool(np.array_equal(ref, cand))
    result = {
        "equal": pcm_equal,
        "file_equal": ref_sha == cand_sha,
        "reference_sha256": ref_sha,
        "candidate_sha256": cand_sha,
        "reference_samples": int(ref.size),
        "candidate_samples": int(cand.size),
        "reference_sample_rate": ref_sr,
        "candidate_sample_rate": cand_sr,
    }
    if not pcm_equal and ref.shape == cand.shape and ref_sr == cand_sr:
        difference = ref.astype(np.int32) - cand.astype(np.int32)
        result["max_abs_pcm_difference"] = int(np.max(np.abs(difference)))
        result["different_samples"] = int(np.count_nonzero(difference))
    return result


def _compare_manifest(reference: dict, candidate: dict, plan: dict) -> dict:
    expected_frames = round(float(plan["window_s"]) * float(plan["frame_hz"]))
    fields = (
        "run_id",
        "session_id",
        "user_wav_sha256",
        "user_channel",
        "seed",
        "condition",
        "layer",
        "scale_rule",
        "window_s",
        "temperature",
        "text_temperature",
        "top_k",
        "top_k_text",
        "first_emitted_frame",
        "n_frames_in",
        "n_agent_samples",
        "directions_sha256",
    )
    mismatches = {
        field: {
            "reference": reference.get(field),
            "candidate": candidate.get(field),
        }
        for field in fields
        if reference.get(field) != candidate.get(field)
    }
    if int(candidate.get("n_frames_in", -1)) != expected_frames:
        mismatches["candidate_expected_frames"] = {
            "expected": expected_frames,
            "candidate": candidate.get("n_frames_in"),
        }
    profile = candidate.get("execution_profile", {})
    if profile.get("equivalence_contract") != "reference_exact":
        mismatches["equivalence_contract"] = {
            "expected": "reference_exact",
            "candidate": profile.get("equivalence_contract"),
        }
    if candidate.get("execution_backend") == "full_graph":
        for field in (
            "cuda_graph_main",
            "cuda_graph_depth",
            "cuda_graph_mimi_encode",
            "cuda_graph_mimi_decoder",
            "cuda_graph_mimi_decoder_transformer",
        ):
            if candidate.get(field) is not True:
                mismatches[field] = {
                    "expected": True,
                    "candidate": candidate.get(field),
                }
    return {
        "equal": not mismatches,
        "mismatches": mismatches,
    }


def _load_manifest(run_dir: Path) -> dict | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("completed") else None


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-lite 优化后端逐运行数值等价验收")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--reference-root", default=None, help="参考 runs/ 根目录")
    parser.add_argument("--candidate-root", default=None, help="候选 runs/ 根目录")
    parser.add_argument("--session-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--activation-atol", type=float, default=0.0)
    parser.add_argument("--activation-rtol", type=float, default=0.0)
    parser.add_argument(
        "--report-tag",
        default="",
        help="报告文件名后缀，用于保存多组优化矩阵结果",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.activation_atol < 0 or args.activation_rtol < 0:
        raise SystemExit("激活容差必须为非负数")
    report_tag = str(args.report_tag).strip()
    if report_tag and not all(char.isalnum() or char in {"-", "_"} for char in report_tag):
        raise SystemExit("--report-tag 只允许字母、数字、连字符和下划线")
    report_suffix = f"_{report_tag}" if report_tag else ""

    cfg = load_config("grids")["e1"]["e2_lite"]
    plan_path = Path(args.plan) if args.plan else data_root() / str(cfg["out_group"]) / "e2_lite.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reference_root = Path(args.reference_root) if args.reference_root else Path(plan["out_root"]) / "runs"
    candidate_root = (
        Path(args.candidate_root)
        if args.candidate_root
        else Path(plan["out_root"]).with_name(Path(plan["out_root"]).name + "_optimized") / "runs"
    )

    selected_sessions = set(args.session_id or [])
    run_ids = [
        f"{session['session_id']}__{condition['name']}"
        for session in plan["sessions"]
        if not selected_sessions or session["session_id"] in selected_sessions
        for condition in plan["conditions"]
    ]
    if args.limit is not None:
        run_ids = run_ids[: int(args.limit)]

    results: list[dict] = []
    missing_reference: list[str] = []
    missing_candidate: list[str] = []
    reference_wall = 0.0
    candidate_groups: dict[str, float] = {}
    candidate_encode_observed: dict[str, float] = {}
    candidate_encode_cold: dict[str, float] = {}
    for run_id in run_ids:
        ref_dir = reference_root / run_id
        cand_dir = candidate_root / run_id
        ref_manifest = _load_manifest(ref_dir)
        cand_manifest = _load_manifest(cand_dir)
        if ref_manifest is None:
            missing_reference.append(run_id)
            continue
        if cand_manifest is None:
            missing_candidate.append(run_id)
            continue

        manifest_contract = _compare_manifest(ref_manifest, cand_manifest, plan)
        text = _compare_array(
            ref_dir / "text_tokens.npy",
            cand_dir / "text_tokens.npy",
        )
        wav = _compare_wav(ref_dir / "agent.wav", cand_dir / "agent.wav")
        acts = {}
        if ref_manifest.get("condition", {}).get("cache_acts"):
            for layer in plan["cache_layers_baseline"]:
                name = f"acts_L{int(layer)}.npy"
                acts[f"L{int(layer)}"] = _compare_array(
                    ref_dir / name,
                    cand_dir / name,
                    atol=float(args.activation_atol),
                    rtol=float(args.activation_rtol),
                )
        hook_equal = (
            int(ref_manifest.get("hook_calls", -1))
            == int(cand_manifest.get("hook_calls", -2))
            == round(float(plan["window_s"]) * float(plan["frame_hz"]))
        )
        exact = (
            manifest_contract["equal"]
            and text["equal"]
            and wav["equal"]
            and hook_equal
            and all(item["equal"] for item in acts.values())
        )
        equivalent = (
            manifest_contract["equal"]
            and text["equal"]
            and wav["equal"]
            and hook_equal
            and all(item.get("equivalent", item["equal"]) for item in acts.values())
        )
        results.append(
            {
                "run_id": run_id,
                "exact": exact,
                "equivalent": equivalent,
                "manifest_contract": manifest_contract,
                "text_tokens": text,
                "agent_wav": wav,
                "activations": acts,
                "hook_calls_equal": hook_equal,
                "reference_wall_s": ref_manifest.get("wall_s"),
                "candidate_wall_s": cand_manifest.get("wall_s"),
                "candidate_backend": cand_manifest.get("execution_backend"),
                "candidate_group_id": cand_manifest.get("group_id"),
                "candidate_group_size": cand_manifest.get("group_size"),
            }
        )
        reference_wall += float(ref_manifest.get("wall_s", 0.0))
        group_id = str(cand_manifest.get("group_id", run_id))
        candidate_groups[group_id] = float(cand_manifest.get("group_wall_s", cand_manifest.get("wall_s", 0.0)))
        session_id = str(cand_manifest.get("session_id", run_id.split("__", 1)[0]))
        candidate_encode_observed[session_id] = float(cand_manifest.get("encode_wall_s", 0.0) or 0.0)
        candidate_encode_cold[session_id] = float(
            cand_manifest.get(
                "encode_source_wall_s",
                cand_manifest.get("encode_wall_s", 0.0),
            )
            or 0.0
        )

    exact_count = sum(bool(item["exact"]) for item in results)
    equivalent_count = sum(bool(item["equivalent"]) for item in results)
    candidate_group_wall = sum(candidate_groups.values())
    candidate_observed_wall = candidate_group_wall + sum(candidate_encode_observed.values())
    candidate_cold_wall = candidate_group_wall + sum(candidate_encode_cold.values())
    payload = {
        "schema": "e2_lite_optimization_validation_v1",
        "plan": str(plan_path),
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "requested_runs": len(run_ids),
        "compared_runs": len(results),
        "exact_runs": exact_count,
        "equivalent_runs": equivalent_count,
        "all_exact": bool(results) and exact_count == len(results),
        "all_equivalent": bool(results) and equivalent_count == len(results),
        "activation_atol": float(args.activation_atol),
        "activation_rtol": float(args.activation_rtol),
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
        "reference_wall_s": reference_wall,
        "candidate_unique_group_wall_s": candidate_group_wall,
        "candidate_observed_encode_wall_s": sum(candidate_encode_observed.values()),
        "candidate_cold_encode_wall_s": sum(candidate_encode_cold.values()),
        "candidate_observed_wall_s": candidate_observed_wall,
        "candidate_cold_wall_s": candidate_cold_wall,
        "observed_speedup": (reference_wall / candidate_observed_wall if candidate_observed_wall > 0 else None),
        "cold_speedup": (reference_wall / candidate_cold_wall if candidate_cold_wall > 0 else None),
        "runs": results,
    }
    write_report_json(
        f"wp_e2_lite_optimization_validation{report_suffix}.json",
        payload,
    )
    print(
        f"等价验收：{exact_count}/{len(results)} 精确一致，"
        f"{equivalent_count}/{len(results)} 数值等价；"
        f"参考缺 {len(missing_reference)}，候选缺 {len(missing_candidate)}；"
        f"冷启动加速比 {payload['cold_speedup']}"
    )
    if args.require_complete and (missing_reference or missing_candidate):
        raise SystemExit(2)
    if not payload["all_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
