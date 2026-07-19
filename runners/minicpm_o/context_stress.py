"""MiniCPM-o 4.5 主干上下文应力运行器。

该工具先从短音频提取真实的 1 秒输入单元嵌入，再循环打乱这些单元，对主干 KV
缓存执行长流压力测试。正式候选边界包含官方 40960 位置及其两倍；计划窗阶段还会
检查 600 秒在最坏 32 位置/秒消耗下所需的位置预算。

大数组只写 ``--out`` 指定目录，建议放在
``D:\\data_storage\\The_Floor_Control_Circuit\\context_stress\\minicpm_o``。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SHARED_ROOT = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED_ROOT))

from context_stress_runner import (  # noqa: E402
    GpuTemperatureGuard,
    SelectedLayerCapture,
    atomic_save_npz,
    atomic_write_json,
    binary_entropy,
    build_sample_positions,
    cache_key,
    cosine_to_reference,
    file_sha256,
    load_binary_labels,
    parse_int_csv,
    validate_layers,
)

INPUT_SAMPLE_RATE = 16000
CHUNK_SAMPLES = INPUT_SAMPLE_RATE
TRACE_SCHEMA_VERSION = 1
ANALYSIS_TARGET_SECONDS = 600.0


def _load_standard_runner():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("minicpm_o_standard_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_hidden_from_feed(result: Any):
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("decoder.feed(return_logits=True) 返回格式异常")
    return result[0], result[1]


def _cat_for_decoder(decoder, parts: list[Any]):
    """把跨卡的 token/audio 嵌入统一搬到主干首层设备后拼接。"""

    import torch

    input_device = next(decoder.m.model.layers[0].parameters()).device
    return torch.cat([part.to(input_device) for part in parts], dim=0)


def _cosine_numpy(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def _repeat_audio_chunk(audio: np.ndarray, chunk_index: int) -> np.ndarray:
    if len(audio) == 0:
        raise ValueError("音频为空")
    start = (chunk_index * CHUNK_SAMPLES) % len(audio)
    indices = (start + np.arange(CHUNK_SAMPLES)) % len(audio)
    return np.asarray(audio[indices], dtype=np.float32)


def _collect_embedding_bank(
    duplex,
    audio: np.ndarray,
    *,
    bank_units: int,
    max_new_speak_tokens: int,
) -> tuple[list[list[Any]], dict[str, Any]]:
    """用官方分段路径提取完整单元嵌入，并保存首单元作为批处理对拍参考。"""

    decoder = duplex.decoder
    original_feed = decoder.feed
    bank: list[list[Any]] = []
    first_logits = None
    first_hidden = None
    current_parts: list[Any] = []
    current_decision = None

    def capture_feed(embeds, *args, **kwargs):
        nonlocal current_decision
        current_parts.append(embeds.detach().clone())
        result = original_feed(embeds, *args, **kwargs)
        if kwargs.get("return_logits", False) and current_decision is None:
            current_decision = result
        return result

    decoder.feed = capture_feed
    old_force_count = duplex.force_listen_count
    duplex.force_listen_count = bank_units + 1
    try:
        for index in range(bank_units):
            current_parts = []
            current_decision = None
            prefill = duplex.streaming_prefill(audio_waveform=_repeat_audio_chunk(audio, index))
            if not prefill.get("success", False):
                raise RuntimeError(f"嵌入库第 {index} 单元预填失败：{prefill.get('reason', '')}")
            duplex.streaming_generate(
                max_new_speak_tokens_per_chunk=max_new_speak_tokens,
                decode_mode="greedy",
            )
            lengths = [int(part.shape[0]) for part in current_parts]
            if len(current_parts) != 4 or lengths[0] != 1 or lengths[-2:] != [1, 1]:
                raise RuntimeError(
                    f"第 {index} 单元的官方 feed 分段异常：调用数={len(current_parts)}，长度={lengths}"
                )
            if current_decision is None:
                raise RuntimeError(f"第 {index} 单元没有捕获决策 logits")
            bank.append([part.detach() for part in current_parts])
            if index == 0:
                logits, hidden = _extract_hidden_from_feed(current_decision)
                first_logits = logits.detach().float().cpu().numpy()
                first_hidden = hidden[:, -1, :].detach().float().cpu().numpy()
    finally:
        decoder.feed = original_feed
        duplex.force_listen_count = old_force_count
    if first_logits is None or first_hidden is None:
        raise RuntimeError("嵌入库为空")
    return bank, {
        "official_logits": first_logits,
        "official_hidden": first_hidden,
        "part_lengths": [int(part.shape[0]) for part in bank[0]],
        "unit_positions": int(sum(part.shape[0] for part in bank[0])),
        "audio_positions": int(bank[0][1].shape[0]),
    }


def _run_batched_parity(duplex, first_unit: list[Any], reference: dict[str, Any]) -> dict[str, Any]:
    """首单元对拍：官方四次 feed 与批处理两次 feed。"""

    decoder = duplex.decoder
    prefill = _cat_for_decoder(decoder, first_unit[:2])
    tail = _cat_for_decoder(decoder, first_unit[2:])
    logits, hidden = _extract_hidden_from_feed(decoder.feed(prefill, return_logits=True))
    decoder.feed(tail)
    logits_np = logits.detach().float().cpu().numpy()
    hidden_np = hidden[:, -1, :].detach().float().cpu().numpy()
    reference_logits = reference["official_logits"]
    reference_hidden = reference["official_hidden"]
    return {
        "hidden_cosine": _cosine_numpy(hidden_np, reference_hidden),
        "hidden_max_abs": float(np.max(np.abs(hidden_np - reference_hidden))),
        "logits_cosine": _cosine_numpy(logits_np, reference_logits),
        "logits_max_abs": float(np.max(np.abs(logits_np - reference_logits))),
        "argmax_equal": bool(int(np.argmax(logits_np)) == int(np.argmax(reference_logits))),
    }


def _run_grouped_parity(
    duplex,
    units: list[list[Any]],
    *,
    system_prompt: str,
) -> dict[str, Any]:
    """对拍逐单元前向与跨单元分块前向。"""

    if len(units) < 2:
        raise ValueError("分块对拍至少需要两个嵌入单元")

    duplex.prepare(prefix_system_prompt=system_prompt)
    decoder = duplex.decoder
    reference_logits = None
    reference_hidden = None
    for unit in units[:2]:
        logits, hidden = _extract_hidden_from_feed(
            decoder.feed(
                _cat_for_decoder(decoder, unit[:2]),
                return_logits=True,
            )
        )
        decoder.feed(_cat_for_decoder(decoder, unit[2:]))
        reference_logits = logits.detach().float().cpu().numpy()
        reference_hidden = hidden[:, -1, :].detach().float().cpu().numpy()

    duplex.prepare(prefix_system_prompt=system_prompt)
    decoder = duplex.decoder
    grouped_parts = [*units[0], *units[1][:2]]
    logits, hidden = _extract_hidden_from_feed(
        decoder.feed(
            _cat_for_decoder(decoder, grouped_parts),
            return_logits=True,
        )
    )
    decoder.feed(_cat_for_decoder(decoder, units[1][2:]))
    grouped_logits = logits.detach().float().cpu().numpy()
    grouped_hidden = hidden[:, -1, :].detach().float().cpu().numpy()
    if reference_logits is None or reference_hidden is None:
        raise RuntimeError("分块对拍没有产生参考输出")
    return {
        "hidden_cosine": _cosine_numpy(grouped_hidden, reference_hidden),
        "hidden_max_abs": float(
            np.max(np.abs(grouped_hidden - reference_hidden))
        ),
        "logits_cosine": _cosine_numpy(grouped_logits, reference_logits),
        "logits_max_abs": float(np.max(np.abs(grouped_logits - reference_logits))),
        "argmax_equal": bool(
            int(np.argmax(grouped_logits)) == int(np.argmax(reference_logits))
        ),
    }


def _trace_arrays(records: dict[str, list[Any]], n_layers: int, hidden_dim: int) -> dict[str, np.ndarray]:
    if not records["logical_positions"]:
        return {
            "logical_positions": np.empty(0, dtype=np.int64),
            "cache_lengths": np.empty(0, dtype=np.int64),
            "position_offsets": np.empty(0, dtype=np.int64),
            "hidden": np.empty((0, n_layers, hidden_dim), dtype=np.float16),
            "decision_probs": np.empty((0, 2), dtype=np.float32),
            "decision_ids": np.empty(0, dtype=np.int16),
            "top_token_ids": np.empty(0, dtype=np.int64),
            "task_labels": np.empty(0, dtype=np.int8),
            "task_scores": np.empty(0, dtype=np.float32),
            "dynamic_key_cosines": np.empty((0, n_layers), dtype=np.float32),
            "all_finite": np.empty((0, n_layers), dtype=bool),
            "sliding_events": np.empty(0, dtype=np.int64),
        }
    return {
        "logical_positions": np.asarray(records["logical_positions"], dtype=np.int64),
        "cache_lengths": np.asarray(records["cache_lengths"], dtype=np.int64),
        "position_offsets": np.asarray(records["position_offsets"], dtype=np.int64),
        "hidden": np.stack(records["hidden"]).astype(np.float16),
        "decision_probs": np.asarray(records["decision_probs"], dtype=np.float32),
        "decision_ids": np.asarray(records["decision_ids"], dtype=np.int16),
        "top_token_ids": np.asarray(records["top_token_ids"], dtype=np.int64),
        "task_labels": np.asarray(records["task_labels"], dtype=np.int8),
        "task_scores": np.asarray(records["task_scores"], dtype=np.float32),
        "dynamic_key_cosines": np.asarray(records["dynamic_key_cosines"], dtype=np.float32),
        "all_finite": np.asarray(records["all_finite"], dtype=bool),
        "sliding_events": np.asarray(records["sliding_events"], dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 上下文应力诊断")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--audio", required=True, help="用于构造真实音频嵌入库的短音频")
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=["planned", "official", "double"], default="planned")
    parser.add_argument("--max-position", type=int, default=None)
    parser.add_argument("--boundaries", default=None)
    parser.add_argument("--layers", default=None, help="零基层号，逗号分隔；默认四个深度")
    parser.add_argument("--bank-units", type=int, default=16)
    parser.add_argument("--bank-labels", default=None, help="与嵌入库逐单元对齐的 .npy 或 JSON 0/1 标签")
    parser.add_argument(
        "--bank-order",
        choices=["randomized", "sequential"],
        default="randomized",
        help="带时间目标标签的正式性能诊断必须使用 sequential",
    )
    parser.add_argument("--feed-mode", choices=["batched", "segmented"], default="batched")
    parser.add_argument(
        "--execution-mode",
        choices=["grouped", "unit"],
        default="grouped",
        help="grouped 合并相邻采样点间的完整单元；unit 逐单元执行",
    )
    parser.add_argument("--sample-every", type=int, default=256)
    parser.add_argument("--dense-radius", type=int, default=256)
    parser.add_argument("--boundary-window-positions", type=int, default=512)
    parser.add_argument("--sliding-window-mode", choices=["off", "basic"], default="off")
    parser.add_argument("--system-prompt", default="Streaming audio conversation. Please answer naturally and briefly.")
    parser.add_argument("--max-new-speak-tokens", type=int, default=20)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=64)
    parser.add_argument("--parity-min-cosine", type=float, default=0.999)
    parser.add_argument("--temperature-limit", type=int, default=95)
    args = parser.parse_args()
    if args.bank_units < 2:
        parser.error("--bank-units 至少为 2，才能执行跨单元对拍")
    if args.execution_mode == "grouped" and args.sliding_window_mode != "off":
        parser.error("主动滑窗诊断必须使用 --execution-mode unit")

    import torch
    from transformers import AutoModel

    standard = _load_standard_runner()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.npz"
    manifest_path = out_dir / "manifest.json"
    started = time.time()
    rng = np.random.default_rng(args.seed)
    guard = GpuTemperatureGuard(limit_c=args.temperature_limit)
    records: dict[str, list[Any]] = {
        "logical_positions": [],
        "cache_lengths": [],
        "position_offsets": [],
        "hidden": [],
        "decision_probs": [],
        "decision_ids": [],
        "top_token_ids": [],
        "task_labels": [],
        "task_scores": [],
        "dynamic_key_cosines": [],
        "all_finite": [],
        "sliding_events": [],
    }
    manifest: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": "context_stress_v1",
        "model": "minicpm_o_4_5",
        "run_id": args.run_id,
        "complete": False,
        "started_unix": started,
        "source_audio": {
            "path": str(Path(args.audio).resolve()),
            "sha256": file_sha256(Path(args.audio)),
        },
        "feed_mode": args.feed_mode,
        "execution_mode": args.execution_mode,
        "seed": args.seed,
    }
    atomic_write_json(manifest_path, manifest)

    layers: list[int] = []
    hidden_dim = 0

    def flush() -> None:
        if not layers or hidden_dim <= 0:
            return
        atomic_save_npz(trace_path, **_trace_arrays(records, len(layers), hidden_dim))

    try:
        guard.check(force=True)
        model_root = standard._sanitize_model_root(args.model_root)
        model = AutoModel.from_pretrained(
            model_root,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=False,
            device_map=args.device_map,
        ).eval()
        duplex = standard._as_duplex_without_unused_tts(
            model,
            generate_audio=False,
            chunk_ms=1000,
            first_chunk_ms=1035,
            max_new_speak_tokens_per_chunk=args.max_new_speak_tokens,
            force_listen_count=0,
            sliding_window_mode=args.sliding_window_mode,
        )
        duplex.prepare(prefix_system_prompt=args.system_prompt)
        audio = standard.load_audio_16k(args.audio)
        bank, bank_meta = _collect_embedding_bank(
            duplex,
            audio,
            bank_units=args.bank_units,
            max_new_speak_tokens=args.max_new_speak_tokens,
        )
        bank_labels = load_binary_labels(args.bank_labels, len(bank))
        if args.bank_labels is not None and args.bank_order != "sequential":
            raise ValueError("带时间目标标签时必须使用 --bank-order sequential")

        # 清空嵌入库采集阶段的所有主干状态，再对拍批处理路径。
        duplex.prepare(prefix_system_prompt=args.system_prompt)
        parity = _run_batched_parity(duplex, bank[0], bank_meta)
        if args.feed_mode == "batched" and (
            parity["hidden_cosine"] < args.parity_min_cosine
            or parity["logits_cosine"] < args.parity_min_cosine
            or not parity["argmax_equal"]
        ):
            raise RuntimeError(
                "批处理 feed 与官方分段路径对拍失败："
                f"hidden cosine={parity['hidden_cosine']:.6f}，"
                f"logits cosine={parity['logits_cosine']:.6f}，"
                f"argmax_equal={parity['argmax_equal']}；请改用 --feed-mode segmented"
            )
        grouped_parity = _run_grouped_parity(
            duplex,
            bank,
            system_prompt=args.system_prompt,
        )
        if args.execution_mode == "grouped" and (
            grouped_parity["hidden_cosine"] < args.parity_min_cosine
            or grouped_parity["logits_cosine"] < args.parity_min_cosine
            or not grouped_parity["argmax_equal"]
        ):
            raise RuntimeError(
                "分块路径与逐单元路径对拍失败："
                f"hidden cosine={grouped_parity['hidden_cosine']:.6f}，"
                f"logits cosine={grouped_parity['logits_cosine']:.6f}，"
                f"argmax_equal={grouped_parity['argmax_equal']}；"
                "请改用 --execution-mode unit"
            )
        # 对拍本身已写入一单元，正式应力测试再次从干净会话开始。
        duplex.prepare(prefix_system_prompt=args.system_prompt)
        decoder = duplex.decoder
        modules = list(decoder.m.model.layers)
        layers = validate_layers(parse_int_csv(args.layers), len(modules))
        hidden_dim = int(decoder.m.config.hidden_size)
        official_max = int(decoder.m.config.max_position_embeddings)
        start_position = int(decoder.get_cache_length())
        unit_positions = int(bank_meta["unit_positions"])
        prefill_positions = int(sum(part.shape[0] for part in bank[0][:2]))
        max_rate = float(prefill_positions + args.max_new_speak_tokens + 1)
        min_rate = float(unit_positions)
        required = start_position + math.ceil(ANALYSIS_TARGET_SECONDS * max_rate)

        default_boundaries = [required, official_max, official_max * 2]
        if args.sliding_window_mode == "basic":
            default_boundaries = [8000, required, official_max, official_max * 2]
        boundaries = parse_int_csv(args.boundaries) or default_boundaries
        if args.max_position is not None:
            max_position = int(args.max_position)
        elif args.profile == "planned":
            max_position = required + args.dense_radius
        elif args.profile == "official":
            max_position = official_max + args.dense_radius
        else:
            max_position = official_max * 2 + args.dense_radius
        sample_positions = build_sample_positions(
            start_position=start_position,
            max_position=max_position,
            quantum=unit_positions,
            coarse_stride=args.sample_every,
            boundaries=boundaries,
            dense_radius=args.dense_radius,
        )
        sample_set = set(int(value) for value in sample_positions)
        logical_position = start_position
        dynamic_reference: dict[int, Any] = {}
        order = (
            rng.permutation(len(bank)).tolist()
            if args.bank_order == "randomized"
            else list(range(len(bank)))
        )
        order_index = 0
        listen_id = int(decoder.listen_id)

        manifest.update(
            {
                "layers": layers,
                "hidden_dim": hidden_dim,
                "planned_boundaries": boundaries,
                "sampling": {
                    "coarse_stride_positions": args.sample_every,
                    "dense_radius_positions": args.dense_radius,
                    "boundary_window_positions": args.boundary_window_positions,
                    "planned_sample_count": len(sample_positions),
                },
                "context_spec": {
                    "official_max_positions": official_max,
                    "start_position": start_position,
                    "logical_step_positions": unit_positions,
                    "positions_per_second_min": min_rate,
                    "positions_per_second_max": max_rate,
                    "positions_per_second_stress": min_rate,
                    "analysis_target_seconds": ANALYSIS_TARGET_SECONDS,
                    "analysis_target_required_positions": required,
                    "cache_policy": args.sliding_window_mode,
                    "basic_window_low_positions": 6000,
                    "basic_window_high_positions": 8000,
                },
                "embedding_bank": {
                    "units": len(bank),
                    "part_lengths": bank_meta["part_lengths"],
                    "unit_positions": unit_positions,
                    "audio_positions": bank_meta["audio_positions"],
                    "order": args.bank_order,
                    "labels_path": (
                        str(Path(args.bank_labels).resolve()) if args.bank_labels else None
                    ),
                    "labels_sha256": (
                        file_sha256(Path(args.bank_labels)) if args.bank_labels else None
                    ),
                },
                "parity": parity,
                "grouped_parity": grouped_parity,
            }
        )
        atomic_write_json(manifest_path, manifest)

        with SelectedLayerCapture(modules, layers) as capture, torch.inference_mode():
            while logical_position + unit_positions <= int(sample_positions[-1]):
                target_position = (
                    int(sample_positions[len(records["logical_positions"])])
                    if args.execution_mode == "grouped"
                    else logical_position + unit_positions
                )
                needed_units = (target_position - logical_position) // unit_positions
                if needed_units <= 0:
                    raise RuntimeError(
                        f"目标位置 {target_position} 与当前位置 {logical_position} 无法按单元对齐"
                    )
                unit_indices = []
                for _ in range(needed_units):
                    if order_index >= len(order):
                        order = (
                            rng.permutation(len(bank)).tolist()
                            if args.bank_order == "randomized"
                            else list(range(len(bank)))
                        )
                        order_index = 0
                    unit_indices.append(int(order[order_index]))
                    order_index += 1
                bank_index = unit_indices[-1]
                parts = bank[bank_index]
                projected_position = logical_position + needed_units * unit_positions
                sample_now = projected_position in sample_set

                if args.execution_mode == "unit":
                    decoder.register_unit_start()
                capture.clear()
                if args.execution_mode == "grouped":
                    grouped_parts = []
                    for index in unit_indices[:-1]:
                        grouped_parts.extend(bank[index])
                    grouped_parts.extend(parts[:2])
                    logits, _last_hidden = _extract_hidden_from_feed(
                        decoder.feed(
                            _cat_for_decoder(decoder, grouped_parts),
                            return_logits=True,
                        )
                    )
                elif args.feed_mode == "batched":
                    prefill = _cat_for_decoder(decoder, parts[:2])
                    logits, _last_hidden = _extract_hidden_from_feed(
                        decoder.feed(prefill, return_logits=True)
                    )
                else:
                    decoder.feed(parts[0])
                    logits, _last_hidden = _extract_hidden_from_feed(
                        decoder.feed(parts[1], return_logits=True)
                    )

                sampled_hidden = capture.last_vectors()[0] if sample_now else None
                if sample_now:
                    log_probs = torch.log_softmax(logits.detach().float()[0], dim=-1)
                    listen_probability = float(torch.exp(log_probs[listen_id]).item())
                    top_token_id = int(torch.argmax(log_probs).item())

                if args.feed_mode == "batched":
                    decoder.feed(_cat_for_decoder(decoder, parts[2:]))
                else:
                    decoder.feed(parts[2])
                    decoder.feed(parts[3])
                if args.execution_mode == "unit":
                    decoder.register_unit_end(
                        input_type="audio",
                        generated_tokens=[listen_id],
                        is_listen=True,
                        generated_text="",
                    )
                if args.sliding_window_mode == "basic":
                    decoder.enforce_window()
                logical_position = projected_position

                cache = decoder.cache
                dynamic_position = int(decoder._system_preserve_length)
                if not dynamic_reference and decoder.get_cache_length() > dynamic_position:
                    dynamic_reference = {
                        layer: cache_key(cache, layer, dynamic_position) for layer in layers
                    }
                if sample_now:
                    key_cosines = []
                    for layer in layers:
                        if decoder.get_cache_length() <= dynamic_position:
                            key_cosines.append(float("nan"))
                        else:
                            key_cosines.append(
                                cosine_to_reference(
                                    cache_key(cache, layer, dynamic_position),
                                    dynamic_reference[layer],
                                )
                            )
                    records["logical_positions"].append(logical_position)
                    records["cache_lengths"].append(int(decoder.get_cache_length()))
                    records["position_offsets"].append(int(decoder._position_offset))
                    records["hidden"].append(sampled_hidden)
                    records["decision_probs"].append(
                        [listen_probability, 1.0 - listen_probability]
                    )
                    records["decision_ids"].append(0 if top_token_id == listen_id else 1)
                    records["top_token_ids"].append(top_token_id)
                    records["task_labels"].append(int(bank_labels[bank_index]))
                    records["task_scores"].append(1.0 - listen_probability)
                    records["dynamic_key_cosines"].append(key_cosines)
                    records["all_finite"].append(
                        np.isfinite(sampled_hidden).all(axis=-1)
                    )
                    records["sliding_events"].append(int(decoder._sliding_event_count))
                    if len(records["logical_positions"]) % args.flush_every == 0:
                        flush()
                        guard.check()

        flush()
        guard.check(force=True)
        manifest.update(
            {
                "complete": True,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "actual_max_position": int(records["logical_positions"][-1]),
                "actual_sample_count": len(records["logical_positions"]),
                "max_gpu_temperature_c": guard.max_seen_c,
                "trace_sha256": file_sha256(trace_path),
                "final_cache_length": int(decoder.get_cache_length()),
                "final_position_offset": int(decoder._position_offset),
                "sliding_event_count": int(decoder._sliding_event_count),
                "binary_entropy_example": binary_entropy(
                    float(records["decision_probs"][-1][0])
                ),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"[minicpm-context] 完成：{out_dir}；位置 {records['logical_positions'][-1]}，"
            f"样本 {len(records['logical_positions'])}，滑窗 {decoder._sliding_event_count} 次"
        )
    except BaseException as exc:
        flush()
        manifest.update(
            {
                "complete": False,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "actual_sample_count": len(records["logical_positions"]),
                "max_gpu_temperature_c": guard.max_seen_c,
            }
        )
        if trace_path.is_file():
            manifest["trace_sha256"] = file_sha256(trace_path)
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    # 避免 Windows 多进程/动态模型导入时重复执行。
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
