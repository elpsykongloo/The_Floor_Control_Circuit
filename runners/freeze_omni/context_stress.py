"""Freeze-Omni 主干上下文应力运行器。

运行器用官方语音编码器和 adapter 从短音频建立真实的 12.5 Hz 嵌入库，随后以
Qwen2 的持续 KV 缓存处理长流。候选边界默认包含计划 600 秒位置、官方 32768
位置、两倍官方位置；每个候选点两侧加密采样。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SHARED_ROOT = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED_ROOT))

from context_stress_runner import (  # noqa: E402
    GpuTemperatureGuard,
    SelectedLayerCapture,
    atomic_save_npz,
    atomic_write_json,
    build_sample_positions,
    cache_key,
    cache_length,
    cosine_to_reference,
    file_sha256,
    load_binary_labels,
    parse_int_csv,
    validate_layers,
)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2560
TRACE_SCHEMA_VERSION = 1
ANALYSIS_TARGET_SECONDS = 600.0
POSITIONS_PER_SECOND = 12.5


def _load_standard_runner():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("freeze_omni_standard_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repeat_audio_chunk(audio: np.ndarray, chunk_index: int) -> np.ndarray:
    if len(audio) == 0:
        raise ValueError("音频为空")
    start = (chunk_index * CHUNK_SAMPLES) % len(audio)
    indices = (start + np.arange(CHUNK_SAMPLES)) % len(audio)
    return np.asarray(audio[indices], dtype=np.float32)


def _encode_audio_bank(pipeline, audio: np.ndarray, *, bank_chunks: int):
    """只走官方语音编码器与 adapter，返回每块两个主干嵌入及首块 fbank。"""

    import torch

    standard = _load_standard_runner()
    processor = standard.AudioEncoderProcessor()
    model = pipeline.model
    buffer = [None] * model.encoder.enc[1].num_blocks
    cnn_cache = None
    pe_index = 0
    bank = []
    first_features = None
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    ):
        for index in range(bank_chunks):
            features = processor.process(_repeat_audio_chunk(audio, index))
            if first_features is None:
                first_features = features.clone()
            speech = features.to("cuda")
            encoder_out, buffer, _, _, pe_index = model.encoder.infer(
                speech,
                buffer,
                0,
                None,
                pe_index,
            )
            encoder_mask = torch.full(
                encoder_out.shape[:2],
                True,
                device=encoder_out.device,
            ).unsqueeze(1)
            embeds, _mask, cnn_cache = model.adpter(
                encoder_out,
                encoder_mask,
                cache=cnn_cache,
                return_cache=True,
            )
            if embeds.shape[0] != 1 or embeds.shape[1] != 2:
                raise RuntimeError(
                    f"Freeze-Omni 每 160 ms 应产生 2 个嵌入，实际 {tuple(embeds.shape)}"
                )
            bank.append(embeds[0].detach().clone())
    if first_features is None:
        raise RuntimeError("音频嵌入库为空")
    return bank, first_features


def _first_audio_inputs(model, audio_embeds):
    """复现 ``AudioLLM.recognize`` 在首个 sl 块上的提示拼接。"""

    import torch

    inputs = audio_embeds.unsqueeze(0)
    if model.prompt_finetune and model.add_prompt_before:
        prompt_ids = model.prompt_ids.repeat(1, 1).to(inputs.device)
        prompt = model.prompt_embeddings(prompt_ids)
        inputs = torch.cat((prompt, inputs), dim=1)
    if model.chat_template is not None:
        chat_prefix = model.chat_template["prefix"].to(inputs.device)
        chat_prefix = torch.cat(
            (
                torch.tensor([[model.tokenizer.eod_id]], device=inputs.device),
                chat_prefix,
            ),
            dim=1,
        )
        chat_prefix_embeds = model.llm_decoder.transformer.wte(chat_prefix)
        inputs = torch.cat((chat_prefix_embeds, inputs), dim=1)
    return inputs.to(dtype=model.llm_decoder.dtype)


def _direct_forward(model, cache, inputs_embeds):
    """按官方 `_generate_one_step` 的主干调用方式前向一个嵌入块。"""

    import torch

    past_length = cache_length(cache)
    attention_mask = torch.ones(
        (1, past_length + inputs_embeds.shape[1]),
        dtype=torch.bool,
        device=inputs_embeds.device,
    )
    return model.llm_decoder.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


def _official_first_chunk(pipeline, pre_outputs: dict[str, Any], first_features):
    """运行官方首块并捕获 predictor_head 的输入，供直接路径对拍。"""

    captured = {}

    def hook(_module, inputs, output) -> None:
        captured["hidden"] = inputs[0].detach().float().cpu().numpy()
        captured["logits"] = output.detach().float().cpu().numpy()

    handle = pipeline.model.predictor_head.register_forward_hook(hook)
    try:
        outputs = pipeline.speech_dialogue(first_features, **pre_outputs)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("官方首块没有触发 predictor_head")
    probabilities = np.asarray(pipeline.model.last_state_prob, dtype=np.float64)
    return {
        "hidden": captured["hidden"][:, -1, :],
        "probabilities": probabilities,
        "cache_length": cache_length(outputs["past_key_values"]),
    }


def _cosine_numpy(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)


def _trace_arrays(records: dict[str, list[Any]], n_layers: int, hidden_dim: int) -> dict[str, np.ndarray]:
    if not records["logical_positions"]:
        return {
            "logical_positions": np.empty(0, dtype=np.int64),
            "cache_lengths": np.empty(0, dtype=np.int64),
            "position_offsets": np.empty(0, dtype=np.int64),
            "hidden": np.empty((0, n_layers, hidden_dim), dtype=np.float16),
            "decision_probs": np.empty((0, 3), dtype=np.float32),
            "decision_ids": np.empty(0, dtype=np.int16),
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
        "task_labels": np.asarray(records["task_labels"], dtype=np.int8),
        "task_scores": np.asarray(records["task_scores"], dtype=np.float32),
        "dynamic_key_cosines": np.asarray(records["dynamic_key_cosines"], dtype=np.float32),
        "all_finite": np.asarray(records["all_finite"], dtype=bool),
        "sliding_events": np.asarray(records["sliding_events"], dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze-Omni 上下文应力诊断")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", default="You are a helpful assistant.")
    parser.add_argument("--profile", choices=["planned", "official", "double"], default="planned")
    parser.add_argument("--max-position", type=int, default=None)
    parser.add_argument("--boundaries", default=None)
    parser.add_argument("--layers", default=None, help="零基层号，逗号分隔；默认四个深度")
    parser.add_argument("--bank-chunks", type=int, default=64)
    parser.add_argument("--bank-labels", default=None, help="与嵌入库逐块对齐的 .npy 或 JSON 0/1 标签")
    parser.add_argument(
        "--bank-order",
        choices=["randomized", "sequential"],
        default="randomized",
        help="带时间目标标签的正式性能诊断必须使用 sequential",
    )
    parser.add_argument("--sample-every", type=int, default=256)
    parser.add_argument("--dense-radius", type=int, default=128)
    parser.add_argument("--boundary-window-positions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=64)
    parser.add_argument("--parity-min-cosine", type=float, default=0.999)
    parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--temperature-limit", type=int, default=95)
    args = parser.parse_args()

    os.environ["FREEZE_OMNI_ATTN_IMPLEMENTATION"] = args.attn_implementation
    import torch

    standard = _load_standard_runner()
    source_root = args.source_root or str(Path(args.model_root) / "_source")
    sys.path.insert(0, source_root)
    from models.pipeline import inferencePipeline

    root = Path(args.model_root)
    pipeline_args = SimpleNamespace(
        model_path=str(root / "_assets" / "checkpoints"),
        llm_path=str(root / "_assets" / "Qwen2-7B-Instruct"),
        top_p=0.8,
        top_k=20,
        temperature=0.8,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.npz"
    manifest_path = out_dir / "manifest.json"
    started = time.time()
    guard = GpuTemperatureGuard(limit_c=args.temperature_limit)
    rng = np.random.default_rng(args.seed)
    records: dict[str, list[Any]] = {
        "logical_positions": [],
        "cache_lengths": [],
        "position_offsets": [],
        "hidden": [],
        "decision_probs": [],
        "decision_ids": [],
        "task_labels": [],
        "task_scores": [],
        "dynamic_key_cosines": [],
        "all_finite": [],
        "sliding_events": [],
    }
    manifest: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": "context_stress_v1",
        "model": "freeze_omni",
        "run_id": args.run_id,
        "complete": False,
        "started_unix": started,
        "source_audio": {
            "path": str(Path(args.audio).resolve()),
            "sha256": file_sha256(Path(args.audio)),
        },
        "feed_mode": "direct_batched_audio_embeddings",
        "seed": args.seed,
        "attention_implementation": args.attn_implementation,
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
        pipeline = inferencePipeline(pipeline_args)
        model = pipeline.model
        audio = standard.load_audio_16k(args.audio)
        bank, first_features = _encode_audio_bank(
            pipeline,
            audio,
            bank_chunks=args.bank_chunks,
        )
        bank_labels = load_binary_labels(args.bank_labels, len(bank))
        if args.bank_labels is not None and args.bank_order != "sequential":
            raise ValueError("带时间目标标签时必须使用 --bank-order sequential")

        # 官方首块参考。
        official_pre = pipeline.speech_dialogue(None, stat="pre", role=args.role)
        official = _official_first_chunk(pipeline, official_pre, first_features)

        # 直接路径首块对拍。
        direct_pre = pipeline.speech_dialogue(None, stat="pre", role=args.role)
        initial_system_cache = direct_pre["past_key_values"]
        first_inputs = _first_audio_inputs(model, bank[0])
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        ):
            direct_out = _direct_forward(model, initial_system_cache, first_inputs)
            direct_logits = model.predictor_head(direct_out.last_hidden_state)[0, -1, :-1]
            direct_probabilities = torch.softmax(direct_logits, dim=-1).detach().float().cpu().numpy()
        direct_hidden = direct_out.last_hidden_state[:, -1, :].detach().float().cpu().numpy()
        parity = {
            "hidden_cosine": _cosine_numpy(direct_hidden, official["hidden"]),
            "hidden_max_abs": float(np.max(np.abs(direct_hidden - official["hidden"]))),
            "probability_max_abs": float(
                np.max(np.abs(direct_probabilities - official["probabilities"]))
            ),
            "argmax_equal": bool(
                int(np.argmax(direct_probabilities)) == int(np.argmax(official["probabilities"]))
            ),
            "cache_length_equal": bool(
                cache_length(direct_out.past_key_values) == official["cache_length"]
            ),
        }
        if (
            parity["hidden_cosine"] < args.parity_min_cosine
            or not parity["argmax_equal"]
            or not parity["cache_length_equal"]
        ):
            raise RuntimeError(
                "直接批处理路径与官方首块对拍失败："
                f"hidden cosine={parity['hidden_cosine']:.6f}，"
                f"argmax_equal={parity['argmax_equal']}，"
                f"cache_length_equal={parity['cache_length_equal']}"
            )

        # 正式运行从第三份干净系统缓存开始。
        stress_pre = pipeline.speech_dialogue(None, stat="pre", role=args.role)
        cache = stress_pre["past_key_values"]
        system_positions = cache_length(cache)
        first_inputs = _first_audio_inputs(model, bank[0])
        fixed_prefix_positions = system_positions + int(first_inputs.shape[1]) - 2
        official_max = int(model.llm_decoder.config.max_position_embeddings)
        required = fixed_prefix_positions + math.ceil(
            ANALYSIS_TARGET_SECONDS * POSITIONS_PER_SECOND
        )
        boundaries = parse_int_csv(args.boundaries) or [
            required,
            official_max,
            official_max * 2,
        ]
        if args.max_position is not None:
            max_position = int(args.max_position)
        elif args.profile == "planned":
            max_position = required + args.dense_radius
        elif args.profile == "official":
            max_position = official_max + args.dense_radius
        else:
            max_position = official_max * 2 + args.dense_radius
        sample_positions = build_sample_positions(
            start_position=fixed_prefix_positions,
            max_position=max_position,
            quantum=2,
            coarse_stride=args.sample_every,
            boundaries=boundaries,
            dense_radius=args.dense_radius,
        )
        modules = list(model.llm_decoder.transformer.h)
        layers = validate_layers(parse_int_csv(args.layers), len(modules))
        hidden_dim = int(model.llm_decoder.config.hidden_size)
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
                    "start_position": fixed_prefix_positions,
                    "logical_step_positions": 2,
                    "positions_per_second_min": POSITIONS_PER_SECOND,
                    "positions_per_second_max": POSITIONS_PER_SECOND,
                    "positions_per_second_stress": POSITIONS_PER_SECOND,
                    "analysis_target_seconds": ANALYSIS_TARGET_SECONDS,
                    "analysis_target_required_positions": required,
                    "cache_policy": "no_sliding",
                    "encoder_local_context_seconds": 2.72,
                    "training_filter_seconds": 20.0,
                },
                "embedding_bank": {
                    "chunks": len(bank),
                    "positions_per_chunk": 2,
                    "chunk_seconds": 0.16,
                    "order": args.bank_order,
                    "labels_path": (
                        str(Path(args.bank_labels).resolve()) if args.bank_labels else None
                    ),
                    "labels_sha256": (
                        file_sha256(Path(args.bank_labels)) if args.bank_labels else None
                    ),
                },
                "parity": parity,
            }
        )
        atomic_write_json(manifest_path, manifest)

        order = (
            rng.permutation(len(bank)).tolist()
            if args.bank_order == "randomized"
            else list(range(len(bank)))
        )
        order_index = 1 if args.bank_order == "sequential" else 0
        logical_position = fixed_prefix_positions
        dynamic_audio_position = fixed_prefix_positions
        dynamic_reference: dict[int, Any] = {}

        with SelectedLayerCapture(modules, layers) as capture, torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        ):
            for target_position in sample_positions:
                target_position = int(target_position)
                needed_positions = target_position - logical_position
                if needed_positions <= 0 or needed_positions % 2:
                    raise RuntimeError(
                        f"采样计划位置 {target_position} 与当前 {logical_position} 无法按 2 位置对齐"
                    )
                chunks = []
                chunk_indices = []
                for _ in range(needed_positions // 2):
                    if logical_position == fixed_prefix_positions and not chunks:
                        chunks.append(bank[0])
                        chunk_indices.append(0)
                        continue
                    if order_index >= len(order):
                        order = (
                            rng.permutation(len(bank)).tolist()
                            if args.bank_order == "randomized"
                            else list(range(len(bank)))
                        )
                        order_index = 0
                    bank_index = int(order[order_index])
                    chunks.append(bank[bank_index])
                    chunk_indices.append(bank_index)
                    order_index += 1
                audio_inputs = torch.cat(chunks, dim=0).unsqueeze(0)
                if logical_position == fixed_prefix_positions:
                    # 首块需要一次性加入 chat/prompt 固定前缀。
                    first = _first_audio_inputs(model, chunks[0])
                    if len(chunks) > 1:
                        audio_inputs = torch.cat(
                            (first, torch.cat(chunks[1:], dim=0).unsqueeze(0)),
                            dim=1,
                        )
                    else:
                        audio_inputs = first
                capture.clear()
                outputs = _direct_forward(model, cache, audio_inputs)
                cache = outputs.past_key_values
                logical_position = target_position
                hidden = capture.last_vectors()[0]
                state_logits = model.predictor_head(outputs.last_hidden_state)[0, -1, :-1]
                probabilities = torch.softmax(state_logits, dim=-1).detach().float().cpu().numpy()

                if not dynamic_reference:
                    dynamic_reference = {
                        layer: cache_key(cache, layer, dynamic_audio_position)
                        for layer in layers
                    }
                key_cosines = [
                    cosine_to_reference(
                        cache_key(cache, layer, dynamic_audio_position),
                        dynamic_reference[layer],
                    )
                    for layer in layers
                ]
                records["logical_positions"].append(logical_position)
                records["cache_lengths"].append(cache_length(cache))
                records["position_offsets"].append(0)
                records["hidden"].append(hidden)
                records["decision_probs"].append(probabilities)
                records["decision_ids"].append(int(np.argmax(probabilities)))
                records["task_labels"].append(int(bank_labels[chunk_indices[-1]]))
                # 官方状态 1 触发 sl/cl → ss，作为原生“开始说”分数。
                records["task_scores"].append(float(probabilities[1]))
                records["dynamic_key_cosines"].append(key_cosines)
                records["all_finite"].append(np.isfinite(hidden).all(axis=-1))
                records["sliding_events"].append(0)
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
                "final_cache_length": cache_length(cache),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"[freeze-context] 完成：{out_dir}；位置 {records['logical_positions'][-1]}，"
            f"样本 {len(records['logical_positions'])}"
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
