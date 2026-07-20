"""MiniCPM-o 4.5 密集全双工三角对照上下文测量运行器。

实验包含三部分：

1. 通过官方 ``streaming_prefill + streaming_generate`` 路径建立真实音频嵌入和
   模型生成 token 库；密集库只屏蔽倾听与提前终止 token，正文 token 仍由模型
   贪心选择，每个一秒单元固定消耗 32 个主干位置。
2. 长流侧持续重放密集全双工单元，在预定秒数插入未干预的自然决策探针。
3. 每个探针建立两个近期后缀复位侧：一个使用低位置，另一个把相同后缀平移到
   与长流相同的绝对 RoPE 位置。三路近期内容逐项一致，从而分离远端历史内容
   与绝对位置对隐藏状态、决策分布和生成行为的影响。

大数组只写 ``--out``，必须放在 D 盘数据目录。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
    file_sha256,
    parse_int_csv,
    validate_layers,
)

TRACE_SCHEMA_VERSION = 2
DEFAULT_CHECKPOINT_SECONDS = (
    "64,256,512,768,1024,1200,1260,1280,1300,1400,"
    "1600,1800,1950,2000,2050,2200,2400,2560,2600"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_helpers():
    current = Path(__file__).resolve().parent
    standard = _load_module(current / "run.py", "minicpm_dense_standard_runner")
    context = _load_module(current / "context_stress.py", "minicpm_dense_context_helpers")
    return standard, context


def _parse_positive_csv(value: str, *, name: str) -> list[int]:
    values = parse_int_csv(value)
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} 必须包含正整数")
    if values != sorted(set(values)):
        raise ValueError(f"{name} 必须严格递增且无重复")
    return values


@dataclass(frozen=True)
class ReplayUnit:
    """一个可精确重放的完整一秒单元。"""

    bank_index: int
    tail_ids: tuple[int, ...]
    mode: str
    group_id: int
    is_listen: bool
    generated_text: str


@dataclass
class LongObservation:
    """长流探针及其复位对照所需的近期历史。"""

    target_seconds: int
    long_input_seconds: int
    long_position: int
    probe_index: int
    long_hidden: np.ndarray
    long_logits: np.ndarray
    long_generated_ids: tuple[int, ...]
    long_cache_length: int
    long_all_finite: np.ndarray
    recent_history: tuple[ReplayUnit, ...]


@contextmanager
def _force_dense_decode(duplex):
    """只屏蔽倾听和提前终止，保留模型对正文 token 的真实贪心选择。"""

    decoder = duplex.decoder
    original_decode = decoder.decode
    masked_ids = sorted(
        {
            int(duplex.listen_token_id),
            int(duplex.chunk_eos_token_id),
            int(duplex.chunk_tts_eos_token_id),
            int(duplex.turn_eos_token_id),
        }
    )

    def dense_decode(logits, *args, **kwargs):
        import torch

        adjusted = logits.clone()
        adjusted[:, masked_ids] = -torch.inf
        kwargs["mode"] = "greedy"
        kwargs["listen_prob_scale"] = 1.0
        kwargs["text_repetition_penalty"] = 1.0
        return original_decode(adjusted, *args, **kwargs)

    decoder.decode = dense_decode
    try:
        yield
    finally:
        decoder.decode = original_decode


@contextmanager
def _shift_decoder_positions(decoder, position_shift: int):
    """保持 KV 紧凑存放，同时把动态后缀映射到更高的绝对 RoPE 位置。"""

    if position_shift < 0:
        raise ValueError("绝对位置平移量不能为负")
    original_feed = decoder.feed

    def shifted_feed(embeds, return_logits: bool = False):
        import torch

        length = embeds.size(0)
        past_length = decoder.get_cache_length()
        position_ids = torch.arange(
            past_length + position_shift,
            past_length + position_shift + length,
            device=embeds.device,
        ).unsqueeze(0)
        output = decoder.m(
            inputs_embeds=embeds.unsqueeze(0),
            position_ids=position_ids,
            past_key_values=decoder.cache,
            return_dict=True,
            output_hidden_states=True,
        )
        decoder.cache = output.past_key_values
        if return_logits:
            logits = decoder.m.lm_head(output.hidden_states[-1])[:, -1]
            return logits, output.hidden_states[-1]
        return None

    decoder.feed = shifted_feed
    try:
        yield
    finally:
        decoder.feed = original_feed


def _feed_prefill(decoder, context_helpers, parts: list[Any]):
    return context_helpers._extract_hidden_from_feed(
        decoder.feed(
            context_helpers._cat_for_decoder(decoder, parts[:2]),
            return_logits=True,
        )
    )


def _run_official_generation(
    duplex,
    *,
    logits,
    max_new_speak_tokens: int,
    dense: bool,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    duplex.pending_logits = logits
    duplex.current_mode = "AUDIO"
    duplex.audio_chunk_idx += 1
    before = len(duplex.total_ids)
    generation_kwargs = {
        "max_new_speak_tokens_per_chunk": max_new_speak_tokens,
        "decode_mode": "greedy",
        "listen_prob_scale": 1.0,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 1,
    }
    if dense:
        duplex.current_turn_ended = False
        with _force_dense_decode(duplex):
            result = duplex.streaming_generate(**generation_kwargs)
    else:
        # 每个自然探针都从“上一话轮已结束”状态开始，允许模型自由选择听或说。
        duplex.current_turn_ended = True
        result = duplex.streaming_generate(**generation_kwargs)
    tail_ids = tuple(int(value) for value in duplex.total_ids[before:])
    return result, tail_ids


def _build_dense_tail_bank(
    duplex,
    context_helpers,
    bank: list[list[Any]],
    *,
    system_prompt: str,
    max_new_speak_tokens: int,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """以官方生成状态机产生固定 32 位置单元所需的真实正文 token。"""

    duplex.prepare(prefix_system_prompt=system_prompt)
    decoder = duplex.decoder
    start = decoder.get_cache_length()
    tails: list[tuple[int, ...]] = []
    texts: list[str] = []
    for bank_index, parts in enumerate(bank):
        before = decoder.get_cache_length()
        decoder.register_unit_start()
        logits, _hidden = _feed_prefill(decoder, context_helpers, parts)
        result, tail_ids = _run_official_generation(
            duplex,
            logits=logits,
            max_new_speak_tokens=max_new_speak_tokens,
            dense=True,
        )
        added = decoder.get_cache_length() - before
        expected = sum(int(part.shape[0]) for part in parts[:2]) + max_new_speak_tokens + 1
        if added != expected:
            raise RuntimeError(
                f"密集库单元 {bank_index} 位置数为 {added}，期望 {expected}"
            )
        if bool(result.get("is_listen", True)):
            raise RuntimeError(f"密集库单元 {bank_index} 意外进入倾听态")
        if len(tail_ids) != max_new_speak_tokens + 1:
            raise RuntimeError(
                f"密集库单元 {bank_index} 尾部 token 数为 {len(tail_ids)}，"
                f"期望 {max_new_speak_tokens + 1}"
            )
        tails.append(tail_ids)
        texts.append(str(result.get("text", "")))
    flat_content = [
        token_id
        for tail in tails
        for token_id in tail[: max_new_speak_tokens - 1]
    ]
    return tails, {
        "start_position": start,
        "end_position": decoder.get_cache_length(),
        "unit_positions": int((decoder.get_cache_length() - start) / len(bank)),
        "tail_positions": max_new_speak_tokens + 1,
        "content_token_count": len(flat_content),
        "unique_content_token_count": len(set(flat_content)),
        "unique_content_token_ratio": (
            float(len(set(flat_content)) / len(flat_content))
            if flat_content
            else 0.0
        ),
        "decoded_examples": texts[: min(4, len(texts))],
    }


def _unit_tensor(decoder, context_helpers, parts: list[Any], tail_ids: tuple[int, ...]):
    token_embeds = decoder.embed_tokens(list(tail_ids))
    return context_helpers._cat_for_decoder(
        decoder,
        [*parts[:2], token_embeds],
    )


def _dense_replay_parity(
    duplex,
    context_helpers,
    bank: list[list[Any]],
    dense_tails: list[tuple[int, ...]],
    *,
    system_prompt: str,
    group_units: int,
) -> dict[str, Any]:
    """对拍官方逐 token 密集路径与长流批量重放路径。"""

    n_units = min(group_units, len(bank) - 1)
    if n_units < 2:
        raise ValueError("密集重放对拍至少需要三个嵌入单元")

    duplex.prepare(prefix_system_prompt=system_prompt)
    decoder = duplex.decoder
    for index in range(n_units):
        decoder.feed(context_helpers._cat_for_decoder(decoder, bank[index][:2]))
        for token_id in dense_tails[index]:
            decoder.feed(decoder.embed_token(token_id))
    reference_logits, reference_hidden = _feed_prefill(
        decoder,
        context_helpers,
        bank[n_units],
    )

    duplex.prepare(prefix_system_prompt=system_prompt)
    decoder = duplex.decoder
    grouped = [
        _unit_tensor(decoder, context_helpers, bank[index], dense_tails[index])
        for index in range(n_units)
    ]
    decoder.feed(context_helpers._cat_for_decoder(decoder, grouped))
    grouped_logits, grouped_hidden = _feed_prefill(
        decoder,
        context_helpers,
        bank[n_units],
    )

    reference_logits_np = reference_logits.detach().float().cpu().numpy()
    grouped_logits_np = grouped_logits.detach().float().cpu().numpy()
    reference_hidden_np = reference_hidden[:, -1, :].detach().float().cpu().numpy()
    grouped_hidden_np = grouped_hidden[:, -1, :].detach().float().cpu().numpy()
    return {
        "units": n_units,
        "hidden_cosine": context_helpers._cosine_numpy(
            reference_hidden_np,
            grouped_hidden_np,
        ),
        "hidden_max_abs": float(
            np.max(np.abs(reference_hidden_np - grouped_hidden_np))
        ),
        "logits_cosine": context_helpers._cosine_numpy(
            reference_logits_np,
            grouped_logits_np,
        ),
        "logits_max_abs": float(
            np.max(np.abs(reference_logits_np - grouped_logits_np))
        ),
        "argmax_equal": bool(
            int(np.argmax(reference_logits_np)) == int(np.argmax(grouped_logits_np))
        ),
    }


def _append_filler_group(
    duplex,
    context_helpers,
    bank: list[list[Any]],
    dense_tails: list[tuple[int, ...]],
    bank_indices: list[int],
    *,
    history: list[ReplayUnit],
    group_id: int,
) -> None:
    decoder = duplex.decoder
    tensors = [
        _unit_tensor(decoder, context_helpers, bank[index], dense_tails[index])
        for index in bank_indices
    ]
    decoder.feed(context_helpers._cat_for_decoder(decoder, tensors))
    duplex.audio_chunk_idx += len(bank_indices)
    duplex.current_turn_ended = False
    for bank_index in bank_indices:
        tail_ids = dense_tails[bank_index]
        duplex.total_ids.extend(tail_ids)
        history.append(
            ReplayUnit(
                bank_index=bank_index,
                tail_ids=tail_ids,
                mode="filler",
                group_id=group_id,
                is_listen=False,
                generated_text="",
            )
        )


def _run_probe(
    duplex,
    context_helpers,
    capture: SelectedLayerCapture,
    bank: list[list[Any]],
    *,
    bank_index: int,
    max_new_speak_tokens: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, Any]]:
    decoder = duplex.decoder
    parts = bank[bank_index]
    decoder.register_unit_start()
    capture.clear()
    logits, _last_hidden = _feed_prefill(decoder, context_helpers, parts)
    sampled_hidden = capture.last_vectors()[0]
    position = int(decoder.get_cache_length())
    result, tail_ids = _run_official_generation(
        duplex,
        logits=logits,
        max_new_speak_tokens=max_new_speak_tokens,
        dense=False,
    )
    return (
        sampled_hidden.astype(np.float16),
        logits.detach().float().cpu().numpy()[0].astype(np.float16),
        tail_ids,
        {
            "position": position,
            "cache_after": int(decoder.get_cache_length()),
            "is_listen": bool(result.get("is_listen", False)),
            "generated_text": str(result.get("text", "")),
        },
    )


def _feed_replay_units(
    duplex,
    context_helpers,
    bank: list[list[Any]],
    units: tuple[ReplayUnit, ...],
    *,
    max_group_units: int,
) -> None:
    """按长流的单元类型重放近期历史；连续 filler 仍用同类批处理。"""

    decoder = duplex.decoder
    index = 0
    while index < len(units):
        unit = units[index]
        if unit.mode == "filler":
            group = [unit]
            next_index = index + 1
            while (
                next_index < len(units)
                and len(group) < max_group_units
                and units[next_index].mode == "filler"
                and units[next_index].group_id == unit.group_id
            ):
                group.append(units[next_index])
                next_index += 1
            tensors = [
                _unit_tensor(
                    decoder,
                    context_helpers,
                    bank[item.bank_index],
                    item.tail_ids,
                )
                for item in group
            ]
            decoder.feed(context_helpers._cat_for_decoder(decoder, tensors))
            duplex.audio_chunk_idx += len(group)
            duplex.total_ids.extend(
                token_id
                for item in group
                for token_id in item.tail_ids
            )
            index = next_index
            continue

        decoder.feed(
            context_helpers._cat_for_decoder(
                decoder,
                bank[unit.bank_index][:2],
            )
        )
        for token_id in unit.tail_ids:
            decoder.feed(decoder.embed_token(token_id))
        duplex.audio_chunk_idx += 1
        duplex.total_ids.extend(unit.tail_ids)
        index += 1


def _pad_token_rows(rows: list[tuple[int, ...]], width: int) -> np.ndarray:
    out = np.full((len(rows), width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        if len(row) > width:
            raise ValueError(f"生成 token 数 {len(row)} 超过固定宽度 {width}")
        out[index, : len(row)] = np.asarray(row, dtype=np.int32)
    return out


def _pack_trace(
    observations: list[LongObservation],
    controls: list[dict[str, Any]],
    position_controls: list[dict[str, Any]],
    *,
    dense_tails: list[tuple[int, ...]],
    max_tail_tokens: int,
) -> dict[str, np.ndarray]:
    if not (
        len(observations) == len(controls) == len(position_controls)
    ):
        raise ValueError("长流观测、低位复位对照与绝对位置对照数量不一致")
    return {
        "target_seconds": np.asarray(
            [item.target_seconds for item in observations],
            dtype=np.int32,
        ),
        "long_input_seconds": np.asarray(
            [item.long_input_seconds for item in observations],
            dtype=np.int32,
        ),
        "long_positions": np.asarray(
            [item.long_position for item in observations],
            dtype=np.int64,
        ),
        "control_positions": np.asarray(
            [item["position"] for item in controls],
            dtype=np.int64,
        ),
        "position_control_positions": np.asarray(
            [item["position"] for item in position_controls],
            dtype=np.int64,
        ),
        "position_control_absolute_positions": np.asarray(
            [item["absolute_position"] for item in position_controls],
            dtype=np.int64,
        ),
        "position_control_shifts": np.asarray(
            [item["position_shift"] for item in position_controls],
            dtype=np.int64,
        ),
        "probe_indices": np.asarray(
            [item.probe_index for item in observations],
            dtype=np.int16,
        ),
        "long_hidden": np.stack([item.long_hidden for item in observations]),
        "control_hidden": np.stack([item["hidden"] for item in controls]),
        "position_control_hidden": np.stack(
            [item["hidden"] for item in position_controls]
        ),
        "long_logits": np.stack([item.long_logits for item in observations]),
        "control_logits": np.stack([item["logits"] for item in controls]),
        "position_control_logits": np.stack(
            [item["logits"] for item in position_controls]
        ),
        "long_generated_ids": _pad_token_rows(
            [item.long_generated_ids for item in observations],
            max_tail_tokens,
        ),
        "control_generated_ids": _pad_token_rows(
            [item["generated_ids"] for item in controls],
            max_tail_tokens,
        ),
        "position_control_generated_ids": _pad_token_rows(
            [item["generated_ids"] for item in position_controls],
            max_tail_tokens,
        ),
        "long_generated_lengths": np.asarray(
            [len(item.long_generated_ids) for item in observations],
            dtype=np.int16,
        ),
        "control_generated_lengths": np.asarray(
            [len(item["generated_ids"]) for item in controls],
            dtype=np.int16,
        ),
        "position_control_generated_lengths": np.asarray(
            [len(item["generated_ids"]) for item in position_controls],
            dtype=np.int16,
        ),
        "long_cache_lengths": np.asarray(
            [item.long_cache_length for item in observations],
            dtype=np.int64,
        ),
        "control_cache_lengths": np.asarray(
            [item["cache_after"] for item in controls],
            dtype=np.int64,
        ),
        "position_control_cache_lengths": np.asarray(
            [item["cache_after"] for item in position_controls],
            dtype=np.int64,
        ),
        "long_all_finite": np.stack(
            [item.long_all_finite for item in observations]
        ).astype(bool),
        "control_all_finite": np.stack(
            [item["all_finite"] for item in controls]
        ).astype(bool),
        "position_control_all_finite": np.stack(
            [item["all_finite"] for item in position_controls]
        ).astype(bool),
        "dense_tail_ids": _pad_token_rows(dense_tails, max_tail_tokens),
        "dense_tail_lengths": np.asarray(
            [len(item) for item in dense_tails],
            dtype=np.int16,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-o 4.5 密集全双工三角对照上下文测量"
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", default=None, help="零基层号，逗号分隔")
    parser.add_argument("--bank-units", type=int, default=32)
    parser.add_argument("--checkpoint-seconds", default=DEFAULT_CHECKPOINT_SECONDS)
    parser.add_argument("--max-seconds", type=int, default=2600)
    parser.add_argument("--probes-per-checkpoint", type=int, default=3)
    parser.add_argument("--probe-indices", default=None, help="嵌入库索引，逗号分隔")
    parser.add_argument("--control-suffix-units", type=int, default=64)
    parser.add_argument("--filler-group-units", type=int, default=8)
    parser.add_argument("--max-new-speak-tokens", type=int, default=20)
    parser.add_argument(
        "--system-prompt",
        default="Streaming audio conversation. Please answer naturally and briefly.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity-min-cosine", type=float, default=0.999)
    parser.add_argument("--temperature-limit", type=int, default=95)
    args = parser.parse_args()

    checkpoints = _parse_positive_csv(
        args.checkpoint_seconds,
        name="--checkpoint-seconds",
    )
    if args.max_seconds < checkpoints[-1]:
        parser.error("--max-seconds 不能小于最后一个检查点")
    if args.bank_units < 4:
        parser.error("--bank-units 至少为 4")
    if args.probes_per_checkpoint < 2:
        parser.error("--probes-per-checkpoint 至少为 2")
    if args.control_suffix_units < 1:
        parser.error("--control-suffix-units 至少为 1")
    if args.filler_group_units < 1:
        parser.error("--filler-group-units 至少为 1")
    if args.max_new_speak_tokens < 3:
        parser.error("--max-new-speak-tokens 至少为 3")

    import torch
    from transformers import AutoModel

    standard, context_helpers = _load_helpers()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    trace_path = out_dir / "trace.npz"
    started = time.time()
    guard = GpuTemperatureGuard(
        limit_c=args.temperature_limit,
        check_interval_s=15.0,
    )
    manifest: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "protocol": "minicpm_dense_context_v1",
        "model": "minicpm_o_4_5",
        "run_id": args.run_id,
        "complete": False,
        "started_unix": started,
        "source_audio": {
            "path": str(Path(args.audio).resolve()),
            "sha256": file_sha256(Path(args.audio)),
        },
        "seed": args.seed,
    }
    atomic_write_json(manifest_path, manifest)

    observations: list[LongObservation] = []
    controls: list[dict[str, Any]] = []
    position_controls: list[dict[str, Any]] = []
    dense_tails: list[tuple[int, ...]] = []
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
            sliding_window_mode="off",
            text_repetition_penalty=1.0,
        )
        duplex.prepare(prefix_system_prompt=args.system_prompt)
        audio = standard.load_audio_16k(args.audio)
        bank, bank_meta = context_helpers._collect_embedding_bank(
            duplex,
            audio,
            bank_units=args.bank_units,
            max_new_speak_tokens=args.max_new_speak_tokens,
        )
        dense_tails, dense_bank_meta = _build_dense_tail_bank(
            duplex,
            context_helpers,
            bank,
            system_prompt=args.system_prompt,
            max_new_speak_tokens=args.max_new_speak_tokens,
        )
        dense_unit_positions = int(dense_bank_meta["unit_positions"])
        if dense_unit_positions != 32 and args.max_new_speak_tokens == 20:
            raise RuntimeError(
                f"默认密集单元应为 32 位置，实测为 {dense_unit_positions}"
            )
        parity = _dense_replay_parity(
            duplex,
            context_helpers,
            bank,
            dense_tails,
            system_prompt=args.system_prompt,
            group_units=args.filler_group_units,
        )
        if (
            parity["hidden_cosine"] < args.parity_min_cosine
            or parity["logits_cosine"] < args.parity_min_cosine
            or not parity["argmax_equal"]
        ):
            raise RuntimeError(
                "密集批量重放与逐 token 路径对拍失败："
                f"hidden cosine={parity['hidden_cosine']:.6f}，"
                f"logits cosine={parity['logits_cosine']:.6f}，"
                f"argmax_equal={parity['argmax_equal']}"
            )

        duplex.prepare(prefix_system_prompt=args.system_prompt)
        decoder = duplex.decoder
        modules = list(decoder.m.model.layers)
        layers = validate_layers(parse_int_csv(args.layers), len(modules))
        hidden_dim = int(decoder.m.config.hidden_size)
        official_max = int(decoder.m.config.max_position_embeddings)
        start_position = int(decoder.get_cache_length())
        prefill_positions = int(sum(part.shape[0] for part in bank[0][:2]))
        if args.probe_indices is None:
            probe_indices = np.linspace(
                0,
                len(bank) - 1,
                num=args.probes_per_checkpoint,
                dtype=np.int64,
            ).tolist()
        else:
            probe_indices = parse_int_csv(args.probe_indices)
            if len(probe_indices) != args.probes_per_checkpoint:
                parser.error("--probe-indices 数量必须等于 --probes-per-checkpoint")
            if any(index < 0 or index >= len(bank) for index in probe_indices):
                parser.error("--probe-indices 超出嵌入库范围")

        manifest.update(
            {
                "layers": layers,
                "hidden_dim": hidden_dim,
                "vocab_size": int(decoder.m.config.vocab_size),
                "special_token_ids": {
                    "listen": int(duplex.listen_token_id),
                    "speak": int(duplex.speak_token_id),
                    "chunk_eos": int(duplex.chunk_eos_token_id),
                    "turn_eos": int(duplex.turn_eos_token_id),
                },
                "context_spec": {
                    "official_max_positions": official_max,
                    "start_position": start_position,
                    "dense_unit_positions": dense_unit_positions,
                    "dense_positions_per_second": float(dense_unit_positions),
                    "formal_dense_seconds": float(
                        (official_max - start_position) / dense_unit_positions
                    ),
                    "cache_policy": "off",
                },
                "design": {
                    "checkpoint_seconds": checkpoints,
                    "max_seconds": args.max_seconds,
                    "checkpoint_clock": (
                        "按完整密集单元的位置预算定位；目标位置="
                        "start_position+target_seconds*dense_unit_positions"
                    ),
                    "probes_per_checkpoint": args.probes_per_checkpoint,
                    "probe_indices": probe_indices,
                    "control_suffix_units": args.control_suffix_units,
                    "filler_group_units": args.filler_group_units,
                    "max_new_speak_tokens": args.max_new_speak_tokens,
                    "prefill_positions": prefill_positions,
                    "dense_tail_positions": args.max_new_speak_tokens + 1,
                    "filler_decode": (
                        "greedy；只屏蔽 listen/chunk_eos/chunk_tts_eos/turn_eos，"
                        "最后一步由官方路径写入 chunk_eos"
                    ),
                    "probe_decode": "官方未干预 greedy，text_repetition_penalty=1.0",
                    "generate_audio": False,
                    "pairing": (
                        "三角对照：长流、近期后缀低位复位、近期后缀绝对位置匹配"
                    ),
                },
                "embedding_bank": {
                    "units": len(bank),
                    "official_part_lengths": bank_meta["part_lengths"],
                    "audio_positions": bank_meta["audio_positions"],
                },
                "dense_bank": dense_bank_meta,
                "dense_replay_parity": parity,
            }
        )
        atomic_write_json(manifest_path, manifest)

        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(bank)).tolist()
        order_index = 0
        group_id = 0
        history: list[ReplayUnit] = []

        def next_bank_indices(count: int) -> list[int]:
            nonlocal order, order_index
            selected = []
            for _ in range(count):
                if order_index >= len(order):
                    order = rng.permutation(len(bank)).tolist()
                    order_index = 0
                selected.append(int(order[order_index]))
                order_index += 1
            return selected

        with SelectedLayerCapture(modules, layers) as capture, torch.inference_mode():
            for target_seconds in checkpoints:
                target_position = (
                    start_position + target_seconds * dense_unit_positions
                )
                while decoder.get_cache_length() < target_position:
                    remaining_positions = target_position - decoder.get_cache_length()
                    remaining_units = max(
                        1,
                        (remaining_positions + dense_unit_positions - 1)
                        // dense_unit_positions,
                    )
                    count = min(
                        args.filler_group_units,
                        remaining_units,
                    )
                    group_id += 1
                    _append_filler_group(
                        duplex,
                        context_helpers,
                        bank,
                        dense_tails,
                        next_bank_indices(count),
                        history=history,
                        group_id=group_id,
                    )
                for probe_index in probe_indices:
                    recent_history = tuple(history[-args.control_suffix_units :])
                    hidden, logits, tail_ids, meta = _run_probe(
                        duplex,
                        context_helpers,
                        capture,
                        bank,
                        bank_index=probe_index,
                        max_new_speak_tokens=args.max_new_speak_tokens,
                    )
                    descriptor = ReplayUnit(
                        bank_index=probe_index,
                        tail_ids=tail_ids,
                        mode="probe",
                        group_id=-1,
                        is_listen=bool(meta["is_listen"]),
                        generated_text=str(meta["generated_text"]),
                    )
                    history.append(descriptor)
                    observations.append(
                        LongObservation(
                            target_seconds=target_seconds,
                            long_input_seconds=len(history),
                            long_position=int(meta["position"]),
                            probe_index=probe_index,
                            long_hidden=hidden,
                            long_logits=logits,
                            long_generated_ids=tail_ids,
                            long_cache_length=int(meta["cache_after"]),
                            long_all_finite=np.isfinite(hidden).all(axis=-1),
                            recent_history=recent_history,
                        )
                    )
                manifest["progress"] = {
                    "phase": "long_stream",
                    "last_completed_target_seconds": target_seconds,
                    "long_input_seconds": len(history),
                    "long_cache_length": int(decoder.get_cache_length()),
                    "long_dense_equivalent_seconds": float(
                        (decoder.get_cache_length() - start_position)
                        / dense_unit_positions
                    ),
                    "observation_count": len(observations),
                }
                atomic_write_json(manifest_path, manifest)
                guard.check()
                # 长序列 SDPA 会保留较大的临时分配缓存；检查点间只释放未占用块，
                # 不触碰仍由 KV cache 持有的张量。
                torch.cuda.empty_cache()

            # 最后一个检查点通常已经达到 max_seconds；仍以位置预算做最终硬校验。
            max_target_position = (
                start_position + args.max_seconds * dense_unit_positions
            )
            if decoder.get_cache_length() < max_target_position:
                while decoder.get_cache_length() < max_target_position:
                    remaining_positions = (
                        max_target_position - decoder.get_cache_length()
                    )
                    remaining_units = max(
                        1,
                        (remaining_positions + dense_unit_positions - 1)
                        // dense_unit_positions,
                    )
                    count = min(
                        args.filler_group_units,
                        remaining_units,
                    )
                    group_id += 1
                    _append_filler_group(
                        duplex,
                        context_helpers,
                        bank,
                        dense_tails,
                        next_bank_indices(count),
                        history=history,
                        group_id=group_id,
                    )
            long_final_cache_length = int(decoder.get_cache_length())
            long_final_input_seconds = len(history)

            # 释放长流 KV 引用，再依次运行短后缀复位对照。
            duplex.prepare(prefix_system_prompt=args.system_prompt)
            gc.collect()
            torch.cuda.empty_cache()
            for observation_index, observation in enumerate(observations):
                duplex.prepare(prefix_system_prompt=args.system_prompt)
                decoder = duplex.decoder
                _feed_replay_units(
                    duplex,
                    context_helpers,
                    bank,
                    observation.recent_history,
                    max_group_units=args.filler_group_units,
                )
                hidden, logits, tail_ids, meta = _run_probe(
                    duplex,
                    context_helpers,
                    capture,
                    bank,
                    bank_index=observation.probe_index,
                    max_new_speak_tokens=args.max_new_speak_tokens,
                )
                controls.append(
                    {
                        "position": int(meta["position"]),
                        "hidden": hidden,
                        "logits": logits,
                        "generated_ids": tail_ids,
                        "cache_after": int(meta["cache_after"]),
                        "all_finite": np.isfinite(hidden).all(axis=-1),
                    }
                )
                low_position = int(meta["position"])

                # 第二个复位侧使用相同的近期完整单元，并把动态 token 的 RoPE
                # 位置整体平移到长流当前绝对位置。这样可以把远端历史内容效应
                # 与绝对位置外推效应拆开。
                position_shift = observation.long_position - low_position
                duplex.prepare(prefix_system_prompt=args.system_prompt)
                decoder = duplex.decoder
                with _shift_decoder_positions(decoder, position_shift):
                    _feed_replay_units(
                        duplex,
                        context_helpers,
                        bank,
                        observation.recent_history,
                        max_group_units=args.filler_group_units,
                    )
                    shifted_hidden, shifted_logits, shifted_tail_ids, shifted_meta = (
                        _run_probe(
                            duplex,
                            context_helpers,
                            capture,
                            bank,
                            bank_index=observation.probe_index,
                            max_new_speak_tokens=args.max_new_speak_tokens,
                        )
                    )
                absolute_position = int(shifted_meta["position"]) + position_shift
                if absolute_position != observation.long_position:
                    raise RuntimeError(
                        "绝对位置对照未与长流探针对齐："
                        f"{absolute_position} != {observation.long_position}"
                    )
                position_controls.append(
                    {
                        "position": int(shifted_meta["position"]),
                        "absolute_position": absolute_position,
                        "position_shift": position_shift,
                        "hidden": shifted_hidden,
                        "logits": shifted_logits,
                        "generated_ids": shifted_tail_ids,
                        "cache_after": int(shifted_meta["cache_after"]),
                        "all_finite": np.isfinite(shifted_hidden).all(axis=-1),
                    }
                )
                if (observation_index + 1) % 8 == 0:
                    manifest["progress"] = {
                        "phase": "paired_controls",
                        "completed_controls": observation_index + 1,
                        "total_controls": len(observations),
                    }
                    atomic_write_json(manifest_path, manifest)
                    guard.check()

        trace = _pack_trace(
            observations,
            controls,
            position_controls,
            dense_tails=dense_tails,
            max_tail_tokens=args.max_new_speak_tokens + 1,
        )
        atomic_save_npz(trace_path, **trace)
        guard.check(force=True)
        manifest.update(
            {
                "complete": True,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "observation_count": len(observations),
                "position_control_count": len(position_controls),
                "long_final_input_seconds": long_final_input_seconds,
                "long_final_cache_length": long_final_cache_length,
                "long_observed_positions_per_second": float(
                    (long_final_cache_length - start_position)
                    / max(long_final_input_seconds, 1)
                ),
                "long_dense_equivalent_seconds": float(
                    (long_final_cache_length - start_position)
                    / dense_unit_positions
                ),
                "official_boundary_crossed": long_final_cache_length >= official_max,
                "seconds_2000_crossed": (
                    long_final_cache_length
                    >= start_position + 2000 * dense_unit_positions
                ),
                "max_gpu_temperature_c": guard.max_seen_c,
                "trace_sha256": file_sha256(trace_path),
                "progress": {
                    "phase": "complete",
                    "completed_controls": len(controls),
                },
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"[minicpm-dense] 完成：{out_dir}；"
            f"长流 {long_final_input_seconds} 秒，位置 {long_final_cache_length}，"
            f"配对探针 {len(observations)}"
        )
    except BaseException as exc:
        if (
            observations
            and controls
            and position_controls
            and len(observations) == len(controls) == len(position_controls)
        ):
            atomic_save_npz(
                trace_path,
                **_pack_trace(
                    observations,
                    controls,
                    position_controls,
                    dense_tails=dense_tails,
                    max_tail_tokens=args.max_new_speak_tokens + 1,
                ),
            )
        manifest.update(
            {
                "complete": False,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "observation_count": len(observations),
                "control_count": len(controls),
                "position_control_count": len(position_controls),
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
