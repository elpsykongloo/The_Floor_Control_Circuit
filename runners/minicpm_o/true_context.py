"""MiniCPM-o 因果双工听说与远端记忆上下文测量运行器。

``floor`` 任务在因果自然双工历史中轮换完整/未完整和显式听说最小对，保存
``is_listen``、话轮结束状态、隐藏状态和完整生成。

``memory`` 任务先通过语音写入无先验口令，再在指定年龄用语音查询。每个查询包含
完整长流、无事实近期低位、无事实近期高位、近期事实低位和近期事实高位五路，
从而把远端检索、绝对位置和猜测率拆开。

大数组只写 ``--out``，必须放在 D 盘数据目录。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(SHARED_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import dense_context as dense  # noqa: E402
from context_stress_runner import (  # noqa: E402
    GpuTemperatureGuard,
    SelectedLayerCapture,
    atomic_save_npz,
    atomic_write_json,
    file_sha256,
    parse_int_csv,
    validate_layers,
)

from floor_circuit.minicpm_true_context import (  # noqa: E402
    PROBE_KIND_FLOOR,
    PROBE_KIND_MEMORY,
    TRUE_CONTEXT_PROTOCOL,
    TRUE_CONTEXT_SCHEMA_VERSION,
    answer_token_f1,
    contains_answer,
)

DEFAULT_FLOOR_CHECKPOINTS = (
    "256,384,480,512,544,576,600,624,640,672,704,736,768,896"
)
DEFAULT_MEMORY_AGES = (
    "256,384,480,512,544,576,600,624,640,672,704,736,768,896"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are in a streaming turn-taking and memory test. "
    "When asked to remember a password, listen silently and store it. "
    "When asked for a password, answer with only the password. "
    "Keep listening to unfinished utterances and respond briefly after complete ones."
)


@dataclass
class Outcome:
    """一次探针最终音频单元的完整读出。"""

    position: int
    cache_after: int
    hidden: np.ndarray
    logits: np.ndarray
    generated_ids: tuple[int, ...]
    generated_text: str
    upstream_text: str
    is_listen: bool
    end_of_turn: bool
    all_finite: np.ndarray


@dataclass
class Observation:
    """长流探针及构建复位对照所需的信息。"""

    target_seconds: int
    history_input_seconds: int
    lane_index: int
    probe_kind: int
    probe_id: int
    expected_is_listen: int
    long: Outcome
    recent_history: tuple[dense.ReplayUnit, ...]
    sequence_indices: tuple[int, ...]
    memory_age_seconds: int = -1
    memory_aliases: tuple[str, ...] = ()
    memory_statement: tuple[dense.ReplayUnit, ...] = ()


@dataclass
class InteractionEvent:
    """完整长流中的一个真实输入单元及模型当时的听说状态。"""

    lane_index: int
    history_unit_index: int
    bank_index: int
    mode: str
    forced_action: str
    position_before: int
    decision_position: int
    cache_after: int
    is_listen: bool
    end_of_turn: bool
    generated_length: int
    generated_text: str


def _parse_positive_csv(value: str, *, name: str) -> list[int]:
    values = parse_int_csv(value)
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} 必须包含正整数")
    if values != sorted(set(values)):
        raise ValueError(f"{name} 必须严格递增且无重复")
    return values


def _load_stimuli(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("protocol") != "minicpm_true_context_stimuli_v1":
        raise ValueError("刺激清单 protocol 不匹配")
    if not bool(manifest.get("complete", False)):
        raise ValueError("刺激清单尚未完整生成音频")
    for candidate in manifest["floor_candidates"]:
        audio = candidate["audio"]
        source = Path(audio["path"])
        if not source.is_file() or file_sha256(source) != audio["sha256"]:
            raise ValueError(f"floor 音频缺失或哈希不匹配：{source}")
    for item in manifest["memory_items"]:
        for field in ("statement_audio", "query_audio"):
            audio = item[field]
            source = Path(audio["path"])
            if not source.is_file() or file_sha256(source) != audio["sha256"]:
                raise ValueError(f"memory 音频缺失或哈希不匹配：{source}")
    return manifest


@contextmanager
def _force_first_token(duplex, token_id: int):
    """临时把本单元首个解码 token 固定为指定特殊 token。"""

    import torch

    decoder = duplex.decoder
    original_decode = decoder.decode
    forced = False

    def forced_decode(*args, **kwargs):
        nonlocal forced
        if not forced:
            forced = True
            return torch.tensor([token_id], dtype=torch.long, device=duplex.device)
        return original_decode(*args, **kwargs)

    decoder.decode = forced_decode
    try:
        yield
    finally:
        decoder.decode = original_decode


def _run_generation(
    duplex,
    *,
    logits,
    max_new_speak_tokens: int,
    reset_turn: bool | None,
    force_token_id: int | None = None,
) -> tuple[dict[str, Any], tuple[int, ...], str]:
    """调用官方生成状态机，同时保留完整 token 与解码文本。"""

    duplex.pending_logits = logits
    duplex.current_mode = "AUDIO"
    duplex.audio_chunk_idx += 1
    if reset_turn is not None:
        duplex.current_turn_ended = reset_turn
    before = len(duplex.total_ids)
    kwargs = {
        "max_new_speak_tokens_per_chunk": max_new_speak_tokens,
        "decode_mode": "greedy",
        "listen_prob_scale": 1.0,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 1,
    }
    if force_token_id is None:
        result = duplex.streaming_generate(**kwargs)
    else:
        with _force_first_token(duplex, force_token_id):
            result = duplex.streaming_generate(**kwargs)
    tail_ids = tuple(int(value) for value in duplex.total_ids[before:])
    decoded = duplex.tokenizer.decode(tail_ids, skip_special_tokens=True).strip()
    return result, tail_ids, decoded


def _append_sequence(
    duplex,
    context_helpers,
    capture: SelectedLayerCapture,
    unit_bank: list[list[Any]],
    sequence_indices: tuple[int, ...],
    *,
    history: list[dense.ReplayUnit],
    max_new_speak_tokens: int,
    mode: str,
    final_force_token_id: int | None = None,
    interaction_events: list[InteractionEvent] | None = None,
    lane_index: int = -1,
) -> Outcome:
    """写入多单元语音序列；前序单元强制听，末单元执行自然决策。"""

    if not sequence_indices:
        raise ValueError("探针序列不能为空")
    decoder = duplex.decoder
    final_outcome: Outcome | None = None
    for offset, bank_index in enumerate(sequence_indices):
        final = offset == len(sequence_indices) - 1
        parts = unit_bank[bank_index]
        position_before = int(decoder.get_cache_length())
        decoder.register_unit_start()
        if final:
            capture.clear()
        logits, _ = dense._feed_prefill(decoder, context_helpers, parts)
        decision_position = int(decoder.get_cache_length())
        if final:
            sampled_hidden = capture.last_vectors()[0].astype(np.float16)
            result, tail_ids, decoded = _run_generation(
                duplex,
                logits=logits,
                max_new_speak_tokens=max_new_speak_tokens,
                reset_turn=True,
                force_token_id=final_force_token_id,
            )
        else:
            result, tail_ids, decoded = _run_generation(
                duplex,
                logits=logits,
                max_new_speak_tokens=max_new_speak_tokens,
                reset_turn=True,
                force_token_id=int(duplex.listen_token_id),
            )
        history.append(
            dense.ReplayUnit(
                bank_index=bank_index,
                tail_ids=tail_ids,
                mode=mode,
                group_id=-1,
                is_listen=bool(result.get("is_listen", False)),
                generated_text=decoded,
            )
        )
        if interaction_events is not None:
            interaction_events.append(
                InteractionEvent(
                    lane_index=lane_index,
                    history_unit_index=len(history) - 1,
                    bank_index=bank_index,
                    mode=mode,
                    forced_action=(
                        "forced_speak"
                        if final and final_force_token_id is not None
                        else "natural"
                        if final
                        else "forced_listen"
                    ),
                    position_before=position_before,
                    decision_position=decision_position,
                    cache_after=int(decoder.get_cache_length()),
                    is_listen=bool(result.get("is_listen", False)),
                    end_of_turn=bool(result.get("end_of_turn", False)),
                    generated_length=len(tail_ids),
                    generated_text=decoded,
                )
            )
        if final:
            logits_array = logits.detach().float().cpu().numpy()[0].astype(np.float16)
            final_outcome = Outcome(
                position=decision_position,
                cache_after=int(decoder.get_cache_length()),
                hidden=sampled_hidden,
                logits=logits_array,
                generated_ids=tail_ids,
                generated_text=decoded,
                upstream_text=str(result.get("text", "")),
                is_listen=bool(result.get("is_listen", False)),
                end_of_turn=bool(result.get("end_of_turn", False)),
                all_finite=(
                    np.isfinite(sampled_hidden).all(axis=-1)
                    & np.asarray([np.isfinite(logits_array).all()] * len(sampled_hidden))
                ),
            )
    if final_outcome is None:
        raise RuntimeError("探针没有产生最终读出")
    return final_outcome


def _append_forced_listen_sequence(
    duplex,
    context_helpers,
    unit_bank: list[list[Any]],
    sequence_indices: tuple[int, ...],
    *,
    history: list[dense.ReplayUnit],
    max_new_speak_tokens: int,
    mode: str,
    interaction_events: list[InteractionEvent] | None = None,
    lane_index: int = -1,
) -> tuple[dense.ReplayUnit, ...]:
    """以官方状态机把整段记忆陈述写入历史，并固定为倾听。"""

    decoder = duplex.decoder
    start = len(history)
    for bank_index in sequence_indices:
        position_before = int(decoder.get_cache_length())
        decoder.register_unit_start()
        logits, _ = dense._feed_prefill(decoder, context_helpers, unit_bank[bank_index])
        decision_position = int(decoder.get_cache_length())
        result, tail_ids, decoded = _run_generation(
            duplex,
            logits=logits,
            max_new_speak_tokens=max_new_speak_tokens,
            reset_turn=True,
            force_token_id=int(duplex.listen_token_id),
        )
        history.append(
            dense.ReplayUnit(
                bank_index=bank_index,
                tail_ids=tail_ids,
                mode=mode,
                group_id=-1,
                is_listen=bool(result.get("is_listen", False)),
                generated_text=decoded,
            )
        )
        if interaction_events is not None:
            interaction_events.append(
                InteractionEvent(
                    lane_index=lane_index,
                    history_unit_index=len(history) - 1,
                    bank_index=bank_index,
                    mode=mode,
                    forced_action="forced_listen",
                    position_before=position_before,
                    decision_position=decision_position,
                    cache_after=int(decoder.get_cache_length()),
                    is_listen=bool(result.get("is_listen", False)),
                    end_of_turn=bool(result.get("end_of_turn", False)),
                    generated_length=len(tail_ids),
                    generated_text=decoded,
                )
            )
    return tuple(history[start:])


def _append_causal_filler(
    duplex,
    context_helpers,
    unit_bank: list[list[Any]],
    *,
    bank_index: int,
    history: list[dense.ReplayUnit],
    max_new_speak_tokens: int,
    group_id: int,
    interaction_events: list[InteractionEvent],
    lane_index: int,
) -> None:
    """写入一秒真实音频并按当前完整历史自然生成模型响应。"""

    decoder = duplex.decoder
    position_before = int(decoder.get_cache_length())
    decoder.register_unit_start()
    logits, _ = dense._feed_prefill(decoder, context_helpers, unit_bank[bank_index])
    decision_position = int(decoder.get_cache_length())
    result, tail_ids, decoded = _run_generation(
        duplex,
        logits=logits,
        max_new_speak_tokens=max_new_speak_tokens,
        reset_turn=None,
    )
    history.append(
        dense.ReplayUnit(
            bank_index=bank_index,
            tail_ids=tail_ids,
            mode="filler",
            group_id=group_id,
            is_listen=bool(result.get("is_listen", False)),
            generated_text=decoded,
        )
    )
    interaction_events.append(
        InteractionEvent(
            lane_index=lane_index,
            history_unit_index=len(history) - 1,
            bank_index=bank_index,
            mode="filler",
            forced_action="natural",
            position_before=position_before,
            decision_position=decision_position,
            cache_after=int(decoder.get_cache_length()),
            is_listen=bool(result.get("is_listen", False)),
            end_of_turn=bool(result.get("end_of_turn", False)),
            generated_length=len(tail_ids),
            generated_text=decoded,
        )
    )


def _collect_clip(
    duplex,
    standard,
    context_helpers,
    *,
    path: Path,
    system_prompt: str,
    max_new_speak_tokens: int,
) -> list[list[Any]]:
    """把任意语音补零到整秒，并提取可精确重放的官方音频嵌入。"""

    audio = standard.load_audio_16k(path)
    units = max(1, math.ceil(len(audio) / 16000))
    padded = np.pad(audio, (0, units * 16000 - len(audio)))
    duplex.prepare(prefix_system_prompt=system_prompt)
    bank, _ = context_helpers._collect_embedding_bank(
        duplex,
        padded,
        bank_units=units,
        max_new_speak_tokens=max_new_speak_tokens,
    )
    return bank


def _register_clip(
    unit_bank: list[list[Any]],
    clip_parts: list[list[Any]],
) -> tuple[int, ...]:
    start = len(unit_bank)
    unit_bank.extend(clip_parts)
    return tuple(range(start, len(unit_bank)))


def _calibrate_floor(
    duplex,
    context_helpers,
    capture,
    unit_bank,
    candidates,
    sequences,
    *,
    system_prompt: str,
    max_new_speak_tokens: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    correct_by_id: dict[str, bool] = {}
    for candidate in candidates:
        duplex.prepare(prefix_system_prompt=system_prompt)
        outcome = _append_sequence(
            duplex,
            context_helpers,
            capture,
            unit_bank,
            sequences[candidate["probe_id"]],
            history=[],
            max_new_speak_tokens=max_new_speak_tokens,
            mode="calibration",
        )
        correct = outcome.is_listen == bool(candidate["expected_is_listen"])
        correct_by_id[candidate["probe_id"]] = correct
        records.append(
            {
                "probe_id": candidate["probe_id"],
                "pair_id": candidate["pair_id"],
                "family": candidate["family"],
                "expected_is_listen": bool(candidate["expected_is_listen"]),
                "observed_is_listen": outcome.is_listen,
                "end_of_turn": outcome.end_of_turn,
                "generated_text": outcome.generated_text,
                "correct": correct,
            }
        )
    pairs: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        pairs.setdefault(str(candidate["pair_id"]), []).append(candidate)
    valid_pairs = []
    for pair_id, items in pairs.items():
        labels = {bool(item["expected_is_listen"]) for item in items}
        if labels == {False, True} and all(correct_by_id[item["probe_id"]] for item in items):
            valid_pairs.append(pair_id)
    return records, sorted(valid_pairs)


def _calibrate_memory(
    duplex,
    context_helpers,
    capture,
    unit_bank,
    items,
    statement_sequences,
    query_sequences,
    *,
    system_prompt: str,
    max_new_speak_tokens: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    valid = []
    for item in items:
        memory_id = str(item["memory_id"])
        duplex.prepare(prefix_system_prompt=system_prompt)
        history: list[dense.ReplayUnit] = []
        _append_forced_listen_sequence(
            duplex,
            context_helpers,
            unit_bank,
            statement_sequences[memory_id],
            history=history,
            max_new_speak_tokens=max_new_speak_tokens,
            mode="memory_statement",
        )
        outcome = _append_sequence(
            duplex,
            context_helpers,
            capture,
            unit_bank,
            query_sequences[memory_id],
            history=history,
            max_new_speak_tokens=max_new_speak_tokens,
            mode="memory_query",
            final_force_token_id=int(duplex.speak_token_id),
        )
        aliases = tuple(str(value) for value in item["answer_aliases"])
        correct = contains_answer(outcome.generated_text, aliases)
        records.append(
            {
                "memory_id": memory_id,
                "generated_text": outcome.generated_text,
                "is_listen": outcome.is_listen,
                "end_of_turn": outcome.end_of_turn,
                "correct": correct,
                "token_f1": answer_token_f1(outcome.generated_text, aliases),
            }
        )
        if correct:
            valid.append(memory_id)
    return records, valid


def _run_control(
    duplex,
    context_helpers,
    capture,
    unit_bank,
    *,
    system_prompt: str,
    recent_history: tuple[dense.ReplayUnit, ...],
    sequence_indices: tuple[int, ...],
    max_new_speak_tokens: int,
    filler_group_units: int,
    position_shift: int = 0,
    memory_statement: tuple[dense.ReplayUnit, ...] = (),
    final_force_token_id: int | None = None,
) -> Outcome:
    duplex.prepare(prefix_system_prompt=system_prompt)
    decoder = duplex.decoder
    temporary_history: list[dense.ReplayUnit] = []
    context = (
        dense._shift_decoder_positions(decoder, position_shift)
        if position_shift
        else nullcontext()
    )
    with context:
        dense._feed_replay_units(
            duplex,
            context_helpers,
            unit_bank,
            recent_history,
            max_group_units=filler_group_units,
        )
        if memory_statement:
            dense._feed_replay_units(
                duplex,
                context_helpers,
                unit_bank,
                memory_statement,
                max_group_units=filler_group_units,
            )
        outcome = _append_sequence(
            duplex,
            context_helpers,
            capture,
            unit_bank,
            sequence_indices,
            history=temporary_history,
            max_new_speak_tokens=max_new_speak_tokens,
            mode="control_probe",
            final_force_token_id=final_force_token_id,
        )
    return outcome


def _pad_token_rows(rows: list[tuple[int, ...]], width: int) -> np.ndarray:
    output = np.full((len(rows), width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        if len(row) > width:
            raise ValueError(f"生成 token 数 {len(row)} 超过固定宽度 {width}")
        output[index, : len(row)] = np.asarray(row, dtype=np.int32)
    return output


def _outcome_arrays(
    prefix: str,
    outcomes: list[Outcome],
    *,
    max_tail_tokens: int,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_positions": np.asarray(
            [item.position for item in outcomes],
            dtype=np.int64,
        ),
        f"{prefix}_hidden": np.stack([item.hidden for item in outcomes]),
        f"{prefix}_logits": np.stack([item.logits for item in outcomes]),
        f"{prefix}_generated_ids": _pad_token_rows(
            [item.generated_ids for item in outcomes],
            max_tail_tokens,
        ),
        f"{prefix}_generated_lengths": np.asarray(
            [len(item.generated_ids) for item in outcomes],
            dtype=np.int16,
        ),
        f"{prefix}_generated_text": np.asarray(
            [item.generated_text for item in outcomes],
            dtype="U256",
        ),
        f"{prefix}_is_listen": np.asarray(
            [item.is_listen for item in outcomes],
            dtype=bool,
        ),
        f"{prefix}_end_of_turn": np.asarray(
            [item.end_of_turn for item in outcomes],
            dtype=bool,
        ),
        f"{prefix}_all_finite": np.stack(
            [item.all_finite for item in outcomes]
        ).astype(bool),
    }


def _placeholder_outcome(reference: Outcome) -> Outcome:
    return Outcome(
        position=-1,
        cache_after=-1,
        hidden=np.zeros_like(reference.hidden),
        logits=np.zeros_like(reference.logits),
        generated_ids=(),
        generated_text="",
        upstream_text="",
        is_listen=False,
        end_of_turn=False,
        all_finite=np.ones_like(reference.all_finite, dtype=bool),
    )


def _pack_trace(
    observations: list[Observation],
    lows: list[Outcome],
    highs: list[Outcome],
    high_shifts: list[int],
    oracle_lows: list[Outcome],
    oracle_highs: list[Outcome],
    oracle_high_shifts: list[int],
    *,
    max_tail_tokens: int,
) -> dict[str, np.ndarray]:
    if not (
        len(observations)
        == len(lows)
        == len(highs)
        == len(high_shifts)
        == len(oracle_lows)
        == len(oracle_highs)
        == len(oracle_high_shifts)
    ):
        raise ValueError("观测和控制臂数量不一致")
    arrays: dict[str, np.ndarray] = {
        "target_seconds": np.asarray(
            [item.target_seconds for item in observations],
            dtype=np.int32,
        ),
        "history_input_seconds": np.asarray(
            [item.history_input_seconds for item in observations],
            dtype=np.int32,
        ),
        "lane_indices": np.asarray(
            [item.lane_index for item in observations],
            dtype=np.int16,
        ),
        "probe_kinds": np.asarray(
            [item.probe_kind for item in observations],
            dtype=np.int8,
        ),
        "probe_ids": np.asarray(
            [item.probe_id for item in observations],
            dtype=np.int16,
        ),
        "expected_is_listen": np.asarray(
            [item.expected_is_listen for item in observations],
            dtype=np.int8,
        ),
        "high_absolute_positions": np.asarray(
            [item.long.position for item in observations],
            dtype=np.int64,
        ),
        "high_position_shifts": np.asarray(high_shifts, dtype=np.int64),
        "memory_available": np.asarray(
            [item.probe_kind == PROBE_KIND_MEMORY for item in observations],
            dtype=bool,
        ),
        "memory_age_seconds": np.asarray(
            [item.memory_age_seconds for item in observations],
            dtype=np.int32,
        ),
        "oracle_high_absolute_positions": np.asarray(
            [
                item.long.position if item.probe_kind == PROBE_KIND_MEMORY else -1
                for item in observations
            ],
            dtype=np.int64,
        ),
        "oracle_high_position_shifts": np.asarray(
            oracle_high_shifts,
            dtype=np.int64,
        ),
    }
    arrays.update(
        _outcome_arrays(
            "long",
            [item.long for item in observations],
            max_tail_tokens=max_tail_tokens,
        )
    )
    arrays.update(_outcome_arrays("low", lows, max_tail_tokens=max_tail_tokens))
    arrays.update(_outcome_arrays("high", highs, max_tail_tokens=max_tail_tokens))
    arrays.update(
        {
            "oracle_low_positions": np.asarray(
                [item.position for item in oracle_lows],
                dtype=np.int64,
            ),
            "oracle_high_positions": np.asarray(
                [item.position for item in oracle_highs],
                dtype=np.int64,
            ),
            "oracle_low_generated_text": np.asarray(
                [item.generated_text for item in oracle_lows],
                dtype="U256",
            ),
            "oracle_high_generated_text": np.asarray(
                [item.generated_text for item in oracle_highs],
                dtype="U256",
            ),
            "oracle_low_is_listen": np.asarray(
                [item.is_listen for item in oracle_lows],
                dtype=bool,
            ),
            "oracle_high_is_listen": np.asarray(
                [item.is_listen for item in oracle_highs],
                dtype=bool,
            ),
            "oracle_low_end_of_turn": np.asarray(
                [item.end_of_turn for item in oracle_lows],
                dtype=bool,
            ),
            "oracle_high_end_of_turn": np.asarray(
                [item.end_of_turn for item in oracle_highs],
                dtype=bool,
            ),
        }
    )
    score_names = (
        ("memory_long", [item.long for item in observations]),
        ("memory_low", lows),
        ("memory_high", highs),
        ("memory_oracle_low", oracle_lows),
        ("memory_oracle_high", oracle_highs),
    )
    for name, outcomes in score_names:
        arrays[f"{name}_correct"] = np.asarray(
            [
                contains_answer(outcome.generated_text, observation.memory_aliases)
                if observation.probe_kind == PROBE_KIND_MEMORY
                else False
                for observation, outcome in zip(observations, outcomes, strict=True)
            ],
            dtype=bool,
        )
    arrays["memory_long_token_f1"] = np.asarray(
        [
            answer_token_f1(item.long.generated_text, item.memory_aliases)
            if item.probe_kind == PROBE_KIND_MEMORY
            else 0.0
            for item in observations
        ],
        dtype=np.float32,
    )
    arrays["memory_oracle_high_token_f1"] = np.asarray(
        [
            answer_token_f1(outcome.generated_text, observation.memory_aliases)
            if observation.probe_kind == PROBE_KIND_MEMORY
            else 0.0
            for observation, outcome in zip(observations, oracle_highs, strict=True)
        ],
        dtype=np.float32,
    )
    return arrays


def _pack_interactions(events: list[InteractionEvent]) -> dict[str, np.ndarray]:
    """把完整长流逐单元状态写成无 pickle 的审计数组。"""

    if not events:
        raise ValueError("完整长流没有交互事件")
    return {
        "interaction_lane_indices": np.asarray(
            [item.lane_index for item in events],
            dtype=np.int16,
        ),
        "interaction_history_unit_indices": np.asarray(
            [item.history_unit_index for item in events],
            dtype=np.int32,
        ),
        "interaction_bank_indices": np.asarray(
            [item.bank_index for item in events],
            dtype=np.int32,
        ),
        "interaction_modes": np.asarray(
            [item.mode for item in events],
            dtype="U32",
        ),
        "interaction_forced_actions": np.asarray(
            [item.forced_action for item in events],
            dtype="U16",
        ),
        "interaction_position_before": np.asarray(
            [item.position_before for item in events],
            dtype=np.int64,
        ),
        "interaction_decision_positions": np.asarray(
            [item.decision_position for item in events],
            dtype=np.int64,
        ),
        "interaction_cache_after": np.asarray(
            [item.cache_after for item in events],
            dtype=np.int64,
        ),
        "interaction_is_listen": np.asarray(
            [item.is_listen for item in events],
            dtype=bool,
        ),
        "interaction_end_of_turn": np.asarray(
            [item.end_of_turn for item in events],
            dtype=bool,
        ),
        "interaction_generated_lengths": np.asarray(
            [item.generated_length for item in events],
            dtype=np.int16,
        ),
        "interaction_generated_text": np.asarray(
            [item.generated_text for item in events],
            dtype="U256",
        ),
    }


def _summarize_interactions(events: list[InteractionEvent]) -> dict[str, Any]:
    """汇总自然填充段，确认历史中同时出现真实倾听和模型发言。"""

    filler = [item for item in events if item.mode == "filler"]
    natural = [item for item in events if item.forced_action == "natural"]

    def summarize(items: list[InteractionEvent]) -> dict[str, Any]:
        if not items:
            return {
                "events": 0,
                "listen_events": 0,
                "speak_events": 0,
                "listen_rate": float("nan"),
                "end_of_turn_rate": float("nan"),
                "mean_generated_tokens": float("nan"),
            }
        listen_count = sum(item.is_listen for item in items)
        return {
            "events": len(items),
            "listen_events": listen_count,
            "speak_events": len(items) - listen_count,
            "listen_rate": float(listen_count / len(items)),
            "end_of_turn_rate": float(
                sum(item.end_of_turn for item in items) / len(items)
            ),
            "mean_generated_tokens": float(
                sum(item.generated_length for item in items) / len(items)
            ),
        }

    return {
        "all_events": len(events),
        "filler": summarize(filler),
        "all_natural_decisions": summarize(natural),
    }


def _load_calibration_cache(
    path: Path,
    *,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """读取经过精确指纹约束的短上下文校准缓存。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "minicpm_true_context_calibration_v1":
        raise ValueError(f"校准缓存 protocol 不匹配：{path}")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"校准缓存指纹不匹配：{path}")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"校准缓存缺少 calibration：{path}")
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-o 因果双工听说与远端记忆上下文测量"
    )
    parser.add_argument("--task", choices=("floor", "memory"), required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--source-audio", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--recent-units", type=int, default=64)
    parser.add_argument("--filler-bank-units", type=int, default=0)
    parser.add_argument("--filler-group-units", type=int, default=8)
    parser.add_argument("--max-new-speak-tokens", type=int, default=20)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--calibration-cache", default=None)
    parser.add_argument(
        "--floor-family",
        choices=("explicit", "all"),
        default="explicit",
        help=(
            "话轮探针族；严格协议默认只使用显式听说指令，"
            "避免短上下文可用但近期上下文不稳的自然句式污染边界"
        ),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature-limit", type=int, default=95)
    args = parser.parse_args()

    checkpoint_text = args.checkpoints or (
        DEFAULT_FLOOR_CHECKPOINTS if args.task == "floor" else DEFAULT_MEMORY_AGES
    )
    checkpoints = _parse_positive_csv(checkpoint_text, name="--checkpoints")
    if args.lanes < 1:
        parser.error("--lanes 至少为 1")
    if args.recent_units < 8:
        parser.error("--recent-units 至少为 8")
    if args.filler_group_units < 1:
        parser.error("--filler-group-units 至少为 1")
    if args.max_new_speak_tokens < 3:
        parser.error("--max-new-speak-tokens 至少为 3")

    import torch
    from transformers import AutoModel

    standard, context_helpers = dense._load_helpers()
    stimuli = _load_stimuli(Path(args.stimuli))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    trace_path = out_dir / "trace.npz"
    started = time.time()
    guard = GpuTemperatureGuard(
        limit_c=args.temperature_limit,
        check_interval_s=15.0,
    )
    source_audio_path = Path(args.source_audio)
    manifest: dict[str, Any] = {
        "schema_version": TRUE_CONTEXT_SCHEMA_VERSION,
        "protocol": TRUE_CONTEXT_PROTOCOL,
        "model": "minicpm_o_4_5",
        "task": args.task,
        "run_id": args.run_id,
        "complete": False,
        "started_unix": started,
        "source_audio": {
            "path": str(source_audio_path.resolve()),
            "sha256": file_sha256(source_audio_path),
        },
        "stimuli": {
            "path": str(Path(args.stimuli).resolve()),
            "sha256": file_sha256(Path(args.stimuli)),
        },
        "seed": args.seed,
        "checkpoints": checkpoints,
        "lanes": args.lanes,
    }
    atomic_write_json(manifest_path, manifest)

    observations: list[Observation] = []
    lows: list[Outcome] = []
    highs: list[Outcome] = []
    high_shifts: list[int] = []
    oracle_lows: list[Outcome] = []
    oracle_highs: list[Outcome] = []
    oracle_high_shifts: list[int] = []
    interaction_events: list[InteractionEvent] = []
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
        decoder = duplex.decoder
        modules = list(decoder.m.model.layers)
        layers = validate_layers(parse_int_csv(args.layers), len(modules))
        hidden_dim = int(decoder.m.config.hidden_size)
        vocab_size = int(decoder.m.config.vocab_size)
        start_position = int(decoder.get_cache_length())
        official_max = int(decoder.m.config.max_position_embeddings)

        source_audio = standard.load_audio_16k(source_audio_path)
        available_source_units = max(1, len(source_audio) // 16000)
        requested_filler_units = args.filler_bank_units or (
            checkpoints[-1] + 256
        )
        filler_units = min(available_source_units, requested_filler_units)
        if filler_units < checkpoints[-1]:
            raise ValueError(
                f"源音频只有 {available_source_units} 个完整秒，"
                f"不足以覆盖检查点 {checkpoints[-1]}"
            )
        filler_audio = source_audio[: filler_units * 16000]
        duplex.prepare(prefix_system_prompt=args.system_prompt)
        filler_parts, filler_meta = context_helpers._collect_embedding_bank(
            duplex,
            filler_audio,
            bank_units=filler_units,
            max_new_speak_tokens=args.max_new_speak_tokens,
        )
        unit_bank: list[list[Any]] = list(filler_parts)
        filler_indices = tuple(range(filler_units))

        all_floor_candidates = list(stimuli["floor_candidates"])
        floor_candidates = (
            [
                item
                for item in all_floor_candidates
                if str(item["family"]) == "explicit_floor_instruction"
            ]
            if args.task == "floor" and args.floor_family == "explicit"
            else all_floor_candidates
        )
        if args.task == "floor" and not floor_candidates:
            raise RuntimeError(
                f"floor_family={args.floor_family} 没有可用的话轮探针"
            )
        memory_items = list(stimuli["memory_items"])
        probe_catalog = [
            {
                "probe_id": item["probe_id"],
                "kind": "floor",
                "expected_is_listen": bool(item["expected_is_listen"]),
                "family": item["family"],
            }
            for item in floor_candidates
        ] + [
            {
                "probe_id": item["memory_id"],
                "kind": "memory",
                "answer_aliases": item["answer_aliases"],
            }
            for item in memory_items
        ]
        probe_index_by_id = {
            str(item["probe_id"]): index for index, item in enumerate(probe_catalog)
        }

        floor_sequences: dict[str, tuple[int, ...]] = {}
        memory_statement_sequences: dict[str, tuple[int, ...]] = {}
        memory_query_sequences: dict[str, tuple[int, ...]] = {}
        if args.task == "floor":
            for index, candidate in enumerate(floor_candidates):
                clip = _collect_clip(
                    duplex,
                    standard,
                    context_helpers,
                    path=Path(candidate["audio"]["path"]),
                    system_prompt=args.system_prompt,
                    max_new_speak_tokens=args.max_new_speak_tokens,
                )
                floor_sequences[str(candidate["probe_id"])] = _register_clip(
                    unit_bank,
                    clip,
                )
                manifest["progress"] = {
                    "phase": "prepare_floor_embeddings",
                    "completed": index + 1,
                    "total": len(floor_candidates),
                }
                atomic_write_json(manifest_path, manifest)
                guard.check()
        else:
            for index, item in enumerate(memory_items):
                memory_id = str(item["memory_id"])
                statement = _collect_clip(
                    duplex,
                    standard,
                    context_helpers,
                    path=Path(item["statement_audio"]["path"]),
                    system_prompt=args.system_prompt,
                    max_new_speak_tokens=args.max_new_speak_tokens,
                )
                query = _collect_clip(
                    duplex,
                    standard,
                    context_helpers,
                    path=Path(item["query_audio"]["path"]),
                    system_prompt=args.system_prompt,
                    max_new_speak_tokens=args.max_new_speak_tokens,
                )
                memory_statement_sequences[memory_id] = _register_clip(
                    unit_bank,
                    statement,
                )
                memory_query_sequences[memory_id] = _register_clip(unit_bank, query)
                manifest["progress"] = {
                    "phase": "prepare_memory_embeddings",
                    "completed": index + 1,
                    "total": len(memory_items),
                }
                atomic_write_json(manifest_path, manifest)
                guard.check()

        calibration_cache_path = (
            Path(args.calibration_cache) if args.calibration_cache else None
        )
        calibration_fingerprint = {
            "task": args.task,
            "stimuli_sha256": manifest["stimuli"]["sha256"],
            "model_config_sha256": file_sha256(Path(model_root) / "config.json"),
            "system_prompt": args.system_prompt,
            "max_new_speak_tokens": args.max_new_speak_tokens,
            "floor_family": (
                args.floor_family if args.task == "floor" else "不适用"
            ),
            "memory_query_readout": (
                "force_first_speak_token_v1"
                if args.task == "memory"
                else "natural_v1"
            ),
        }
        cached_calibration = (
            _load_calibration_cache(
                calibration_cache_path,
                fingerprint=calibration_fingerprint,
            )
            if calibration_cache_path is not None
            and calibration_cache_path.is_file()
            else None
        )
        rng = np.random.default_rng(args.seed)
        with SelectedLayerCapture(modules, layers) as capture, torch.inference_mode():
            if args.task == "floor":
                if cached_calibration is None:
                    calibration, valid_pairs = _calibrate_floor(
                        duplex,
                        context_helpers,
                        capture,
                        unit_bank,
                        floor_candidates,
                        floor_sequences,
                        system_prompt=args.system_prompt,
                        max_new_speak_tokens=args.max_new_speak_tokens,
                    )
                else:
                    calibration = list(cached_calibration["records"])
                    valid_pairs = [
                        str(value)
                        for value in cached_calibration["valid_pairs"]
                    ]
                if len(valid_pairs) < 2:
                    raise RuntimeError(
                        f"只有 {len(valid_pairs)} 个听说最小对通过短上下文校准"
                    )
                candidates_by_pair = {
                    pair_id: [
                        item
                        for item in floor_candidates
                        if str(item["pair_id"]) == pair_id
                    ]
                    for pair_id in valid_pairs
                }
                manifest["calibration"] = {
                    "records": calibration,
                    "valid_pairs": valid_pairs,
                }
            else:
                if cached_calibration is None:
                    calibration, valid_memory_ids = _calibrate_memory(
                        duplex,
                        context_helpers,
                        capture,
                        unit_bank,
                        memory_items,
                        memory_statement_sequences,
                        memory_query_sequences,
                        system_prompt=args.system_prompt,
                        max_new_speak_tokens=args.max_new_speak_tokens,
                    )
                else:
                    calibration = list(cached_calibration["records"])
                    valid_memory_ids = [
                        str(value)
                        for value in cached_calibration["valid_memory_ids"]
                    ]
                if len(valid_memory_ids) < len(checkpoints):
                    raise RuntimeError(
                        f"只有 {len(valid_memory_ids)} 个口令通过即时校准，"
                        f"需要 {len(checkpoints)} 个"
                    )
                memory_by_id = {
                    str(item["memory_id"]): item for item in memory_items
                }
                manifest["calibration"] = {
                    "records": calibration,
                    "valid_memory_ids": valid_memory_ids,
                }
            if calibration_cache_path is not None and cached_calibration is None:
                atomic_write_json(
                    calibration_cache_path,
                    {
                        "protocol": "minicpm_true_context_calibration_v1",
                        "fingerprint": calibration_fingerprint,
                        "calibration": manifest["calibration"],
                    },
                )
            manifest["calibration_cache"] = {
                "path": (
                    str(calibration_cache_path.resolve())
                    if calibration_cache_path is not None
                    else None
                ),
                "reused": cached_calibration is not None,
                "fingerprint": calibration_fingerprint,
            }
            manifest["probe_catalog"] = probe_catalog
            manifest["layers"] = layers
            manifest["hidden_dim"] = hidden_dim
            manifest["vocab_size"] = vocab_size
            manifest["context_spec"] = {
                "official_max_positions": official_max,
                "start_position": start_position,
                "sliding_window_mode": "off",
            }
            manifest["design"] = {
                "history": "逐秒真实音频嵌入 + 官方自然 greedy 因果生成",
                "recent_units": args.recent_units,
                "filler_source_units": filler_units,
                "filler_official_part_lengths": filler_meta["part_lengths"],
                "max_new_speak_tokens": args.max_new_speak_tokens,
                "system_prompt": args.system_prompt,
                "floor_family": (
                    args.floor_family if args.task == "floor" else "不适用"
                ),
                "floor_probe_sampling": (
                    "每个检查点、每条 lane 使用一个完整最小对，"
                    "期望倾听与期望发言各一次"
                    if args.task == "floor"
                    else "不适用"
                ),
                "memory_query_readout": (
                    "首 token 固定为 <|speak|>，其余 token 保持官方 greedy"
                    if args.task == "memory"
                    else "不适用"
                ),
            }
            atomic_write_json(manifest_path, manifest)

            for lane_index in range(args.lanes):
                duplex.prepare(prefix_system_prompt=args.system_prompt)
                history: list[dense.ReplayUnit] = []
                filler_cursor = (
                    args.seed * 97 + lane_index * 131
                ) % len(filler_indices)
                filler_count = 0

                def append_filler_until(
                    target_units: int,
                    history_ref: list[dense.ReplayUnit] = history,
                    lane_index_ref: int = lane_index,
                ) -> None:
                    nonlocal filler_cursor, filler_count
                    while len(history_ref) < target_units:
                        bank_index = filler_indices[filler_cursor]
                        filler_cursor = (filler_cursor + 1) % len(filler_indices)
                        group_id = filler_count // args.filler_group_units
                        _append_causal_filler(
                            duplex,
                            context_helpers,
                            unit_bank,
                            bank_index=bank_index,
                            history=history_ref,
                            max_new_speak_tokens=args.max_new_speak_tokens,
                            group_id=group_id,
                            interaction_events=interaction_events,
                            lane_index=lane_index_ref,
                        )
                        filler_count += 1
                        if filler_count % 16 == 0:
                            guard.check()

                if args.task == "floor":
                    pair_order = list(valid_pairs)
                    rng.shuffle(pair_order)
                    for checkpoint_index, target in enumerate(checkpoints):
                        append_filler_until(target)
                        pair_id = pair_order[
                            (checkpoint_index + lane_index) % len(pair_order)
                        ]
                        pair_items = list(candidates_by_pair[pair_id])
                        pair_items.sort(
                            key=lambda item: bool(item["expected_is_listen"]),
                            reverse=bool((checkpoint_index + lane_index) % 2),
                        )
                        for candidate in pair_items:
                            recent = tuple(history[-args.recent_units :])
                            outcome = _append_sequence(
                                duplex,
                                context_helpers,
                                capture,
                                unit_bank,
                                floor_sequences[str(candidate["probe_id"])],
                                history=history,
                                max_new_speak_tokens=args.max_new_speak_tokens,
                                mode="floor_probe",
                                interaction_events=interaction_events,
                                lane_index=lane_index,
                            )
                            observations.append(
                                Observation(
                                    target_seconds=target,
                                    history_input_seconds=len(history),
                                    lane_index=lane_index,
                                    probe_kind=PROBE_KIND_FLOOR,
                                    probe_id=probe_index_by_id[
                                        str(candidate["probe_id"])
                                    ],
                                    expected_is_listen=int(
                                        bool(candidate["expected_is_listen"])
                                    ),
                                    long=outcome,
                                    recent_history=recent,
                                    sequence_indices=floor_sequences[
                                        str(candidate["probe_id"])
                                    ],
                                )
                            )
                        manifest["progress"] = {
                            "phase": "long_floor",
                            "lane": lane_index,
                            "target_seconds": target,
                            "history_input_seconds": len(history),
                            "cache_length": int(duplex.decoder.get_cache_length()),
                            "observations": len(observations),
                        }
                        atomic_write_json(manifest_path, manifest)
                        torch.cuda.empty_cache()
                else:
                    ordered_ids = list(valid_memory_ids)
                    rng.shuffle(ordered_ids)
                    selected_ids = ordered_ids[: len(checkpoints)]
                    statement_units: dict[str, tuple[dense.ReplayUnit, ...]] = {}
                    statement_end: dict[str, int] = {}
                    for memory_id in selected_ids:
                        statement_units[memory_id] = _append_forced_listen_sequence(
                            duplex,
                            context_helpers,
                            unit_bank,
                            memory_statement_sequences[memory_id],
                            history=history,
                            max_new_speak_tokens=args.max_new_speak_tokens,
                            mode="memory_statement",
                            interaction_events=interaction_events,
                            lane_index=lane_index,
                        )
                        statement_end[memory_id] = len(history)
                    tasks = []
                    for target, memory_id in zip(
                        checkpoints,
                        selected_ids,
                        strict=True,
                    ):
                        query_units = len(memory_query_sequences[memory_id])
                        due_before_query = (
                            statement_end[memory_id] + target - query_units
                        )
                        tasks.append(
                            (due_before_query, target, memory_id)
                        )
                    tasks.sort()
                    for due_before_query, target, memory_id in tasks:
                        append_filler_until(due_before_query)
                        recent = tuple(history[-args.recent_units :])
                        outcome = _append_sequence(
                            duplex,
                            context_helpers,
                            capture,
                            unit_bank,
                            memory_query_sequences[memory_id],
                            history=history,
                            max_new_speak_tokens=args.max_new_speak_tokens,
                            mode="memory_query",
                            final_force_token_id=int(duplex.speak_token_id),
                            interaction_events=interaction_events,
                            lane_index=lane_index,
                        )
                        actual_age = len(history) - statement_end[memory_id]
                        item = memory_by_id[memory_id]
                        observations.append(
                            Observation(
                                target_seconds=target,
                                history_input_seconds=len(history),
                                lane_index=lane_index,
                                probe_kind=PROBE_KIND_MEMORY,
                                probe_id=probe_index_by_id[memory_id],
                                expected_is_listen=-1,
                                long=outcome,
                                recent_history=recent,
                                sequence_indices=memory_query_sequences[memory_id],
                                memory_age_seconds=actual_age,
                                memory_aliases=tuple(
                                    str(value) for value in item["answer_aliases"]
                                ),
                                memory_statement=statement_units[memory_id],
                            )
                        )
                        manifest["progress"] = {
                            "phase": "long_memory",
                            "lane": lane_index,
                            "target_seconds": target,
                            "memory_age_seconds": actual_age,
                            "history_input_seconds": len(history),
                            "cache_length": int(duplex.decoder.get_cache_length()),
                            "observations": len(observations),
                        }
                        atomic_write_json(manifest_path, manifest)
                        torch.cuda.empty_cache()

                # 当前 lane 的完整长流已经存入 CPU 数组，释放长 KV 后运行复位臂。
                lane_observations = [
                    item for item in observations if item.lane_index == lane_index
                ]
                duplex.prepare(prefix_system_prompt=args.system_prompt)
                gc.collect()
                torch.cuda.empty_cache()
                for lane_offset, observation in enumerate(lane_observations):
                    low = _run_control(
                        duplex,
                        context_helpers,
                        capture,
                        unit_bank,
                        system_prompt=args.system_prompt,
                        recent_history=observation.recent_history,
                        sequence_indices=observation.sequence_indices,
                        max_new_speak_tokens=args.max_new_speak_tokens,
                        filler_group_units=args.filler_group_units,
                        final_force_token_id=(
                            int(duplex.speak_token_id)
                            if observation.probe_kind == PROBE_KIND_MEMORY
                            else None
                        ),
                    )
                    shift = observation.long.position - low.position
                    if shift < 0:
                        raise RuntimeError("近期高位臂需要负位置平移")
                    high = _run_control(
                        duplex,
                        context_helpers,
                        capture,
                        unit_bank,
                        system_prompt=args.system_prompt,
                        recent_history=observation.recent_history,
                        sequence_indices=observation.sequence_indices,
                        max_new_speak_tokens=args.max_new_speak_tokens,
                        filler_group_units=args.filler_group_units,
                        position_shift=shift,
                        final_force_token_id=(
                            int(duplex.speak_token_id)
                            if observation.probe_kind == PROBE_KIND_MEMORY
                            else None
                        ),
                    )
                    if high.position + shift != observation.long.position:
                        raise RuntimeError("近期高位臂未与长流探针对齐")
                    lows.append(low)
                    highs.append(high)
                    high_shifts.append(shift)

                    if observation.probe_kind == PROBE_KIND_MEMORY:
                        oracle_low = _run_control(
                            duplex,
                            context_helpers,
                            capture,
                            unit_bank,
                            system_prompt=args.system_prompt,
                            recent_history=observation.recent_history,
                            memory_statement=observation.memory_statement,
                            sequence_indices=observation.sequence_indices,
                            max_new_speak_tokens=args.max_new_speak_tokens,
                            filler_group_units=args.filler_group_units,
                            final_force_token_id=int(duplex.speak_token_id),
                        )
                        oracle_shift = (
                            observation.long.position - oracle_low.position
                        )
                        if oracle_shift < 0:
                            raise RuntimeError("近期记忆高位臂需要负位置平移")
                        oracle_high = _run_control(
                            duplex,
                            context_helpers,
                            capture,
                            unit_bank,
                            system_prompt=args.system_prompt,
                            recent_history=observation.recent_history,
                            memory_statement=observation.memory_statement,
                            sequence_indices=observation.sequence_indices,
                            max_new_speak_tokens=args.max_new_speak_tokens,
                            filler_group_units=args.filler_group_units,
                            position_shift=oracle_shift,
                            final_force_token_id=int(duplex.speak_token_id),
                        )
                        if (
                            oracle_high.position + oracle_shift
                            != observation.long.position
                        ):
                            raise RuntimeError("近期记忆高位臂未与长流探针对齐")
                    else:
                        oracle_low = _placeholder_outcome(observation.long)
                        oracle_high = _placeholder_outcome(observation.long)
                        oracle_shift = -1
                    oracle_lows.append(oracle_low)
                    oracle_highs.append(oracle_high)
                    oracle_high_shifts.append(oracle_shift)
                    if (lane_offset + 1) % 4 == 0:
                        manifest["progress"] = {
                            "phase": "controls",
                            "lane": lane_index,
                            "completed": lane_offset + 1,
                            "total": len(lane_observations),
                        }
                        atomic_write_json(manifest_path, manifest)
                        guard.check()
                        torch.cuda.empty_cache()

            trace = _pack_trace(
                observations,
                lows,
                highs,
                high_shifts,
                oracle_lows,
                oracle_highs,
                oracle_high_shifts,
                max_tail_tokens=args.max_new_speak_tokens + 1,
            )
            trace.update(_pack_interactions(interaction_events))
            atomic_save_npz(trace_path, **trace)
            trace_sha = file_sha256(trace_path)
            finite = bool(
                trace["long_all_finite"].all()
                and trace["low_all_finite"].all()
                and trace["high_all_finite"].all()
            )
            memory_mask = trace["memory_available"]
            oracle_alignment = bool(
                np.all(
                    trace["oracle_low_positions"][memory_mask]
                    + trace["oracle_high_position_shifts"][memory_mask]
                    == trace["oracle_high_absolute_positions"][memory_mask]
                )
            )
            manifest.update(
                {
                    "complete": True,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "observation_count": len(observations),
                    "all_finite": finite,
                    "position_alignment": bool(
                        np.array_equal(
                            trace["high_positions"] + trace["high_position_shifts"],
                            trace["high_absolute_positions"],
                        )
                    ),
                    "oracle_position_alignment": oracle_alignment,
                    "interaction_summary": _summarize_interactions(
                        interaction_events
                    ),
                    "max_gpu_temperature_c": guard.max_seen_c,
                    "trace_sha256": trace_sha,
                    "progress": {"phase": "complete"},
                }
            )
            atomic_write_json(manifest_path, manifest)
            print(
                "[minicpm-true-context] "
                f"完成 task={args.task}，观测={len(observations)}，"
                f"trace={trace_path}，最高温度={guard.max_seen_c}°C"
            )
    except Exception as error:
        manifest.update(
            {
                "complete": False,
                "failed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "max_gpu_temperature_c": guard.max_seen_c,
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
