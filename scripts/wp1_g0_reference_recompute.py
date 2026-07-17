"""WP1：按 DualTurn 官方源码独立复算 G0 四类金标，并做逐帧全等检查。

本脚本用于协议取证，不调用 ``floor_circuit.events.g0_official``，从而避免“被测实现
调用自身”的循环验证。标签生成函数逐句移植自 DualTurn 官方仓库：

- 仓库：https://github.com/anyreachai/dualturn
- 提交：2d0db21e767b953f5017c1cc697928b54161d645
- 文件：dualturn/data/relabel_context_aware.py
- 原文件 SHA-256：357b377cbf538d5feca9da2f0b07ca1f6e82d3de27ad8f086328f7d381d93089

生成阶段只读取两条 VAD 轨；官方 EOT/HOLD/BOT/BC 仅在比较阶段读取，避免标签泄漏。

用法：
  uv run python scripts/wp1_g0_reference_recompute.py --split test
  uv run python scripts/wp1_g0_reference_recompute.py --split test --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import write_report_json

from floor_circuit.config import data_root

SOURCE_REPOSITORY = "https://github.com/anyreachai/dualturn"
SOURCE_COMMIT = "2d0db21e767b953f5017c1cc697928b54161d645"
SOURCE_FILE = "dualturn/data/relabel_context_aware.py"
SOURCE_SHA256 = "357b377cbf538d5feca9da2f0b07ca1f6e82d3de27ad8f086328f7d381d93089"
LABELS = ("eot", "hold", "bot", "bc")


def get_speech_segments(vad: np.ndarray) -> list[tuple[int, int]]:
    """提取连续语音段，返回左闭右开区间 ``(start, end)``。"""
    segments: list[tuple[int, int]] = []
    in_speech = False
    start = 0
    for frame in range(len(vad)):
        if vad[frame] > 0.5 and not in_speech:
            start = frame
            in_speech = True
        elif vad[frame] <= 0.5 and in_speech:
            segments.append((start, frame))
            in_speech = False
    if in_speech:
        segments.append((start, len(vad)))
    return segments


def compute_reference_labels(
    vad_ch0: np.ndarray,
    vad_ch1: np.ndarray,
    min_turn_frames: int = 12,
    max_bc_duration: int = 12,
    min_bc_silence: int = 12,
    max_gap_frames: int = 50,
) -> dict[str, np.ndarray]:
    """逐句复现官方 ``compute_context_aware_labels``。

    该函数只接收 VAD，不接收任何目标标签。边界行为有意保持官方实现原样，包括：
    录音首尾不具备足量可见静音时不判 BC；所有未到录音末尾的非 BC 段都落 EOT 或
    HOLD；BOT 的最近发言者比较使用原始活动轨。
    """
    vad_ch0 = np.asarray(vad_ch0)
    vad_ch1 = np.asarray(vad_ch1)
    if vad_ch0.ndim != 1 or vad_ch1.ndim != 1:
        raise ValueError("VAD 必须是一维轨")
    if len(vad_ch0) != len(vad_ch1):
        raise ValueError(f"双通道 VAD 长度不同：{len(vad_ch0)} != {len(vad_ch1)}")

    total_frames = len(vad_ch0)
    eot = [np.zeros(total_frames, dtype=np.int8), np.zeros(total_frames, dtype=np.int8)]
    hold = [np.zeros(total_frames, dtype=np.int8), np.zeros(total_frames, dtype=np.int8)]
    bot = [np.zeros(total_frames, dtype=np.int8), np.zeros(total_frames, dtype=np.int8)]
    bc = [np.zeros(total_frames, dtype=np.int8), np.zeros(total_frames, dtype=np.int8)]
    vads = [vad_ch0, vad_ch1]

    for channel in range(2):
        other = 1 - channel
        segments_self = get_speech_segments(vads[channel])
        segments_other = get_speech_segments(vads[other])
        if not segments_self:
            continue

        vad_binary = (vads[channel] > 0.5).astype(np.int8)

        # BC：短段、同通道前后静音充足，且附近存在对方的有效话轮。
        for segment_start, segment_end in segments_self:
            segment_duration = segment_end - segment_start
            if segment_duration > max_bc_duration:
                continue

            if segment_start > 0:
                previous_speech = np.where(vad_binary[:segment_start] > 0)[0]
                silence_before = (
                    segment_start - (previous_speech[-1] + 1)
                    if len(previous_speech) > 0
                    else segment_start
                )
            else:
                silence_before = 0
            if silence_before < min_bc_silence:
                continue

            if segment_end < total_frames:
                next_speech = np.where(vad_binary[segment_end:] > 0)[0]
                silence_after = (
                    int(next_speech[0]) if len(next_speech) > 0 else total_frames - segment_end
                )
            else:
                silence_after = 0
            if silence_after < min_bc_silence:
                continue

            context_start = max(0, segment_start - max_gap_frames)
            context_end = min(total_frames, segment_end + max_gap_frames)
            other_holds_floor = False
            for other_start, other_end in segments_other:
                if other_end - other_start < min_turn_frames:
                    continue
                if other_end > context_start and other_start < context_end:
                    other_holds_floor = True
                    break
            if other_holds_floor:
                bc[channel][segment_start:segment_end] = 1

        other_binary = (vads[other] > 0.5).astype(np.int8)
        self_binary = vad_binary

        # EOT/HOLD：官方实现对每个未触及录音末尾的非 BC 段二选一。
        for segment_start, segment_end in segments_self:
            if segment_end >= total_frames:
                continue
            offset_frame = segment_end - 1
            if bc[channel][segment_start:segment_end].any():
                continue

            gap_end = min(total_frames, segment_end + max_gap_frames)
            gap_region_other = other_binary[segment_end:gap_end]
            gap_region_self = self_binary[segment_end:gap_end]
            other_onsets = np.where(gap_region_other > 0)[0]
            self_onsets = np.where(gap_region_self > 0)[0]
            gap_length = gap_end - segment_end
            first_other = int(other_onsets[0]) if len(other_onsets) > 0 else gap_length
            first_self = int(self_onsets[0]) if len(self_onsets) > 0 else gap_length

            other_takes_floor = False
            if first_other < first_self and len(other_onsets) > 0:
                absolute_start = segment_end + first_other
                absolute_end = absolute_start
                while absolute_end < total_frames and other_binary[absolute_end] > 0:
                    absolute_end += 1
                true_start = absolute_start
                while true_start > 0 and other_binary[true_start - 1] > 0:
                    true_start -= 1
                if absolute_end - true_start >= min_turn_frames:
                    other_takes_floor = True

            if other_takes_floor:
                eot[channel][offset_frame] = 1
            else:
                hold[channel][offset_frame] = 1

        # BOT：当前段达标，且过去 4 秒内最后活动来自已持续至少 12 帧的对方段。
        for segment_start, segment_end in segments_self:
            if bc[channel][segment_start:segment_end].any():
                continue
            if segment_end - segment_start < min_turn_frames:
                continue

            lookback_start = max(0, segment_start - max_gap_frames)
            lookback_other = other_binary[lookback_start:segment_start]
            lookback_self = self_binary[lookback_start:segment_start]
            other_last_indices = np.where(lookback_other > 0)[0]
            self_last_indices = np.where(lookback_self > 0)[0]
            last_other = int(other_last_indices[-1]) if len(other_last_indices) > 0 else -1
            last_self = int(self_last_indices[-1]) if len(self_last_indices) > 0 else -1

            other_was_speaking = False
            if last_other > last_self and len(other_last_indices) > 0:
                absolute_position = lookback_start + last_other
                absolute_start = absolute_position
                while absolute_start > 0 and other_binary[absolute_start - 1] > 0:
                    absolute_start -= 1
                if absolute_position + 1 - absolute_start >= min_turn_frames:
                    other_was_speaking = True
            if other_was_speaking:
                bot[channel][segment_start] = 1

    return {
        "eot_ch0": eot[0],
        "eot_ch1": eot[1],
        "hold_ch0": hold[0],
        "hold_ch1": hold[1],
        "bot_ch0": bot[0],
        "bot_ch1": bot[1],
        "bc_ch0": bc[0],
        "bc_ch1": bc[1],
    }


def session_dirs(root: Path, split: str | None, limit: int | None) -> list[Path]:
    """筛选两通道金标齐全且溯源划分匹配的会话目录。"""
    selected: list[Path] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        if not all((path / f"gold_ch{channel}.npz").exists() for channel in (0, 1)):
            continue
        if split is not None:
            try:
                metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path} 的 meta.json 不可读，拒绝静默跳过") from exc
            if metadata.get("split") != split:
                continue
        selected.append(path)
    return selected[:limit] if limit is not None else selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="默认使用 <data_root>/dualturn_prep")
    parser.add_argument("--split", default="test", help="按 meta.json 的 split 字段过滤")
    parser.add_argument("--all-sessions", action="store_true", help="忽略 split 字段")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-sessions", type=int, default=None, help="严格核对预期会话数")
    args = parser.parse_args()

    root = Path(args.root) if args.root else data_root() / "dualturn_prep"
    split = None if args.all_sessions else args.split
    directories = session_dirs(root, split, args.limit)
    if not directories:
        raise SystemExit(f"{root} 下没有符合条件的双通道金标；请先跑 wp1_g0_prepare.py")
    if args.expected_sessions is not None and len(directories) != args.expected_sessions:
        raise SystemExit(
            f"会话数不符：发现 {len(directories)}，预期 {args.expected_sessions}；拒绝部分集全等"
        )

    mismatch = {label: 0 for label in LABELS}
    label_totals = {label: {"pred": 0, "gold": 0} for label in LABELS}
    total_frames = 0
    mismatched_sessions: list[dict] = []

    for directory in directories:
        # 先单独读取 VAD 并完成预测；此时进程内尚未持有四类目标轨。
        with np.load(directory / "gold_ch0.npz", allow_pickle=False) as file_ch0:
            vad_ch0 = np.asarray(file_ch0["vad"])
        with np.load(directory / "gold_ch1.npz", allow_pickle=False) as file_ch1:
            vad_ch1 = np.asarray(file_ch1["vad"])

        for channel, vad_track in ((0, vad_ch0), (1, vad_ch1)):
            vad_values = set(np.unique(vad_track).tolist())
            if not vad_values <= {0, 1}:
                raise ValueError(
                    f"{directory.name} ch{channel} VAD 含非二值：{sorted(vad_values)}"
                )
        predicted = compute_reference_labels(vad_ch0, vad_ch1)

        # 预测完成后才读取目标轨，使输入隔离在执行顺序上也清晰可审计。
        with np.load(directory / "gold_ch0.npz", allow_pickle=False) as file_ch0:
            gold_ch0 = {key: np.asarray(file_ch0[key]) for key in LABELS}
        with np.load(directory / "gold_ch1.npz", allow_pickle=False) as file_ch1:
            gold_ch1 = {key: np.asarray(file_ch1[key]) for key in LABELS}

        session_mismatch = {label: 0 for label in LABELS}
        for channel, gold, channel_frames in (
            (0, gold_ch0, len(vad_ch0)),
            (1, gold_ch1, len(vad_ch1)),
        ):
            total_frames += channel_frames
            for label in LABELS:
                pred_track = predicted[f"{label}_ch{channel}"]
                gold_track = gold[label]
                if pred_track.shape != gold_track.shape:
                    raise ValueError(
                        f"{directory.name} ch{channel} {label} 形状不同："
                        f"{pred_track.shape} != {gold_track.shape}"
                    )
                pred_values = set(np.unique(pred_track).tolist())
                gold_values = set(np.unique(gold_track).tolist())
                if not pred_values <= {0, 1} or not gold_values <= {0, 1}:
                    raise ValueError(
                        f"{directory.name} ch{channel} {label} 含非二值："
                        f"pred={sorted(pred_values)} gold={sorted(gold_values)}"
                    )
                count = int(np.count_nonzero(pred_track != gold_track))
                mismatch[label] += count
                session_mismatch[label] += count
                label_totals[label]["pred"] += int(np.count_nonzero(pred_track))
                label_totals[label]["gold"] += int(np.count_nonzero(gold_track))
        if sum(session_mismatch.values()) > 0:
            mismatched_sessions.append(
                {"session_id": directory.name, "mismatch": session_mismatch}
            )

    total_mismatch = sum(mismatch.values())
    report = {
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "file": SOURCE_FILE,
            "sha256": SOURCE_SHA256,
        },
        "input_contract": "生成阶段仅向参考函数传入 vad_ch0/vad_ch1；四类目标轨仅用于事后比较",
        "root": str(root),
        "split": split,
        "limit": args.limit,
        "expected_sessions": args.expected_sessions,
        "n_sessions": len(directories),
        "n_channel_frames": total_frames,
        "label_totals": label_totals,
        "mismatch": mismatch,
        "total_mismatch": total_mismatch,
        "exact_equal": total_mismatch == 0,
        "n_mismatched_sessions": len(mismatched_sessions),
        "mismatched_sessions_truncated": len(mismatched_sessions) > 20,
        "mismatched_sessions": mismatched_sessions[:20],
    }
    write_report_json("g0_reference_recompute.json", report)

    if total_mismatch:
        print(f"参考复算仍有 {total_mismatch} 帧不等：{mismatch}")
        raise SystemExit(1)
    print(
        f"参考复算逐帧全等：{len(directories)} 会话，"
        f"{total_frames} 个通道帧，EOT/HOLD/BOT/BC 总残差均为 0"
    )


if __name__ == "__main__":
    main()
