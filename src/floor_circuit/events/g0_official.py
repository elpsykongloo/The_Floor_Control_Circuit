"""DualTurn 官方标签算法（库版，2026-07-17 定稿）。

语义权威 = 官方源码的逐句移植（用户已在 138/138 会话、1,030,734 通道帧上验证逐帧全等，
并与官方源码做 2,000 组随机 VAD 对拍一致，见 scripts/wp1_g0_reference_recompute.py 与
reports/g0_reference_recompute.json）：

- 仓库：https://github.com/anyreachai/dualturn
- 提交：2d0db21e767b953f5017c1cc697928b54161d645
- 文件：dualturn/data/relabel_context_aware.py（SHA-256
  357b377cbf538d5feca9da2f0b07ca1f6e82d3de27ad8f086328f7d381d93089）

官方语义要点（区别于早期错误复现）：
- EOT/HOLD 互补：每个未触及录音末尾的非-BC 段末**必落其一**——gap 窗内对方先活跃且其
  完整段 ≥ min_turn 帧 → EOT，否则（含无人恢复）→ HOLD；事件帧 = 段末最后语音帧；
- BC：段长 ≤ max_bc；前静音 = 距上一语音帧的距离（录音首段记 0 → 不判 BC）；后静音 =
  距下一语音帧的距离（无下一段时 = 到录音末尾；触及末尾记 0）；附近（±gap 窗）须有
  ≥ min_turn 帧的对方段；BC 覆盖整段；
- BOT：非-BC 且 ≥ min_turn 帧的段起点；lookback 窗内**原始二值轨**（不排除 BC）最后
  活跃帧对方严格晚于本人，且对方该活跃点所在段（向前延伸、截至该点）≥ min_turn 帧。

常量冻结：min_turn=12、max_bc=12、min_bc_silence=12、gap=50（帧，12.5 Hz）。
本模块与取证脚本刻意保持两份独立实现——修改本文件后必须重跑
`wp1_g0_calibrate.py --protocol-check` 与参考脚本对拍。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.schemas import Seg

FRAME_HZ = 12.5
OFFICIAL_CLASSES = ("eot", "hold", "bot", "bc")

MIN_TURN_FRAMES = 12
MAX_BC_DURATION = 12
MIN_BC_SILENCE = 12
MAX_GAP_FRAMES = 50

SOURCE_COMMIT = "2d0db21e767b953f5017c1cc697928b54161d645"
SOURCE_SHA256 = "357b377cbf538d5feca9da2f0b07ca1f6e82d3de27ad8f086328f7d381d93089"


def get_speech_segments(vad: np.ndarray) -> list[tuple[int, int]]:
    """提取连续语音段（阈值 0.5，兼容软 VAD），左闭右开 (start, end)。"""
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


def compute_official_labels(
    vad_ch0: np.ndarray,
    vad_ch1: np.ndarray,
    min_turn_frames: int = MIN_TURN_FRAMES,
    max_bc_duration: int = MAX_BC_DURATION,
    min_bc_silence: int = MIN_BC_SILENCE,
    max_gap_frames: int = MAX_GAP_FRAMES,
) -> dict[str, np.ndarray]:
    """双通道 VAD → 官方八条标签轨 {eot/hold/bot/bc}_ch{0,1}（逐句移植官方实现）。"""
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

        # BC：短段、同通道前后静音充足，且附近存在对方的有效话轮
        for segment_start, segment_end in segments_self:
            if segment_end - segment_start > max_bc_duration:
                continue
            if segment_start > 0:
                previous_speech = np.where(vad_binary[:segment_start] > 0)[0]
                silence_before = (
                    segment_start - (previous_speech[-1] + 1) if len(previous_speech) > 0 else segment_start
                )
            else:
                silence_before = 0
            if silence_before < min_bc_silence:
                continue
            if segment_end < total_frames:
                next_speech = np.where(vad_binary[segment_end:] > 0)[0]
                silence_after = int(next_speech[0]) if len(next_speech) > 0 else total_frames - segment_end
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

        # EOT/HOLD：每个未触及录音末尾的非-BC 段二选一
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

        # BOT：达标段起点，过去 gap 窗内最后活动来自已持续 ≥ min_turn 帧的对方段
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
        "eot_ch0": eot[0], "eot_ch1": eot[1],
        "hold_ch0": hold[0], "hold_ch1": hold[1],
        "bot_ch0": bot[0], "bot_ch1": bot[1],
        "bc_ch0": bc[0], "bc_ch1": bc[1],
    }


def official_tracks(vad_self: np.ndarray, vad_other: np.ndarray) -> dict[str, np.ndarray]:
    """单通道视角便捷包装：返回 self 通道的 {eot, hold, bot, bc}。"""
    full = compute_official_labels(vad_self, vad_other)
    return {cls: full[f"{cls}_ch0"] for cls in OFFICIAL_CLASSES}


def segments_to_frame_track(
    segs: list[Seg], n_frames: int, hz: float = FRAME_HZ, rule: str = "majority"
) -> np.ndarray:
    """秒域 VAD 段 → 12.5 Hz 二值轨。majority：帧内活跃占比 ≥ 0.5；any：有任何活跃。"""
    track = np.zeros(n_frames, dtype=np.int8)
    frame_len = 1.0 / hz
    for seg in segs:
        f0 = max(0, int(np.floor(seg.start * hz)))
        f1 = min(n_frames - 1, int(np.ceil(seg.end * hz)))
        for f in range(f0, f1 + 1):
            t0, t1 = f * frame_len, (f + 1) * frame_len
            overlap = max(0.0, min(seg.end, t1) - max(seg.start, t0))
            # 浮点容差：恰好半帧的覆盖按多数计入
            if (rule == "majority" and overlap >= 0.5 * frame_len - 1e-9) or (rule == "any" and overlap > 1e-12):
                track[f] = 1
    return track


def track_prf(pred: np.ndarray, gold: np.ndarray) -> dict:
    """帧级二值 P/R/F1（VAD 一致性层用）。"""
    n = min(len(pred), len(gold))
    p, g = np.asarray(pred[:n]) > 0, np.asarray(gold[:n]) > 0
    tp = int(np.sum(p & g))
    prec = tp / int(p.sum()) if p.sum() else (1.0 if not g.sum() else 0.0)
    rec = tp / int(g.sum()) if g.sum() else (1.0 if not p.sum() else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_pred": int(p.sum()), "n_gold": int(g.sum())}


def exact_mismatches(pred: dict[str, np.ndarray], gold: dict[str, np.ndarray]) -> dict[str, int]:
    """逐帧不等的帧数（协议正确性层：目标全零）。长度不等直接报错，绝不截断漏报。"""
    out = {}
    for cls in OFFICIAL_CLASSES:
        p, g = np.asarray(pred[cls]), np.asarray(gold[cls])
        if len(p) != len(g):
            raise ValueError(f"{cls} 轨长不等：pred {len(p)} != gold {len(g)}（拒绝截断比较）")
        out[cls] = int(np.sum((p > 0) != (g > 0)))
    return out
