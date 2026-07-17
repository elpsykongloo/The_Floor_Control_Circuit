"""官方 DualTurn 标签算法（定稿语义）单测：手工构造 VAD → 按官方规则手算预期。

语义要点（与早期错误复现的差异，均有对应用例）：
- 每个未触及录音末尾的非-BC 段必落 EOT 或 HOLD（无人恢复 → HOLD）；
- 触及录音末尾的段不标 EOT/HOLD；
- BC 前静音在录音首段记 0（首段不判 BC）；
- BOT 的最近发言者比较用原始二值轨（本人的 BC 活跃也算）。
"""

from __future__ import annotations

import numpy as np

from floor_circuit.events.g0 import _match_sparse
from floor_circuit.events.g0_official import (
    compute_official_labels,
    exact_mismatches,
    get_speech_segments,
    official_tracks,
    segments_to_frame_track,
    track_prf,
)
from floor_circuit.schemas import Seg

N = 200


def make_vad(n: int, segs: list[tuple[int, int]]) -> np.ndarray:
    v = np.zeros(n, dtype=np.int8)
    for s, e in segs:
        v[s:e] = 1
    return v


class TestSegments:
    def test_basic(self):
        assert get_speech_segments(make_vad(10, [(2, 5), (7, 9)])) == [(2, 5), (7, 9)]
        assert get_speech_segments(np.zeros(5, dtype=np.int8)) == []
        assert get_speech_segments(np.ones(4, dtype=np.int8)) == [(0, 4)]

    def test_soft_vad_threshold(self):
        v = np.array([0.6, 0.4, 0.7, 0.7], dtype=np.float32)
        assert get_speech_segments(v) == [(0, 1), (2, 4)]


class TestOfficialSemantics:
    def test_eot_when_other_takes_floor(self):
        self_v = make_vad(N, [(10, 40)])
        other_v = make_vad(N, [(45, 65)])  # 20 帧 ≥ 12 → 有效接管
        tr = official_tracks(self_v, other_v)
        assert tr["eot"][39] == 1 and tr["eot"].sum() == 1
        assert tr["hold"].sum() == 0
        # 对方通道：其段末 4 s 内无人恢复 → HOLD（官方互补语义）；且其起点为 BOT
        full = compute_official_labels(self_v, other_v)
        assert full["hold_ch1"][64] == 1
        assert full["bot_ch1"][45] == 1
        assert full["bot_ch0"].sum() == 0  # ch0 起点前无人说话

    def test_hold_when_nobody_resumes(self):
        # 关键差异用例：4 s 内无人恢复 → HOLD（早期实现漏标）
        tr = official_tracks(make_vad(N, [(10, 40)]), np.zeros(N, dtype=np.int8))
        assert tr["hold"][39] == 1 and tr["eot"].sum() == 0

    def test_hold_when_self_resumes(self):
        tr = official_tracks(make_vad(N, [(10, 40), (50, 80)]), np.zeros(N, dtype=np.int8))
        assert tr["hold"][39] == 1 and tr["hold"][79] == 1
        assert tr["eot"].sum() == 0 and tr["bot"].sum() == 0

    def test_segment_touching_end_unlabeled(self):
        # 触及录音末尾的段不落 EOT/HOLD（早期实现的"末帧钳位"方向是错的）
        tr = official_tracks(make_vad(N, [(150, N)]), np.zeros(N, dtype=np.int8))
        assert tr["eot"].sum() == 0 and tr["hold"].sum() == 0

    def test_bc_isolated_short_with_other_floor(self):
        self_v = make_vad(N, [(30, 38)])  # 8 帧 ≤ 12，前后静音充足
        other_v = make_vad(N, [(10, 60)])  # 附近对方有效话轮
        tr = official_tracks(self_v, other_v)
        assert tr["bc"][30:38].all() and tr["bc"].sum() == 8
        assert tr["eot"].sum() == 0 and tr["hold"].sum() == 0 and tr["bot"].sum() == 0

    def test_recording_initial_segment_never_bc(self):
        # 录音首段 silence_before 记 0 → 不判 BC；随后按互补语义落 EOT/HOLD
        self_v = make_vad(N, [(0, 8)])
        other_v = make_vad(N, [(20, 40)])  # 20 帧有效，8 帧末端后 12 帧起
        tr = official_tracks(self_v, other_v)
        assert tr["bc"].sum() == 0
        assert tr["eot"][7] == 1  # 对方在 gap 窗内先恢复且段长达标

    def test_bc_needs_silence_after(self):
        # 短段后 5 帧本人又说话（< 12 帧静音）→ 非 BC
        self_v = make_vad(N, [(30, 38), (43, 70)])
        other_v = make_vad(N, [(10, 60)])
        tr = official_tracks(self_v, other_v)
        assert tr["bc"].sum() == 0

    def test_overlap_other_ongoing_counts(self):
        # 段末时对方已在说话（重叠）：first_other=0 → 立即接管 → EOT
        self_v = make_vad(N, [(10, 40)])
        other_v = make_vad(N, [(30, 60)])
        tr = official_tracks(self_v, other_v)
        assert tr["eot"][39] == 1

    def test_bot_uses_raw_track_including_own_bc(self):
        # 本人的 BC 活跃出现在候选 BOT 段的 lookback 窗内且晚于对方 → 阻断 BOT（原始轨语义）
        self_v = make_vad(N, [(70, 78), (100, 120)])  # 70-78 为 BC（孤立短段）
        other_v = make_vad(N, [(10, 60)])
        full = compute_official_labels(self_v, other_v)
        assert full["bc_ch0"][70:78].all()  # 确认 70-78 判为 BC
        assert full["bot_ch0"][100] == 0  # last_self(77) > last_other(59) → 无 BOT
        # 移除本人 BC 后同一段应为 BOT
        full2 = compute_official_labels(make_vad(N, [(100, 120)]), other_v)
        assert full2["bot_ch0"][100] == 1

    def test_short_segment_no_bot(self):
        self_v = make_vad(N, [(80, 90)])  # 10 帧 < 12
        other_v = make_vad(N, [(10, 60)])
        tr = official_tracks(self_v, other_v)
        assert tr["bot"].sum() == 0

    def test_length_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError, match="长度不同"):
            compute_official_labels(np.zeros(10, dtype=np.int8), np.zeros(9, dtype=np.int8))
        with pytest.raises(ValueError, match="轨长不等"):
            exact_mismatches(
                {c: np.zeros(5, dtype=np.int8) for c in ("eot", "hold", "bot", "bc")},
                {c: np.zeros(6, dtype=np.int8) for c in ("eot", "hold", "bot", "bc")},
            )

    def test_exact_mismatch_zero_on_self(self):
        tr = official_tracks(make_vad(N, [(10, 40)]), make_vad(N, [(45, 65)]))
        assert exact_mismatches(tr, {k: v.copy() for k, v in tr.items()}) == {
            "eot": 0, "hold": 0, "bot": 0, "bc": 0,
        }


class TestFrameTrackAndPrf:
    def test_majority_downsample(self):
        track = segments_to_frame_track([Seg(0.0, 0.12)], 10, 12.5, rule="majority")
        assert track[0] == 1 and track[1] == 1 and track[2] == 0
        t2 = segments_to_frame_track([Seg(0.0, 0.10)], 10, 12.5, rule="majority")
        assert t2[1] == 0
        t3 = segments_to_frame_track([Seg(0.0, 0.10)], 10, 12.5, rule="any")
        assert t3[1] == 1

    def test_track_prf(self):
        gold = np.array([0, 1, 1, 1, 0], dtype=np.int8)
        pred = np.array([0, 1, 1, 0, 0], dtype=np.int8)
        prf = track_prf(pred, gold)
        assert prf["precision"] == 1.0 and abs(prf["recall"] - 2 / 3) < 1e-9


class TestMaxMatching:
    def test_counterexample_from_review(self):
        assert _match_sparse(np.array([0, 3]), np.array([2, 4]), 2) == 2

    def test_basic_cases(self):
        assert _match_sparse(np.array([5]), np.array([5]), 0) == 1
        assert _match_sparse(np.array([5]), np.array([8]), 2) == 0
        assert _match_sparse(np.array([1, 2, 3]), np.array([2]), 1) == 1
        assert _match_sparse(np.array([]), np.array([1, 2]), 2) == 0
