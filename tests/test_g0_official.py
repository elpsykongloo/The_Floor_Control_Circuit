"""官方 DualTurn 标签算法复现的单测：手工构造 VAD → 手算预期标签（非同源，独立验证）。"""

from __future__ import annotations

import numpy as np

from floor_circuit.events.g0 import _match_sparse
from floor_circuit.events.g0_official import (
    OfficialParams,
    exact_mismatches,
    official_tracks,
    param_grid,
    segments_to_frame_track,
    track_prf,
    vad_segments,
)
from floor_circuit.schemas import Seg

P = OfficialParams()  # lookahead 50、有效段 ≥13、bc ≤12 且前后 ≥13 静音


def make_vad(n: int, segs: list[tuple[int, int]]) -> np.ndarray:
    v = np.zeros(n, dtype=np.int8)
    for s, e in segs:
        v[s:e] = 1
    return v


class TestVadSegments:
    def test_basic(self):
        assert vad_segments(make_vad(10, [(2, 5), (7, 9)])) == [(2, 5), (7, 9)]
        assert vad_segments(np.zeros(5, dtype=np.int8)) == []
        assert vad_segments(np.ones(4, dtype=np.int8)) == [(0, 4)]


class TestOfficialTracks:
    def test_eot_other_takes_floor(self):
        # 本人 [10,40)，对方在 45 起说 20 帧（有效）；本人 4 s 内未恢复 → EOT@39
        n = 200
        self_v = make_vad(n, [(10, 40)])
        other_v = make_vad(n, [(45, 65)])
        tr = official_tracks(self_v, other_v, P)
        assert tr["eot"][39] == 1
        assert tr["hold"].sum() == 0
        # 对方段 ≥1 s 且此前 4 s 内本人是最近发言者 → 对方通道 BOT@45
        tr_o = official_tracks(other_v, self_v, P)
        assert tr_o["bot"][45] == 1

    def test_hold_self_resumes(self):
        # 本人 [10,40) 停顿后 [50,80) 恢复；对方全程安静 → HOLD@39，且第二段无 BOT（对方非最近发言者）
        n = 200
        self_v = make_vad(n, [(10, 40), (50, 80)])
        other_v = np.zeros(n, dtype=np.int8)
        tr = official_tracks(self_v, other_v, P)
        assert tr["hold"][39] == 1
        assert tr["eot"].sum() == 0
        assert tr["bot"].sum() == 0

    def test_no_label_when_nobody_resumes(self):
        # 本人 [10,40)，此后 4 s 内无人说话 → 段末无 eot/hold
        n = 200
        tr = official_tracks(make_vad(n, [(10, 40)]), np.zeros(n, dtype=np.int8), P)
        assert tr["eot"].sum() == 0 and tr["hold"].sum() == 0

    def test_bc_isolated_short(self):
        # 对方长段 [10,60)；本人 [30,38)（8 帧 ≤12，前后静音充足）→ BC 覆盖 [30,38)
        n = 200
        self_v = make_vad(n, [(30, 38)])
        other_v = make_vad(n, [(10, 60)])
        tr = official_tracks(self_v, other_v, P)
        assert tr["bc"][30:38].all() and tr["bc"].sum() == 8
        # BC 段不产生 eot/hold/bot
        assert tr["eot"].sum() == 0 and tr["hold"].sum() == 0 and tr["bot"].sum() == 0

    def test_short_seg_near_own_speech_not_bc(self):
        # 短段距本人上一段仅 5 帧（< bc_gap 13）→ 非 BC
        n = 200
        self_v = make_vad(n, [(10, 30), (35, 40)])
        other_v = make_vad(n, [(50, 70)])
        tr = official_tracks(self_v, other_v, P)
        assert tr["bc"].sum() == 0

    def test_overlap_other_ongoing_counts_as_eot(self):
        # 对方有效段 [30,60) 覆盖本人段末 40 → other_ongoing_counts=True 时立即接管 → EOT@39
        n = 200
        self_v = make_vad(n, [(10, 40)])
        other_v = make_vad(n, [(30, 60)])
        tr = official_tracks(self_v, other_v, P)
        assert tr["eot"][39] == 1

    def test_bot_requires_min_duration(self):
        # 对方先说；本人 10 帧短段（<13）不算 BOT；后续 20 帧段算 BOT
        n = 300
        self_v = make_vad(n, [(80, 90), (120, 140)])
        other_v = make_vad(n, [(10, 60)])
        tr = official_tracks(self_v, other_v, P)
        assert tr["bot"][80] == 0
        # 120 处：过去 4 s（70-120）内最近发言者是本人自己的 80-90 段？80-90 仅 10 帧非有效段
        # → 最近**有效**发言者仍是对方（10-60 在 lookback 之外，70-120 窗内对方无活跃）→ 无 BOT
        assert tr["bot"][120] == 0
        # 把对方段挪近：对方 [100,115) 有效（15 帧）→ 本人 [120,140) 应为 BOT
        other_v2 = make_vad(n, [(100, 115)])
        tr2 = official_tracks(self_v, other_v2, P)
        assert tr2["bot"][120] == 1

    def test_exact_mismatch_and_grid(self):
        n = 200
        self_v = make_vad(n, [(10, 40)])
        other_v = make_vad(n, [(45, 65)])
        tr = official_tracks(self_v, other_v, P)
        assert exact_mismatches(tr, {k: v.copy() for k, v in tr.items()}) == {
            "eot": 0, "hold": 0, "bot": 0, "bc": 0,
        }
        grid = param_grid()
        assert len(grid) == 64 and len({tuple(sorted(p.as_dict().items())) for p in grid}) == 64


class TestFrameTrackAndPrf:
    def test_majority_downsample(self):
        # 段 [0.0, 0.12)：帧 0 覆盖 0.08/0.08=100%、帧 1 覆盖 0.04/0.08=50% → majority 两帧皆 1
        track = segments_to_frame_track([Seg(0.0, 0.12)], 10, 12.5, rule="majority")
        assert track[0] == 1 and track[1] == 1 and track[2] == 0
        # 段 [0.0, 0.10)：帧 1 覆盖 0.02（25%）→ majority 不亮，any 亮
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
        # 旧贪心命中 1，最大匹配应为 2（pred [0,3]、gold [2,4]、tol 2）
        assert _match_sparse(np.array([0, 3]), np.array([2, 4]), 2) == 2

    def test_basic_cases(self):
        assert _match_sparse(np.array([5]), np.array([5]), 0) == 1
        assert _match_sparse(np.array([5]), np.array([8]), 2) == 0
        assert _match_sparse(np.array([1, 2, 3]), np.array([2]), 1) == 1
        assert _match_sparse(np.array([]), np.array([1, 2]), 2) == 0
