"""E2-lite 二次分析库（PREREG #40(a)）与掩码工具的定向测试：合成掩码、CPU 可跑。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from floor_circuit.e1x import sdt as sx
from floor_circuit.e1x.mask_cache import shifted_agent_mask

REPO = Path(__file__).resolve().parents[1]

EV_CFG = {
    "onset_pre_silence_s": 0.4,
    "offset_post_silence_s": 0.4,
}


def _mask(total_s: float, dt: float, segments: list[tuple[float, float]]) -> np.ndarray:
    n = round(total_s / dt)
    mask = np.zeros(n, dtype=bool)
    for start, end in segments:
        mask[round(start / dt) : round(end / dt)] = True
    return mask


def test_onset_context_split_three_way():
    dt = 0.01
    # 用户说 [1,3)；agent onset：2.2（用户话中）、3.5（offset=3.0 后 0.5 s 内 → respond）、8.0（gap）
    mask_user = _mask(12.0, dt, [(1.0, 3.0)])
    mask_agent = _mask(12.0, dt, [(2.2, 2.6), (3.5, 4.5), (8.0, 9.0)])
    result = sx.onset_context_split(mask_user, mask_agent, dt, EV_CFG, respond_window_s=1.2)
    assert result["n_agent_onsets"] == 3
    assert result["n_during_user"] == 1
    assert result["n_respond"] == 1
    assert result["n_gap"] == 1
    assert result["respond_share"] == pytest.approx(1 / 3)
    assert result["n_user_offsets"] == 1


def test_sdt_decision_stats_hit_and_false_alarm_rates():
    dt = 0.01
    # 用户两段：[1,4)、[6,9)；agent 只响应第一段末（4.3），从不在段中闯入。
    mask_user = _mask(14.0, dt, [(1.0, 4.0), (6.0, 9.0)])
    mask_agent = _mask(14.0, dt, [(4.3, 5.0)])
    stats = sx.sdt_decision_stats(
        mask_user, mask_agent, dt, EV_CFG,
        window_s=1.2, noise_stride_s=1.0, noise_guard_s=1.2,
    )
    assert stats["n_signal"] == 2
    assert stats["hits"] == 1
    assert stats["n_noise"] > 0
    assert stats["false_alarms"] == 0
    # 校正后 H=(1+0.5)/3=0.5、F<0.5 → d′>0；c>0（保守判据）
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["d_prime"] > 0
    assert stats["criterion_c"] > 0


def test_sdt_degenerate_inputs_return_none():
    dt = 0.01
    empty = np.zeros(500, dtype=bool)
    stats = sx.sdt_decision_stats(
        empty, empty, dt, EV_CFG, window_s=1.2, noise_stride_s=1.0, noise_guard_s=1.2
    )
    assert stats["d_prime"] is None and stats["n_signal"] == 0


def test_z_transform_symmetry():
    assert sx._z(0.5) == pytest.approx(0.0, abs=1e-9)
    assert sx._z(0.8413) == pytest.approx(1.0, abs=1e-3)
    assert sx._z(0.1587) == pytest.approx(-1.0, abs=1e-3)


def test_token_stats_pad_runs_and_entropy():
    tokens = np.array([3, 3, 5, 6, 6, 3, 7])
    stats = sx.token_stats(tokens, pad_id=3)
    assert stats["pad_frac"] == pytest.approx(3 / 7)
    # 非 PAD 段：[5,6,6]（长 3）与 [7]（长 1）→ 均值 2.0
    assert stats["mean_nonpad_run"] == pytest.approx(2.0)
    assert stats["max_nonpad_run"] == 3
    assert stats["entropy_bits"] > 0
    assert stats["top_tokens"][3] == 3
    with pytest.raises(ValueError):
        sx.token_stats(np.array([]), pad_id=3)


def test_exchange_gaps_and_overlap_durations():
    dt = 0.01
    mask_from = _mask(10.0, dt, [(1.0, 3.0)])
    mask_to = _mask(10.0, dt, [(3.6, 5.0)])
    gaps = sx.exchange_gaps(mask_from, mask_to, dt, EV_CFG, max_gap_s=3.0)
    assert gaps == pytest.approx([0.6], abs=0.02)
    overlaps = sx.overlap_durations(_mask(10.0, dt, [(1.0, 4.0)]), _mask(10.0, dt, [(3.0, 5.0)]), dt)
    assert overlaps == pytest.approx([1.0], abs=0.02)


def test_histogram_and_jsd_properties():
    bins = [0.0, 1.0, 2.0]
    counts = sx.histogram_counts([0.5, 1.5, 2.7], bins)  # 2.7 入末桶
    assert counts.tolist() == [1, 2]
    equal = sx.jensen_shannon_divergence(np.array([5, 5]), np.array([10, 10]))
    assert equal == pytest.approx(0.0, abs=1e-12)
    disjoint = sx.jensen_shannon_divergence(np.array([10, 0]), np.array([0, 10]))
    assert disjoint == pytest.approx(1.0, abs=1e-9)
    ab = sx.jensen_shannon_divergence(np.array([8, 2]), np.array([2, 8]))
    ba = sx.jensen_shannon_divergence(np.array([2, 8]), np.array([8, 2]))
    assert ab == pytest.approx(ba)
    assert sx.jensen_shannon_divergence(np.array([0, 0]), np.array([1, 1])) is None


def test_shifted_agent_mask_offsets_timeline():
    local = np.array([True, True, False, False], dtype=bool)
    # first_emitted=1 帧、帧长 0.08 s、dt=0.04 s → 平移 2 个 dt 样本
    shifted = shifted_agent_mask(local, first_emitted_frame=1, frame_s=0.08, dt=0.04)
    assert shifted.tolist() == [False, False, True, True]
    all_out = shifted_agent_mask(local, first_emitted_frame=100, frame_s=0.08, dt=0.04)
    assert not all_out.any()


def test_leadtime_checkpoint_guard():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from wp_e1x_suite import leadtime_checkpoint_valid
    finally:
        sys.path.pop(0)
    good = {"min_anchor_step": 62, "per_seed": {"0": {}, "1": {}, "2": {}}}
    assert leadtime_checkpoint_valid(good, 62, [0, 1, 2])
    assert not leadtime_checkpoint_valid(good, 25, [0, 1, 2])  # 网格扩展前的旧断点
    assert not leadtime_checkpoint_valid({"per_seed": {"0": {}}}, 62, [0, 1, 2])
    assert not leadtime_checkpoint_valid(None, 62, [0, 1, 2])
