"""时间对齐哨兵测试（PREREG #7）：任何表征都不得读到观测截止之后的信息。

约定：标签步 s 的观测截止 = s·τ。acts 读行 s；Mimi/hazard/声学读行 s−1。
每个哨兵都带反例（旧实现的行映射确实泄漏未来一帧），防止测试退化为恒真。
"""

from __future__ import annotations

import numpy as np
import pytest

from floor_circuit.mve.alignment import (
    FEATURE_OBSERVED_THROUGH_OFFSET,
    MIN_ELIGIBLE_STEP,
    RUNNER_TIME_ALIGNMENT,
    feature_row_indices,
)
from floor_circuit.probes.baselines import hazard_features
from floor_circuit.probes.gru import make_windows


class TestRowMapping:
    def test_offsets_are_frozen(self):
        assert FEATURE_OBSERVED_THROUGH_OFFSET == {
            "acts": 0,
            "mimi": 1,
            "hazard": 1,
            "acoustic": 1,
        }
        assert MIN_ELIGIBLE_STEP == 1
        assert RUNNER_TIME_ALIGNMENT == {
            "initial_token_position": 0,
            "acts_observed_through_offset_steps": 0,
        }

    def test_acts_reads_row_s_and_mimi_reads_row_s_minus_1(self):
        steps = np.array([1, 4, 9], dtype=np.int64)
        np.testing.assert_array_equal(feature_row_indices("acts", steps), [1, 4, 9])
        np.testing.assert_array_equal(feature_row_indices("mimi", steps), [0, 3, 8])
        np.testing.assert_array_equal(feature_row_indices("hazard", steps), [0, 3, 8])
        np.testing.assert_array_equal(feature_row_indices("acoustic", steps), [0, 3, 8])

    def test_step_zero_is_rejected(self):
        with pytest.raises(ValueError, match="MIN_ELIGIBLE_STEP"):
            feature_row_indices("mimi", np.array([0, 1], dtype=np.int64))

    def test_unknown_feature_is_rejected(self):
        with pytest.raises(ValueError, match="未知表征类型"):
            feature_row_indices("depformer", np.array([1], dtype=np.int64))


def _random_states(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 6, size=n).astype(np.int64)


class TestHazardCausality:
    """hazard 行 s−1 只能依赖 states[0..s−1]（观测截止 s·τ）。"""

    def test_selected_rows_are_causal(self):
        rng = np.random.default_rng(20260718)
        for _trial in range(20):
            n = int(rng.integers(8, 40))
            states = _random_states(rng, n)
            full = hazard_features(states, step_s=0.08)
            for step in range(MIN_ELIGIBLE_STEP, n):
                row = feature_row_indices("hazard", np.array([step]))[0]
                truncated = hazard_features(states[:step], step_s=0.08)
                np.testing.assert_array_equal(
                    full[row],
                    truncated[row],
                    err_msg=f"step {step}: hazard 行读到了 states[{step}:] 的信息",
                )

    def test_old_mapping_leaks_current_step_state(self):
        """反例：旧实现读行 s，state[s] 一变该行就变（证明泄漏真实存在）。"""
        states = np.array([0, 0, 1, 1, 3, 3, 0, 0], dtype=np.int64)
        perturbed = states.copy()
        step = 4
        perturbed[step] = 4  # OVERLAP_HOLD → GAP：改变当前步的活跃状态
        old_row_original = hazard_features(states, 0.08)[step]
        old_row_perturbed = hazard_features(perturbed, 0.08)[step]
        assert not np.array_equal(old_row_original, old_row_perturbed)
        # 新映射（行 s−1）对同一扰动不变
        new_row_original = hazard_features(states, 0.08)[step - 1]
        new_row_perturbed = hazard_features(perturbed, 0.08)[step - 1]
        np.testing.assert_array_equal(new_row_original, new_row_perturbed)


class TestAcousticWindowCausality:
    """声学窗口（窗尾 s−1）不得包含帧 s 及之后的内容。"""

    def test_windows_ignore_future_frames(self):
        rng = np.random.default_rng(7)
        feats = rng.normal(size=(30, 4)).astype(np.float32)
        steps = np.arange(MIN_ELIGIBLE_STEP, 30, dtype=np.int64)
        rows = feature_row_indices("acoustic", steps)
        windows = make_windows(feats, rows)
        for index, step in enumerate(steps):
            perturbed = feats.copy()
            perturbed[step:] += 100.0
            perturbed_windows = make_windows(perturbed, rows)
            np.testing.assert_array_equal(
                windows[index],
                perturbed_windows[index],
                err_msg=f"step {step}: 声学窗口读到了帧 {step} 及之后的内容",
            )

    def test_old_window_end_contains_current_frame(self):
        """反例：旧实现窗尾 = s，会随帧 s 变化。"""
        rng = np.random.default_rng(9)
        feats = rng.normal(size=(12, 4)).astype(np.float32)
        step = 5
        perturbed = feats.copy()
        perturbed[step] += 100.0
        old = make_windows(feats, np.array([step]))
        old_perturbed = make_windows(perturbed, np.array([step]))
        assert not np.array_equal(old, old_perturbed)


class TestAcousticFrameFootprint:
    """acoustic_frames 的帧 i 观测不得越过 (i+1)·hop（对两个 F0 后端都成立）。

    这是 alignment.py 声学 offset=1 前提的实体核验：撤回复盘审查发现旧的
    librosa.yin 回退（frame_length=4·hop）会观测到 (i+2)·hop。
    """

    @staticmethod
    def _wav(seconds: float = 1.6, sr: int = 16_000) -> tuple[np.ndarray, int, int]:
        rng = np.random.default_rng(20260718)
        n = int(seconds * sr)
        t = np.arange(n) / sr
        wav = (0.3 * np.sin(2 * np.pi * 140.0 * t) + 0.05 * rng.normal(size=n)).astype(np.float32)
        hop = round(sr * 80.0 / 1000.0)
        return wav, sr, hop

    def _assert_causal(self, monkeypatch=None, expect_backend: str = "parselmouth") -> None:
        from floor_circuit.probes import baselines

        wav, sr, hop = self._wav()
        feats, meta = baselines.acoustic_frames(wav, sr, return_meta=True)
        assert meta["f0_backend"] == expect_backend
        for frame in (3, 7, 12):
            perturbed = wav.copy()
            perturbed[(frame + 1) * hop :] += 0.5
            perturbed_feats, _ = baselines.acoustic_frames(perturbed, sr, return_meta=True)
            np.testing.assert_array_equal(
                feats[frame],
                perturbed_feats[frame],
                err_msg=f"帧 {frame}（后端 {expect_backend}）观测越过 (i+1)·hop",
            )

    def test_parselmouth_backend_is_causal(self):
        pytest.importorskip("parselmouth")
        self._assert_causal(expect_backend="parselmouth")

    def test_yin_fallback_is_causal(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "parselmouth", None)  # 强制 import 失败 → yin 回退
        self._assert_causal(expect_backend="yin")


class TestMimiRowShiftEndToEnd:
    """迷你 zarr 世界：行值编码行号，端到端验证 acts=s、mimi=s−1。"""

    def test_dataset_assembly_uses_shifted_rows(self, tmp_path):
        import pandas as pd
        import zarr

        from floor_circuit.mve.dataset import load_role_xy, run_dir_for

        n_steps = 6
        runs = tmp_path / "runs"
        for channel in (0, 1):
            run_dir = run_dir_for(runs, "s0", channel)
            run_dir.mkdir(parents=True)
            group = zarr.open_group(str(run_dir), mode="w")
            ramp = np.arange(n_steps, dtype=np.float16).reshape(-1, 1)
            for name in ("acts_L4", "mimi_latent"):
                array = group.create_array(
                    name,
                    shape=(n_steps, 1),
                    dtype="float16",
                    chunks=(n_steps, 1),
                )
                array[:] = ramp + (100 if channel == 1 else 0)
        labels = pd.DataFrame(
            {
                "target": ["T1"] * n_steps,
                "agent_channel": [0] * n_steps,
                "step": list(range(n_steps)),
                "label": [0, 1] * (n_steps // 2),
                "delta_ms": [240] * n_steps,
            }
        )

        X_acts, y = load_role_xy(runs, labels, "s0", 0, 4, "T1", 240, feature="acts")
        assert y.tolist() == [1, 0, 1, 0, 1]  # 步 1..5
        np.testing.assert_array_equal(X_acts[:, 0], [1, 2, 3, 4, 5])  # acts 行 = s

        X_mimi, _ = load_role_xy(runs, labels, "s0", 0, -1, "T1", 240, feature="mimi")
        np.testing.assert_array_equal(X_mimi[:, 0], [0, 1, 2, 3, 4])  # 自通道行 = s−1
        np.testing.assert_array_equal(X_mimi[:, 1], [100, 101, 102, 103, 104])  # 对方通道同移
