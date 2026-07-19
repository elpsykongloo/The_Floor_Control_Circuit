"""时间对齐哨兵测试（PREREG #7/#8）：四路表征观测截止统一为 (s+1)·τ。

锚定（#8，用户依 LMGen 源码裁定）：标签步 s = 在线系统刚接收完对方帧 s 的决策
状态；acts 读行 s+1（数组首位是 initial），Mimi/hazard/声学读行 s；acts[0] 永不
使用，末标签步（无对应 acts 行）丢弃。每个哨兵都带反例，防止测试退化为恒真。
"""

from __future__ import annotations

import numpy as np
import pytest

from floor_circuit.mve.alignment import (
    ANALYSIS_MAX_LABEL_STEP,
    ANALYSIS_TIME_ALIGNMENT,
    CONTEXT_TRUNCATION,
    FEATURE_OBSERVED_THROUGH_OFFSET,
    LABEL_STEP_OBSERVED_THROUGH_OFFSET,
    MIN_ELIGIBLE_STEP,
    MODEL_CONTEXT_STEPS,
    RUNNER_TIME_ALIGNMENT,
    feature_row_indices,
    min_eligible_step_for,
    usable_label_steps,
)
from floor_circuit.probes.baselines import hazard_features
from floor_circuit.probes.gru import make_windows


class TestRowMapping:
    def test_offsets_are_frozen(self):
        assert LABEL_STEP_OBSERVED_THROUGH_OFFSET == 1
        assert FEATURE_OBSERVED_THROUGH_OFFSET == {
            "acts": 0,
            "mimi": 1,
            "hazard": 1,
            "acoustic": 1,
            "mimi_prev": 2,
        }
        assert MIN_ELIGIBLE_STEP == 0
        assert RUNNER_TIME_ALIGNMENT == {
            "initial_token_position": 0,
            "acts_observed_through_offset_steps": 0,
        }
        assert ANALYSIS_TIME_ALIGNMENT["acts_row_for_step"] == "s+1"
        assert ANALYSIS_TIME_ALIGNMENT["baseline_row_for_step"] == "s"
        assert ANALYSIS_TIME_ALIGNMENT["last_label_step_dropped"] is True

    def test_context_truncation_is_frozen(self):
        # PREREG #11：Moshi 官方 context=3000；主判据窗标签步 ≤ 2998（acts 行 ≤ 2999）
        assert MODEL_CONTEXT_STEPS == 3000
        assert ANALYSIS_MAX_LABEL_STEP == 2998
        assert CONTEXT_TRUNCATION == {
            "context_steps": 3000,
            "analysis_max_label_step": 2998,
            "prereg": "#11",
        }

    def test_spike_row_3000_is_never_selected(self):
        # 全窗 7500 步缓存：可用步 0..2998 → acts 行 1..2999；行 3000（淘汰尖峰）不可达
        usable = usable_label_steps(7500)
        assert usable == 2999
        steps = np.arange(0, usable, dtype=np.int64)
        rows = feature_row_indices("acts", steps)
        assert int(rows.max()) == 2999
        assert int(rows.max()) < MODEL_CONTEXT_STEPS

    def test_mimi_prev_reads_row_s_minus_1(self):
        steps = np.array([1, 4, 9], dtype=np.int64)
        np.testing.assert_array_equal(feature_row_indices("mimi_prev", steps), [0, 3, 8])
        assert min_eligible_step_for("mimi_prev") == 1
        assert min_eligible_step_for("acts") == 0
        assert min_eligible_step_for("mimi") == 0
        with pytest.raises(ValueError, match="最小合法步"):
            feature_row_indices("mimi_prev", np.array([0, 1], dtype=np.int64))

    def test_acts_reads_row_s_plus_1_and_baselines_read_row_s(self):
        steps = np.array([0, 4, 9], dtype=np.int64)
        np.testing.assert_array_equal(feature_row_indices("acts", steps), [1, 5, 10])
        np.testing.assert_array_equal(feature_row_indices("mimi", steps), [0, 4, 9])
        np.testing.assert_array_equal(feature_row_indices("hazard", steps), [0, 4, 9])
        np.testing.assert_array_equal(feature_row_indices("acoustic", steps), [0, 4, 9])

    def test_initial_state_row_zero_is_never_selected_for_acts(self):
        # acts 行 0 是纯 initial 状态；任何合法标签步映射到的 acts 行都 ≥ 1
        steps = np.arange(0, 50, dtype=np.int64)
        assert int(feature_row_indices("acts", steps).min()) == 1

    def test_last_label_step_is_dropped_and_window_is_capped(self):
        # #8 末步丢弃 + #11 截断：可用步数 = min(n−1, 2999)
        assert usable_label_steps(7500) == 2999  # 全窗缓存截到规格内（步 0..2998）
        assert usable_label_steps(3001) == 2999
        assert usable_label_steps(3000) == 2999
        assert usable_label_steps(2999) == 2998  # 短于截断窗时仍是末步丢弃
        assert usable_label_steps(100) == 99
        assert usable_label_steps(1) == 0
        assert usable_label_steps(0) == 0

    def test_negative_step_is_rejected(self):
        with pytest.raises(ValueError, match="最小合法步"):
            feature_row_indices("mimi", np.array([-1, 1], dtype=np.int64))

    def test_unknown_feature_is_rejected(self):
        with pytest.raises(ValueError, match="未知表征类型"):
            feature_row_indices("depformer", np.array([1], dtype=np.int64))


def _random_states(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 6, size=n).astype(np.int64)


class TestHazardCausality:
    """hazard 行 s 只能依赖 states[0..s]（观测截止 (s+1)·τ）。"""

    def test_selected_rows_are_causal(self):
        rng = np.random.default_rng(20260718)
        for _trial in range(20):
            n = int(rng.integers(8, 40))
            states = _random_states(rng, n)
            full = hazard_features(states, step_s=0.08)
            for step in range(MIN_ELIGIBLE_STEP, n - 1):
                row = int(feature_row_indices("hazard", np.array([step]))[0])
                truncated = hazard_features(states[: step + 1], step_s=0.08)
                np.testing.assert_array_equal(
                    full[row],
                    truncated[row],
                    err_msg=f"step {step}: hazard 行读到了 states[{step + 1}:] 的信息",
                )

    def test_row_beyond_cutoff_leaks_next_state(self):
        """反例：读行 s+1 会随 states[s+1] 变化（证明越界读取真实泄漏）。"""
        states = np.array([0, 0, 1, 1, 3, 3, 0, 0], dtype=np.int64)
        perturbed = states.copy()
        step = 4
        perturbed[step + 1] = 4  # 改变截止之后的状态
        leaked_original = hazard_features(states, 0.08)[step + 1]
        leaked_perturbed = hazard_features(perturbed, 0.08)[step + 1]
        assert not np.array_equal(leaked_original, leaked_perturbed)
        # 正确映射（行 s）对同一扰动不变
        row_original = hazard_features(states, 0.08)[step]
        row_perturbed = hazard_features(perturbed, 0.08)[step]
        np.testing.assert_array_equal(row_original, row_perturbed)


class TestAcousticWindowCausality:
    """声学窗口（窗尾 s）不得包含帧 s+1 及之后的内容。"""

    def test_windows_ignore_future_frames(self):
        rng = np.random.default_rng(7)
        feats = rng.normal(size=(30, 4)).astype(np.float32)
        steps = np.arange(MIN_ELIGIBLE_STEP, 29, dtype=np.int64)
        rows = feature_row_indices("acoustic", steps)
        windows = make_windows(feats, rows)
        for index, step in enumerate(steps):
            perturbed = feats.copy()
            perturbed[step + 1 :] += 100.0
            perturbed_windows = make_windows(perturbed, rows)
            np.testing.assert_array_equal(
                windows[index],
                perturbed_windows[index],
                err_msg=f"step {step}: 声学窗口读到了帧 {step + 1} 及之后的内容",
            )

    def test_window_end_beyond_cutoff_contains_future_frame(self):
        """反例：窗尾 = s+1 会随帧 s+1 变化。"""
        rng = np.random.default_rng(9)
        feats = rng.normal(size=(12, 4)).astype(np.float32)
        step = 5
        perturbed = feats.copy()
        perturbed[step + 1] += 100.0
        leaked = make_windows(feats, np.array([step + 1]))
        leaked_perturbed = make_windows(perturbed, np.array([step + 1]))
        assert not np.array_equal(leaked, leaked_perturbed)


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

    def _assert_causal(self, expect_backend: str) -> None:
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


class TestRowShiftEndToEnd:
    """迷你 zarr 世界：行值编码行号，端到端验证 acts=s+1、mimi=s、末步丢弃。"""

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
        assert y.tolist() == [0, 1, 0, 1, 0]  # 可用步 0..4（末步 5 丢弃）
        np.testing.assert_array_equal(X_acts[:, 0], [1, 2, 3, 4, 5])  # acts 行 = s+1，行 0 弃用

        X_mimi, _ = load_role_xy(runs, labels, "s0", 0, -1, "T1", 240, feature="mimi")
        np.testing.assert_array_equal(X_mimi[:, 0], [0, 1, 2, 3, 4])  # 自通道行 = s
        np.testing.assert_array_equal(X_mimi[:, 1], [100, 101, 102, 103, 104])  # 对方通道同行

    def test_context_truncation_and_mimi_prev_end_to_end(self, tmp_path):
        """n_steps>3000 的世界：#11 截断到步 2998；mimi_prev 读行 s−1、步域 1 起。"""
        from types import SimpleNamespace

        import pandas as pd
        import zarr

        from floor_circuit.mve.dataset import load_role_xy, load_session_feature, run_dir_for

        n_steps = 3005
        runs = tmp_path / "runs"
        for channel in (0, 1):
            run_dir = run_dir_for(runs, "s0", channel)
            run_dir.mkdir(parents=True)
            group = zarr.open_group(str(run_dir), mode="w")
            ramp = (np.arange(n_steps, dtype=np.float32) % 1000).reshape(-1, 1)
            for name in ("acts_L4", "mimi_latent"):
                array = group.create_array(
                    name,
                    shape=(n_steps, 1),
                    dtype="float32",
                    chunks=(n_steps, 1),
                )
                array[:] = ramp + (0.5 if channel == 1 else 0.0)
        labels = pd.DataFrame(
            {
                "target": ["T4"] * n_steps,
                "agent_channel": [0] * n_steps,
                "step": list(range(n_steps)),
                "label": [i % 2 for i in range(n_steps)],
                "delta_ms": [-1] * n_steps,
            }
        )

        X_acts, y = load_role_xy(runs, labels, "s0", 0, 4, "T4", None, feature="acts")
        assert len(y) == 2999  # 步 0..2998（#11：行 3000 尖峰及其后不可达）
        assert float(X_acts[0, 0]) == 1.0  # 步 0 → acts 行 1
        assert float(X_acts[-1, 0]) == float(2999 % 1000)  # 步 2998 → acts 行 2999

        labels_root = tmp_path / "labels"
        labels_root.mkdir()
        labels.to_parquet(labels_root / "s0.parquet")
        specs = {
            ("s0", 0): SimpleNamespace(n_steps=n_steps),
            ("s0", 1): SimpleNamespace(n_steps=n_steps),
        }
        X_mimi, y_mimi = load_session_feature(
            runs, labels_root, "s0", specs, -1, "T4", None, feature="mimi"
        )
        assert len(y_mimi) == 2999
        assert float(X_mimi[0, 0]) == 0.0  # 步 0 → mimi 行 0
        X_prev, y_prev = load_session_feature(
            runs, labels_root, "s0", specs, -1, "T4", None, feature="mimi_prev", min_step=1
        )
        assert len(y_prev) == 2998  # 步 1..2998（mimi_prev 无步 0）
        assert float(X_prev[0, 0]) == 0.0  # 步 1 → mimi 行 0
        assert float(X_prev[0, 1]) == 0.5  # 对方通道同行
        assert float(X_prev[-1, 0]) == float(2997 % 1000)  # 步 2998 → 行 2997
