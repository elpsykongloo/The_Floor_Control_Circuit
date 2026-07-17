"""库版官方算法 vs 取证参考实现的随机对拍（永久护栏）。

参考实现（scripts/wp1_g0_reference_recompute.py）已在本机对官方源码做过 2,000 组随机
对拍并在 138/138 真实会话逐帧全等；本测试保证库版（events/g0_official.py）与参考实现
在随机 VAD 上八条输出轨全等——任何一侧被改动而另一侧未同步都会在此失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from floor_circuit.events.g0_official import compute_official_labels

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wp1_g0_reference_recompute import compute_reference_labels

KEYS = [f"{cls}_ch{ch}" for cls in ("eot", "hold", "bot", "bc") for ch in (0, 1)]


def random_vad(rng: np.random.Generator, n: int) -> np.ndarray:
    """块状随机 VAD：交替静音/语音段，段长几何分布，覆盖边界形态。"""
    v = np.zeros(n, dtype=np.int8)
    pos = 0
    state = rng.integers(0, 2)
    while pos < n:
        seg = int(rng.geometric(0.08)) if state else int(rng.geometric(0.05))
        if state:
            v[pos : pos + seg] = 1
        pos += seg
        state = 1 - state
    return v


def test_random_parity_500_pairs():
    rng = np.random.default_rng(20260717)
    for trial in range(500):
        n = int(rng.integers(20, 400))
        vad0, vad1 = random_vad(rng, n), random_vad(rng, n)
        lib = compute_official_labels(vad0, vad1)
        ref = compute_reference_labels(vad0, vad1)
        for key in KEYS:
            assert np.array_equal(lib[key], ref[key]), (
                f"trial {trial} n={n} {key} 不一致：lib={np.nonzero(lib[key])[0]} ref={np.nonzero(ref[key])[0]}"
            )


def test_edge_patterns_parity():
    patterns = [
        (np.zeros(50, dtype=np.int8), np.zeros(50, dtype=np.int8)),
        (np.ones(50, dtype=np.int8), np.ones(50, dtype=np.int8)),
        (np.ones(50, dtype=np.int8), np.zeros(50, dtype=np.int8)),
    ]
    for vad0, vad1 in patterns:
        lib = compute_official_labels(vad0, vad1)
        ref = compute_reference_labels(vad0, vad1)
        for key in KEYS:
            assert np.array_equal(lib[key], ref[key])
