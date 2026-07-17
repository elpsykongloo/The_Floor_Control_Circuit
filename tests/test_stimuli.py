"""刺激工程单测：S1 文本库、S2 试次清单、音频操作与质检。"""

from __future__ import annotations

import numpy as np
import pytest

from floor_circuit.stimuli.audio_ops import (
    clip_ratio,
    energy_match,
    fit_length,
    mix_at_snr,
    reverse,
    rms,
)
from floor_circuit.stimuli.build import s2_trials_manifest
from floor_circuit.stimuli.textgen import generate_s1


class TestS1Text:
    @pytest.mark.parametrize("lang", ["en", "zh"])
    def test_counts_and_uniqueness(self, lang):
        items = generate_s1(lang)
        assert len(items) == 300
        assert len({it["complete"] for it in items}) == 300
        assert len({it["incomplete"] for it in items}) == 300  # 话头机制保证前缀互异

    @pytest.mark.parametrize("lang", ["en", "zh"])
    def test_incomplete_is_proper_prefix(self, lang):
        for it in generate_s1(lang):
            assert it["complete"].startswith(it["incomplete"].rstrip())
            assert len(it["complete"]) > len(it["incomplete"])

    def test_en_incomplete_ends_projecting(self):
        # 不完整前缀不得以句末标点结束
        for it in generate_s1("en"):
            assert not it["incomplete"].rstrip().endswith((".", "!", "?"))

    def test_zh_incomplete_no_terminal_punct(self):
        for it in generate_s1("zh"):
            assert not it["incomplete"].endswith(("。", "！", "？"))


class TestS2Manifest:
    def test_trials(self, stimuli_cfg, tmp_path):
        df = s2_trials_manifest("en", tmp_path, stimuli_cfg, seed=42)
        assert len(df) == 5 * 200
        assert set(df["type"]) == {"instruct", "babble", "reversed", "backchannel", "crosslingual"}
        assert df["onset_pct"].between(20, 80).all()
        xl = df[df["type"] == "crosslingual"]
        assert (xl["source_lang"] == "zh").all()
        df2 = s2_trials_manifest("en", tmp_path, stimuli_cfg, seed=42)
        assert np.allclose(df["onset_pct"], df2["onset_pct"])  # 种子固定可复现


class TestAudioOps:
    def test_reverse(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(reverse(x), [3.0, 2.0, 1.0])

    def test_mix_at_snr_hits_target(self):
        rng = np.random.default_rng(0)
        sig = np.sin(2 * np.pi * 220 * np.arange(24000) / 24000).astype(np.float32) * 0.3
        noise = rng.normal(0, 1, 48000).astype(np.float32)
        mixed = mix_at_snr(sig, noise, snr_db=10.0, rng=np.random.default_rng(1))
        residual = mixed - sig
        snr_est = 20 * np.log10(rms(sig) / rms(residual))
        assert abs(snr_est - 10.0) < 0.5

    def test_energy_match(self):
        a = np.ones(1000, dtype=np.float32) * 0.1
        b = np.ones(1000, dtype=np.float32) * 0.5
        assert abs(rms(energy_match(a, b)) - rms(b)) < 1e-6

    def test_fit_length(self):
        rng = np.random.default_rng(0)
        assert len(fit_length(np.ones(10, np.float32), 25, rng)) == 25
        assert len(fit_length(np.ones(100, np.float32), 25, rng)) == 25

    def test_clip_ratio(self):
        x = np.array([0.5, 1.0, -1.0, 0.2], dtype=np.float32)
        assert clip_ratio(x) == 0.5


class TestLoudness:
    def test_normalize_lufs(self):
        pyloudnorm = pytest.importorskip("pyloudnorm")
        from floor_circuit.stimuli.audio_ops import loudness_lufs, normalize_lufs

        sr = 24000
        rng = np.random.default_rng(0)
        wav = (rng.normal(0, 0.1, sr * 3)).astype(np.float32)
        out = normalize_lufs(wav, sr, -20.0)
        assert abs(loudness_lufs(out, sr) - (-20.0)) < 0.3
        _ = pyloudnorm
