"""Tests for audio utilities."""

import numpy as np

from src.evaluation.audio_utils import mix_audio


def test_mix_audio_high_sir():
    """At high SIR, interference should be very quiet."""
    target = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    interference = np.sin(2 * np.pi * 880 * np.arange(16000) / 16000).astype(np.float32)

    mixed = mix_audio(target, interference, sir_db=40)

    # Mixed should be close to target at 40 dB SIR
    correlation = np.corrcoef(target, mixed)[0, 1]
    assert correlation > 0.99


def test_mix_audio_zero_sir():
    """At 0 dB SIR, both signals should be equally loud."""
    target = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    interference = np.sin(2 * np.pi * 880 * np.arange(16000) / 16000).astype(np.float32)

    mixed = mix_audio(target, interference, sir_db=0)

    # Mixed should differ noticeably from target
    correlation = np.corrcoef(target, mixed)[0, 1]
    assert correlation < 0.95


def test_mix_audio_no_clipping():
    """Output should not exceed [-1, 1]."""
    target = np.ones(16000, dtype=np.float32) * 0.9
    interference = np.ones(16000, dtype=np.float32) * 0.9

    mixed = mix_audio(target, interference, sir_db=0)
    assert np.abs(mixed).max() <= 1.0
