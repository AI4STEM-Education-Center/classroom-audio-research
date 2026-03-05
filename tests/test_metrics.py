"""Tests for evaluation metrics."""

import numpy as np

from src.evaluation.metrics import signal_to_distortion_ratio, word_error_rate


def test_wer_perfect():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_completely_wrong():
    assert word_error_rate("hello world", "foo bar") == 1.0


def test_wer_partial():
    result = word_error_rate("the cat sat on the mat", "the cat on the mat")
    assert 0.0 < result < 1.0


def test_wer_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "something") == 1.0


def test_sdr_identical():
    audio = np.random.randn(16000).astype(np.float32)
    sdr = signal_to_distortion_ratio(audio, audio)
    assert sdr == float("inf")


def test_sdr_noisy():
    ref = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    noise = np.random.randn(16000) * 0.01
    est = ref + noise
    sdr = signal_to_distortion_ratio(ref, est)
    # With small noise, SDR should be high
    assert sdr > 20
