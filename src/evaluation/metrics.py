"""Evaluation metrics for the audio pipeline.

Provides WER (word error rate), SDR (signal-to-distortion ratio),
and EER (equal error rate) computation for evaluating ASR, TSE,
and speaker verification models.
"""

from jiwer import wer as compute_wer


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis transcripts.

    Args:
        reference: Ground truth transcript.
        hypothesis: Model output transcript.

    Returns:
        WER as a float between 0.0 (perfect) and 1.0+ (bad).
    """
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return compute_wer(reference, hypothesis)


def signal_to_distortion_ratio(reference_audio, estimated_audio) -> float:
    """Compute SDR between reference (clean) and estimated (extracted) audio.

    Args:
        reference_audio: numpy array of clean target audio.
        estimated_audio: numpy array of extracted/processed audio.

    Returns:
        SDR in decibels. Higher is better.
    """
    import numpy as np

    # Ensure same length
    min_len = min(len(reference_audio), len(estimated_audio))
    ref = reference_audio[:min_len].astype(np.float64)
    est = estimated_audio[:min_len].astype(np.float64)

    # SDR = 10 * log10(||ref||^2 / ||ref - est||^2)
    noise = ref - est
    ref_energy = np.sum(ref ** 2)
    noise_energy = np.sum(noise ** 2)

    if noise_energy == 0:
        return float("inf")
    if ref_energy == 0:
        return 0.0

    return 10 * np.log10(ref_energy / noise_energy)
