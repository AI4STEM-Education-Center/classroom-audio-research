"""Audio utilities for loading, mixing, and resampling.

Used by evaluation scripts and the mix_audio.py tool.
"""

import numpy as np
import soundfile as sf


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate.

    Args:
        path: Path to audio file (WAV, FLAC, etc.)
        target_sr: Target sample rate (default 16kHz for most models).

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    data, sr = sf.read(path, dtype="float32")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        sr = target_sr

    return data, sr


def mix_audio(
    target: np.ndarray,
    interference: np.ndarray,
    sir_db: float = 20.0,
) -> np.ndarray:
    """Mix target and interference audio at a given signal-to-interference ratio.

    Args:
        target: Target speaker audio (numpy array).
        interference: Interfering speaker audio (numpy array).
        sir_db: Signal-to-interference ratio in dB.
            20-40 dB simulates headset bleed (target much louder).
            0-10 dB simulates same-table conversation.

    Returns:
        Mixed audio as numpy array.
    """
    # Match lengths (truncate longer one)
    min_len = min(len(target), len(interference))
    target = target[:min_len].copy()
    interference = interference[:min_len].copy()

    # Scale interference to achieve desired SIR
    target_power = np.mean(target ** 2)
    interference_power = np.mean(interference ** 2)

    if interference_power == 0:
        return target

    # SIR = 10 * log10(target_power / scaled_interference_power)
    # scaled_interference_power = target_power / 10^(SIR/10)
    desired_interference_power = target_power / (10 ** (sir_db / 10))
    scale = np.sqrt(desired_interference_power / interference_power)

    mixed = target + scale * interference

    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed


def save_audio(path: str, audio: np.ndarray, sr: int = 16000) -> None:
    """Save audio array to WAV file."""
    sf.write(path, audio, sr)
