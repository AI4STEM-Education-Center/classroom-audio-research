"""
enhancement.py — Multi-channel audio enhancement: Wiener filter, noisereduce, spectral subtraction.

Functions:
  _spectral_subtraction(y_target, y_interference, alpha) -> ndarray
  _enhance_wiener(y_a, y_b, sr) -> (y_a_clean, y_b_clean)
  _enhance_noisereduce(y_a, y_b, sr) -> (y_a_clean, y_b_clean)
  enhance_channels(y_a, y_b, sr, method) -> (y_a_clean, y_b_clean)
"""

import numpy as np
import librosa

from config import (
    ENHANCE_BETA,
    ENHANCE_SMOOTH_FRAMES,
    ENHANCE_N_FFT,
    ENHANCE_HOP,
    CROSSTALK_ALPHA,
    _NOISEREDUCE_AVAILABLE,
    _noisereduce_lib,
)


def _spectral_subtraction(y_target: np.ndarray, y_interference: np.ndarray,
                           alpha: float = CROSSTALK_ALPHA) -> np.ndarray:
    """
    Simple time-domain crosstalk reduction (used inside VAD frame loops).
    Subtracts alpha * (level-scaled interference) from the target signal.
    """
    n = min(len(y_target), len(y_interference))
    yt = y_target[:n].copy()
    yi = y_interference[:n]
    scale = np.std(yt) / (np.std(yi) + 1e-10)
    return yt - alpha * scale * yi


def _enhance_wiener(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> tuple:
    """
    Frequency-domain cross-channel Wiener filter.

    For every STFT time-frequency bin the filter estimates how much of channel B
    leaks into channel A (and vice-versa) using the cross-power spectral density,
    then applies a per-bin Wiener gain that suppresses just that contribution
    while leaving the primary speaker's energy intact.

    Why this is better than time-domain subtraction (_spectral_subtraction):
      - Bleed is frequency-dependent (affected by room acoustics, distance,
        speaker angle).  A global scalar cannot model that.
      - The Wiener gain is bounded: it can never amplify the signal or produce
        negative energy, so it avoids the musical-noise artefacts that naive
        subtraction introduces.

    Returns (y_a_clean, y_b_clean) as float32 arrays of the original lengths.
    """
    from scipy.ndimage import uniform_filter1d

    n_fft = ENHANCE_N_FFT
    hop   = ENHANCE_HOP
    sf_k  = ENHANCE_SMOOTH_FRAMES
    beta  = ENHANCE_BETA

    S_a = librosa.stft(y_a, n_fft=n_fft, hop_length=hop)
    S_b = librosa.stft(y_b, n_fft=n_fft, hop_length=hop)

    P_a = np.abs(S_a) ** 2
    P_b = np.abs(S_b) ** 2

    P_a_s = uniform_filter1d(P_a, size=sf_k, axis=1) + 1e-10
    P_b_s = uniform_filter1d(P_b, size=sf_k, axis=1) + 1e-10

    C_ab = uniform_filter1d(np.real(S_a * np.conj(S_b)), size=sf_k, axis=1)
    C_ba = uniform_filter1d(np.real(S_b * np.conj(S_a)), size=sf_k, axis=1)

    H_ab = np.clip(C_ab / P_b_s, 0, None)
    H_ba = np.clip(C_ba / P_a_s, 0, None)

    P_interf_a = H_ab ** 2 * P_b_s
    P_interf_b = H_ba ** 2 * P_a_s

    G_a = np.sqrt(np.maximum(P_a_s - beta * P_interf_a, 0.0) / P_a_s)
    G_b = np.sqrt(np.maximum(P_b_s - beta * P_interf_b, 0.0) / P_b_s)

    y_a_clean = librosa.istft(G_a * S_a, hop_length=hop, n_fft=n_fft, length=len(y_a))
    y_b_clean = librosa.istft(G_b * S_b, hop_length=hop, n_fft=n_fft, length=len(y_b))

    return y_a_clean.astype(np.float32), y_b_clean.astype(np.float32)


def _enhance_noisereduce(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> tuple:
    """
    Spectral-gating noise reduction using `noisereduce` (pip install noisereduce).
    Each channel uses the other as a cross-channel noise reference — the bleed
    from the opposite speaker is treated as a stationary-ish noise profile.
    """
    ya_clean = _noisereduce_lib.reduce_noise(
        y=y_a, y_noise=y_b, sr=sr, stationary=False, prop_decrease=0.8
    )
    yb_clean = _noisereduce_lib.reduce_noise(
        y=y_b, y_noise=y_a, sr=sr, stationary=False, prop_decrease=0.8
    )
    return ya_clean.astype(np.float32), yb_clean.astype(np.float32)


def enhance_channels(y_a: np.ndarray, y_b: np.ndarray, sr: int,
                     method: str = "wiener") -> tuple:
    """
    Multi-channel audio enhancement dispatcher.

    Methods:
      'wiener'      — frequency-domain per-bin Wiener filter (no extra deps)
      'noisereduce' — spectral gating via noisereduce (pip install noisereduce)
      'combined'    — Wiener first, then noisereduce for residual noise removal
                       (best quality; falls back to Wiener if noisereduce missing)

    Returns (y_a_clean, y_b_clean) as float32 arrays of the same lengths.
    """
    if method == "noisereduce":
        if not _NOISEREDUCE_AVAILABLE:
            raise RuntimeError("noisereduce not installed: pip install noisereduce")
        return _enhance_noisereduce(y_a, y_b, sr)

    ya_enh, yb_enh = _enhance_wiener(y_a, y_b, sr)

    if method == "combined" and _NOISEREDUCE_AVAILABLE:
        ya_enh, yb_enh = _enhance_noisereduce(ya_enh, yb_enh, sr)

    return ya_enh, yb_enh
