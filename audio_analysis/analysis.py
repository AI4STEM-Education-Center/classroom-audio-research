"""
analysis.py — Per-channel audio quality metrics: SNR and bandwidth.

Functions:
  estimate_snr(y, sr) -> dict
  compute_bandwidth_metrics(y, sr) -> dict
  channel_report_to_dict(cr) -> dict
"""

from dataclasses import asdict

import numpy as np
import librosa

from config import (
    FRAME_MS,
    HOP_MS,
    SILENCE_DB_OFFSET,
    compute_bandwidth,
    _AQC_AVAILABLE,
    analyze_channel,
)


def estimate_snr(y: np.ndarray, sr: int) -> dict:
    """
    Blind SNR estimator using percentile frame energy.
      noise_floor  = 10th percentile of frame RMS (dB) — quietest frames ≈ ambient
      signal_level = 90th percentile               — typical loud speech
      SNR          = signal_level − noise_floor    (dB)
    Also reports voiced_fraction: share of frames > noise_floor + 10 dB.
    """
    frame_samples = max(1, int(sr * FRAME_MS / 1000))
    hop_samples   = max(1, int(sr * HOP_MS  / 1000))
    if len(y) < frame_samples:
        return {"snr_db": 0.0, "noise_floor_db": -120.0,
                "signal_level_db": -120.0, "voiced_fraction": 0.0}

    n_frames = (len(y) - frame_samples) // hop_samples + 1
    energies_db = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_samples
        chunk = y[s: s + frame_samples]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies_db[i] = 20.0 * np.log10(rms + 1e-10)

    noise_floor  = float(np.percentile(energies_db, 10))
    signal_level = float(np.percentile(energies_db, 90))
    voiced_fraction = float(np.mean(energies_db > (noise_floor + SILENCE_DB_OFFSET)))

    return {
        "snr_db":          round(signal_level - noise_floor, 2),
        "noise_floor_db":  round(noise_floor,  2),
        "signal_level_db": round(signal_level, 2),
        "voiced_fraction": round(voiced_fraction, 3),
    }


def compute_bandwidth_metrics(y: np.ndarray, sr: int) -> dict:
    """
    Spectral bandwidth and per-band energy distribution.

    Two bandwidth measures:
      - bandwidth_3db_hz / bandwidth_10db_hz:
          Full-spectrum measure (same as audio_quality_check). Dominated by DC/low-freq
          noise if the recording has strong subsonic content (HVAC, mic self-noise).
          Low values (< 200 Hz) indicate the spectrum is dominated by near-DC energy.
      - speech_band_bw_90pct_hz / speech_band_bw_50pct_hz:
          Measured within the speech band (100 Hz – 8 kHz). Robust to DC/low-freq noise.
    """
    n_fft = 2048
    hop   = 512
    S        = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    S_power  = S ** 2
    S_db     = librosa.power_to_db(S_power, ref=float(np.max(S_power)) + 1e-20)
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # ── Full-spectrum bandwidth (reference: global peak) ──────────────────────
    mean_spec = np.mean(S_db, axis=1)
    peak_db   = float(np.max(mean_spec))
    bw_3db    = float(compute_bandwidth(S_db, freqs, peak_db, 3))
    bw_10db   = float(compute_bandwidth(S_db, freqs, peak_db, 10))

    # ── Speech-band 90th-percentile cumulative energy frequency ───────────────
    sb_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    if sb_mask.any():
        sb_power    = np.mean(S_power[sb_mask, :], axis=1)
        sb_freqs    = freqs[sb_mask]
        cumsum      = np.cumsum(sb_power)
        total_sb    = cumsum[-1] + 1e-20
        idx_90      = np.searchsorted(cumsum, 0.90 * total_sb)
        idx_50      = np.searchsorted(cumsum, 0.50 * total_sb)
        sb_bw_90pct = float(sb_freqs[min(idx_90, len(sb_freqs) - 1)])
        sb_bw_50pct = float(sb_freqs[min(idx_50, len(sb_freqs) - 1)])
    else:
        sb_bw_90pct = sb_bw_50pct = 0.0

    # ── Band energy distribution ──────────────────────────────────────────────
    total = float(np.sum(np.mean(S_power, axis=1))) + 1e-20

    def band_pct(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.sum(np.mean(S_power[mask, :], axis=1)) / total * 100) if mask.any() else 0.0

    return {
        "bandwidth_3db_hz":        round(bw_3db),
        "bandwidth_10db_hz":       round(bw_10db),
        "speech_band_bw_90pct_hz": round(sb_bw_90pct),
        "speech_band_bw_50pct_hz": round(sb_bw_50pct),
        "energy_below_1k_pct":     round(band_pct(0,    1000), 1),
        "energy_1k_3k_pct":        round(band_pct(1000, 3000), 1),
        "energy_3k_8k_pct":        round(band_pct(3000, 8000), 1),
        "energy_above_8k_pct":     round(band_pct(8000, sr / 2), 1),
    }


def channel_report_to_dict(cr) -> dict:
    """Convert a ChannelReport dataclass to a JSON-safe plain dict."""
    if cr is None:
        return {}
    d = asdict(cr)
    return {
        k: (bool(v)  if isinstance(v, (bool,  np.bool_))    else
            float(v) if isinstance(v, (float, np.floating)) else
            int(v)   if isinstance(v, (int,   np.integer))  else v)
        for k, v in d.items()
    }
