"""
vad.py — Voice Activity Detection and speaker diarization.

Functions:
  _vad_label(vad_a, vad_b) -> str
  _compute_vad_frames(y_a, y_b, sr, snr_a, snr_b, method) -> list
  _compute_silero_vad_frames(y_a, y_b, sr, silero_model, method) -> list
  _vad_frames_to_segments(frames, hop_samples, sr) -> list
  _vad_summary(frames) -> dict
  run_diarization_testbed(y_a, y_b, sr, snr_a, snr_b) -> dict
  run_diarization(y_a, y_b, sr, snr_a, snr_b) -> dict
"""

import numpy as np
import librosa

from config import (
    FRAME_MS,
    HOP_MS,
    SILENCE_DB_OFFSET,
    DIARIZATION_SMOOTH_K,
    MIN_SEGMENT_FRAMES,
    BLEED_FACTOR,
    SILERO_THRESHOLD,
    SILERO_SR,
    CROSSTALK_ALPHA,
    L_SILENCE, L_A, L_B, L_OVERLAP,
)
from enhancement import _spectral_subtraction, enhance_channels


def _vad_label(vad_a: bool, vad_b: bool) -> str:
    """Map dual-boolean VAD state to a single label string."""
    if vad_a and vad_b:  return L_OVERLAP
    if vad_a:            return L_A
    if vad_b:            return L_B
    return L_SILENCE


def _compute_vad_frames(y_a: np.ndarray, y_b: np.ndarray,
                        sr: int, snr_a: dict, snr_b: dict,
                        method: str = "m2m3") -> list:
    """
    Compute per-frame independent dual-channel VAD.

    Each frame outputs { time_sec, vad_a, vad_b, energy_a_db, energy_b_db }.
    Four states: SILENCE (F,F) / SPEAKER_A (T,F) / SPEAKER_B (F,T) / OVERLAP (T,T).

    Methods:
      m1   — raw signals, fixed per-channel threshold (baseline)
      m3   — raw signals, adaptive threshold (raises when other mic is loud)
      m2   — crosstalk-subtracted signals, fixed threshold
      m2m3 — crosstalk-subtracted signals + adaptive threshold [default]
    """
    frame_samples = max(1, int(sr * FRAME_MS / 1000))
    hop_samples   = max(1, int(sr * HOP_MS  / 1000))
    n             = min(len(y_a), len(y_b))
    n_frames      = max(1, (n - frame_samples) // hop_samples + 1)

    thresh_a = snr_a.get("noise_floor_db", -60.0) + SILENCE_DB_OFFSET
    thresh_b = snr_b.get("noise_floor_db", -60.0) + SILENCE_DB_OFFSET

    # Pre-compute crosstalk-corrected signals for M2 / M2+M3
    if method in ("m2", "m2m3"):
        ya_src = _spectral_subtraction(y_a, y_b, alpha=CROSSTALK_ALPHA)
        yb_src = _spectral_subtraction(y_b, y_a, alpha=CROSSTALK_ALPHA)
    else:
        ya_src, yb_src = y_a, y_b

    frames: list = []
    for i in range(n_frames):
        s = i * hop_samples
        e = s + frame_samples

        ca = ya_src[s: min(e, len(ya_src))]
        cb = yb_src[s: min(e, len(yb_src))]
        db_a = 20.0 * np.log10(float(np.sqrt(np.mean(ca ** 2) + 1e-20)) + 1e-10)
        db_b = 20.0 * np.log10(float(np.sqrt(np.mean(cb ** 2) + 1e-20)) + 1e-10)

        # Raw energy for display (always from original signals so plots are comparable)
        if method in ("m2", "m2m3"):
            ra = y_a[s: min(e, len(y_a))]
            rb = y_b[s: min(e, len(y_b))]
            raw_db_a = 20.0 * np.log10(float(np.sqrt(np.mean(ra ** 2) + 1e-20)) + 1e-10)
            raw_db_b = 20.0 * np.log10(float(np.sqrt(np.mean(rb ** 2) + 1e-20)) + 1e-10)
        else:
            raw_db_a, raw_db_b = db_a, db_b

        # Adaptive threshold (M3 / M2+M3): raise threshold when other mic is loud
        if method in ("m3", "m2m3"):
            ta = thresh_a + max(0.0, (db_b - thresh_b) * BLEED_FACTOR)
            tb = thresh_b + max(0.0, (db_a - thresh_a) * BLEED_FACTOR)
        else:
            ta, tb = thresh_a, thresh_b

        frames.append({
            "time_sec":    round(i * hop_samples / sr, 3),
            "vad_a":       bool(db_a > ta),
            "vad_b":       bool(db_b > tb),
            "energy_a_db": round(raw_db_a, 2),
            "energy_b_db": round(raw_db_b, 2),
        })

    return frames


def _compute_silero_vad_frames(y_a: np.ndarray, y_b: np.ndarray,
                                sr: int, silero_model,
                                method: str = "m4") -> list:
    """
    Silero-based dual-channel VAD (M4, M2+M4).
    method='m4'   — raw signals fed to Silero
    method='m2m4' — crosstalk-subtracted signals fed to Silero
    """
    import torch
    from silero_vad import get_speech_timestamps

    if method == "m2m4":
        ya_src = _spectral_subtraction(y_a, y_b, alpha=CROSSTALK_ALPHA)
        yb_src = _spectral_subtraction(y_b, y_a, alpha=CROSSTALK_ALPHA)
    else:
        ya_src, yb_src = y_a, y_b

    def active_intervals(y: np.ndarray) -> list:
        y16    = librosa.resample(y, orig_sr=sr, target_sr=SILERO_SR)
        tensor = torch.FloatTensor(y16)
        ts = get_speech_timestamps(tensor, silero_model, sampling_rate=SILERO_SR,
                                   threshold=SILERO_THRESHOLD,
                                   min_speech_duration_ms=50,
                                   min_silence_duration_ms=50)
        ratio = sr / SILERO_SR
        return [(int(t["start"] * ratio), int(t["end"] * ratio)) for t in ts]

    def in_intervals(s: int, e: int, ivs: list) -> bool:
        return any(s < b and e > a for a, b in ivs)

    ivs_a = active_intervals(ya_src)
    ivs_b = active_intervals(yb_src)

    frame_samples = max(1, int(sr * FRAME_MS / 1000))
    hop_samples   = max(1, int(sr * HOP_MS  / 1000))
    n             = min(len(y_a), len(y_b))
    n_frames      = max(1, (n - frame_samples) // hop_samples + 1)

    frames: list = []
    for i in range(n_frames):
        s = i * hop_samples
        e = s + frame_samples
        ra = y_a[s: min(e, len(y_a))]
        rb = y_b[s: min(e, len(y_b))]
        raw_db_a = 20.0 * np.log10(float(np.sqrt(np.mean(ra ** 2) + 1e-20)) + 1e-10)
        raw_db_b = 20.0 * np.log10(float(np.sqrt(np.mean(rb ** 2) + 1e-20)) + 1e-10)
        frames.append({
            "time_sec":    round(i * hop_samples / sr, 3),
            "vad_a":       in_intervals(s, e, ivs_a),
            "vad_b":       in_intervals(s, e, ivs_b),
            "energy_a_db": round(raw_db_a, 2),
            "energy_b_db": round(raw_db_b, 2),
        })

    return frames


def _vad_frames_to_segments(frames: list, hop_samples: int, sr: int) -> list:
    """Merge consecutive same-label frames into contiguous segments (≥ MIN_SEGMENT_FRAMES)."""
    if not frames:
        return []
    segs: list = []
    cur_lbl   = _vad_label(frames[0]["vad_a"], frames[0]["vad_b"])
    cur_start = 0
    n_cur     = 1

    for i in range(1, len(frames)):
        lbl = _vad_label(frames[i]["vad_a"], frames[i]["vad_b"])
        if lbl == cur_lbl:
            n_cur += 1
        else:
            if n_cur >= MIN_SEGMENT_FRAMES:
                segs.append({
                    "start_sec": round(frames[cur_start]["time_sec"], 3),
                    "end_sec":   round(frames[i]["time_sec"],         3),
                    "label":     cur_lbl,
                })
            cur_lbl   = lbl
            cur_start = i
            n_cur     = 1

    if n_cur >= MIN_SEGMENT_FRAMES:
        last_time = frames[-1]["time_sec"] + hop_samples / sr
        segs.append({
            "start_sec": round(frames[cur_start]["time_sec"], 3),
            "end_sec":   round(last_time,                     3),
            "label":     cur_lbl,
        })
    return segs


def _vad_summary(frames: list) -> dict:
    """Compute per-speaker and overlap statistics from dual-boolean VAD frames."""
    if not frames:
        return {"a_only_pct": 0.0, "b_only_pct": 0.0, "both_pct": 0.0,
                "silence_pct": 0.0, "a_total_pct": 0.0, "b_total_pct": 0.0,
                "overlap_ratio": 0.0}
    total  = len(frames)
    n_a    = sum(1 for f in frames if     f["vad_a"] and not f["vad_b"])
    n_b    = sum(1 for f in frames if not f["vad_a"] and     f["vad_b"])
    n_both = sum(1 for f in frames if     f["vad_a"] and     f["vad_b"])
    n_sil  = sum(1 for f in frames if not f["vad_a"] and not f["vad_b"])
    a_tot  = n_a + n_both
    b_tot  = n_b + n_both
    denom  = a_tot + b_tot - n_both
    return {
        "a_only_pct":    round(n_a    / total * 100, 1),
        "b_only_pct":    round(n_b    / total * 100, 1),
        "both_pct":      round(n_both / total * 100, 1),
        "silence_pct":   round(n_sil  / total * 100, 1),
        "a_total_pct":   round(a_tot  / total * 100, 1),
        "b_total_pct":   round(b_tot  / total * 100, 1),
        "overlap_ratio": round(n_both / denom * 100, 1) if denom > 0 else 0.0,
    }


def run_diarization_testbed(y_a: np.ndarray, y_b: np.ndarray,
                             sr: int, snr_a: dict, snr_b: dict) -> dict:
    """
    Run all available VAD methods and return results keyed by method name.
    Each value is { frames, segments, summary }; plot_timeline is added
    later in orchestrator once speaker names are known.
    """
    from analysis import estimate_snr  # imported here to avoid circular dep

    hop_samples = max(1, int(sr * HOP_MS / 1000))
    energy_methods = [
        ("m1_energy",     "m1"),
        ("m3_adaptive",   "m3"),
        ("m2_crosstalk",  "m2"),
        ("m2m3_combined", "m2m3"),
    ]

    _silero_model = None
    try:
        import torch as _torch  # noqa
        from silero_vad import load_silero_vad as _load_sv
        _silero_model, _ = _load_sv()
    except Exception:
        pass

    results: dict = {}
    for key, mcode in energy_methods:
        frames = _compute_vad_frames(y_a, y_b, sr, snr_a, snr_b, method=mcode)
        segs   = _vad_frames_to_segments(frames, hop_samples, sr)
        results[key] = {"frames": frames, "segments": segs, "summary": _vad_summary(frames)}

    # Wiener-enhanced VAD
    try:
        ya_enh, yb_enh = enhance_channels(y_a, y_b, sr, method="wiener")
        snr_a_enh = estimate_snr(ya_enh, sr)
        snr_b_enh = estimate_snr(yb_enh, sr)
        frames_enh = _compute_vad_frames(ya_enh, yb_enh, sr,
                                          snr_a_enh, snr_b_enh, method="m1")
        segs_enh = _vad_frames_to_segments(frames_enh, hop_samples, sr)
        results["m_wiener"] = {"frames": frames_enh, "segments": segs_enh,
                               "summary": _vad_summary(frames_enh)}
    except Exception:
        pass

    if _silero_model is not None:
        for key, mcode in [("m4_silero", "m4"), ("m2m4_best", "m2m4")]:
            try:
                frames = _compute_silero_vad_frames(y_a, y_b, sr, _silero_model, method=mcode)
                segs   = _vad_frames_to_segments(frames, hop_samples, sr)
                results[key] = {"frames": frames, "segments": segs,
                                "summary": _vad_summary(frames)}
            except Exception:
                pass

    return results


def run_diarization(y_a: np.ndarray, y_b: np.ndarray,
                    sr: int, snr_a: dict, snr_b: dict) -> dict:
    """
    Backward-compatible wrapper: runs the default VAD method (m2m3_combined)
    and returns { frames, segments, summary } for the transcription pipelines.
    """
    hop_samples = max(1, int(sr * HOP_MS / 1000))
    frames = _compute_vad_frames(y_a, y_b, sr, snr_a, snr_b, method="m2m3")
    segs   = _vad_frames_to_segments(frames, hop_samples, sr)
    return {"frames": frames, "segments": segs, "summary": _vad_summary(frames)}
