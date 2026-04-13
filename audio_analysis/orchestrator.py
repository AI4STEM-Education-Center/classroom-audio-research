"""
orchestrator.py — Full analysis pipeline coordinator and JSON serialization helper.

Functions:
  _resample(y, orig_sr) -> (y, sr)
  run_full_analysis(headset_wav, array_wav, headset_meta, array_meta) -> dict
  _json_safe(obj) -> obj
"""

from pathlib import Path

import numpy as np
import librosa

from config import (
    ANALYSIS_SR,
    _AQC_AVAILABLE,
    _AQC_ERROR,
    analyze_channel,
)
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from analysis import estimate_snr, compute_bandwidth_metrics, channel_report_to_dict
from enhancement import enhance_channels
from vad import run_diarization_testbed
from visualizations import (
    plot_waveform_overlay,
    plot_spectrograms,
    plot_enhancement_comparison,
    plot_diarization_timeline,
    plot_energy_ratio,
)


def _resample(y: np.ndarray, orig_sr: int) -> tuple:
    if orig_sr != ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=ANALYSIS_SR)
    return y, ANALYSIS_SR


def run_full_analysis(headset_wav: Path, array_wav: Path,
                      headset_meta: dict, array_meta: dict) -> dict:
    """
    Main analysis pipeline:
      1. Load both WAVs as mono float32
      2. Resample to ANALYSIS_SR (22050 Hz)
      3. Compute sync offset from metadata and align
      4. Per-mic: channel quality (audio_quality_check), SNR, bandwidth
      5. Two-mic VAD diarization testbed (all methods)
      6. Generate matplotlib plots as base64 PNGs
    """
    warns: list = []

    # 1. Load
    y_h_raw, sr_h = load_mono_wav(headset_wav)
    y_a_raw, sr_a = load_mono_wav(array_wav)

    # 2. Resample
    y_h, sr = _resample(y_h_raw, sr_h)
    y_a, _  = _resample(y_a_raw, sr_a)

    # 3. Sync + align
    offset_sec = compute_sync_offset(headset_meta, array_meta)
    if abs(offset_sec) > 30:
        warns.append(
            f"Large sync offset ({offset_sec:.1f} s) — verify both files are from the same session."
        )
    y_h, y_a = align_signals(y_h, y_a, sr, offset_sec)
    duration  = len(y_h) / sr

    # 4a. Quality (audio_quality_check.analyze_channel — may be unavailable)
    cr_h = cr_a = None
    if _AQC_AVAILABLE:
        cr_h = analyze_channel(y_h, sr, 0)
        cr_a = analyze_channel(y_a, sr, 0)
    else:
        warns.append(f"audio_quality_check not available ({_AQC_ERROR}); quality metrics skipped.")

    # 4b. SNR
    snr_h = estimate_snr(y_h, sr)
    snr_a = estimate_snr(y_a, sr)

    # 4c. Bandwidth
    bw_h = compute_bandwidth_metrics(y_h, sr)
    bw_a = compute_bandwidth_metrics(y_a, sr)

    name_a = headset_meta.get("tag") or headset_meta.get("source") or "Speaker A"
    name_b = array_meta.get("tag")   or array_meta.get("source")   or "Speaker B"

    # 5. VAD testbed: run all methods, then add per-method timeline plots
    testbed = run_diarization_testbed(y_h, y_a, sr, snr_h, snr_a)
    for mdata in testbed.values():
        mdata["plot_timeline"] = plot_diarization_timeline(
            mdata, duration, name_a=name_a, name_b=name_b
        )

    diarization = testbed.get("m2m3_combined") or next(iter(testbed.values()))

    # 6. Plots
    y_h_clean, y_a_clean = enhance_channels(y_h, y_a, sr, method="wiener")
    plots = {
        "waveform":               plot_waveform_overlay(y_h, y_a, sr),
        "spectrograms":           plot_spectrograms(y_h, y_a, sr),
        "enhancement_comparison": plot_enhancement_comparison(
            y_h, y_a, y_h_clean, y_a_clean, sr, name_a=name_a, name_b=name_b
        ),
        "diarization_timeline":   diarization["plot_timeline"],
        "energy_ratio":           plot_energy_ratio(diarization),
    }

    return {
        "headset": {
            "file":      headset_wav.name,
            "device":    headset_meta.get("deviceLabel", ""),
            "speaker":   name_a,
            "snr":       snr_h,
            "bandwidth": bw_h,
            "quality":   channel_report_to_dict(cr_h),
        },
        "array": {
            "file":      array_wav.name,
            "device":    array_meta.get("deviceLabel", ""),
            "speaker":   name_b,
            "snr":       snr_a,
            "bandwidth": bw_a,
            "quality":   channel_report_to_dict(cr_a),
        },
        "diarization":         diarization,
        "diarization_methods": testbed,
        "sync_offset_sec":     round(offset_sec, 3),
        "duration_sec":        round(duration,   2),
        "plots":               plots,
        "warnings":            warns,
    }


def _json_safe(obj):
    """Recursively convert numpy types so Flask's jsonify can serialize."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
