"""
pipeline_adaptive.py — Adaptive spectral subtraction pipeline.

Steps:
  1. Load + align
  2. Measure per-channel bleed ratio
  3. Derive proportional alpha per channel (capped at 0.4)
  4. Transcribe each channel twice (raw + subtracted); keep whichever
     Whisper scores higher (lower no_speech_prob)
"""

from pathlib import Path

from config import _OPENAI_AVAILABLE, _openai_lib
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from enhancement import _spectral_subtraction
from transcription_utils import _measure_bleed, _transcribe_best, _format_transcript, _mark_simultaneous_vad
from evaluation import evaluate_transcript


def run_transcription_adaptive(headset_wav: Path, array_wav: Path,
                                headset_meta: dict, array_meta: dict,
                                api_key: str) -> dict:
    """
    Adaptive pipeline — measures bleed per channel, derives proportional alpha,
    then picks raw vs. subtracted per mic based on Whisper's no_speech_prob.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    y_h_raw, sr_h = load_mono_wav(headset_wav)
    y_a_raw, sr_a = load_mono_wav(array_wav)
    y_h, sr  = _resample(y_h_raw, sr_h)
    y_a, _   = _resample(y_a_raw, sr_a)
    offset   = compute_sync_offset(headset_meta, array_meta)
    y_h, y_a = align_signals(y_h, y_a, sr, offset)

    name_a = headset_meta.get("tag") or headset_meta.get("source") or "A"
    name_b = array_meta.get("tag")   or array_meta.get("source")   or "B"

    bleed_h = _measure_bleed(y_h, y_a, sr)
    bleed_a = _measure_bleed(y_a, y_h, sr)
    alpha_h = round(min(bleed_h * 0.6, 0.4), 3)
    alpha_a = round(min(bleed_a * 0.6, 0.4), 3)

    y_h_clean = _spectral_subtraction(y_h, y_a, alpha=alpha_h)
    y_a_clean = _spectral_subtraction(y_a, y_h, alpha=alpha_a)

    client = _openai_lib.OpenAI(api_key=api_key)
    lines_a, label_a, nsp_raw_a, nsp_sub_a = _transcribe_best(y_h, y_h_clean, name_a, client, sr)
    lines_b, label_b, nsp_raw_b, nsp_sub_b = _transcribe_best(y_a, y_a_clean, name_b, client, sr)

    lines = lines_a + lines_b
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, [])

    pipeline_note = (
        f"adaptive | {name_a}: bleed={bleed_h:.0%} alpha={alpha_h} used={label_a} "
        f"(no_speech_prob raw={nsp_raw_a} sub={nsp_sub_a}) | "
        f"{name_b}: bleed={bleed_a:.0%} alpha={alpha_a} used={label_b} "
        f"(no_speech_prob raw={nsp_raw_b} sub={nsp_sub_b})"
    )

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline":             pipeline_note,
        "evaluation":           evaluate_transcript(lines, name_a, name_b),
    }
