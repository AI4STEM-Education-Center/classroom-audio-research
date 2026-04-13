"""
pipeline_full.py — Full pipeline with light spectral subtraction (alpha=0.15).

Steps:
  1. Load + resample + align
  2. Apply light spectral subtraction to attenuate cross-channel bleed
  3. Send each speaker's full aligned audio to Whisper
  4. Merge both streams sorted by time; flag temporal overlaps
"""

from pathlib import Path

import numpy as np

from config import WHISPER_MODEL, _OPENAI_AVAILABLE, _openai_lib
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from enhancement import _spectral_subtraction
from transcription_utils import _chunk_to_wav_bytes, _format_transcript, _mark_simultaneous_vad
from evaluation import evaluate_transcript

LIGHT_ALPHA = 0.15


def run_transcription(headset_wav: Path, array_wav: Path,
                      headset_meta: dict, array_meta: dict,
                      api_key: str) -> dict:
    """
    Full pipeline — gentle bleed-reduction approach:
    light spectral subtraction (alpha=0.15) → Whisper.
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

    y_h_clean = _spectral_subtraction(y_h, y_a, alpha=LIGHT_ALPHA)
    y_a_clean = _spectral_subtraction(y_a, y_h, alpha=LIGHT_ALPHA)

    client = _openai_lib.OpenAI(api_key=api_key)

    def _transcribe_full(y: np.ndarray, name: str) -> list:
        buf = _chunk_to_wav_bytes(y, sr)
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.wav", buf, "audio/wav"),
            language="en",
            response_format="verbose_json",
        )
        segs = getattr(result, "segments", None) or []
        out = []
        for s in segs:
            text = (getattr(s, "text", None) or "").strip()
            if text:
                out.append({
                    "speaker":      name,
                    "start":        round(float(getattr(s, "start", 0)), 2),
                    "end":          round(float(getattr(s, "end",   0)), 2),
                    "text":         text,
                    "simultaneous": False,
                })
        return out

    lines = _transcribe_full(y_h_clean, name_a) + _transcribe_full(y_a_clean, name_b)
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, [])

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline":             f"full pipeline (light bleed reduction, alpha={LIGHT_ALPHA})",
        "evaluation":           evaluate_transcript(lines, name_a, name_b),
    }
