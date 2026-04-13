"""
pipeline_raw.py — Aligned-only transcription (no preprocessing).

Sends each speaker's full aligned audio directly to Whisper with no VAD,
no enhancement, and no spectral preprocessing. Uses verbose_json for per-segment
timestamps so both streams can be interleaved.
"""

from pathlib import Path

import numpy as np

from config import WHISPER_MODEL, _OPENAI_AVAILABLE, _openai_lib
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from transcription_utils import _chunk_to_wav_bytes, _format_transcript, _mark_simultaneous_vad
from evaluation import evaluate_transcript


def run_transcription_raw(headset_wav: Path, array_wav: Path,
                          headset_meta: dict, array_meta: dict,
                          api_key: str) -> dict:
    """
    Minimal transcription pipeline — load + align only.
    No VAD, no enhancement, no spectral preprocessing.
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

    lines = _transcribe_full(y_h, name_a) + _transcribe_full(y_a, name_b)
    lines.sort(key=lambda x: x["start"])

    _mark_simultaneous_vad(lines, [])

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline":             "aligned only (no VAD / no enhancement)",
        "evaluation":           evaluate_transcript(lines, name_a, name_b),
    }
