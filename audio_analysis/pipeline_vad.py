"""
pipeline_vad.py — VAD-gated + text-diff transcription pipeline.

Steps:
  - Speaker A (clean channel): VAD-gated audio → Whisper (solo frames only)
  - Speaker B (bleed channel): full audio → Whisper, then n-gram text subtraction
    removes any verbatim run of ≥3 words that appears in Speaker A's transcript
"""

from pathlib import Path

import numpy as np

from config import WHISPER_MODEL, _OPENAI_AVAILABLE, _openai_lib
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from analysis import estimate_snr
from vad import run_diarization
from transcription_utils import (
    _chunk_to_wav_bytes,
    _vad_gated_audio,
    _local_to_orig,
    _tokenize,
    _subtract_ngrams,
    _format_transcript,
    _mark_simultaneous_vad,
)
from config import HOP_MS
from evaluation import evaluate_transcript


def run_transcription_vad(headset_wav: Path, array_wav: Path,
                          headset_meta: dict, array_meta: dict,
                          api_key: str) -> dict:
    """
    VAD-gated + text-diff pipeline:
    Speaker A: VAD-gated (solo frames) → Whisper.
    Speaker B: full audio → Whisper, then remove A's words via n-gram matching.
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

    snr_h  = estimate_snr(y_h, sr)
    snr_a  = estimate_snr(y_a, sr)
    diar   = run_diarization(y_h, y_a, sr, snr_h, snr_a)
    frames = diar["frames"]

    n_single_a = sum(1 for f in frames if f["vad_a"] and not f["vad_b"])
    n_single_b = sum(1 for f in frames if f["vad_b"] and not f["vad_a"])
    n_overlap  = sum(1 for f in frames if f["vad_a"] and f["vad_b"])

    audio_a, offsets_a = _vad_gated_audio(y_h, frames, sr, "vad_a", "vad_b")
    gated_sec_a = sum(dur for _, dur, _ in offsets_a) if offsets_a else 0.0

    client = _openai_lib.OpenAI(api_key=api_key)

    def _transcribe_gated(audio: np.ndarray, offsets: list, name: str) -> list:
        if len(audio) == 0:
            return []
        buf    = _chunk_to_wav_bytes(audio, sr)
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.wav", buf, "audio/wav"),
            language="en",
            response_format="verbose_json",
        )
        segs = getattr(result, "segments", None) or []
        out  = []
        for s in segs:
            text = (getattr(s, "text", None) or "").strip()
            if not text:
                continue
            local_start = float(getattr(s, "start", 0))
            local_end   = float(getattr(s, "end",   0))
            orig_start  = _local_to_orig(local_start, offsets)
            orig_end    = _local_to_orig(local_end,   offsets)
            out.append({
                "speaker":      name,
                "start":        round(orig_start, 2),
                "end":          round(orig_end,   2),
                "text":         text,
                "simultaneous": False,
            })
        return out

    def _transcribe_full(y: np.ndarray, name: str) -> list:
        buf    = _chunk_to_wav_bytes(y, sr)
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.wav", buf, "audio/wav"),
            language="en",
            response_format="verbose_json",
        )
        segs = getattr(result, "segments", None) or []
        out  = []
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

    lines_a = _transcribe_gated(audio_a, offsets_a, name_a)
    lines_b = _transcribe_full(y_a, name_b)

    ref_words = _tokenize(" ".join(seg["text"] for seg in lines_a))

    words_before  = sum(len(seg["text"].split()) for seg in lines_b)
    lines_b       = _subtract_ngrams(lines_b, ref_words, min_match=3)
    words_after   = sum(len(seg["text"].split()) for seg in lines_b)
    words_removed = words_before - words_after

    hop_samples = max(1, int(sr * HOP_MS / 1000))
    lines = lines_a + lines_b
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, frames, hop_samples, sr)

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline": (
            f"VAD-gated+text-diff | {name_a}: {gated_sec_a:.1f}s gated, "
            f"{name_b}: full audio, {words_removed} bleed words removed "
            f"(overlap {n_overlap} frames)"
        ),
        "evaluation": evaluate_transcript(lines, name_a, name_b),
    }
