"""
pipeline_reftext.py — Clean-reference text subtraction pipeline.

Hypothesis: when bleed is strong and asymmetric, Whisper transcribes the
other speaker's words *verbatim* in the contaminated channel. Text-domain
n-gram matching can then remove those exact phrases surgically — without
any audio signal processing that might distort the target speaker's voice.

Why this is different from MVP (pipeline_mvp.py):
  - MVP runs AEC first, then text-diff on the cleaned audio. The AEC already
    removes the verbatim bleed, so text-diff finds nothing (0-1 matches).
  - This pipeline skips audio enhancement entirely: transcribe raw → text-diff.
    The bleed is still verbatim in the raw transcript → n-grams match reliably.

Robustness guards:
  1. Asymmetry check: if |bleed_a - bleed_b| < ASYMMETRY_MIN (10 pp), the
     channels are too similarly contaminated to trust one as a clean reference.
     Text-diff is skipped; both transcripts are returned as-is (safe fallback).
  2. Timestamp gate: only attempts removal on noisy segments that temporally
     overlap with the clean speaker's solo-active intervals (from ratio-VAD
     frames). Prevents removing the noisy speaker's own words when they repeat
     a phrase the clean speaker said at a different time.

Steps:
  1. Load + align (no audio enhancement)
  2. Measure bleed both directions
  3. Asymmetry guard → skip text-diff when channels are too similar
  4. Auto-detect clean (lower bleed) vs noisy channel
  5. Compute ratio-VAD frames for timestamp gating
  6. Transcribe both channels raw with Whisper (parallel)
  7. Build reference word list from clean channel transcript
  8. Extract clean speaker's solo-active intervals from VAD frames
  9. Timestamp-gated n-gram subtraction on noisy transcript
 10. Combine, sort, flag simultaneous, evaluate
"""

from pathlib import Path

import numpy as np

from config import (
    WHISPER_MODEL, HOP_MS,
    _OPENAI_AVAILABLE, _openai_lib,
)
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from transcription_utils import (
    _chunk_to_wav_bytes,
    _measure_bleed,
    _format_transcript,
    _tokenize,
    _subtract_ngrams_timed,
    _vad_active_intervals,
    _mark_simultaneous_vad,
)
from pipeline_ratiovat import _ratio_vad_frames, RATIO_MARGIN
from evaluation import evaluate_transcript

# If the two bleed measurements are within this margin, neither channel is
# clean enough to serve as a trustworthy reference → skip text-diff.
ASYMMETRY_MIN = 0.10


def run_transcription_reftext(headset_wav: Path, array_wav: Path,
                               headset_meta: dict, array_meta: dict,
                               api_key: str) -> dict:
    """
    Raw transcription of both channels + timestamp-gated n-gram subtraction.
    Falls back to raw output when bleed is symmetric.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    # ── Load + align ──────────────────────────────────────────────────────────
    y_h_raw, sr_h = load_mono_wav(headset_wav)
    y_a_raw, sr_a = load_mono_wav(array_wav)
    y_h, sr  = _resample(y_h_raw, sr_h)
    y_a, _   = _resample(y_a_raw, sr_a)
    offset   = compute_sync_offset(headset_meta, array_meta)
    y_h, y_a = align_signals(y_h, y_a, sr, offset)

    name_h = headset_meta.get("tag") or headset_meta.get("source") or "A"
    name_a = array_meta.get("tag")   or array_meta.get("source")   or "B"

    # ── Measure bleed both directions ─────────────────────────────────────────
    bleed_h = _measure_bleed(y_h, y_a, sr)   # fraction of array bleeding into headset
    bleed_a = _measure_bleed(y_a, y_h, sr)   # fraction of headset bleeding into array

    asymmetry = abs(bleed_h - bleed_a)
    text_diff_applied = asymmetry >= ASYMMETRY_MIN

    # ── Auto-detect clean vs noisy channel ───────────────────────────────────
    if bleed_h <= bleed_a:
        clean_y,   noisy_y   = y_h, y_a
        clean_name, noisy_name = name_h, name_a
        clean_bleed, noisy_bleed = bleed_h, bleed_a
        clean_vad_key, noisy_vad_key = "vad_a", "vad_b"
    else:
        clean_y,   noisy_y   = y_a, y_h
        clean_name, noisy_name = name_a, name_h
        clean_bleed, noisy_bleed = bleed_a, bleed_h
        clean_vad_key, noisy_vad_key = "vad_b", "vad_a"

    # ── Ratio-VAD frames for timestamp gate ───────────────────────────────────
    frames = _ratio_vad_frames(y_h, y_a, sr, bleed_h, bleed_a, margin=RATIO_MARGIN)
    hop_samples = max(1, int(sr * HOP_MS / 1000))

    # ── Transcribe both channels raw (no audio enhancement) ───────────────────
    client = _openai_lib.OpenAI(api_key=api_key)

    def _transcribe_raw(y: np.ndarray, name: str) -> list:
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

    clean_segs = _transcribe_raw(clean_y, clean_name)
    noisy_segs = _transcribe_raw(noisy_y, noisy_name)

    n_noisy_before = len(noisy_segs)
    n_removed = 0

    if text_diff_applied:
        # ── Build reference word list from clean transcript ───────────────────
        clean_words = []
        for seg in clean_segs:
            clean_words.extend(_tokenize(seg["text"]))

        # ── Extract clean speaker's solo-active intervals ─────────────────────
        active_intervals = _vad_active_intervals(
            frames, clean_vad_key, noisy_vad_key, hop_samples, sr
        )

        # ── Timestamp-gated n-gram subtraction ───────────────────────────────
        noisy_segs = _subtract_ngrams_timed(
            noisy_segs, clean_words,
            min_match=3,
            active_intervals=active_intervals,
        )
        n_removed = n_noisy_before - len(noisy_segs)
    else:
        active_intervals = []

    # ── Combine + mark simultaneous ───────────────────────────────────────────
    lines = clean_segs + noisy_segs
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, frames, hop_samples, sr)

    if text_diff_applied:
        diff_note = (
            f"text-diff applied | clean={clean_name} (bleed {clean_bleed:.0%}), "
            f"noisy={noisy_name} (bleed {noisy_bleed:.0%}) | "
            f"asymmetry={asymmetry:.0%} ≥ {ASYMMETRY_MIN:.0%} threshold | "
            f"active intervals: {len(active_intervals)} | "
            f"noisy segs removed: {n_removed}"
        )
    else:
        diff_note = (
            f"text-diff SKIPPED (symmetric bleed) | "
            f"{name_h}: {bleed_h:.0%}, {name_a}: {bleed_a:.0%} | "
            f"asymmetry={asymmetry:.0%} < {ASYMMETRY_MIN:.0%} threshold"
        )

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline": f"ref-text (raw → text-diff) | {diff_note}",
        "evaluation": evaluate_transcript(lines, name_h, name_a),
        "insights": {
            "clean_channel":      clean_name if text_diff_applied else None,
            "noisy_channel":      noisy_name if text_diff_applied else None,
            "clean_bleed":        round(clean_bleed, 4) if text_diff_applied else None,
            "noisy_bleed":        round(noisy_bleed, 4) if text_diff_applied else None,
            "asymmetry":          round(asymmetry, 4),
            "text_diff_applied":  text_diff_applied,
            "active_intervals":   len(active_intervals),
            "noisy_segs_before":  n_noisy_before,
            "noisy_segs_removed": n_removed,
        },
    }
