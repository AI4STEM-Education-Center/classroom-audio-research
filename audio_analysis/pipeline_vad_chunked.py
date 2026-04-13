"""
pipeline_vad_chunked.py — VAD-chunked Whisper transcription pipeline.

The core structural fix for the "everything is simultaneous" problem:
instead of sending the full 90s recording to Whisper (which produces 1-3
giant segments spanning the whole recording), this pipeline:

  1. Runs ratio-VAD (bleed-calibrated) to identify per-frame speaker activity
  2. Extracts SOLO frames for each speaker into audio chunks (10–30s each)
  3. Sends each speaker's chunks to Whisper separately
  4. Maps Whisper's local timestamps back to original recording time
  5. Marks simultaneous using the VAD overlap frames (not Whisper timestamps)

Why this improves on all current pipelines:
  - Whisper gets a focused, single-speaker clip → shorter, more precise segments
  - Simultaneous flagging is finally accurate (VAD-based, not timestamp-based)
  - No audio enhancement needed: the bleed is isolated by selecting only solo frames
    so there's minimal contamination in each chunk's audio
  - Overlap frames are correctly detected from VAD and marked on the segments that
    fall in those windows, without artificially inflating the count

Caveat: words spoken during overlap frames are NOT transcribed by either speaker's
chunk. This is conservative — overlap speech is ambiguous anyway. Those segments
appear in the combined transcript with simultaneous=True when a VAD overlap window
coincides with a transcribed segment's timestamp.
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
    _vad_gated_audio,
    _local_to_orig,
    _format_transcript,
    _mark_simultaneous_vad,
)
from pipeline_ratiovat import _ratio_vad_frames, RATIO_MARGIN
from evaluation import evaluate_transcript


def run_transcription_vad_chunked(headset_wav: Path, array_wav: Path,
                                   headset_meta: dict, array_meta: dict,
                                   api_key: str) -> dict:
    """
    VAD-chunked transcription: ratio-VAD solo frames → per-speaker Whisper chunks
    → timestamp remapping → VAD-based simultaneous flagging.
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

    # ── Bleed-calibrated ratio VAD ────────────────────────────────────────────
    bleed_h = _measure_bleed(y_h, y_a, sr)
    bleed_a = _measure_bleed(y_a, y_h, sr)
    frames  = _ratio_vad_frames(y_h, y_a, sr, bleed_h, bleed_a, margin=RATIO_MARGIN)

    hop_samples = max(1, int(sr * HOP_MS / 1000))

    n_solo_h  = sum(1 for f in frames if     f["vad_a"] and not f["vad_b"])
    n_solo_a  = sum(1 for f in frames if not f["vad_a"] and     f["vad_b"])
    n_overlap = sum(1 for f in frames if     f["vad_a"] and     f["vad_b"])
    n_silence = sum(1 for f in frames if not f["vad_a"] and not f["vad_b"])

    # ── Extract solo-speaker audio chunks ─────────────────────────────────────
    # headset speaker (vad_a): solo when vad_a=True, vad_b=False
    # array speaker  (vad_b): solo when vad_b=True, vad_a=False
    audio_h, offsets_h = _vad_gated_audio(y_h, frames, sr, "vad_a", "vad_b")
    audio_a, offsets_a = _vad_gated_audio(y_a, frames, sr, "vad_b", "vad_a")

    gated_sec_h = sum(dur for _, dur, _ in offsets_h) if offsets_h else 0.0
    gated_sec_a = sum(dur for _, dur, _ in offsets_a) if offsets_a else 0.0

    # ── Transcribe each speaker's chunks ──────────────────────────────────────
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

    lines_h = _transcribe_gated(audio_h, offsets_h, name_h)
    lines_a = _transcribe_gated(audio_a, offsets_a, name_a)

    # ── Combine + VAD-based simultaneous flagging ─────────────────────────────
    lines = lines_h + lines_a
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, frames, hop_samples, sr)

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline": (
            f"VAD-chunked | margin={RATIO_MARGIN} | "
            f"bleed {name_h}←{name_a}: {bleed_h:.0%}, "
            f"bleed {name_a}←{name_h}: {bleed_a:.0%} | "
            f"solo-{name_h}: {n_solo_h} frames ({gated_sec_h:.1f}s), "
            f"solo-{name_a}: {n_solo_a} frames ({gated_sec_a:.1f}s), "
            f"overlap: {n_overlap} frames, silence: {n_silence} frames"
        ),
        "evaluation": evaluate_transcript(lines, name_h, name_a),
        "insights": {
            "bleed_h":       round(bleed_h, 4),
            "bleed_a":       round(bleed_a, 4),
            "solo_h_frames": n_solo_h,
            "solo_a_frames": n_solo_a,
            "overlap_frames": n_overlap,
            "gated_sec_h":   round(gated_sec_h, 2),
            "gated_sec_a":   round(gated_sec_a, 2),
        },
    }
