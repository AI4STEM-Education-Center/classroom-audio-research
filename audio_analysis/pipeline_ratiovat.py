"""
pipeline_ratiovat.py — Bleed-calibrated ratio VAD + AEC pipeline.

The core insight: energy-based VAD is fooled by cross-channel bleed.
When Speaker A talks, ~20-30% of their energy bleeds into Speaker B's mic,
and the naive VAD marks those frames as "both speaking" → AEC freezes → no
echo cancellation happens.

This pipeline fixes the VAD by using the RATIO between channels, calibrated
by the measured bleed, to correctly classify frames:

  - If Ch_A >> bleed_a × Ch_B  →  Speaker A is talking (Ch_A is primary)
  - If Ch_B >> bleed_b × Ch_A  →  Speaker B is talking (Ch_B is primary)
  - Both above expected bleed   →  genuine simultaneous speech (rare)

With a correct VAD, the AEC filter gets many more genuine solo frames to
adapt on, giving substantially better echo cancellation.

Steps:
  1. Load + align
  2. Measure bleed both directions
  3. Compute bleed-calibrated ratio VAD frames
  4. Run bidirectional AEC using corrected VAD
  5. Apply light spectral subtraction (belt + suspenders)
  6. Transcribe each channel with Whisper
"""

from pathlib import Path

import numpy as np

from config import (
    WHISPER_MODEL, HOP_MS, FRAME_MS,
    _OPENAI_AVAILABLE, _openai_lib,
)
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from enhancement import _spectral_subtraction
from transcription_utils import (
    _chunk_to_wav_bytes,
    _measure_bleed,
    _adaptive_aec,
    _format_transcript,
    _mark_simultaneous_vad,
)
from evaluation import evaluate_transcript

# How much higher than expected bleed energy a channel must be
# to count as that speaker being genuinely active.
# 1.5 = 50% above the bleed level → conservative, avoids false positives.
RATIO_MARGIN = 1.5

# Light post-AEC spectral subtraction (belt + suspenders).
# Kept low so AEC does the real work.
POST_AEC_ALPHA = 0.10


def _ratio_vad_frames(y_h: np.ndarray, y_a: np.ndarray,
                      sr: int,
                      bleed_h: float, bleed_a: float,
                      margin: float = RATIO_MARGIN) -> list:
    """
    Bleed-calibrated ratio VAD.

    For each frame, compute RMS energy of both channels. A channel is
    considered "active" only if its energy exceeds what would be explained
    by the other speaker's bleed alone (with a safety margin).

    Parameters
    ----------
    y_h, y_a  : aligned audio arrays (headset = A, array = B)
    sr        : sample rate
    bleed_h   : fraction of y_a that bleeds into y_h  (e.g. 0.07)
    bleed_a   : fraction of y_h that bleeds into y_a  (e.g. 0.29)
    margin    : multiplicative safety factor above expected-bleed level

    Returns list of frame dicts compatible with _adaptive_aec:
      { time_sec, vad_a, vad_b, energy_a_db, energy_b_db }
    """
    frame_samples = max(1, int(sr * FRAME_MS / 1000))
    hop_samples   = max(1, int(sr * HOP_MS   / 1000))
    n             = min(len(y_h), len(y_a))
    n_frames      = max(1, (n - frame_samples) // hop_samples + 1)

    frames: list = []
    for i in range(n_frames):
        s = i * hop_samples
        e = min(s + frame_samples, n)

        rms_h = float(np.sqrt(np.mean(y_h[s:e] ** 2) + 1e-20))
        rms_a = float(np.sqrt(np.mean(y_a[s:e] ** 2) + 1e-20))

        # Expected bleed in each channel if only the other speaker is active:
        # y_h contains bleed_h × y_a's energy
        # y_a contains bleed_a × y_h's energy
        expected_h_from_a = bleed_h * rms_a   # what y_h would have if only A spoke
        expected_a_from_h = bleed_a * rms_h   # what y_a would have if only H spoke

        # Speaker H (headset) is active if y_h energy is well above what
        # bleed from A alone would explain.
        vad_h = rms_h > expected_h_from_a * margin

        # Speaker A (array) is active if y_a energy is well above what
        # bleed from H alone would explain.
        vad_a = rms_a > expected_a_from_h * margin

        db_h = 20.0 * np.log10(rms_h + 1e-10)
        db_a = 20.0 * np.log10(rms_a + 1e-10)

        frames.append({
            "time_sec":    round(i * hop_samples / sr, 3),
            "vad_a":       vad_h,   # convention: vad_a = headset channel active
            "vad_b":       vad_a,   # vad_b = array channel active
            "energy_a_db": round(db_h, 2),
            "energy_b_db": round(db_a, 2),
        })

    return frames


def run_transcription_ratiovat(headset_wav: Path, array_wav: Path,
                                headset_meta: dict, array_meta: dict,
                                api_key: str) -> dict:
    """
    Bleed-calibrated ratio VAD + AEC pipeline.

    Uses measured cross-channel bleed to set rational VAD thresholds,
    then runs the existing bidirectional NLMS AEC with the corrected frames.
    A light spectral subtraction is applied afterwards.
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

    # Step 1: measure bleed both directions
    bleed_h = _measure_bleed(y_h, y_a, sr)   # fraction of y_a bleeding into y_h
    bleed_a = _measure_bleed(y_a, y_h, sr)   # fraction of y_h bleeding into y_a

    # Step 2: bleed-calibrated ratio VAD
    frames = _ratio_vad_frames(y_h, y_a, sr, bleed_h, bleed_a, margin=RATIO_MARGIN)

    n_solo_h  = sum(1 for f in frames if     f["vad_a"] and not f["vad_b"])
    n_solo_a  = sum(1 for f in frames if not f["vad_a"] and     f["vad_b"])
    n_overlap = sum(1 for f in frames if     f["vad_a"] and     f["vad_b"])
    n_silence = sum(1 for f in frames if not f["vad_a"] and not f["vad_b"])

    # Step 3: bidirectional AEC using the corrected VAD
    # Channel H: target=y_h, ref=y_a — adapt on frames where vad_a=T, vad_b=F
    # Channel A: target=y_a, ref=y_h — flip frame labels for the B pass
    frames_flipped = [{"vad_a": f["vad_b"], "vad_b": f["vad_a"]} for f in frames]
    y_h_aec = _adaptive_aec(y_h, y_a, frames,         sr)
    y_a_aec = _adaptive_aec(y_a, y_h, frames_flipped, sr)

    # Step 4: light post-AEC spectral subtraction
    y_h_clean = _spectral_subtraction(y_h_aec, y_a_aec, alpha=POST_AEC_ALPHA)
    y_a_clean = _spectral_subtraction(y_a_aec, y_h_aec, alpha=POST_AEC_ALPHA)

    # Step 5: transcribe
    client = _openai_lib.OpenAI(api_key=api_key)

    def _transcribe(y: np.ndarray, name: str) -> list:
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

    hop_samples = max(1, int(sr * HOP_MS / 1000))
    lines = _transcribe(y_h_clean, name_a) + _transcribe(y_a_clean, name_b)
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, frames, hop_samples, sr)

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline": (
            f"ratio-VAD+AEC | margin={RATIO_MARGIN} | "
            f"bleed {name_a}←{name_b}: {bleed_h:.0%}, "
            f"bleed {name_b}←{name_a}: {bleed_a:.0%} | "
            f"solo-{name_a}: {n_solo_h} frames, "
            f"solo-{name_b}: {n_solo_a} frames, "
            f"overlap: {n_overlap} frames, "
            f"silence: {n_silence} frames | "
            f"post-AEC α={POST_AEC_ALPHA}"
        ),
        "evaluation": evaluate_transcript(lines, name_a, name_b),
    }
