"""
pipeline_aec.py — AEC (Adaptive Echo Cancellation) transcription pipeline.

Steps:
  1. Load + align
  2. Run VAD to identify single-speaker vs double-talk frames
  3. Apply bidirectional adaptive NLMS filter (each mic cleans the other's bleed)
  4. Send cleaned channels to Whisper
"""

from pathlib import Path

import numpy as np

from config import WHISPER_MODEL, _OPENAI_AVAILABLE, _openai_lib
from audio_io import load_mono_wav, compute_sync_offset, align_signals
from orchestrator import _resample
from analysis import estimate_snr
from vad import run_diarization
from transcription_utils import _chunk_to_wav_bytes, _adaptive_aec, _format_transcript, _mark_simultaneous_vad
from config import HOP_MS
from evaluation import evaluate_transcript


def run_transcription_aec(headset_wav: Path, array_wav: Path,
                           headset_meta: dict, array_meta: dict,
                           api_key: str) -> dict:
    """
    AEC pipeline — VAD-guided bidirectional NLMS filter → Whisper.
    Best for recordings with heavy cross-channel bleed.
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

    # Bidirectional AEC: each channel cleans the other's bleed
    # Channel A: target=y_h, ref=y_a — adapt on frames where vad_a=T, vad_b=F
    # Channel B: target=y_a, ref=y_h — flip frame labels for the B pass
    frames_flipped = [{"vad_a": f["vad_b"], "vad_b": f["vad_a"]} for f in frames]
    y_h_clean = _adaptive_aec(y_h, y_a, frames,         sr)
    y_a_clean = _adaptive_aec(y_a, y_h, frames_flipped, sr)

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

    n_single_a = sum(1 for f in frames if f["vad_a"] and not f["vad_b"])
    n_single_b = sum(1 for f in frames if f["vad_b"] and not f["vad_a"])
    n_overlap  = sum(1 for f in frames if f["vad_a"] and f["vad_b"])

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline": (
            f"AEC (bidirectional adaptive NLMS) | "
            f"single-{name_a}: {n_single_a} frames, "
            f"single-{name_b}: {n_single_b} frames, "
            f"overlap: {n_overlap} frames used for freeze"
        ),
        "evaluation": evaluate_transcript(lines, name_a, name_b),
    }
