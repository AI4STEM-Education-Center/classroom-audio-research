"""
pipeline_mvp.py — Two-stage adaptive pipeline: Ratio-VAD+AEC + Text-domain bleed subtraction.

Stage 1 — Audio domain (inherits from pipeline_ratiovat):
  - Measure bleed both directions → self-calibrating VAD thresholds
  - Run bleed-calibrated ratio VAD to classify frames correctly
  - Bidirectional NLMS AEC trained only on solo frames
  - Light post-AEC spectral subtraction

Stage 2 — Text domain (new):
  - Auto-detect the cleaner channel (lower bleed → less contamination)
  - Transcribe BOTH channels with Whisper after audio cleaning
  - Extract active intervals for the clean speaker from ratio-VAD frames
  - Run timestamp-gated n-gram subtraction: remove clean speaker's verbatim
    phrases from the noisy channel — but ONLY in time windows where the clean
    speaker was actually active (prevents false-positives when noisy speaker
    independently repeats the same phrase)

Why this works where audio-only approaches fall short:
  - Audio separation fails when two voices overlap continuously (near-0 dB SNR)
  - Text n-gram subtraction operates after Whisper decodes speech: bleed is
    transcribed verbatim (it's intelligible), so matching runs are exact
  - The timestamp gate makes this general: it doesn't assume the noisy speaker
    never says the same words — it only removes matches at times the clean
    speaker was speaking
  - No fixed thresholds: bleed measurement and VAD are self-calibrating per recording
"""

from pathlib import Path
import numpy as np

from config import (
    WHISPER_MODEL, HOP_MS,
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
    _tokenize,
    _subtract_ngrams_timed,
    _vad_active_intervals,
    _mark_simultaneous_vad,
)
from pipeline_ratiovat import _ratio_vad_frames, RATIO_MARGIN, POST_AEC_ALPHA
from evaluation import evaluate_transcript


def run_transcription_mvp(headset_wav: Path, array_wav: Path,
                           headset_meta: dict, array_meta: dict,
                           api_key: str) -> dict:
    """
    Two-stage pipeline: Ratio-VAD+AEC (audio) + timestamp-gated n-gram
    subtraction (text). Fully adaptive — no fixed thresholds.
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

    # ── Stage 1a: measure bleed both directions ───────────────────────────────
    bleed_h = _measure_bleed(y_h, y_a, sr)   # fraction of array bleeding into headset
    bleed_a = _measure_bleed(y_a, y_h, sr)   # fraction of headset bleeding into array

    # Auto-detect which channel is cleaner (lower incoming bleed = less contamination)
    if bleed_h <= bleed_a:
        clean_y,   noisy_y   = y_h, y_a
        clean_name, noisy_name = name_h, name_a
        clean_bleed, noisy_bleed = bleed_h, bleed_a
        # In ratio-VAD frame convention: vad_a = headset, vad_b = array
        clean_vad_key, noisy_vad_key = "vad_a", "vad_b"
    else:
        clean_y,   noisy_y   = y_a, y_h
        clean_name, noisy_name = name_a, name_h
        clean_bleed, noisy_bleed = bleed_a, bleed_h
        clean_vad_key, noisy_vad_key = "vad_b", "vad_a"

    # ── Stage 1b: ratio VAD (uses headset/array convention internally) ────────
    frames = _ratio_vad_frames(y_h, y_a, sr, bleed_h, bleed_a, margin=RATIO_MARGIN)

    hop_samples = max(1, int(sr * HOP_MS / 1000))

    n_solo_h  = sum(1 for f in frames if     f["vad_a"] and not f["vad_b"])
    n_solo_a  = sum(1 for f in frames if not f["vad_a"] and     f["vad_b"])
    n_overlap = sum(1 for f in frames if     f["vad_a"] and     f["vad_b"])
    n_silence = sum(1 for f in frames if not f["vad_a"] and not f["vad_b"])

    # ── Stage 1c: bidirectional AEC ───────────────────────────────────────────
    frames_flipped = [{"vad_a": f["vad_b"], "vad_b": f["vad_a"]} for f in frames]
    y_h_aec = _adaptive_aec(y_h, y_a, frames,         sr)
    y_a_aec = _adaptive_aec(y_a, y_h, frames_flipped, sr)

    # ── Stage 1d: light post-AEC spectral subtraction ────────────────────────
    y_h_clean = _spectral_subtraction(y_h_aec, y_a_aec, alpha=POST_AEC_ALPHA)
    y_a_clean = _spectral_subtraction(y_a_aec, y_h_aec, alpha=POST_AEC_ALPHA)

    # Map cleaned arrays back to clean/noisy roles
    if bleed_h <= bleed_a:
        clean_clean, noisy_clean = y_h_clean, y_a_clean
    else:
        clean_clean, noisy_clean = y_a_clean, y_h_clean

    # ── Stage 2a: transcribe both channels ───────────────────────────────────
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

    clean_segs = _transcribe(clean_clean, clean_name)
    noisy_segs = _transcribe(noisy_clean, noisy_name)

    # ── Stage 2b: build reference word list from clean channel ────────────────
    clean_words = []
    for seg in clean_segs:
        clean_words.extend(_tokenize(seg["text"]))

    # ── Stage 2c: extract active intervals for clean speaker (solo only) ──────
    active_intervals = _vad_active_intervals(
        frames, clean_vad_key, noisy_vad_key, hop_samples, sr
    )

    # ── Stage 2d: timestamp-gated n-gram subtraction on noisy channel ─────────
    n_noisy_before = len(noisy_segs)
    noisy_segs_clean = _subtract_ngrams_timed(
        noisy_segs, clean_words,
        min_match=3,
        active_intervals=active_intervals,
    )
    n_removed = n_noisy_before - len(noisy_segs_clean)

    # ── Combine + mark simultaneous ───────────────────────────────────────────
    lines = clean_segs + noisy_segs_clean
    lines.sort(key=lambda x: x["start"])
    _mark_simultaneous_vad(lines, frames, hop_samples, sr)

    return {
        "transcript":           lines,
        "formatted":            _format_transcript(lines),
        "segments_transcribed": len(lines),
        "sync_offset_sec":      round(offset, 3),
        "pipeline": (
            f"MVP (ratio-VAD+AEC + text-diff) | margin={RATIO_MARGIN} | "
            f"clean={clean_name} (bleed {clean_bleed:.0%}), "
            f"noisy={noisy_name} (bleed {noisy_bleed:.0%}) | "
            f"solo-{name_h}: {n_solo_h} frames, "
            f"solo-{name_a}: {n_solo_a} frames, "
            f"overlap: {n_overlap} frames, "
            f"silence: {n_silence} frames | "
            f"clean active intervals: {len(active_intervals)} | "
            f"noisy segs removed by text-diff: {n_removed} | "
            f"post-AEC α={POST_AEC_ALPHA}"
        ),
        "evaluation": evaluate_transcript(lines, name_h, name_a),
        "insights": {
            "clean_channel":       clean_name,
            "noisy_channel":       noisy_name,
            "clean_bleed":         round(clean_bleed, 4),
            "noisy_bleed":         round(noisy_bleed, 4),
            "active_intervals":    len(active_intervals),
            "noisy_segs_before":   n_noisy_before,
            "noisy_segs_removed":  n_removed,
            "solo_clean_frames":   n_solo_h if bleed_h <= bleed_a else n_solo_a,
            "solo_noisy_frames":   n_solo_a if bleed_h <= bleed_a else n_solo_h,
            "overlap_frames":      n_overlap,
        },
    }
