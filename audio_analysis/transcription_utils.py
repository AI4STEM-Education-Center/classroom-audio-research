"""
transcription_utils.py — Shared helpers used by all transcription pipelines.

Functions:
  _chunk_to_wav_bytes(y, sr) -> BytesIO
  _extract_speaker_segments(frames, hop_samples, sr, speaker) -> list
  _transcribe_chunk(y, sr, api_key, prompt) -> str
  _format_transcript(lines) -> str
  _measure_bleed(y_target, y_ref, sr) -> float
  _transcribe_best(y_raw, y_clean, name, client, sr) -> tuple
  _adaptive_aec(y_target, y_ref, frames, sr, n_taps, mu, block_size) -> ndarray
  _vad_gated_audio(y, frames, sr, speaker_key, other_key, min_dur, merge_gap) -> tuple
  _local_to_orig(t, offsets) -> float
  _tokenize(text) -> list
  _subtract_ngrams(xin_segs, ref_words, min_match) -> list
"""

import io
import re
from typing import Optional

import numpy as np
import soundfile as sf

from config import (
    WHISPER_MODEL,
    MIN_TRANSCRIBE_SEC,
    MIN_MERGE_GAP_SEC,
    MIN_SEGMENT_FRAMES,
    HOP_MS,
    _OPENAI_AVAILABLE,
    _openai_lib,
)

_VAD_GATE_PAD_SEC = 0.2   # silence inserted between chunks for clean Whisper boundaries


def _chunk_to_wav_bytes(y: np.ndarray, sr: int) -> io.BytesIO:
    """Encode a numpy audio array as WAV bytes (float32) for the OpenAI API."""
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf


def _extract_speaker_segments(frames: list, hop_samples: int, sr: int,
                               speaker: str) -> list:
    """
    Build continuous active-speech segments for one speaker from dual-VAD frames.
    speaker = 'a' or 'b'.

    Steps:
      1. Find runs of frames where vad_<speaker> is True.
      2. Merge gaps shorter than MIN_MERGE_GAP_SEC.
      3. Drop segments shorter than MIN_TRANSCRIBE_SEC.
    """
    vad_key  = f"vad_{speaker}"
    raw_segs: list = []
    in_speech = False
    seg_start = 0

    for i, f in enumerate(frames):
        if f[vad_key] and not in_speech:
            seg_start = i
            in_speech = True
        elif not f[vad_key] and in_speech:
            if i - seg_start >= MIN_SEGMENT_FRAMES:
                raw_segs.append({
                    "start_sec": round(frames[seg_start]["time_sec"], 3),
                    "end_sec":   round(frames[i]["time_sec"],         3),
                })
            in_speech = False

    if in_speech and (len(frames) - seg_start) >= MIN_SEGMENT_FRAMES:
        last_time = frames[-1]["time_sec"] + hop_samples / sr
        raw_segs.append({
            "start_sec": round(frames[seg_start]["time_sec"], 3),
            "end_sec":   round(last_time,                     3),
        })

    merged: list = []
    for seg in raw_segs:
        if seg["end_sec"] - seg["start_sec"] < MIN_TRANSCRIBE_SEC:
            continue
        if merged and seg["start_sec"] - merged[-1]["end_sec"] <= MIN_MERGE_GAP_SEC:
            merged[-1]["end_sec"] = seg["end_sec"]
        else:
            merged.append({"start_sec": seg["start_sec"], "end_sec": seg["end_sec"]})

    return [s for s in merged if s["end_sec"] - s["start_sec"] >= MIN_TRANSCRIBE_SEC]


def _transcribe_chunk(y: np.ndarray, sr: int, api_key: str,
                      prompt: str = "") -> str:
    """
    Send one audio chunk to OpenAI Whisper and return transcribed text.
    Uses a rolling prompt (last ~200 chars of previous turn) for context.
    """
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = _openai_lib.OpenAI(api_key=api_key)
    buf = _chunk_to_wav_bytes(y, sr)
    kwargs: dict = dict(
        model=WHISPER_MODEL,
        file=("chunk.wav", buf, "audio/wav"),
        language="en",
        response_format="text",
    )
    if prompt:
        kwargs["prompt"] = prompt[:224]
    result = client.audio.transcriptions.create(**kwargs)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return text.strip()


def _format_transcript(lines: list) -> str:
    """Render transcript lines as a human-readable string."""
    out: list = []
    prev_sim_ts: Optional[float] = None
    for line in lines:
        ts = f"[{int(line['start'] // 60):02d}:{int(line['start'] % 60):02d}]"
        if line["simultaneous"]:
            if prev_sim_ts is None or abs(line["start"] - prev_sim_ts) > 0.1:
                out.append(f"\n{ts} [simultaneous]")
                prev_sim_ts = line["start"]
            out.append(f'  Speaker {line["speaker"]}: "{line["text"]}"')
        else:
            prev_sim_ts = None
            out.append(f'\n{ts} Speaker {line["speaker"]}: "{line["text"]}"')
    return "\n".join(out).strip()


def _measure_bleed(y_target: np.ndarray, y_ref: np.ndarray, sr: int = 22050) -> float:
    """
    Estimate how much of y_ref bleeds into y_target.
    Uses only single-speaker windows (ref active, target passive) so overlap
    frames don't inflate the ratio.  Returns 0=no bleed, 1=equal energy.
    """
    frame = int(sr * 0.5)
    n = (min(len(y_target), len(y_ref)) // frame) * frame
    tgt = np.sqrt(np.mean(y_target[:n].reshape(-1, frame) ** 2, axis=1))
    ref = np.sqrt(np.mean(y_ref[:n].reshape(-1, frame) ** 2, axis=1))

    ref_active = ref > np.percentile(ref, 60)
    tgt_quiet  = tgt < np.percentile(tgt, 50)
    mask = ref_active & tgt_quiet

    if mask.sum() < 5:
        return float(np.median(tgt / (ref + 1e-10)))

    return float(np.median(tgt[mask] / (ref[mask] + 1e-10)))


def _transcribe_best(y_raw: np.ndarray, y_clean: np.ndarray,
                     name: str, client, sr: int) -> tuple:
    """
    Transcribe both raw and cleaned audio; pick the version with lower mean
    no_speech_prob (Whisper's per-segment hallucination indicator).
    avg_logprob is used as a tiebreaker when scores are close.

    Returns (segments_list, chosen_label, nsp_raw, nsp_clean).
    """
    def _call(y):
        buf = _chunk_to_wav_bytes(y, sr)
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.wav", buf, "audio/wav"),
            language="en",
            response_format="verbose_json",
        )
        segs = getattr(result, "segments", None) or []
        nsp = float(np.mean([getattr(s, "no_speech_prob", 0.0) for s in segs])) if segs else 1.0
        lp  = float(np.mean([getattr(s, "avg_logprob",   -1.0) for s in segs])) if segs else -99.0
        return segs, nsp, lp

    segs_raw,   nsp_raw,   lp_raw   = _call(y_raw)
    segs_clean, nsp_clean, lp_clean = _call(y_clean)

    if abs(nsp_clean - nsp_raw) < 0.05:
        chosen, label = (segs_clean, "subtracted") if lp_clean > lp_raw else (segs_raw, "raw")
    else:
        chosen, label = (segs_clean, "subtracted") if nsp_clean < nsp_raw else (segs_raw, "raw")

    out = []
    for s in chosen:
        text = (getattr(s, "text", None) or "").strip()
        if text:
            out.append({
                "speaker":      name,
                "start":        round(float(getattr(s, "start", 0)), 2),
                "end":          round(float(getattr(s, "end",   0)), 2),
                "text":         text,
                "simultaneous": False,
            })
    return out, label, round(nsp_raw, 3), round(nsp_clean, 3)


def _adaptive_aec(y_target: np.ndarray, y_ref: np.ndarray,
                   frames: list, sr: int,
                   n_taps: int = 512, mu: float = 0.02,
                   block_size: int = 512) -> np.ndarray:
    """
    Bidirectional adaptive echo canceller using block NLMS.

    Uses VAD frames to detect double-talk: filter weights only update during
    single-speaker segments (target active, reference silent). During double-talk
    the weights freeze so the filter never accidentally removes the target speaker.

    n_taps     : filter length in samples (~23 ms at 22050 Hz)
    mu         : NLMS step size (0 = no adaptation, larger = faster but less stable)
    block_size : samples per adaptation block
    """
    from numpy.lib.stride_tricks import sliding_window_view

    hop = max(1, int(sr * HOP_MS / 1000))
    n   = min(len(y_target), len(y_ref))
    tgt = y_target[:n].astype(np.float64)
    ref = y_ref[:n].astype(np.float64)

    single = np.zeros(n, dtype=bool)
    for fi, f in enumerate(frames):
        s = fi * hop
        e = min(s + hop, n)
        if f.get("vad_a") and not f.get("vad_b"):
            single[s:e] = True

    w      = np.zeros(n_taps)
    output = tgt.copy()
    ref_pad = np.concatenate([np.zeros(n_taps - 1), ref])

    for b in range(0, n, block_size):
        end_b = min(b + block_size, n)
        L     = end_b - b

        ref_block = sliding_window_view(ref_pad[b: b + L + n_taps - 1], n_taps)[:L]
        echo          = ref_block @ w
        e_block       = tgt[b:end_b] - echo
        output[b:end_b] = e_block

        ss_mask = single[b:end_b]
        if ss_mask.any():
            X_ss  = ref_block[ss_mask]
            e_ss  = e_block[ss_mask]
            grad  = (X_ss * e_ss[:, np.newaxis]).mean(axis=0)
            norm  = (X_ss ** 2).sum(axis=1).mean() + 1e-10
            w    += mu * grad / norm

    return output.astype(np.float32)


def _vad_gated_audio(y: np.ndarray, frames: list, sr: int,
                     speaker_key: str, other_key: str,
                     min_dur: float = 0.05,
                     merge_gap: float = 0.5) -> tuple:
    """
    Extract audio segments where `speaker_key` is active and `other_key` is silent.
    Nearby windows within `merge_gap` seconds are joined into one chunk to give
    Whisper more context.  A short silence is inserted between chunks so Whisper
    does not hallucinate across discontinuities.

    Returns (concatenated_audio, offsets) where each offset entry is
    (orig_start_sec, chunk_dur_sec, cum_start_sec).
    """
    hop = int(sr * HOP_MS / 1000)
    min_samples    = int(sr * min_dur)
    merge_samples  = int(sr * merge_gap)
    pad_samples    = int(sr * _VAD_GATE_PAD_SEC)
    silence        = np.zeros(pad_samples, dtype=np.float32)

    raw_chunks = []
    in_chunk = False
    chunk_start = 0
    for i, f in enumerate(frames):
        active = f.get(speaker_key, False) and not f.get(other_key, False)
        if active and not in_chunk:
            in_chunk = True
            chunk_start = i
        elif not active and in_chunk:
            in_chunk = False
            raw_chunks.append((chunk_start * hop, i * hop))
    if in_chunk:
        raw_chunks.append((chunk_start * hop, len(frames) * hop))

    segments = [(s, min(e, len(y))) for s, e in raw_chunks if e - s >= min_samples]

    if not segments:
        return np.array([], dtype=np.float32), []

    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= merge_samples:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    audio_parts = []
    offsets     = []   # (orig_start_sec, chunk_dur_sec, cum_start_sec)
    cum = 0.0
    for s, e in merged:
        chunk_dur = (e - s) / sr
        offsets.append((s / sr, chunk_dur, cum))
        audio_parts.append(y[s:e])
        audio_parts.append(silence)
        cum += chunk_dur + _VAD_GATE_PAD_SEC

    return np.concatenate(audio_parts), offsets


def _local_to_orig(t: float, offsets: list) -> float:
    """Map a timestamp in concatenated (gated) audio back to original recording time."""
    for orig_start, chunk_dur, cum in offsets:
        if t < cum + chunk_dur:
            return orig_start + (t - cum)
        if t < cum + chunk_dur + _VAD_GATE_PAD_SEC:
            return orig_start + chunk_dur
    orig_start, chunk_dur, _ = offsets[-1]
    return orig_start + chunk_dur


def _mark_simultaneous_vad(lines: list, frames: list,
                            hop_samples: int = 0, sr: int = 22050) -> None:
    """
    Mark transcript segments as simultaneous using VAD frame data.

    A segment is flagged [simultaneous] only when the VAD shows BOTH speakers
    were active during the segment's time window — preventing long Whisper
    segments (e.g. a single 90s segment for one speaker) from causing every
    other speaker's segment to be incorrectly flagged.

    When `frames` is empty (pipelines without VAD), falls back to the simpler
    Whisper-timestamp-overlap method so behaviour is unchanged for those pipelines.

    Modifies `lines` in-place; returns None.
    """
    if not frames:
        # Fallback: pure Whisper timestamp overlap
        for li in lines:
            for lj in lines:
                if li is not lj and li["start"] < lj["end"] and li["end"] > lj["start"]:
                    li["simultaneous"] = True
                    break
        return

    frame_dur = hop_samples / sr if hop_samples and sr else 0.0

    for seg in lines:
        seg_start = float(seg["start"])
        seg_end   = float(seg["end"])
        for f in frames:
            f_start = float(f["time_sec"])
            f_end   = f_start + frame_dur
            # This frame overlaps the segment's time window AND both speakers active
            if f_start < seg_end and f_end > seg_start:
                if f.get("vad_a", False) and f.get("vad_b", False):
                    seg["simultaneous"] = True
                    break


def _tokenize(text: str) -> list:
    """Lowercase word tokens stripped of punctuation for n-gram matching."""
    return re.findall(r"[a-z']+", text.lower())


def _subtract_ngrams(xin_segs: list, ref_words: list, min_match: int = 3) -> list:
    """
    Remove verbatim runs of ≥ min_match words from each segment that appear
    in ref_words.  Works on normalised tokens but reconstructs output from the
    original segment text so casing/punctuation are preserved for kept words.

    Returns a new list of segments (empty-text segments are dropped).
    """
    max_n = min(len(ref_words), 20)
    ref_ngrams: set = set()
    for n in range(min_match, max_n + 1):
        for i in range(len(ref_words) - n + 1):
            ref_ngrams.add(tuple(ref_words[i:i + n]))

    cleaned = []
    for seg in xin_segs:
        raw_tokens  = seg["text"].split()
        norm_tokens = _tokenize(seg["text"])

        norm_to_raw: list = []
        ni = 0
        for ri, rt in enumerate(raw_tokens):
            stripped = re.sub(r"[^a-z']", "", rt.lower())
            if stripped and ni < len(norm_tokens) and norm_tokens[ni] == stripped:
                norm_to_raw.append(ri)
                ni += 1

        bleed = [False] * len(norm_tokens)
        i = 0
        while i < len(norm_tokens):
            matched = 0
            for n in range(min(20, len(norm_tokens) - i), min_match - 1, -1):
                if tuple(norm_tokens[i:i + n]) in ref_ngrams:
                    matched = n
                    break
            if matched:
                for j in range(i, i + matched):
                    bleed[j] = True
                i += matched
            else:
                i += 1

        bleed_raw = set()
        for ni2, ri in enumerate(norm_to_raw):
            if bleed[ni2]:
                bleed_raw.add(ri)

        kept = [rt for ri, rt in enumerate(raw_tokens) if ri not in bleed_raw]
        text_clean = " ".join(kept).strip()
        if text_clean:
            cleaned.append({**seg, "text": text_clean})

    return cleaned


def _vad_active_intervals(frames: list, vad_key: str, other_key: str,
                          hop_samples: int, sr: int) -> list:
    """
    Extract (start_sec, end_sec) intervals where `vad_key` is True and
    `other_key` is False — i.e. solo frames of the speaker assigned to
    `vad_key`. Used as the timestamp gate for n-gram subtraction.
    """
    intervals = []
    in_seg = False
    seg_start_sec = 0.0

    for f in frames:
        active = f.get(vad_key, False) and not f.get(other_key, False)
        if active and not in_seg:
            in_seg = True
            seg_start_sec = f["time_sec"]
        elif not active and in_seg:
            in_seg = False
            intervals.append((seg_start_sec, f["time_sec"]))

    if in_seg:
        last_time = frames[-1]["time_sec"] + hop_samples / sr
        intervals.append((seg_start_sec, last_time))

    return intervals


def _subtract_ngrams_timed(xin_segs: list, ref_words: list,
                            min_match: int = 3,
                            active_intervals: list = None) -> list:
    """
    Like _subtract_ngrams, but only attempts removal on segments that
    temporally overlap with `active_intervals` [(start_sec, end_sec), ...].

    Segments that do NOT overlap with any active interval pass through
    unchanged — this prevents removing a noisy speaker's own words when
    they repeat a phrase the clean speaker said at a different time.

    If active_intervals is None or empty, behaves identically to
    _subtract_ngrams (all segments are eligible for removal).
    """
    if not active_intervals:
        return _subtract_ngrams(xin_segs, ref_words, min_match=min_match)

    max_n = min(len(ref_words), 20)
    ref_ngrams: set = set()
    for n in range(min_match, max_n + 1):
        for i in range(len(ref_words) - n + 1):
            ref_ngrams.add(tuple(ref_words[i:i + n]))

    def _overlaps_active(seg_start: float, seg_end: float) -> bool:
        for iv_start, iv_end in active_intervals:
            if seg_start < iv_end and seg_end > iv_start:
                return True
        return False

    cleaned = []
    for seg in xin_segs:
        seg_start = float(seg.get("start", 0))
        seg_end   = float(seg.get("end",   seg_start))

        if not _overlaps_active(seg_start, seg_end):
            cleaned.append(seg)
            continue

        raw_tokens  = seg["text"].split()
        norm_tokens = _tokenize(seg["text"])

        norm_to_raw: list = []
        ni = 0
        for ri, rt in enumerate(raw_tokens):
            stripped = re.sub(r"[^a-z']", "", rt.lower())
            if stripped and ni < len(norm_tokens) and norm_tokens[ni] == stripped:
                norm_to_raw.append(ri)
                ni += 1

        bleed_flag = [False] * len(norm_tokens)
        i = 0
        while i < len(norm_tokens):
            matched = 0
            for n in range(min(20, len(norm_tokens) - i), min_match - 1, -1):
                if tuple(norm_tokens[i:i + n]) in ref_ngrams:
                    matched = n
                    break
            if matched:
                for j in range(i, i + matched):
                    bleed_flag[j] = True
                i += matched
            else:
                i += 1

        bleed_raw = set()
        for ni2, ri in enumerate(norm_to_raw):
            if bleed_flag[ni2]:
                bleed_raw.add(ri)

        kept = [rt for ri, rt in enumerate(raw_tokens) if ri not in bleed_raw]
        text_clean = " ".join(kept).strip()
        if text_clean:
            cleaned.append({**seg, "text": text_clean})

    return cleaned
