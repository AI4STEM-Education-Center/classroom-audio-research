"""
audio_io.py — WAV loading, timestamp parsing, sync-offset computation, and signal alignment.

Functions:
  load_mono_wav(wav_path) -> (y, sr)
  parse_filename_timestamp(wav_filename) -> Optional[int]
  compute_sync_offset(meta_a, meta_b) -> float
  align_signals(y_a, y_b, sr, offset_sec) -> (y_a, y_b)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa


def load_mono_wav(wav_path: Path) -> tuple:
    """Load a WAV file as mono float32. Returns (y, sr)."""
    try:
        data, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:
        data, sr = librosa.load(str(wav_path), sr=None, mono=False)
        data = np.array(data, dtype=np.float32)

    if data.ndim == 2:
        axis = 1 if data.shape[1] < data.shape[0] else 0
        data = np.mean(data, axis=axis)
    return data.astype(np.float32), int(sr)


def parse_filename_timestamp(wav_filename: str) -> Optional[int]:
    """
    Extract HHMMSS suffix from a filename like *_090658.wav.
    Returns seconds since midnight, or None if pattern not found.
    """
    m = re.search(r"_(\d{6})\.\w+$", wav_filename)
    if not m:
        return None
    ts = m.group(1)
    return int(ts[0:2]) * 3600 + int(ts[2:4]) * 60 + int(ts[4:6])


def compute_sync_offset(meta_a: dict, meta_b: dict) -> float:
    """
    Return offset_sec = time_b_start - time_a_start.
    Positive → B started later than A → pad B's start with zeros to align.

    Priority order (highest precision first):
      1. recording_started_at — ISO 8601 set by the browser at the moment
         MediaRecorder.start() / first PCM buffer was captured. Millisecond
         precision, unaffected by upload latency.
      2. filename HHMMSS suffix — server-assigned at upload time, 1-second
         resolution, reflects arrival not start. Used only as fallback for
         recordings made before the client-timestamp fix.
      3. meta["timestamp"] — server receive time, least reliable.
    """
    def _parse_iso(s: str) -> Optional[float]:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        except (ValueError, AttributeError):
            return None

    # 1. Client-side recording start (most reliable)
    t_a = _parse_iso(meta_a.get("recording_started_at", ""))
    t_b = _parse_iso(meta_b.get("recording_started_at", ""))
    if t_a is not None and t_b is not None:
        return round(t_b - t_a, 3)

    # 2. Filename HHMMSS (1-second resolution fallback)
    t_a = parse_filename_timestamp(meta_a.get("wav_file", ""))
    t_b = parse_filename_timestamp(meta_b.get("wav_file", ""))
    if t_a is not None and t_b is not None:
        return float(t_b - t_a)

    # 3. Server receive timestamp (least reliable)
    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        ta = datetime.strptime(meta_a["timestamp"], fmt)
        tb = datetime.strptime(meta_b["timestamp"], fmt)
        return (tb - ta).total_seconds()
    except Exception:
        return 0.0


def align_signals(y_a: np.ndarray, y_b: np.ndarray,
                  sr: int, offset_sec: float) -> tuple:
    """
    Align two signals using the timing offset.
    offset_sec > 0: B started later → prepend zeros to B.
    offset_sec < 0: A started later → prepend zeros to A.
    Both arrays are truncated to the same length after padding.
    """
    offset_samples = int(abs(offset_sec) * sr)
    if offset_sec > 0:
        y_b = np.concatenate([np.zeros(offset_samples, dtype=y_b.dtype), y_b])
    elif offset_sec < 0:
        y_a = np.concatenate([np.zeros(offset_samples, dtype=y_a.dtype), y_a])
    min_len = min(len(y_a), len(y_b))
    return y_a[:min_len], y_b[:min_len]
