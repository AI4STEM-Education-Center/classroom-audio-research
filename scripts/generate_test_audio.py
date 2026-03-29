#!/usr/bin/env python3
"""Generate synthetic test audio fixtures.

Creates simple synthetic audio files for testing the TSE pipeline.
Uses sine waves at different frequencies to simulate different speakers.

Usage:
    python scripts/generate_test_audio.py

Output files are saved to tests/fixtures/.
"""

import io
import struct
import wave
from pathlib import Path

import numpy as np


FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
SAMPLE_RATE = 16000


def make_sine_wav(freq_hz: float, duration_s: float, amplitude: float = 0.5) -> bytes:
    """Generate a sine wave as WAV bytes."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def make_mixed_wav(
    freq1: float, freq2: float, duration_s: float, ratio: float = 0.5
) -> bytes:
    """Generate a mixture of two sine waves as WAV bytes.

    Args:
        freq1: Frequency of speaker 1 (Hz)
        freq2: Frequency of speaker 2 (Hz)
        duration_s: Duration in seconds
        ratio: Mix ratio (0.5 = equal volume)
    """
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    s1 = ratio * np.sin(2 * np.pi * freq1 * t)
    s2 = (1 - ratio) * np.sin(2 * np.pi * freq2 * t)
    mixed = ((s1 + s2) * 0.5 * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(mixed.tobytes())
    return buf.getvalue()


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Clean single-speaker audio (simulates target speaker)
    clean_path = FIXTURES_DIR / "clean_speech.wav"
    clean_path.write_bytes(make_sine_wav(freq_hz=440.0, duration_s=3.0))
    print(f"Created: {clean_path}")

    # Reference clip (short clip of target speaker)
    ref_path = FIXTURES_DIR / "reference.wav"
    ref_path.write_bytes(make_sine_wav(freq_hz=440.0, duration_s=3.0))
    print(f"Created: {ref_path}")

    # Mixed audio (two speakers overlapping)
    mixed_path = FIXTURES_DIR / "mixed_speech.wav"
    mixed_path.write_bytes(make_mixed_wav(freq1=440.0, freq2=880.0, duration_s=3.0))
    print(f"Created: {mixed_path}")

    # Silence (for edge case testing)
    silence_path = FIXTURES_DIR / "silence_1s.wav"
    num_samples = SAMPLE_RATE
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    silence_path.write_bytes(buf.getvalue())
    print(f"Created: {silence_path}")

    print(f"\nAll fixtures saved to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
