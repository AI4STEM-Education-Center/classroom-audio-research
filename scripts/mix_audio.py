#!/usr/bin/env python3
"""Mix audio to simulate classroom conditions.

Creates synthetic mixtures for evaluating TSE performance under controlled
conditions. Supports:
  - Clean speech + WHAM! noise at varying SNR
  - Two-speaker overlap at varying SIR (signal-to-interference ratio)
  - Positive pairs (same speaker enrollment + test)
  - Negative pairs (different speaker enrollment + test)

Usage:
    # Mix two WAV files at 5dB SNR
    python scripts/mix_audio.py --target speaker1.wav --interference noise.wav --snr 5 --output mix.wav

    # Batch: create mixtures at multiple SNRs
    python scripts/mix_audio.py --target speaker1.wav --interference noise.wav --snr -5 0 5 10 15 --output-dir mixtures/

    # Mix with WHAM! noise (requires data/wham/)
    python scripts/mix_audio.py --target speaker1.wav --wham --snr 5 --output mix.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Read audio file and resample to target sample rate."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data)
    return data


def mix_at_snr(target: np.ndarray, interference: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix target and interference signals at a given SNR (dB).

    SNR = 10 * log10(power_target / power_interference)
    Positive SNR = target louder. Negative = interference louder.
    """
    min_len = min(len(target), len(interference))
    target = target[:min_len]
    interference = interference[:min_len]

    power_target = np.mean(target ** 2) + 1e-10
    power_interference = np.mean(interference ** 2) + 1e-10

    desired_ratio = 10 ** (snr_db / 10)
    scale = np.sqrt(power_target / (power_interference * desired_ratio))
    interference_scaled = interference * scale

    mixed = target + interference_scaled

    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * 0.95 / peak

    return mixed


def write_audio(path: str | Path, audio: np.ndarray, sr: int = 16000) -> None:
    """Write audio to WAV file."""
    sf.write(str(path), audio, sr, format="WAV")


def get_random_wham_noise(wham_dir: Path, duration_samples: int, sr: int = 16000) -> np.ndarray:
    """Load a random WHAM! noise segment."""
    noise_files = list(wham_dir.glob("**/*.wav"))
    if not noise_files:
        print(f"ERROR: No WAV files found in {wham_dir}")
        print("Run: python scripts/download_wham.py")
        sys.exit(1)

    rng = np.random.default_rng()
    noise_file = rng.choice(noise_files)
    noise = read_audio(noise_file, sr)

    if len(noise) >= duration_samples:
        start = rng.integers(0, len(noise) - duration_samples + 1)
        return noise[start : start + duration_samples]
    else:
        repeats = (duration_samples // len(noise)) + 1
        noise = np.tile(noise, repeats)
        return noise[:duration_samples]


def main():
    parser = argparse.ArgumentParser(description="Mix audio for TSE evaluation")
    parser.add_argument("--target", required=True, help="Target speaker WAV file")
    parser.add_argument("--interference", help="Interference WAV file (or use --wham)")
    parser.add_argument("--wham", action="store_true", help="Use random WHAM! noise as interference")
    parser.add_argument("--wham-dir", default="data/wham/wav", help="WHAM! noise directory")
    parser.add_argument("--snr", type=float, nargs="+", default=[5.0], help="SNR in dB (can specify multiple)")
    parser.add_argument("--output", help="Output WAV path (for single SNR)")
    parser.add_argument("--output-dir", help="Output directory (for multiple SNRs)")

    args = parser.parse_args()

    if not args.interference and not args.wham:
        parser.error("Specify --interference or --wham")

    target = read_audio(args.target)

    if args.wham:
        interference = get_random_wham_noise(Path(args.wham_dir), len(target))
    else:
        interference = read_audio(args.interference)

    for snr_db in args.snr:
        mixed = mix_at_snr(target, interference, snr_db)

        if args.output and len(args.snr) == 1:
            out_path = Path(args.output)
        elif args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.target).stem
            out_path = out_dir / f"{stem}_snr{snr_db:+.0f}dB.wav"
        else:
            stem = Path(args.target).stem
            out_path = Path(f"{stem}_snr{snr_db:+.0f}dB.wav")

        write_audio(out_path, mixed)
        print(f"Created: {out_path} (SNR={snr_db:+.0f} dB)")


if __name__ == "__main__":
    main()
