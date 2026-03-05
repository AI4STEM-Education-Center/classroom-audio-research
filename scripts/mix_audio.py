#!/usr/bin/env python3
"""Create overlap test audio by mixing two recordings at a given SIR.

Usage:
    python scripts/mix_audio.py --target speaker1/clean_01.wav --interference speaker2/clean_01.wav --sir 20
    python scripts/mix_audio.py --target speaker1/clean_01.wav --interference speaker2/clean_01.wav --sir 0 10 20 40

This simulates classroom audio conditions:
    SIR 20-40 dB: Headset bleed (target speaker is much louder, neighbor faint)
    SIR 0-10 dB:  Same-table conversation without headsets (both voices similar volume)
"""

import argparse
from pathlib import Path

from src.evaluation.audio_utils import load_audio, mix_audio, save_audio


def main():
    parser = argparse.ArgumentParser(description="Mix audio files at specified SIR levels")
    parser.add_argument("--target", required=True, help="Path to target speaker audio")
    parser.add_argument("--interference", required=True, help="Path to interference speaker audio")
    parser.add_argument("--sir", type=float, nargs="+", default=[20.0],
                        help="Signal-to-interference ratio(s) in dB (default: 20)")
    parser.add_argument("--output-dir", default="tests/fixtures/mixed",
                        help="Output directory (default: tests/fixtures/mixed)")
    args = parser.parse_args()

    target_audio, sr = load_audio(args.target)
    interference_audio, _ = load_audio(args.interference, target_sr=sr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_name = Path(args.target).stem
    interference_name = Path(args.interference).stem

    for sir in args.sir:
        mixed = mix_audio(target_audio, interference_audio, sir_db=sir)
        output_path = output_dir / f"{target_name}_plus_{interference_name}_sir{sir:.0f}dB.wav"
        save_audio(str(output_path), mixed, sr)
        print(f"Created: {output_path} (SIR={sir:.0f} dB)")


if __name__ == "__main__":
    main()
