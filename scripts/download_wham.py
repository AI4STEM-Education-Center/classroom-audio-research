#!/usr/bin/env python3
"""Download WHAM! / WHAMR! noise dataset for evaluation.

WHAM! provides real-world ambient noise from urban environments.
WHAMR! adds room impulse responses (reverberation).

Paper: https://wham.whisper.ai/
Download: https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/

Usage:
    python scripts/download_wham.py               # Download WHAM! noise only (~4GB)
    python scripts/download_wham.py --reverb       # Download WHAMR! (~7GB)
    python scripts/download_wham.py --mini         # Download mini version (~500MB)

Files are saved to data/wham/ (gitignored).
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

WHAM_NOISE_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress indication."""
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {desc or url}...")
    print(f"  -> {dest}")

    cmd = ["curl", "-L", "-o", str(dest), "--progress-bar", url]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Download failed. Try manually: {url}")
        sys.exit(1)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.0f} MB")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}")


def print_dataset_stats(wham_dir: Path) -> None:
    """Print summary stats about the downloaded dataset."""
    wav_files = list(wham_dir.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found.")
        return

    total_size = sum(f.stat().st_size for f in wav_files)
    print(f"\nDataset stats:")
    print(f"  WAV files:  {len(wav_files)}")
    print(f"  Total size: {total_size / (1024**3):.1f} GB")

    for split in ["tr", "cv", "tt"]:
        split_files = [f for f in wav_files if f"/{split}/" in str(f) or f"\\{split}\\" in str(f)]
        if split_files:
            split_name = {"tr": "train", "cv": "validation", "tt": "test"}[split]
            print(f"  {split_name}: {len(split_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Download WHAM! noise dataset")
    parser.add_argument("--reverb", action="store_true", help="Download WHAMR! (with reverberation)")
    parser.add_argument("--mini", action="store_true", help="Download mini version (~500MB)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Download directory")

    args = parser.parse_args()
    wham_dir = args.data_dir / "wham"
    wham_dir.mkdir(parents=True, exist_ok=True)

    if list(wham_dir.rglob("*.wav")):
        print("WHAM! data already present.")
        print_dataset_stats(wham_dir)
        return

    zip_path = wham_dir / "wham_noise.zip"
    download_file(WHAM_NOISE_URL, zip_path, "WHAM! noise dataset (~4GB)")
    extract_zip(zip_path, wham_dir)

    if zip_path.exists():
        print(f"Removing archive: {zip_path.name}")
        zip_path.unlink()

    print_dataset_stats(wham_dir)
    print("\nDone. Use with: python scripts/mix_audio.py --target speech.wav --wham --snr 5")


if __name__ == "__main__":
    main()
