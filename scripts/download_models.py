#!/usr/bin/env python3
"""Download pretrained models for evaluation.

Downloads model checkpoints to the models/ directory.
Run this once before starting evaluation tracks.

Usage:
    python scripts/download_models.py              # Download all
    python scripts/download_models.py --whisper     # Whisper models only
    python scripts/download_models.py --ecapa       # ECAPA-TDNN only
"""

import argparse
from pathlib import Path

MODELS_DIR = Path("models")


def download_whisper():
    """Download Whisper models via HuggingFace."""
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        print("Install ML deps first: pip install -e '.[ml]'")
        return

    models = ["openai/whisper-base.en", "openai/whisper-small.en"]
    for model_id in models:
        print(f"Downloading {model_id}...")
        cache_dir = MODELS_DIR / "whisper"
        WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=str(cache_dir))
        WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        print(f"  Done: {model_id}")


def download_ecapa():
    """Download ECAPA-TDNN via SpeechBrain."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("Install ML deps first: pip install -e '.[ml]'")
        return

    print("Downloading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
    cache_dir = MODELS_DIR / "ecapa"
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache_dir),
        run_opts={"device": "cpu"},
    )
    print("  Done: ECAPA-TDNN")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper models only")
    parser.add_argument("--ecapa", action="store_true", help="Download ECAPA-TDNN only")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    # If no specific flag, download all
    download_all = not args.whisper and not args.ecapa

    if args.whisper or download_all:
        download_whisper()
    if args.ecapa or download_all:
        download_ecapa()

    print("\nModels downloaded to models/ directory.")
    print("Note: SE-DiCoW and MeanFlow-TSE require separate setup — see their READMEs.")


if __name__ == "__main__":
    main()
