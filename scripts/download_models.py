#!/usr/bin/env python3
"""Download model checkpoints for TSE service.

Usage:
    python scripts/download_models.py

Downloads to the MODEL_CACHE_DIR directory (default: ./models).

MeanFlow-TSE checkpoints are on Google Drive:
    https://drive.google.com/drive/folders/1pB5IMjjef3irWl9730F-ez_ztA0srmSW
"""

import os
import subprocess
import sys
from pathlib import Path


def get_cache_dir() -> Path:
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "./models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _ensure_gdown():
    """Ensure gdown is available for Google Drive downloads."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download_ecapa_tdnn(cache_dir: Path) -> None:
    """Download ECAPA-TDNN speaker embedding model from SpeechBrain."""
    ecapa_dir = cache_dir / "ecapa-tdnn"
    if ecapa_dir.exists() and any(ecapa_dir.iterdir()):
        print(f"ECAPA-TDNN already downloaded at {ecapa_dir}")
        return

    print("Downloading ECAPA-TDNN from SpeechBrain hub...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ecapa_dir),
        )
        print(f"ECAPA-TDNN saved to {ecapa_dir}")
    except ImportError:
        print(
            "ERROR: speechbrain not installed. "
            "Run: pip install -e '.[ml]'"
        )
    except Exception as e:
        print(f"ERROR downloading ECAPA-TDNN: {e}")


def download_meanflow_tse(cache_dir: Path) -> None:
    """Download MeanFlow-TSE checkpoints from Google Drive.

    Downloads two files:
        best.ckpt          - UDiT model (~1.4GB)
        t_predictor.ckpt   - TPredicter (~50MB)

    Source: https://drive.google.com/drive/folders/1pB5IMjjef3irWl9730F-ez_ztA0srmSW
    """
    meanflow_dir = cache_dir / "meanflow-tse"
    meanflow_dir.mkdir(parents=True, exist_ok=True)

    # Google Drive file IDs from the shared folder
    # These are the pretrained Libri2Mix clean checkpoints
    files = {
        "best.ckpt": {
            "description": "UDiT model (~1.4GB)",
            "folder_url": "https://drive.google.com/drive/folders/1pB5IMjjef3irWl9730F-ez_ztA0srmSW",
        },
        "t_predictor.ckpt": {
            "description": "TPredicter (~50MB)",
            "folder_url": "https://drive.google.com/drive/folders/1pB5IMjjef3irWl9730F-ez_ztA0srmSW",
        },
    }

    all_exist = all((meanflow_dir / name).exists() for name in files)
    if all_exist:
        print(f"MeanFlow-TSE already downloaded at {meanflow_dir}")
        return

    _ensure_gdown()
    import gdown

    # Download the entire folder
    print("Downloading MeanFlow-TSE checkpoints from Google Drive...")
    print(f"  Folder: {files['best.ckpt']['folder_url']}")
    print(f"  Destination: {meanflow_dir}")
    print()

    try:
        gdown.download_folder(
            url=files["best.ckpt"]["folder_url"],
            output=str(meanflow_dir),
            quiet=False,
        )

        # Verify downloads
        for name, info in files.items():
            path = meanflow_dir / name
            if not path.exists():
                # gdown may create subdirectories; search for the file
                found = list(meanflow_dir.rglob(name))
                if found:
                    # Move to expected location
                    found[0].rename(path)
                    print(f"  Moved {name} to {path}")
                else:
                    print(f"  WARNING: {name} not found after download ({info['description']})")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {name}: {size_mb:.1f} MB")

        print("MeanFlow-TSE download complete.")

    except Exception as e:
        print(f"ERROR downloading MeanFlow-TSE: {e}")
        print()
        print("If automatic download fails, download manually:")
        print(f"  1. Visit: {files['best.ckpt']['folder_url']}")
        print(f"  2. Download best.ckpt and t_predictor.ckpt")
        print(f"  3. Place them in: {meanflow_dir}/")


def download_campplus(cache_dir: Path) -> None:
    """Download CAM++ ONNX speaker verification model from HuggingFace.

    WeSpeaker CAM++ (28MB, 512-dim embeddings).
    Source: https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM
    """
    onnx_path = cache_dir / "campplus_lm.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"CAM++ already downloaded at {onnx_path} ({size_mb:.1f} MB)")
        return

    print("Downloading CAM++ ONNX from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="Wespeaker/wespeaker-voxceleb-campplus-LM",
            filename="campplus_lm.onnx",
            local_dir=str(cache_dir),
        )
        print(f"CAM++ saved to {downloaded}")
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
    except Exception as e:
        print(f"ERROR downloading CAM++: {e}")
        print("Manual download: https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM")


def main() -> None:
    cache_dir = get_cache_dir()
    print(f"Model cache directory: {cache_dir}")
    print()

    download_ecapa_tdnn(cache_dir)
    print()
    download_campplus(cache_dir)
    print()
    download_meanflow_tse(cache_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
