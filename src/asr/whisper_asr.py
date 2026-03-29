"""Whisper ASR using HuggingFace transformers.

Provides a working ASR backend so the team can test their TSE output
end-to-end without needing an external API. Uses OpenAI's Whisper model
via the transformers library.

Requires the [ml] optional dependencies:
    pip install -e ".[ml]"
"""

import io
import logging
from typing import Any

import numpy as np
import soundfile as sf

from src.asr.base import BaseASR, TranscriptionResult

logger = logging.getLogger(__name__)

# Maximum reasonable transcript length (characters). Real 5s speech ≈ 50-80 chars.
MAX_TRANSCRIPT_LENGTH = 300


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination patterns.

    Whisper hallucinates repetitive text when fed near-silent or garbage audio.
    Common patterns:
    - "I'm sorry, I'm sorry, I'm sorry, ..." (repeated phrases)
    - "D-D-D-D-D-D-D-D-D-..." (repeated tokens)
    - Very long output from short audio (real 5s speech is ~50-80 chars)
    """
    if not text:
        return False

    # 1. Excessive length — real 5s utterance is never 300+ chars
    if len(text) > MAX_TRANSCRIPT_LENGTH:
        return True

    # 2. Repeated short phrases: split into words, check if any phrase
    #    of 1-5 words repeats 4+ times consecutively
    words = text.split()
    if len(words) < 4:
        return False

    for phrase_len in range(1, min(6, len(words) // 3 + 1)):
        repeats = 1
        for i in range(phrase_len, len(words) - phrase_len + 1, phrase_len):
            chunk = words[i : i + phrase_len]
            prev = words[i - phrase_len : i]
            if chunk == prev:
                repeats += 1
                if repeats >= 4:
                    return True
            else:
                repeats = 1

    # 3. Single character/token stuttering: "D-D-D-D" or "d, d, d, d"
    import re
    stutter_match = re.search(r'((?:\w[-,]){4,})', text)
    if stutter_match:
        return True

    return False


class WhisperASR(BaseASR):
    """Whisper ASR via HuggingFace transformers pipeline.

    Loads the model lazily on first call to load(). Accepts WAV audio bytes.
    """

    def __init__(self, model_size: str = "base.en", cache_dir: str = "./models") -> None:
        self._model_size = model_size
        self._cache_dir = cache_dir
        self._pipeline: Any = None
        self._loaded = False

    async def load(self) -> None:
        try:
            import torch
            from transformers import pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = f"openai/whisper-{self._model_size}"

            # Try to resolve a local snapshot path first (for offline/EFS deployments
            # where HuggingFace cache symlinks may not survive S3 copy)
            local_path = self._resolve_local_model(model_id)
            model_source = local_path or model_id

            logger.info(f"Loading Whisper model {model_source} on {device}")

            kwargs = {"device": device}
            if not local_path:
                kwargs["model_kwargs"] = {"cache_dir": self._cache_dir}

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_source,
                **kwargs,
            )
            self._loaded = True
            logger.info("Whisper model loaded successfully")

        except ImportError:
            raise RuntimeError(
                "Whisper ASR requires ML dependencies. "
                "Install with: pip install -e '.[ml]'"
            )

    def _resolve_local_model(self, model_id: str) -> str | None:
        """Resolve a HuggingFace model to a local path if available.

        Checks two locations:
        1. Flat directory: {cache_dir}/{model_name}/ (e.g. whisper-base.en/)
        2. HF cache structure: {cache_dir}/models--{org}--{model}/snapshots/{rev}/
        """
        import os

        # 1. Check flat directory (EFS-friendly, no symlinks)
        model_name = model_id.split("/")[-1]  # "openai/whisper-base.en" -> "whisper-base.en"
        flat_path = os.path.join(self._cache_dir, model_name)
        if os.path.exists(os.path.join(flat_path, "config.json")):
            logger.info(f"Found local Whisper model at {flat_path}")
            return flat_path

        # 2. Check HF cache structure
        hf_cache_name = f"models--{model_id.replace('/', '--')}"
        cache_path = os.path.join(self._cache_dir, hf_cache_name)
        refs_path = os.path.join(cache_path, "refs", "main")

        if os.path.exists(refs_path):
            with open(refs_path) as f:
                revision = f.read().strip()
            snapshot_path = os.path.join(cache_path, "snapshots", revision)
            if os.path.exists(os.path.join(snapshot_path, "config.json")):
                logger.info(f"Found local Whisper snapshot at {snapshot_path}")
                return snapshot_path

        return None

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        if not self._loaded or self._pipeline is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        # Decode audio bytes to numpy array
        audio_array, sample_rate = sf.read(io.BytesIO(audio))

        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        duration = len(audio_array) / sample_rate

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple linear interpolation resample
            num_samples = int(len(audio_array) * 16000 / sample_rate)
            indices = np.linspace(0, len(audio_array) - 1, num_samples)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

        # Run inference
        result = self._pipeline(
            audio_array.astype(np.float32),
            return_timestamps=False,
        )

        text = result.get("text", "").strip()

        # Detect and reject Whisper hallucinations (repetitive decoder loops)
        if text and _is_hallucination(text):
            logger.warning("Whisper hallucination detected, discarding: %.80s...", text)
            return TranscriptionResult(text="", confidence=0.0, duration=duration)

        return TranscriptionResult(
            text=text,
            confidence=0.9,  # Whisper pipeline doesn't expose per-utterance confidence
            duration=duration,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded
