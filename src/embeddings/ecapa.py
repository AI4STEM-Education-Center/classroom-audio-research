"""ECAPA-TDNN speaker embedding model via SpeechBrain.

This is the STANDALONE embedding model for speaker verification / enrollment.
It uses SpeechBrain's pretrained model from HuggingFace (~35MB).
NOT the same as the vendored ECAPA-TDNN inside MeanFlow-TSE's TPredicter.

Outputs 192-dimensional L2-normalized embeddings.
See: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
"""

import asyncio
import io
import logging

import numpy as np

from src.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class ECAPAEmbedding(BaseEmbedding):
    """ECAPA-TDNN speaker embedding via SpeechBrain."""

    def __init__(self, cache_dir: str = "./models") -> None:
        self._cache_dir = cache_dir
        self._model = None
        self._loaded = False

    async def load(self) -> None:
        """Load ECAPA-TDNN from SpeechBrain hub."""

        def _load():
            import torch

            # Workaround: SpeechBrain 1.0.3 calls torchaudio.list_audio_backends()
            # at import time, but torchaudio >= 2.9 removed that function.
            # Monkey-patch it before importing SpeechBrain.
            # Remove when SpeechBrain ships a fix (tracked: github.com/speechbrain/speechbrain/issues/3012)
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: []

            from speechbrain.inference.speaker import EncoderClassifier

            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=f"{self._cache_dir}/ecapa-tdnn",
                run_opts={"device": "cpu"},
            )
            return model

        logger.info("Loading ECAPA-TDNN from SpeechBrain hub...")
        self._model = await asyncio.to_thread(_load)
        self._loaded = True
        logger.info("ECAPA-TDNN loaded.")

    async def embed(self, audio: bytes) -> np.ndarray:
        """Extract 192-dim L2-normalized speaker embedding from audio.

        Args:
            audio: Raw WAV bytes (16kHz mono expected).

        Returns:
            (192,) numpy array, L2-normalized.
        """
        if not self._loaded:
            raise RuntimeError("ECAPA-TDNN not loaded. Call load() first.")

        def _embed():
            import soundfile as sf
            import torch

            # Decode WAV bytes using soundfile (avoids torchcodec dependency)
            data, sr = sf.read(io.BytesIO(audio), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, time)

            # Resample to 16kHz if needed
            if sr != 16000:
                import torchaudio
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            # SpeechBrain expects (batch, time)
            if waveform.dim() == 2:
                waveform = waveform[0:1, :]  # keep first channel as batch dim

            # Extract embedding
            with torch.no_grad():
                embedding = self._model.encode_batch(waveform)

            # Shape: (1, 1, 192) -> (192,)
            emb = embedding.squeeze().cpu().numpy()

            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        return await asyncio.to_thread(_embed)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
