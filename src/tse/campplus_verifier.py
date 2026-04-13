"""CAM++ Speaker Verifier — SECS post-filter for TSE output.

Uses the WeSpeaker CAM++ ONNX model (28MB, 512-dim embeddings) to verify
that TSE-extracted audio matches the enrollment voiceprint.

This runs alongside MeanFlow-TSE: MeanFlow uses its internal ECAPA-TDNN
for extraction conditioning, CAM++ provides a stronger independent
verification signal via cosine similarity (SECS score).

Paper: https://arxiv.org/abs/2303.00332 (Interspeech 2023)
Model: https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM
"""

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import io

logger = logging.getLogger(__name__)

# Default location
DEFAULT_MODEL_PATH = "models/campplus_lm.onnx"

# Fbank params matching WeSpeaker's preprocessing
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
N_MELS = 80


def _compute_fbank(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute 80-dim log mel filterbank features (matching WeSpeaker preprocessing)."""
    import scipy.signal
    import scipy.fft

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    frame_length = WIN_LENGTH
    frame_step = HOP_LENGTH
    signal_length = len(audio)
    num_frames = 1 + (signal_length - frame_length) // frame_step

    indices = np.arange(frame_length)[None, :] + np.arange(num_frames)[:, None] * frame_step
    frames = audio[indices]

    # Apply hamming window
    frames *= np.hamming(frame_length)

    # FFT
    mag_frames = np.absolute(np.fft.rfft(frames, N_FFT))
    pow_frames = (1.0 / N_FFT) * (mag_frames ** 2)

    # Mel filterbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, N_MELS + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((N_FFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = np.log(filter_banks)

    # CMN (cepstral mean normalization)
    filter_banks -= np.mean(filter_banks, axis=0, keepdims=True)

    return filter_banks.astype(np.float32)


class CAMPlusVerifier:
    """Lightweight CAM++ speaker verifier using ONNX Runtime."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or os.environ.get(
            "CAMPPLUS_MODEL_PATH", DEFAULT_MODEL_PATH
        )
        self._session = None

    def load(self) -> None:
        """Load the ONNX model."""
        import onnxruntime as ort

        path = Path(self._model_path)
        if not path.exists():
            logger.warning("CAM++ model not found at %s — SECS verification disabled", path)
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(path), providers=providers)
        logger.info("CAM++ verifier loaded from %s (providers: %s)", path, self._session.get_providers())

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def embed(self, audio: np.ndarray | bytes) -> np.ndarray | None:
        """Compute 512-dim speaker embedding from audio.

        Args:
            audio: numpy array (float32, 16kHz mono) or WAV bytes

        Returns:
            512-dim embedding vector, or None if model not loaded
        """
        if not self._session:
            return None

        # Convert bytes to numpy if needed
        if isinstance(audio, bytes):
            buf = io.BytesIO(audio)
            audio_np, sr = sf.read(buf, dtype="float32")
            if sr != SAMPLE_RATE:
                # Simple resample via scipy
                import scipy.signal
                audio_np = scipy.signal.resample(
                    audio_np, int(len(audio_np) * SAMPLE_RATE / sr)
                )
        else:
            audio_np = audio

        # Ensure mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Compute fbank features
        fbank = _compute_fbank(audio_np)  # (T, 80)

        # Add batch dimension: (1, T, 80)
        fbank_input = fbank[np.newaxis, :, :]

        # Run inference
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        embedding = self._session.run([output_name], {input_name: fbank_input})[0]

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)

        return embedding.squeeze(0)  # (512,)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))

    def verify(self, audio: np.ndarray | bytes, reference: np.ndarray | bytes) -> float | None:
        """Compute SECS score between extracted audio and enrollment reference.

        Returns cosine similarity (-1 to 1), or None if model not loaded.
        """
        emb_audio = self.embed(audio)
        emb_ref = self.embed(reference)

        if emb_audio is None or emb_ref is None:
            return None

        return self.similarity(emb_audio, emb_ref)
