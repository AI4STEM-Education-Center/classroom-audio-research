"""MeanFlow-TSE: Single-step flow matching target speaker extraction.

Paper: https://arxiv.org/abs/2512.18572
Code:  https://github.com/rikishimizu/MeanFlow-TSE

Pipeline:
    1. TPredicter(mixture_wav, enrollment_wav) → alpha scalar
    2. STFT both signals (n_fft=510, hop=128)
    3. Pad mixture to 3s chunks, run UDiT single-step Euler
    4. iSTFT → output waveform → WAV bytes

CPU-only inference. Single Euler step (NFE=1) ≈ 1-3s per 3s chunk on CPU.
"""

import asyncio
import io
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.tse.base import BaseTSE

logger = logging.getLogger(__name__)

# STFT params matching upstream config (CRITICAL: n_fft=510, NOT 512)
N_FFT = 510
HOP_LENGTH = 128
WIN_LENGTH = 510
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = SAMPLE_RATE * 3  # 3 seconds
# 3s of audio at 16kHz with hop=128 → this many STFT frames (+1 for n_fft)
SEGMENT_FRAMES = SAMPLE_RATE * 3 // HOP_LENGTH + 1  # 376 frames


def _strip_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    """Strip a prefix from all keys in a state_dict.

    Lightning checkpoints wrap model weights under 'model.' prefix.
    We load into bare model instances, so strip it.
    """
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _pad_and_reshape(tensor: torch.Tensor, multiple: int):
    """Pad tensor along last dim to a multiple, then chunk into batch.

    Args:
        tensor: (N, C, L) input
        multiple: target chunk length

    Returns:
        (reshaped, original_length): reshaped is (N*K, C, multiple)
    """
    n, c, l = tensor.shape
    padding_length = (multiple - (l % multiple)) % multiple
    padded = torch.nn.functional.pad(tensor, (0, padding_length))
    chunks = torch.cat(torch.chunk(padded, padded.shape[-1] // multiple, dim=-1), dim=0)
    return chunks, l


def _reshape_and_trim(tensor: torch.Tensor, original_length: int):
    """Reverse pad_and_reshape: reassemble chunks and trim to original length.

    Args:
        tensor: (N*K, C, multiple) chunked output
        original_length: original L before padding

    Returns:
        (N, C, original_length) tensor
    """
    n_k, c, multiple = tensor.shape
    n_chunks = original_length // multiple + (1 if original_length % multiple != 0 else 0)
    reassembled = torch.cat(torch.chunk(tensor, n_chunks, dim=0), dim=-1)
    return reassembled[:, :, :original_length]


def _sample_euler_single_step(model, mixture_spec, enrollment_spec, alpha):
    """Single-step Euler sampling from alpha to 1.0.

    This is the core inference: one forward pass of UDiT to predict velocity,
    then a single Euler step to reach the clean source estimate.

    Args:
        model: UDiT model
        mixture_spec: (B, 512, T) mixture spectrogram at position alpha
        enrollment_spec: (B, 512, T_enroll) enrollment spectrogram
        alpha: (B,) or scalar mixing ratio

    Returns:
        (B, 512, T) predicted clean source spectrogram
    """
    batch_size = mixture_spec.size(0)
    device = mixture_spec.device

    if not torch.is_tensor(alpha):
        alpha = torch.tensor([alpha], device=device)
    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)
    if alpha.shape[0] == 1 and batch_size > 1:
        alpha = alpha.repeat(batch_size)

    # Single step: predict velocity at t=alpha, step to t=1.0
    t_current = alpha
    t_target = torch.ones(batch_size, device=device)

    velocity = model(mixture_spec, t_current, t_target, enrollment_spec)

    dt = (t_target - t_current).view(batch_size, 1, 1)
    z = mixture_spec + dt * velocity

    return z


def _scale_audio(audio: torch.Tensor) -> torch.Tensor:
    """Scale audio to [-1, 1] range."""
    max_val = torch.max(torch.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio


class MeanFlowTSE(BaseTSE):
    """MeanFlow-TSE implementation with CPU-only single-step inference."""

    def __init__(self, model_dir: str = "./models") -> None:
        self._model_dir = Path(model_dir)
        self._udit = None
        self._t_predicter = None
        self._campplus_verifier = None
        self._loaded = False
        self._num_threads = int(os.environ.get("TORCH_NUM_THREADS", "4"))

    async def load(self) -> None:
        """Load UDiT and TPredicter checkpoints.

        Expected checkpoint locations:
            {model_dir}/meanflow-tse/best.ckpt         - UDiT model (~1.4GB)
            {model_dir}/meanflow-tse/t_predictor.ckpt   - TPredicter (~50MB)
        """

        def _load():
            torch.set_num_threads(self._num_threads)

            from src.tse.vendor.meanflow_tse import UDiT, TPredicter

            ckpt_dir = self._model_dir / "meanflow-tse"

            # Support both naming conventions from Google Drive
            udit_candidates = ["best-clean-weights.ckpt", "best.ckpt"]
            tp_candidates = ["t-predictor-clean-weights.ckpt", "t_predictor.ckpt"]

            udit_path = next((ckpt_dir / f for f in udit_candidates if (ckpt_dir / f).exists()), None)
            tp_path = next((ckpt_dir / f for f in tp_candidates if (ckpt_dir / f).exists()), None)

            if udit_path is None:
                raise FileNotFoundError(
                    f"UDiT checkpoint not found in {ckpt_dir}. "
                    "Run: python scripts/download_models.py"
                )
            if tp_path is None:
                raise FileNotFoundError(
                    f"TPredicter checkpoint not found in {ckpt_dir}. "
                    "Run: python scripts/download_models.py"
                )

            # Load UDiT (config from upstream config_MeanFlowTSE_clean.yaml)
            udit = UDiT(
                input_dim=512,
                output_dim=512,
                pos_method="none",
                pos_length=500,
                hidden_size=1024,
                depth=16,
                num_heads=16,
                use_checkpoint=False,  # no gradient checkpointing for inference
            )
            udit_ckpt = torch.load(udit_path, map_location="cpu", weights_only=False)
            udit_sd = udit_ckpt.get("state_dict", udit_ckpt)
            udit_sd = _strip_prefix(udit_sd, "model.")
            udit.load_state_dict(udit_sd, strict=True)
            udit.eval()
            logger.info(f"UDiT loaded from {udit_path}")

            # Load TPredicter (C=1024 from config)
            tp = TPredicter(C=1024)
            tp_ckpt = torch.load(tp_path, map_location="cpu", weights_only=False)
            tp_sd = tp_ckpt.get("state_dict", tp_ckpt)
            tp_sd = _strip_prefix(tp_sd, "model.")
            tp.load_state_dict(tp_sd, strict=True)
            tp.eval()
            logger.info(f"TPredicter loaded from {tp_path}")

            return udit, tp

        logger.info("Loading MeanFlow-TSE models...")
        self._udit, self._t_predicter = await asyncio.to_thread(_load)
        self._loaded = True
        logger.info("MeanFlow-TSE ready.")

        # Load CAM++ verifier for SECS post-filter (optional, graceful fallback)
        try:
            from src.tse.campplus_verifier import CAMPlusVerifier
            campplus_path = self._model_dir / "campplus_lm.onnx"
            if campplus_path.exists():
                self._campplus_verifier = CAMPlusVerifier(str(campplus_path))
                self._campplus_verifier.load()
                if self._campplus_verifier.is_loaded:
                    logger.info("CAM++ verifier loaded for SECS post-filter")
                else:
                    logger.warning("CAM++ verifier failed to load — using ECAPA-TDNN fallback")
            else:
                logger.info("CAM++ model not found at %s — using ECAPA-TDNN for SECS", campplus_path)
        except Exception as e:
            logger.warning("CAM++ verifier initialization failed: %s — using ECAPA-TDNN fallback", e)

    async def extract(
        self, mixed_audio: bytes, reference_audio: bytes | None = None
    ) -> bytes:
        """Extract target speaker from mixed audio.

        Args:
            mixed_audio: WAV bytes of the mixture.
            reference_audio: WAV bytes of the target speaker enrollment.
                If None, returns mixed_audio unchanged (can't extract without reference).

        Returns:
            WAV bytes of the extracted target speaker.
        """
        if not self._loaded:
            raise RuntimeError("MeanFlow-TSE not loaded. Call load() first.")

        if reference_audio is None:
            logger.warning("No reference audio provided, returning input unchanged.")
            return mixed_audio

        def _extract():
            torch.set_num_threads(self._num_threads)

            from src.tse.vendor.meanflow_tse import stft_torch, istft_torch

            # Decode WAV bytes using soundfile (avoids torchcodec dependency)
            mix_data, mix_sr = sf.read(io.BytesIO(mixed_audio), dtype="float32")
            ref_data, ref_sr = sf.read(io.BytesIO(reference_audio), dtype="float32")

            # Mono: take first channel if stereo
            if mix_data.ndim > 1:
                mix_data = mix_data[:, 0]
            if ref_data.ndim > 1:
                ref_data = ref_data[:, 0]

            # Convert to torch tensors: (1, time)
            mix_wav = torch.from_numpy(mix_data).unsqueeze(0)
            ref_wav = torch.from_numpy(ref_data).unsqueeze(0)

            # Resample to 16kHz if needed
            if mix_sr != SAMPLE_RATE:
                import torchaudio
                mix_wav = torchaudio.functional.resample(mix_wav, mix_sr, SAMPLE_RATE)
            if ref_sr != SAMPLE_RATE:
                import torchaudio
                ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, SAMPLE_RATE)

            original_mix_length = mix_wav.shape[-1]

            # Pad enrollment to 3s (TPredicter and UDiT expect 3s enrollment)
            if ref_wav.shape[-1] < SEGMENT_SAMPLES:
                ref_wav = torch.nn.functional.pad(
                    ref_wav, (0, SEGMENT_SAMPLES - ref_wav.shape[-1])
                )
            elif ref_wav.shape[-1] > SEGMENT_SAMPLES:
                ref_wav = ref_wav[:, :SEGMENT_SAMPLES]

            # Pad mixture to 3s minimum for TPredicter
            mix_wav_for_tp = mix_wav
            if mix_wav_for_tp.shape[-1] < SEGMENT_SAMPLES:
                mix_wav_for_tp = torch.nn.functional.pad(
                    mix_wav_for_tp, (0, SEGMENT_SAMPLES - mix_wav_for_tp.shape[-1])
                )

            with torch.inference_mode():
                # Step 1: Predict mixing ratio alpha
                alpha = self._t_predicter(mix_wav_for_tp, ref_wav, aug=False)

                # Step 2: Compute STFT of mixture and enrollment
                mixture_spec = stft_torch(
                    mix_wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
                )
                enroll_spec = stft_torch(
                    ref_wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
                )

                # Step 3: Pad mixture spec to 3s chunks and batch
                mixture_chunks, orig_spec_len = _pad_and_reshape(
                    mixture_spec, SEGMENT_FRAMES
                )
                batch_size = mixture_chunks.shape[0]

                # Repeat enrollment for each chunk
                enroll_batch = enroll_spec.repeat(batch_size, 1, 1)

                # Step 4: Single-step Euler inference
                output_chunks = _sample_euler_single_step(
                    self._udit,
                    mixture_chunks.float(),
                    enroll_batch.float(),
                    alpha,
                )

                # Step 5: Reassemble and iSTFT
                output_spec = _reshape_and_trim(output_chunks, orig_spec_len)
                output_wav = istft_torch(
                    output_spec,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH,
                    length=original_mix_length,
                )
                output_wav = _scale_audio(output_wav)

            # SECS: Speaker Encoder Cosine Similarity (post-filter verification)
            # Use CAM++ (if available) for stronger verification, fall back to ECAPA-TDNN
            speaker_similarity = None
            try:
                if self._campplus_verifier and self._campplus_verifier.is_loaded:
                    # CAM++ SECS (512-dim, better accuracy, 2.5x faster)
                    speaker_similarity = self._campplus_verifier.verify(
                        output_wav.squeeze(0).cpu().numpy(),
                        ref_data,
                    )
                    if speaker_similarity is not None:
                        logger.info(f"SECS (CAM++): {speaker_similarity:.3f}")
                else:
                    # Fallback: ECAPA-TDNN SECS (192-dim, from MeanFlow's TPredicter)
                    ecapa = self._t_predicter.ecapa_tdnn
                    with torch.inference_mode():
                        enroll_emb = ecapa(ref_wav, aug=False)
                        if output_wav.shape[-1] >= SAMPLE_RATE // 2:
                            extract_emb = ecapa(output_wav, aug=False)
                            cos_sim = torch.nn.functional.cosine_similarity(
                                enroll_emb, extract_emb, dim=-1
                            )
                            speaker_similarity = float(cos_sim.item())
                            logger.info(f"SECS (ECAPA-TDNN fallback): {speaker_similarity:.3f}")
            except Exception as e:
                logger.warning(f"SECS computation failed: {e}")

            # Encode as WAV bytes using soundfile
            buf = io.BytesIO()
            sf.write(buf, output_wav.squeeze(0).cpu().numpy(), SAMPLE_RATE, format="WAV")
            return buf.getvalue(), speaker_similarity

        return await asyncio.to_thread(_extract)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
