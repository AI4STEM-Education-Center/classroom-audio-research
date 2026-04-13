import io
import logging
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.schemas import ErrorResponse, HealthResponse, TranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _convert_to_wav(audio_bytes: bytes) -> bytes:
    """Convert audio bytes to 16kHz mono WAV.

    Accepts WAV or any format soundfile can read. For WebM/Opus from the
    browser, falls back to ffmpeg subprocess conversion.
    """
    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        # soundfile can't read WebM/Opus — use ffmpeg directly
        return _ffmpeg_to_wav(audio_bytes)

    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        num_samples = int(len(data) * 16000 / sample_rate)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data)
        sample_rate = 16000

    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return buf.getvalue()


def _ffmpeg_to_wav(audio_bytes: bytes) -> bytes:
    """Convert arbitrary audio to 16kHz mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_out:
            cmd = [
                "ffmpeg", "-y", "-i", tmp_in.name,
                "-ar", "16000", "-ac", "1", "-f", "wav",
                tmp_out.name,
            ]
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace")[:500]
                raise ValueError(f"ffmpeg conversion failed: {stderr}")

            return tmp_out.read()


def _compute_rms(wav_bytes: bytes) -> float:
    """Compute RMS energy of WAV audio. Returns 0.0 on decode failure."""
    try:
        data, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return float(np.sqrt(np.mean(data ** 2)))
    except Exception:
        return 0.0


@router.post(
    "/extract-and-transcribe",
    response_model=TranscriptionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def extract_and_transcribe(
    request: Request,
    audio: UploadFile = File(..., description="Mixed audio to process"),
    referenceAudio: UploadFile | None = File(None, description="Reference clip of target speaker"),
    speakerName: str | None = Form(None, description="Name of the target speaker"),
):
    """Extract target speaker from mixed audio and transcribe.

    This is the main endpoint. It:
    1. Receives mixed audio (and optional reference clip)
    2. Runs TSE to isolate the target speaker
    3. Transcribes the cleaned audio via ASR
    4. Returns the transcript

    The Next.js app calls this endpoint from /api/speech/transcribe-tse.
    """
    tse = request.app.state.tse
    asr = request.app.state.asr

    if not tse.is_loaded or not asr.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded yet. Try again shortly."},
        )

    # Read audio bytes
    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse(status_code=422, content={"error": "Empty audio file"})

    # Convert to WAV for model consumption
    try:
        wav_bytes = _convert_to_wav(audio_bytes)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    # Read optional reference audio
    reference_bytes = None
    if referenceAudio is not None:
        ref_raw = await referenceAudio.read()
        if ref_raw:
            try:
                reference_bytes = _convert_to_wav(ref_raw)
            except ValueError:
                logger.warning("Could not decode reference audio, proceeding without it")

    logger.info(
        "Processing audio: %d bytes, reference: %s, speaker: %s",
        len(wav_bytes),
        "yes" if reference_bytes else "no",
        speakerName or "unknown",
    )

    # Step 1: Target speaker extraction (+ SECS post-filter)
    extract_result = await tse.extract(wav_bytes, reference_bytes)

    # Unpack: new models return (audio_bytes, speaker_similarity), legacy returns just bytes
    if isinstance(extract_result, tuple):
        extracted_audio, speaker_similarity = extract_result
    else:
        extracted_audio, speaker_similarity = extract_result, None

    if speaker_similarity is not None:
        logger.info("SECS speaker similarity: %.3f", speaker_similarity)

    # Step 1.5: Energy gate — if TSE output is near-silent, skip ASR
    rms = _compute_rms(extracted_audio)
    logger.info("Post-extraction RMS: %.6f", rms)
    if rms < 0.005:
        logger.info("Extracted audio is near-silent (RMS=%.6f), skipping ASR", rms)
        return TranscriptionResponse(text="", confidence=0.0, duration=0.0, speakerSimilarity=speaker_similarity)

    # Step 2: Transcribe
    result = await asr.transcribe(extracted_audio)

    logger.info("Transcription: %.60s (confidence=%.2f, secs=%.3f)", result.text, result.confidence, speaker_similarity or 0.0)

    return TranscriptionResponse(
        text=result.text,
        confidence=result.confidence,
        duration=result.duration,
        speakerSimilarity=speaker_similarity,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check endpoint. Returns model loading status."""
    return HealthResponse(
        status="ok",
        tse_model=request.app.state.tse_model_name,
        asr_model=request.app.state.asr_model_name,
        tse_loaded=request.app.state.tse.is_loaded,
        asr_loaded=request.app.state.asr.is_loaded,
    )
