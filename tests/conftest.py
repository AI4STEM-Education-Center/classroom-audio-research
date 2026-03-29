import io
import math
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from src.asr.mock import MockASR
from src.main import app
from src.tse.mock import MockTSE


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate WAV bytes containing a 440Hz sine wave (above energy gate threshold)."""
    num_samples = int(duration_s * sample_rate)
    amplitude = 16000  # ~0.49 of int16 max — well above RMS 0.005 energy gate
    samples = [int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
    return buf.getvalue()


@pytest.fixture
def sample_audio() -> bytes:
    """1 second of silence as 16kHz mono WAV."""
    return _make_wav_bytes(1.0)


@pytest.fixture
def sample_reference() -> bytes:
    """1 second of silence as reference audio."""
    return _make_wav_bytes(1.0)


@pytest.fixture
async def client():
    """FastAPI test client with mock models pre-loaded."""
    tse = MockTSE()
    await tse.load()
    asr = MockASR()
    await asr.load()

    app.state.tse = tse
    app.state.asr = asr
    app.state.tse_model_name = "mock"
    app.state.asr_model_name = "mock"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
