import shutil
import subprocess
import tempfile

import pytest

has_ffmpeg = shutil.which("ffmpeg") is not None


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["tse_model"] == "mock"
    assert data["asr_model"] == "mock"
    assert data["tse_loaded"] is True
    assert data["asr_loaded"] is True


@pytest.mark.asyncio
async def test_extract_and_transcribe(client, sample_audio):
    resp = await client.post(
        "/extract-and-transcribe",
        files={"audio": ("test.wav", sample_audio, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "confidence" in data
    assert "duration" in data
    # Mock ASR returns "[mock transcript]"
    assert data["text"] == "[mock transcript]"
    assert data["confidence"] == 1.0


@pytest.mark.asyncio
async def test_extract_with_reference(client, sample_audio, sample_reference):
    resp = await client.post(
        "/extract-and-transcribe",
        files={
            "audio": ("test.wav", sample_audio, "audio/wav"),
            "referenceAudio": ("ref.wav", sample_reference, "audio/wav"),
        },
        data={"speakerName": "Test Student"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "[mock transcript]"


@pytest.mark.asyncio
async def test_missing_audio_returns_422(client):
    resp = await client.post("/extract-and-transcribe")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_empty_audio_returns_422(client):
    resp = await client.post(
        "/extract-and-transcribe",
        files={"audio": ("test.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_models_not_loaded_returns_503(client):
    """If models aren't loaded, should return 503."""
    from src.tse.mock import MockTSE

    # Create an unloaded TSE
    unloaded = MockTSE()
    client._transport.app.state.tse = unloaded  # type: ignore[attr-defined]

    resp = await client.post(
        "/extract-and-transcribe",
        files={"audio": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Audio conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg not installed")
class TestAudioConversion:
    def test_convert_wav_passthrough(self, sample_audio):
        """WAV input should pass through soundfile path (no ffmpeg needed)."""
        from src.api.routes import _convert_to_wav

        result = _convert_to_wav(sample_audio)
        assert result[:4] == b"RIFF"
        assert len(result) > 44

    def test_convert_webm_via_ffmpeg(self):
        """WebM/Opus input should be converted via ffmpeg fallback."""
        from src.api.routes import _convert_to_wav

        # Generate WebM/Opus using ffmpeg
        webm_bytes = _make_webm_bytes(1.0)
        result = _convert_to_wav(webm_bytes)
        assert result[:4] == b"RIFF"
        assert len(result) > 44

    def test_convert_stereo_to_mono(self):
        """Stereo WAV should be converted to mono."""
        import io
        import struct
        import wave

        from src.api.routes import _convert_to_wav

        sr = 16000
        n = sr  # 1 second
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{n * 2}h", *([0] * n * 2)))
        stereo_wav = buf.getvalue()

        result = _convert_to_wav(stereo_wav)
        assert result[:4] == b"RIFF"

    def test_convert_invalid_audio_raises(self):
        """Invalid audio bytes should raise ValueError."""
        from src.api.routes import _convert_to_wav

        with pytest.raises(ValueError, match="ffmpeg conversion failed"):
            _convert_to_wav(b"not audio data at all")

    @pytest.mark.asyncio
    async def test_webm_through_endpoint(self, client):
        """WebM audio should work through the full API endpoint."""
        webm_bytes = _make_webm_bytes(1.0)
        resp = await client.post(
            "/extract-and-transcribe",
            files={"audio": ("recording.webm", webm_bytes, "audio/webm")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "[mock transcript]"


def _make_webm_bytes(duration_s: float = 1.0) -> bytes:
    """Generate WebM/Opus audio bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}:sample_rate=16000",
            "-c:a", "libopus", tmp.name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return tmp.read()
