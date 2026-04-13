import pytest

from src.asr.mock import MockASR


@pytest.mark.asyncio
async def test_mock_asr_returns_expected_result(sample_audio):
    """Mock ASR should return fixed transcript."""
    asr = MockASR()
    await asr.load()
    assert asr.is_loaded

    result = await asr.transcribe(sample_audio)
    assert result.text == "[mock transcript]"
    assert result.confidence == 1.0
    assert result.duration == 5.0


@pytest.mark.asyncio
async def test_mock_asr_not_loaded():
    """Mock ASR should report not loaded before load() is called."""
    asr = MockASR()
    assert not asr.is_loaded


# --------------------------------------------------------------------------
# Template for Whisper tests (uncomment after installing ML dependencies)
# --------------------------------------------------------------------------
#
# @pytest.mark.skipif(
#     not importlib.util.find_spec("transformers"),
#     reason="transformers not installed (need pip install -e '.[ml]')"
# )
# @pytest.mark.asyncio
# async def test_whisper_asr_loads():
#     from src.asr.whisper_asr import WhisperASR
#     asr = WhisperASR(model_size="tiny.en")
#     await asr.load()
#     assert asr.is_loaded
#
# @pytest.mark.skipif(
#     not importlib.util.find_spec("transformers"),
#     reason="transformers not installed"
# )
# @pytest.mark.asyncio
# async def test_whisper_asr_transcribes(sample_audio):
#     from src.asr.whisper_asr import WhisperASR
#     asr = WhisperASR(model_size="tiny.en")
#     await asr.load()
#     result = await asr.transcribe(sample_audio)
#     assert isinstance(result.text, str)
#     assert result.duration > 0
