import pytest

from src.tse.mock import MockTSE


@pytest.mark.asyncio
async def test_mock_tse_passthrough(sample_audio):
    """Mock TSE should return input audio unchanged."""
    tse = MockTSE()
    await tse.load()
    assert tse.is_loaded

    result = await tse.extract(sample_audio)
    assert result == sample_audio


@pytest.mark.asyncio
async def test_mock_tse_with_reference(sample_audio, sample_reference):
    """Mock TSE should accept and ignore reference audio."""
    tse = MockTSE()
    await tse.load()

    result = await tse.extract(sample_audio, reference_audio=sample_reference)
    assert result == sample_audio


@pytest.mark.asyncio
async def test_mock_tse_not_loaded():
    """Mock TSE should report not loaded before load() is called."""
    tse = MockTSE()
    assert not tse.is_loaded


# --------------------------------------------------------------------------
# Template for real model tests (uncomment when MeanFlow-TSE is implemented)
# --------------------------------------------------------------------------
#
# @pytest.mark.skipif(
#     not Path("./models/meanflow-tse").exists(),
#     reason="MeanFlow-TSE checkpoint not downloaded"
# )
# @pytest.mark.asyncio
# async def test_meanflow_tse_loads():
#     from src.tse.meanflow import MeanFlowTSE
#     tse = MeanFlowTSE(model_dir="./models")
#     await tse.load()
#     assert tse.is_loaded
#
# @pytest.mark.skipif(
#     not Path("./models/meanflow-tse").exists(),
#     reason="MeanFlow-TSE checkpoint not downloaded"
# )
# @pytest.mark.asyncio
# async def test_meanflow_tse_extract(sample_audio, sample_reference):
#     from src.tse.meanflow import MeanFlowTSE
#     tse = MeanFlowTSE(model_dir="./models")
#     await tse.load()
#     result = await tse.extract(sample_audio, reference_audio=sample_reference)
#     # Result should be valid WAV bytes
#     assert result[:4] == b"RIFF"
#     # Result should be different from input (speaker was extracted)
#     assert result != sample_audio
