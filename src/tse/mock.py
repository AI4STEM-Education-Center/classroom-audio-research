from src.tse.base import BaseTSE


class MockTSE(BaseTSE):
    """Mock TSE that passes audio through unchanged.

    Use this for integration testing before a real model is available.
    The service will accept audio, run it through the pipeline, and return
    transcriptions — the TSE step is just a no-op passthrough.
    """

    def __init__(self) -> None:
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def extract(
        self, mixed_audio: bytes, reference_audio: bytes | None = None
    ) -> bytes:
        # Passthrough: return input audio unchanged
        return mixed_audio

    @property
    def is_loaded(self) -> bool:
        return self._loaded
