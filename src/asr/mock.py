from src.asr.base import BaseASR, TranscriptionResult


class MockASR(BaseASR):
    """Mock ASR that returns a fixed transcript.

    Use this for integration testing before a real ASR model is available.
    Always returns "[mock transcript]" with confidence 1.0.
    """

    def __init__(self) -> None:
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        return TranscriptionResult(
            text="[mock transcript]",
            confidence=1.0,
            duration=5.0,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded
