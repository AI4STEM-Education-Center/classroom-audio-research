from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""

    text: str
    confidence: float
    duration: float  # audio duration in seconds


class BaseASR(ABC):
    """Abstract base class for automatic speech recognition models.

    To implement a new ASR backend:
    1. Create a new file in src/asr/ (e.g., my_asr.py)
    2. Subclass BaseASR
    3. Implement load(), transcribe(), and is_loaded
    4. Add your model name to the factory in src/main.py
    """

    @abstractmethod
    async def load(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio: Raw audio bytes (WAV format).

        Returns:
            TranscriptionResult with text, confidence score, and audio duration.
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        ...
