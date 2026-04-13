from abc import ABC, abstractmethod


class BaseTSE(ABC):
    """Abstract base class for target-speaker extraction models.

    To implement a new TSE model:
    1. Create a new file in src/tse/ (e.g., my_model.py)
    2. Subclass BaseTSE
    3. Implement load(), extract(), and is_loaded
    4. Add your model name to the factory in src/main.py

    The extract() method receives raw audio bytes (WAV format) and should
    return cleaned audio bytes (WAV format) with only the target speaker.
    """

    @abstractmethod
    async def load(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    async def extract(
        self, mixed_audio: bytes, reference_audio: bytes | None = None
    ) -> "bytes | tuple[bytes, float | None]":
        """Extract target speaker from mixed audio.

        Args:
            mixed_audio: Raw audio bytes (WAV format) containing mixed speakers.
            reference_audio: Optional reference clip of the target speaker (WAV).
                             Used to create a speaker embedding for conditioning.

        Returns:
            Cleaned audio bytes (WAV format) containing only the target speaker,
            OR a tuple of (audio_bytes, speaker_similarity) where speaker_similarity
            is the SECS score (cosine similarity between extracted and enrollment embeddings).
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        ...
