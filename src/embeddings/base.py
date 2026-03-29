from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedding(ABC):
    """Abstract base class for speaker embedding models.

    Speaker embeddings are fixed-length vectors that capture a speaker's
    vocal characteristics. Used by TSE models to identify which speaker
    to extract from a mixture.

    To implement a new embedding model:
    1. Create a new file in src/embeddings/ (e.g., my_embedding.py)
    2. Subclass BaseEmbedding
    3. Implement load(), embed(), and is_loaded
    """

    @abstractmethod
    async def load(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    async def embed(self, audio: bytes) -> np.ndarray:
        """Extract speaker embedding from audio.

        Args:
            audio: Raw audio bytes (WAV format) of a single speaker.

        Returns:
            1-D numpy array (speaker embedding vector).
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        ...
