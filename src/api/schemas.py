from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """Response from the extract-and-transcribe endpoint."""

    text: str
    confidence: float
    duration: float
    speakerSimilarity: float | None = None  # SECS: cosine similarity between extracted and enrollment embeddings


class ErrorResponse(BaseModel):
    """Error response."""

    error: str


class HealthResponse(BaseModel):
    """Response from the health endpoint."""

    status: str
    tse_model: str
    asr_model: str
    tse_loaded: bool
    asr_loaded: bool
