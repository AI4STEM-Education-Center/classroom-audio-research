import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.config import settings


def _create_tse(model_name: str):
    """Factory for TSE implementations."""
    if model_name == "mock":
        from src.tse.mock import MockTSE

        return MockTSE()
    elif model_name == "meanflow":
        from src.tse.meanflow import MeanFlowTSE

        return MeanFlowTSE(model_dir=settings.model_cache_dir)
    else:
        raise ValueError(f"Unknown TSE model: {model_name}. Use 'mock' or 'meanflow'.")


def _create_asr(model_name: str):
    """Factory for ASR implementations."""
    if model_name == "mock":
        from src.asr.mock import MockASR

        return MockASR()
    elif model_name == "whisper":
        from src.asr.whisper_asr import WhisperASR

        return WhisperASR(
            model_size=settings.whisper_model_size,
            cache_dir=settings.model_cache_dir,
        )
    else:
        raise ValueError(f"Unknown ASR model: {model_name}. Use 'mock' or 'whisper'.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup."""
    logger = logging.getLogger("tse-service")

    # Create model instances
    tse = _create_tse(settings.tse_model)
    asr = _create_asr(settings.asr_model)

    logger.info(f"Loading TSE model: {settings.tse_model}")
    await tse.load()
    logger.info(f"Loading ASR model: {settings.asr_model}")
    await asr.load()

    # Store on app.state for route access
    app.state.tse = tse
    app.state.asr = asr
    app.state.tse_model_name = settings.tse_model
    app.state.asr_model_name = settings.asr_model

    logger.info("All models loaded. Service ready.")

    yield

    logger.info("Shutting down TSE service.")


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="Classroom Audio Research — TSE Service",
    description="Target-Speaker Extraction + ASR for classroom audio research",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)
