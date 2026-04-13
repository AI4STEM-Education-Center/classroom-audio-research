from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """TSE service configuration. All values can be set via environment variables."""

    port: int = 8100
    tse_model: str = "mock"  # "mock" | "meanflow"
    asr_model: str = "mock"  # "mock" | "whisper"
    model_cache_dir: str = "./models"
    whisper_model_size: str = "base.en"
    log_level: str = "info"
    torch_num_threads: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
