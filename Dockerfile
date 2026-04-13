FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_NUM_THREADS=4

# Install audio libraries + curl for healthcheck + git for pip install from github
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash tse
WORKDIR /home/tse/app

# Install dependencies (CPU-only torch via --extra-index-url)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[ml]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY src/ src/
COPY scripts/ scripts/

# Models directory (mount or pre-bake)
RUN mkdir -p models && chown tse:tse models
VOLUME /home/tse/app/models

USER tse

EXPOSE 8100

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8100"]
