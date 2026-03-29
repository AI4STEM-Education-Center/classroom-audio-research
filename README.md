# Classroom Audio Research

Research and evaluation codebase for target-speaker processing, ASR, and speaker verification in classroom environments.

**Research companion to [ArguAgent](https://arguagent.ai4genius.org)** — an AI co-teacher for small-group math discussions that needs to know what each student said, attributed correctly, in near-real-time.

## The Problem

A classroom with 8 students in small groups is noisy. Each student wears a headset mic, but neighboring voices still bleed through. We need to:

1. **Detect** when the target student is speaking (VAD + Speaker Verification)
2. **Extract** their voice from the mix (Target Speaker Extraction)
3. **Transcribe** what they said (ASR)
4. **Verify** the transcript belongs to the right student (SECS post-filter)

## Architecture

```
Headset Mic (48kHz WebM/Opus)
    |
Audio Conversion (-> 16kHz mono WAV)
    |
Target Speaker Extraction (MeanFlow-TSE)
    |-- TPredicter: mixture + enrollment -> mixing ratio
    |-- UDiT: single-step diffusion -> clean spectrogram
    |-- SECS: CAM++ speaker similarity score
    |
Energy Gate (RMS < 0.005 -> skip)
    |
ASR (Whisper base.en)
    |-- Transcription
    |-- Hallucination detection (repetition, length)
    |
{text, confidence, speakerSimilarity}
```

## Quick Start

```bash
# Install (base deps only — fast)
pip install -e ".[dev]"

# Generate synthetic test audio
python scripts/generate_test_audio.py

# Run tests (mock mode — no model downloads needed)
pytest tests/ -v

# Install ML dependencies (slower — downloads PyTorch, Whisper, etc.)
pip install -e ".[all]"

# Download model checkpoints (~2GB)
python scripts/download_models.py

# Start the service
TSE_MODEL=meanflow ASR_MODEL=whisper uvicorn src.main:app --port 8100

# Health check
curl http://localhost:8100/health

# Process audio
curl -X POST http://localhost:8100/extract-and-transcribe \
  -F "audio=@tests/fixtures/mixed_speech.wav" \
  -F "referenceAudio=@tests/fixtures/reference.wav"
```

## Evaluation Datasets

```bash
# Download WHAM! noise (~4GB) for realistic mixing
python scripts/download_wham.py

# Create test mixtures at various SNR levels
python scripts/mix_audio.py \
  --target clean_speech.wav \
  --wham --snr -5 0 5 10 15 \
  --output-dir data/mixtures/
```

## Current Baseline (March 2026)

| Metric | Value | Target |
|--------|-------|--------|
| Success rate | 53% | >90% |
| Speaker similarity (SECS) | 0.160 avg | >0.5 |
| ASR WER | unmeasured | <15% |
| GPU latency (T4) | 0.24s / 5.5s audio | Meets target |

See [docs/CURRENT_RESULTS.md](docs/CURRENT_RESULTS.md) for full analysis.

## Research Tracks

| Track | Focus | Status |
|-------|-------|--------|
| A | SE-DiCoW end-to-end evaluation | Not started |
| B | ASR model comparison (Whisper variants, Canary) | Whisper base.en baseline |
| C | Speaker verification for children's voices | CAM++ integrated, scores too low |
| D | Public test dataset creation (WHAM!, VoxCeleb) | Scripts ready |
| E | Multi-channel processing (cross-channel gating, GSS) | Not started |

See [docs/KICKOFF.md](docs/KICKOFF.md) for details.

## Project Structure

```
src/
|-- main.py                    # FastAPI application
|-- config.py                  # Settings (env vars)
|-- api/                       # HTTP endpoints
|-- tse/                       # Target Speaker Extraction
|   |-- meanflow.py            # MeanFlow-TSE (single-step flow matching)
|   |-- campplus_verifier.py   # CAM++ SECS post-filter
|   |-- vendor/meanflow_tse/   # Vendored model code
|-- asr/                       # Automatic Speech Recognition
|   |-- whisper_asr.py         # Whisper + hallucination detection
|-- embeddings/                # Speaker embeddings
|   |-- ecapa.py               # ECAPA-TDNN via SpeechBrain
|-- evaluation/                # Metrics (WER, SDR, SI-SNR, EER)

scripts/
|-- download_models.py         # Fetch MeanFlow, CAM++, ECAPA-TDNN, Whisper
|-- download_wham.py           # Fetch WHAM! noise dataset
|-- mix_audio.py               # Create SNR-controlled audio mixtures
|-- generate_test_audio.py     # Synthetic test fixtures

results/
|-- baseline-2026-03-28.json   # Production pipeline metrics
```

## Documentation

- **[TECHNICAL_PRIMER.md](docs/TECHNICAL_PRIMER.md)** — How the pipeline works, key algorithms, papers
- **[CURRENT_RESULTS.md](docs/CURRENT_RESULTS.md)** — Baseline numbers and what they mean
- **[KICKOFF.md](docs/KICKOFF.md)** — Research tracks and getting started

## Team

- **Jennifer Kleiman** — PM, system design ([ArguAgent](https://arguagent.ai4genius.org))
- **Arne Bewersdorff** — Hardware architecture, audio engineering
- **AI@UGa undergraduates** — Research

## Key References

- [MeanFlow-TSE](https://arxiv.org/abs/2512.18572) — Single-step flow matching for speaker extraction
- [CAM++](https://arxiv.org/abs/2303.00332) — Context-aware speaker verification
- [WHAM!](https://wham.whisper.ai/) — Noise dataset for speech separation
- [Khan et al. (EDM 2025)](https://arxiv.org/abs/2505.10879) — Classroom VAD evaluation
