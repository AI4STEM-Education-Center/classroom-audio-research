# Classroom Audio Research — Project Kickoff

*AI@UGa undergraduate research team*
*Updated: March 2026*

---

## Mission

Build and evaluate audio processing components that let an AI co-teacher know **what each student said** in a noisy classroom, attributed to the correct speaker, in near-real-time.

This repo is the research companion to [ArguAgent](https://arguagent.ai4genius.org), an AI co-teacher for small-group math discussions. The production pipeline is running, but key components need improvement — that's where your work comes in.

## Team

- **Jennifer Kleiman** — PM, system design, ArguAgent platform
- **Arne Bewersdorff** — Hardware architecture, audio engineering
- **AI@UGa undergraduates** — Research tracks below

## What We Know Now (March 2026)

Before diving into research tracks, here's where the production pipeline stands:

### Working
- **MeanFlow-TSE** extracts target speaker audio from headset mic input
- **Whisper base.en** transcribes extracted audio with hallucination detection
- **CAM++ speaker verification** provides SECS post-filter scores
- **Energy gate** prevents ASR on near-silent TSE output
- **GPU inference** (T4): 0.24s per 5.5s audio — viable for real-time

### Not Working Well
- **Speaker verification scores are too low:** avg SECS = 0.160 across 13 real speakers (need >0.5)
- **53% success rate** on real classroom audio (need >90%)
- **21% of extractions produce silence** — TSE sometimes suppresses the target speaker
- **No standardized evaluation** — we've only tested on production data, not public benchmarks

### Key Technical Findings
- **n_fft=510, not 512** — MeanFlow uses 510 to get exactly 256 freq bins (256x2=512=UDiT input dim)
- **Whisper hallucinates on near-silent audio** — decoder loops produce fluent-sounding garbage. Detection: >300 chars for 5s, 4+ repeated phrases, stuttering patterns
- **CAM++ (512-dim, ONNX) > ECAPA-TDNN (192-dim, PyTorch)** for speaker verification speed and accuracy
- **CPU latency is 50x too slow** for real-time — GPU is mandatory for production
- **Child voices degrade SV** — models trained on VoxCeleb adults, EER 2.9-10.8% on children

See [CURRENT_RESULTS.md](CURRENT_RESULTS.md) for full baseline numbers.

---

## Research Tracks

### Track A: SE-DiCoW Evaluation

**Goal:** Evaluate the Self-Enrolled Diarize-Classify-Whisper pipeline end-to-end.

**Status:** Not started. MeanFlow-TSE is integrated as the extraction backbone. The full SE-DiCoW pipeline (diarization -> classification -> ASR) has not been evaluated as a unit.

**Key questions:**
- How does SE-DiCoW perform vs our current pipeline on WHAM!-mixed speech?
- What's the DER (diarization error rate) with enrolled speakers?
- Does TSE quality affect downstream diarization accuracy?

### Track B: ASR Model Comparison

**Goal:** Compare ASR models on classroom audio (noisy, children's speech, informal language).

**Status:** Whisper base.en is the current model. Hallucination detection is implemented but confidence scoring is hardcoded (0.9).

**Key questions:**
- How does Whisper base.en compare to Whisper small.en, medium.en on our test data?
- What is the actual WER on classroom speech? (Currently unmeasured)
- Can we get real per-utterance confidence scores? (HuggingFace pipeline doesn't expose them)
- How does Canary (NVIDIA) compare for noisy children's speech?
- Does RNNoise preprocessing help or hurt ASR accuracy? (Research suggests speech enhancement can degrade Whisper WER)

### Track C: Speaker Verification Benchmarking

**Goal:** Improve speaker verification for 8th-grade voices.

**Status:** CAM++ integrated (SECS post-filter), but avg score 0.160 is too low. Threshold 0.25 blocks most real speech.

**Key questions:**
- What SECS scores do we get on VoxCeleb test pairs? (Establishes model baseline)
- How much does enrollment quality affect SECS? (Duration, noise level, speaking style)
- Can we fine-tune CAM++ or ECAPA-TDNN on children's speech data?
- Is there a better threshold strategy than fixed 0.25? (Adaptive, percentile-based, per-speaker)
- What does the MyST (My Science Tutor) corpus show for child SV performance?

### Track D: Test Data Creation

**Goal:** Build a reproducible evaluation dataset from public sources.

**Status:** WHAM! download script ready. Mixing script supports SNR-controlled combinations.

**Key deliverables:**
- WHAM!-mixed test set at SNR = {-5, 0, 5, 10, 15} dB
- VoxCeleb same-speaker and different-speaker pairs for SV evaluation
- Synthetic classroom scenarios (2-speaker overlap, 3-speaker, background HVAC noise)
- Ground-truth labels for all test data (who spoke when, what they said)

### Track E: Multi-Channel Processing (NEW)

**Goal:** Exploit the fact that we have multiple synchronized audio streams.

**Status:** Not started. Current system processes each student's stream independently.

**Key research directions:**
- **Cross-channel energy gating:** If Student A's mic has high energy but Student B's has low energy -> gate B's stream. Research shows this can eliminate ~85% of phantom transcriptions.
- **GSS (Guided Source Separation):** Use spatial information across channels. NOTSOFAR-1 challenge showed dramatic improvements.
- **Dia-Sep-ASR:** Joint diarization + separation + ASR across channels.

This track is exploratory — start with energy gating (simple) before attempting neural multi-channel methods.

---

## Getting Started

```bash
# Clone and install
git clone https://github.com/AI4STEM-Education-Center/classroom-audio-research.git
cd classroom-audio-research
pip install -e ".[all]"

# Generate test fixtures
python scripts/generate_test_audio.py

# Run tests (mock mode — no models needed)
pytest tests/ -v

# Download WHAM! noise for evaluation
python scripts/download_wham.py

# Download real models (requires ~2GB disk)
python scripts/download_models.py

# Start the service with real models
TSE_MODEL=meanflow ASR_MODEL=whisper uvicorn src.main:app --port 8100

# Test it
curl -X POST http://localhost:8100/extract-and-transcribe \
  -F "audio=@tests/fixtures/mixed_speech.wav"
```

## Documentation

- [TECHNICAL_PRIMER.md](TECHNICAL_PRIMER.md) — How the pipeline works, key algorithms, research context
- [CURRENT_RESULTS.md](CURRENT_RESULTS.md) — Baseline performance numbers and what they mean
- `results/baseline-2026-03-28.json` — Raw evaluation metrics

## Contributing

1. Pick a research track (or propose a new one)
2. Create a branch: `git checkout -b track-X/your-name/description`
3. Write evaluation scripts in `scripts/`, results go in `results/`
4. Document findings in `docs/results/your-evaluation.md`
5. Open a PR with your results and analysis
