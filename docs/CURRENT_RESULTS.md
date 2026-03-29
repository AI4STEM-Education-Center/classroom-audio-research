# Current Baseline Results

*March 2026 — from ArguAgent production pipeline testing*

---

## Test Setup

- **Speakers:** 23 enrolled (real 8th-grade students), 13 tested with available audio chunks
- **Total chunks:** 57 audio segments (3-15 seconds each)
- **Pipeline:** MeanFlow-TSE -> Energy Gate -> Whisper base.en -> Hallucination Filter
- **Speaker verification:** CAM++ ONNX (512-dim, SECS post-filter)
- **Hardware:** CPU (macOS ARM64) for this test run
- **Raw data:** `results/baseline-2026-03-28.json`

## Summary

| Metric | Value | Target |
|--------|-------|--------|
| Success rate | 53% (30/57) | >90% |
| Avg SECS (speaker similarity) | 0.160 | >0.5 |
| Min SECS | 0.055 | >0.3 |
| Max SECS | 0.393 | >0.7 (confident match) |
| Avg latency (CPU) | 49.6s | <3s (real-time) |
| GPU latency (T4) | 0.24s/5.5s | Meets target |

## What These Numbers Mean

### SECS Scores Are Too Low

The average SECS of 0.160 means the pipeline has weak confidence that extracted audio belongs to the target speaker. For context:

- **>0.7:** High confidence same speaker (what we want)
- **0.3-0.7:** Likely same speaker
- **0.15-0.3:** Ambiguous (where most of our scores land)
- **<0.15:** Likely different speaker

**Hypotheses for low scores:**

1. **Enrollment quality** — Enrollment clips are short (3-5s) and may contain background noise. Longer, cleaner enrollments should produce better embeddings.

2. **Child voices** — CAM++ and ECAPA-TDNN are trained on VoxCeleb (adult speech). Children's voices have different formant structures, higher pitch variability, and less stable vocal characteristics. Published EER for children is 2.9-10.8% vs <1% for adults.

3. **Short segments** — Many test chunks are 3-5 seconds. Speaker embeddings are less reliable on short utterances. The models expect at least 3s for stable embeddings.

4. **TSE distortion** — MeanFlow extraction may subtly alter the speaker's voice characteristics, reducing the SECS match even when it correctly isolates the target speaker.

### Failure Modes

| Mode | Count | Cause |
|------|-------|-------|
| HTTP 500 | 10 | TSE service crash under load (memory) |
| Empty transcription | 12 | Energy gate: TSE output was near-silent |
| Low energy skip | 5 | Input audio was near-silent (student not speaking) |

The high empty-transcription rate (21%) suggests MeanFlow sometimes extracts the wrong speaker or suppresses all speech.

### Latency

CPU latency of ~50s per chunk is not viable for production (need <3s for real-time). **GPU (T4) at 0.24s for 5.5s audio is the path to real-time operation.** The 200x CPU-to-GPU speedup is typical for diffusion-based models.

## Baseline Targets for Research

If you're working on improving any component, here's what "better" looks like:

| Component | Current | Good | Excellent |
|-----------|---------|------|-----------|
| SECS (same speaker) | 0.160 | >0.4 | >0.6 |
| SECS (diff speaker) | untested | <0.15 | <0.10 |
| ASR WER | unmeasured | <15% | <10% |
| TSE SI-SNR improvement | unmeasured | >8 dB | >12 dB |
| Success rate | 53% | >85% | >95% |

## How to Reproduce

```bash
# Start the service with real models
TSE_MODEL=meanflow ASR_MODEL=whisper uvicorn src.main:app --port 8100

# Process a test audio file
curl -X POST http://localhost:8100/extract-and-transcribe \
  -F "audio=@test_chunk.wav" \
  -F "referenceAudio=@enrollment.wav" \
  -F "speakerName=test"
```

## Next: Standardized Evaluation

The baseline above was from production testing with real student audio (not available in this repo for privacy). To establish reproducible benchmarks, use:

1. **WHAM! noise mixing** (`scripts/mix_audio.py`) to create controlled test conditions
2. **VoxCeleb** test pairs for speaker verification evaluation
3. **LibriSpeech** clean speech + WHAM! noise for TSE+ASR evaluation

See `scripts/download_wham.py` to get started.
