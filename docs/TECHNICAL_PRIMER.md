# Technical Primer: Classroom Audio for AI

*For the AI@UGa research team — March 2026*

---

## The Problem

We want an AI co-teacher that participates in small-group math discussions. To do that, the AI needs to know **what each student said** — not what their neighbor said, not what the group across the room said, just what *that specific student* said.

Each student wears a headset with a mic positioned 2-5 cm from their mouth, plugged into their own Chromebook. We know which mic belongs to which student. The headset's cardioid pattern rejects sound from the sides and rear, giving the wearer a **20-40 dB advantage** over neighboring voices.

This is our key architectural bet: individual headset mics make the problem tractable.

---

## The Three Sub-Problems

Every classroom audio challenge falls into one of three situations:

### 1. Student Is Silent, Neighbor Is Talking

The student isn't speaking, but a neighbor's voice bleeds through the headset mic. If we transcribe everything, we attribute the neighbor's words to our silent student. **This is the most common source of phantom transcriptions.** The headset helps (20-40 dB rejection), but loud neighbors still leak through.

### 2. Student Is Talking Alone

Clean audio, easy case. Just transcribe. This is **~85% of classroom time** — conversation research shows <5% simultaneous speech in dyads, and small-group discussions involve significant listening time.

### 3. Student AND Neighbor Are Both Talking

The hardest case. The target student IS talking, but another voice is mixed in. ~5-15% of classroom time with headsets. The neighbor's voice is quieter, but can still produce phantom words.

---

## The Pipeline

```
Microphone (48kHz, WebM/Opus)
    |
Voice Activity Detection (VAD)
    |  "Is anyone speaking?"
Speaker Verification (SV)
    |  "Is it THIS student speaking?"
Target Speaker Extraction (TSE)
    |  "Remove other voices, keep only the target"
Energy Gate
    |  "Is there actually speech in the output?"
Automatic Speech Recognition (ASR)
    |  "What words were said?"
Hallucination Filter
    |  "Is this real transcription or decoder noise?"
Final Transcript
```

### Voice Activity Detection (VAD)

Answers: "Is someone speaking right now?" A small neural network (Silero VAD, ~2MB) classifies each 30ms audio frame as speech/non-speech. With headset mics, VAD eliminates ~60-70% of phantom transcriptions by correctly gating faint neighbor bleed as "no speech."

**Key finding:** Khan et al. (EDM 2025) found VAD quality has p=0.96 correlation with overall diarization error rate in classrooms — getting VAD right is the single most important factor.

**Current model:** Silero VAD v6.2 (Nov 2025) — specifically improved for child voices.

### Speaker Verification (SV)

Answers: "Does this audio match the enrolled student's voice?" Compares a speaker embedding of the current audio against a stored enrollment embedding using cosine similarity. If the similarity is below a threshold, the audio is likely from a different speaker.

**Two models in our pipeline:**

| Model | Dims | Size | Speed | Notes |
|-------|------|------|-------|-------|
| **CAM++** (ONNX) | 512 | 28 MB | Fast | Current primary. WeSpeaker pretrained. |
| **ECAPA-TDNN** (SpeechBrain) | 192 | 35 MB | Medium | Fallback. Lower dimensionality. |

**Open problem:** Both models are trained on adult speech (VoxCeleb). Our users are 8th graders. Speaker verification EER degrades significantly for children (2.9-10.8% vs <1% for adults). Our baseline SECS scores average **0.160** across 13 student speakers — well below the 0.25 threshold. This is a primary research target.

### Target Speaker Extraction (TSE) — MeanFlow

**What it does:** Given a mixture of voices and a reference clip of the target speaker, isolates just the target speaker's voice.

**How MeanFlow-TSE works** (paper: [arXiv:2512.18572](https://arxiv.org/abs/2512.18572)):

1. **TPredicter** takes the mixture spectrogram + enrollment embedding and predicts a mixing ratio alpha
2. **STFT** converts both mixture and enrollment to spectrograms (n_fft=**510**, hop=128, win=510)
   - Note: n_fft=510, NOT 512. This gives 256 frequency bins. Real+imag = 512 = UDiT input_dim.
3. **UDiT** (U-shaped Diffusion Transformer) runs a **single Euler step** to predict the clean target spectrogram from the noisy mixture
4. **iSTFT** converts back to waveform

The single-step approach (NFE=1) is what makes MeanFlow fast enough for near-real-time use. Traditional diffusion models need 50+ steps.

**Performance:**
- CPU (macOS ARM64): 1.6s for 1.5s audio, 2.6s for 7s audio
- CPU (Graviton ARM64): 9.8s for 5.5s audio
- **GPU (T4): 0.24s for 5.5s audio** — viable for real-time

**Chunk processing:** Audio is split into 3-second segments at 16kHz. Each chunk is padded to exactly 48,000 samples, processed independently, then concatenated.

### Energy Gate

After TSE extracts audio, we compute the RMS energy. If RMS < 0.005, the output is near-silent — meaning TSE suppressed everything (the target speaker wasn't actually talking). Skip ASR to avoid transcribing noise.

### Automatic Speech Recognition (ASR) — Whisper

**Model:** OpenAI Whisper (base.en, via HuggingFace transformers). Takes WAV audio and outputs text.

**Hallucination detection** is critical. Whisper's decoder can enter repetitive loops on near-silent or garbage audio, generating fluent-sounding but completely fabricated text. We detect and reject:

- **Long output:** >300 characters for a 5-second segment (real speech is approx 50-80 chars)
- **Repeated phrases:** Same phrase 4+ times in a row
- **Character stuttering:** Patterns like "d-d-d-d" or "d, d, d, d"

**Confidence:** Currently hardcoded at 0.9 — the HuggingFace transformers pipeline doesn't expose per-utterance confidence scores. Improving this is an open research question.

### SECS Post-Filter (Speaker Embedding Cosine Similarity)

After TSE extraction and before returning results, we compute a **SECS score**: cosine similarity between the extracted audio's speaker embedding and the enrollment embedding. This tells us how confident we are that the extracted audio actually belongs to the target speaker.

The CAM++ verifier computes 80-dim log mel filterbank features with cepstral mean normalization, runs ONNX inference to get a 512-dim embedding, then computes cosine similarity against the enrollment embedding.

---

## The Frontier: Multi-Channel Processing

Our biggest untapped advantage is that we have **multiple synchronized audio streams** (one per headset). Current system processes each student's stream independently. The frontier research combines them:

**Cross-channel gating:** Compare energy/VAD across all headsets simultaneously. If Student A's mic shows high energy but Student B's mic shows low energy, Student A is probably the one talking — gate Student B's stream.

**GSS (Guided Source Separation):** Use spatial/spectral information from multiple channels to separate speakers. NOTSOFAR-1 challenge showed this dramatically outperforms single-channel approaches.

**Dia-Sep-ASR:** Joint diarization + separation + ASR in a single model. Processes all channels together. The field is moving from simple energy-based gating to neural multi-channel separation.

---

## Key Papers

| Topic | Paper | Key Finding |
|-------|-------|-------------|
| MeanFlow-TSE | [arXiv:2512.18572](https://arxiv.org/abs/2512.18572) | Single-step flow matching for TSE, 10-50x faster than diffusion |
| CAM++ | [arXiv:2303.00332](https://arxiv.org/abs/2303.00332) | Context-aware masking for speaker verification, SOTA on VoxCeleb |
| SE-DiCoW | Self-Enrolled Diarize-Classify-Whisper | Pipeline using TSE for diarization with enrolled speakers |
| Classroom VAD | Khan et al. (EDM 2025, [arXiv:2505.10879](https://arxiv.org/abs/2505.10879)) | VAD quality has p=0.96 with DER in classrooms |
| Silero VAD v6.2 | [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) | Improved child voice detection, 2MB, MIT license |
| Child SV | Multiple (EER survey) | SV EER 2.9-10.8% for children vs <1% for adults |
| NOTSOFAR-1 | CHiME challenge | Multi-channel GSS outperforms single-channel TSE |

---

## What You're Working On (Research Tracks)

See [KICKOFF.md](KICKOFF.md) for the full research track descriptions and current status.
