# Classroom Audio Pipeline — Project Kickoff

**Team:** Jennifer Kleiman (PM / design), Arne Bewersdorff (hardware / architecture), AI@UGA undergraduates
**Started:** March 2026
**Updated:** March 5, 2026 — added real deployment findings, SE-DiCoW, testing plan
**Status:** MeanFlow-TSE deployed to production, comparative testing needed

---

## The Problem

We're building an AI co-teacher (ArguAgent) that joins small-group math discussions. For the AI to say anything useful, it needs to know what each student said — attributed to the right student, in near-real-time, in a noisy classroom.

This is harder than it sounds. Kids talk over each other, neighboring groups bleed into microphones, children's voices are acoustically different from the adult speech most models train on, and there are real legal constraints around recording minors.

We don't have this solved yet. We have a working prototype with basic VAD and speaker verification in the browser, and we send audio to a commercial ASR API. It works in quiet conditions. It falls apart in realistic classroom noise.

---

## What Exists Today (in ArguAgent)

- **Silero VAD** — runs in-browser, detects whether someone is speaking on each headset channel
- **Speaker verification** — ECAPA-TDNN in-browser, compares audio against an enrolled voiceprint
- **ASR** — currently using OpenAI's `gpt-4o-transcribe-diarize` API (see "Why We Need to Move Off This" below)
- **Headset mics** — each student wears one, plugged into their Chromebook

What's missing: streaming ASR, children's speech optimization, proper confidence scoring, privacy-compliant enrollment, and a testing infrastructure to know how well any of this actually works.

---

## What We've Built and Tested (March 2026)

We deployed a full MeanFlow-TSE + Whisper pipeline to AWS ECS and tested it on real audio. Here's what we found — these are real results, not projections.

### MeanFlow-TSE + Whisper base.en (deployed, CPU)

**Infrastructure:** ECS Fargate, 2 vCPU / 4GB RAM, ARM64 Graviton, ~$6-57/month

**What works:**
- Accuracy is good when it processes speech. Transcripts from TSE-extracted audio are nearly 100% correct on adult speech.
- The pipeline runs end-to-end: browser captures → TSE service → MeanFlow extraction → Whisper transcription → transcript in chat.

**What doesn't work:**
- **Latency is brutal.** ~15s round-trip for a 5s audio chunk on ARM64 CPU. MeanFlow inference alone is ~10s. This means we can only process 1 chunk per 15 seconds, dropping most of the user's speech.
- **TSE quality on real-world audio is poor.** Tested with laptop mic + podcast playing from a phone nearby (simulating classroom bleed). MeanFlow did NOT effectively isolate the target speaker — podcast voices leaked through significantly. Partial extraction when the target was speaking, but non-target voices weren't filtered.
- **Whisper hallucinates on near-silent TSE output.** When the target speaker isn't talking, MeanFlow outputs near-silence. Whisper base.en hallucinates on this: "I'm sorry, I'm sorry, I'm sorry..." (100+ repetitions), "D-D-D-D-D..." (200+ repetitions). Classic decoder loops.
- **Speaker verification doesn't work alongside TSE.** SV similarity scores of 0.05-0.24 (basically random) even for the actual target speaker. This is likely an enrollment quality issue and/or the ECAPA-TDNN ONNX model struggling with real-world conditions.

**Mitigations we've added:**
- Post-extraction RMS energy gate (skip Whisper if output is near-silent)
- Whisper hallucination filter (detect repeated phrases, excessive length)
- 15s capture segments + chunk buffering to improve throughput
- SV and VAD gate bypass for TSE mode (TSE does its own speaker isolation)

**Key takeaway:** MeanFlow-TSE works *in theory* (accuracy is good when it processes the right audio) but fails on real-world conditions. It was trained on Libri2Mix (clean lab data), not laptop mics with real noise. And the latency makes it unusable for real-time classroom use on CPU. GPU would help latency but wouldn't fix the extraction quality.

### SE-DiCoW: A More Promising Architecture

After our MeanFlow testing, we found [SE-DiCoW](https://arxiv.org/abs/2601.19194) (ICASSP 2026, code available at [BUTSpeechFIT/TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)). It takes a fundamentally different approach:

| | MeanFlow-TSE + Whisper | SE-DiCoW |
|---|---|---|
| **Approach** | Separate audio → then transcribe | Condition Whisper to only transcribe target |
| **Enrollment** | Pre-recorded voiceprint required | Self-enrolled from conversation audio |
| **Model** | UDiT 343M + Whisper 74M (two models) | Modified Whisper large-v3-turbo (~800M, one model) |
| **Inference** | Two stages, CPU ~15s/chunk | Single forward pass, GPU ~1-2s/chunk |
| **Quality** | Trained on Libri2Mix only | SOTA: 52% relative improvement on EMMA benchmark |
| **Hallucination risk** | High (near-silent extraction → Whisper loops) | Low (outputs text directly, no intermediate audio) |
| **Cost (classroom scale)** | ~$6/month (CPU Fargate) | ~$28/month (GPU spot instances) |

SE-DiCoW doesn't separate audio — it conditions Whisper's encoder with speaker identity (via diarization + cross-attention) so the ASR model *only attends to the target speaker*. No intermediate waveform, no hallucination risk from silent audio, no two-stage latency.

**This is our top priority model to evaluate.**

---

## The Three Sub-Problems

Every audio challenge in our setup falls into one of three cases. Understanding these is essential to understanding the whole pipeline. Read the [audio primer](./classroom-audio-primer-v4.md) for full detail.

### Sub-Problem 1: Student Is Silent, Neighbor Is Talking
The headset picks up a neighbor's voice faintly (20-40 dB quieter than the wearer). If we transcribe it, we attribute the neighbor's words to the wrong student.

**Current solution:** VAD + speaker verification gate. If the audio doesn't match the enrolled voiceprint, reject it. This mostly works.

### Sub-Problem 2: Student Is Talking, Nobody Else Is
Clean audio, one speaker, known identity. Just transcribe it. This is the easy case and the most common (~85% of classroom time based on conversation analysis research, though not measured in our specific context).

**Current solution:** Works fine today.

### Sub-Problem 3: Student Is Talking AND Neighbor Is Also Talking
Both voices are in the audio. The student's voice is louder (close-talk advantage), but the neighbor's words leak in. This is rare with headsets (~5-15%) but produces the worst errors — jumbled transcripts mixing two speakers' words.

**Current solution:** We don't have one. This is the hard problem.

**What we need:** Target-Speaker Extraction (TSE) — isolate just the target student's voice before transcribing. This is the central research question for this project.

---

## Why We Need to Move Off `gpt-4o-transcribe-diarize`

We're currently sending each student's headset audio to OpenAI's `gpt-4o-transcribe-diarize` model. This is the wrong tool for our setup:

- **We're paying for diarization we don't need.** Each headset channel is one student — we already know who's speaking. Diarization (figuring out "who said what" from mixed audio) is redundant.
- **It's worse at transcription because of it.** The diarize model has ~21% WER vs. the non-diarizing model. OpenAI acknowledges it "smooths transcripts" at the cost of accuracy.
- **It hallucinates speakers.** Community reports of phantom speakers detected in single-speaker audio.
- **No streaming.** Batch-only via the transcriptions API. We need near-real-time.
- **No real confidence scores.** We're hardcoding `confidence: 0.9` because the API doesn't return word-level confidence.
- **Per-request cost.** Self-hosted ASR eliminates this and keeps audio data on our infrastructure.

The right approach: use a non-diarizing ASR for the ~85% clean audio (Sub-Problem 2), and route to TSE only when the VAD/speaker-verification gates detect overlap (Sub-Problem 3). This is what the audio primer recommends.

---

## Why a Separate Repo

The audio pipeline has different concerns than the web app:
- Different runtime (Python, C++, ONNX) vs. the Next.js/TypeScript app
- Needs its own test suite with audio fixtures, ground-truth transcripts, accuracy benchmarks
- Multiple people working on independent components (VAD, ASR, TSE, etc.)
- Arne's edge computing work doesn't need to touch the web codebase

The main ArguAgent repo will consume the pipeline as a service. The interface between them is: audio in, attributed transcripts out.

---

## Architecture Sketch

This is a starting point, not a commitment. Pieces will change as we learn.

```
Student Chromebook (browser)
  ├── Silero VAD (in-browser, ~1ms)
  ├── Speaker verification (in-browser)
  └── Audio stream → server

Server / Edge Device
  ├── Route: overlap detected?
  │     ├── No  → ASR directly (Sub-Problem 2, ~85% of audio)
  │     └── Yes → TSE (extract target voice) → ASR (Sub-Problem 3)
  ├── ASR (Whisper? Canary? Moonshine? TBD)
  ├── Confidence scoring (word-level)
  └── Attributed transcript → ArguAgent app
```

**The TSE / TS-ASR question:** No published model has been evaluated on children's speech for target-speaker processing. We now have real experience with MeanFlow-TSE (see findings above). The models to evaluate, in priority order:
1. **SE-DiCoW** (TS-ASR, Jan 2026) — conditions Whisper on speaker identity, self-enrolling, SOTA results. **Top priority.**
2. **MeanFlow-TSE** (flow-based, Dec 2025) — we have this deployed; accuracy is good but extraction quality on real-world audio needs improvement. Worth optimizing (quantization, ONNX, GPU).
3. **SoloSpeech** (generative) — highest published quality on adults
4. **USEF-TSE** (discriminative) — different architecture, worth comparing

**Arne's edge angle:** Can we push more of the server-side processing onto local hardware (a classroom device) instead of cloud? Open-source models like Moonshine v2 (27M params, runs on a Raspberry Pi) and Silero make this plausible. The NOTSOFAR-1 challenge showed multi-channel separation gives ~51% improvement over single-channel — but that processing has to happen somewhere with access to all channels.

---

## What We Think We Know (and What We Don't)

**Fairly confident:**
- Headset mics are a huge advantage (20-40 dB signal-to-interference ratio, known speaker identity). The technical primer makes a strong case for keeping them.
- VAD on children's speech works well enough with Silero v6.2.
- Overlapping speech is rare with headsets (~5-15% of classroom time) but still needs handling.
- `gpt-4o-transcribe-diarize` is the wrong model for per-channel headset audio.

**Less certain:**
- Whether Whisper or Canary or something else is best for children's math speech. Canary reportedly does better on children's voices but we haven't tested it ourselves.
- How much latency we can tolerate before it breaks the pedagogical experience. Sub-2-second? Sub-5-second?
- Whether speaker verification generalizes to kids — most published results are on adults. SSL models (WavLM, HuBERT) may do better on out-of-distribution voices like children, but they're 4-14x larger than ECAPA-TDNN.
- How to do privacy-compliant voiceprint enrollment for minors (COPPA deadline: April 22, 2026).

**Honestly don't know:**
- Whether target-speaker extraction (TSE) works on children at all. No published evaluation exists. This is the most important open question.
- What error rate is acceptable before the AI co-teacher starts saying unhelpful things. ("Did you say the denominator is six, or fixed?")
- Whether edge deployment is viable for the TSE + ASR pipeline, or if it needs real GPU.

---

## Workstreams

These are somewhat independent — good for parallel work. Workstreams 1-4 are designed so undergrad team members can make progress **in parallel** with our ArguAgent app development.

### 1. Testing Infrastructure (everyone, first priority)
Before we optimize anything, we need to measure it. Build a test harness that can:
- Play audio fixtures through the pipeline
- Compare output against ground-truth transcripts (WER, speaker attribution accuracy)
- Run automatically (CI)
- Include realistic scenarios: clean speech, neighbor bleed, overlap, background noise
- Measure TSE quality: signal-to-distortion ratio (SDR) before and after extraction

We need recordings. Ideally classroom recordings with children, but we can start with simulated setups (record each person on separate channels, mix with known overlap ratios).

### 2. TSE / TS-ASR Model Evaluation (central research question, 2-3 people)
No one has tested whether target-speaker processing works on children. We'd be the first.

**Models to evaluate (priority order, updated with deployment experience):**
1. **SE-DiCoW** (TS-ASR, Jan 2026) — conditions Whisper on speaker identity, self-enrolling from conversation audio, SOTA results. Code available at [BUTSpeechFIT/TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper). **Top priority — see comparison table above.**
2. **MeanFlow-TSE** (flow-based, Dec 2025) — we have this deployed and tested. Accuracy is good but real-world extraction quality is poor (see findings above). Still worth optimizing (quantization, ONNX, GPU) as a baseline.
3. **SoloSpeech** (generative) — highest published quality on adults
4. **USEF-TSE** (discriminative) — different architecture, worth comparing

**What to measure:**
- SDR improvement on children's voices vs. adult voices (does it degrade?)
- Effect of enrollment clip length (how short can we go?)
- Latency per chunk (can it run in near-real-time?)
- Downstream ASR accuracy: WER on TSE-cleaned audio vs. raw overlapping audio
- For SE-DiCoW specifically: does self-enrollment work with children's voices? Does cross-attention conditioning generalize?

**What we need to figure out:**
- How to get or create children's speech test data (MyST corpus, Samrómur Children, CSLU Kids, or record our own with IRB approval)
- Whether a short enrollment clip (2-5 seconds) is enough for TSE conditioning
- Whether the 20-40 dB headset advantage makes TSE easier (the target voice is already louder)

### 3. ASR Model Evaluation (1-2 people)
Compare Whisper variants, Canary, and Moonshine on children's math speech:
- Word error rate on clean vs. noisy audio
- Math vocabulary accuracy ("numerator," "denominator," "equivalent fraction")
- Latency (batch vs. streaming)
- Resource requirements (can it run on what Arne's thinking about for edge?)
- Compare against `gpt-4o-transcribe` (non-diarize) as baseline

### 4. Speaker Verification on Children (1 person)
Does our current ECAPA-TDNN approach work for 13-14-year-olds?
- Measure false accept/reject rates
- Test enrollment with short clips (how little audio do we need?)
- Look into G-IFT adapter fine-tuning for children
- Compare against SSL approaches (WavLM/HuBERT) — better on out-of-distribution voices but much larger models

### 5. Privacy Architecture (Arne + Jennifer)
Design an enrollment and data handling approach that can survive legal review:
- COPPA compliance (voiceprints = biometric identifiers as of April 2026)
- FERPA (voice recordings as education records)
- Session-scoped enrollment with deletion?
- What can stay on-device vs. what has to go to a server?

### 6. Edge Deployment Exploration (Arne)
What's the smallest/cheapest hardware that can run the server-side pipeline?
- Moonshine v2 on Raspberry Pi — does it actually work for our use case?
- Can TSE run on an edge device or does it need GPU?
- Network architecture: Chromebooks → classroom edge device → cloud (if needed)

---

## Key Technical References

Everything we know is documented in [`docs/classroom-audio-primer-v4.md`](./classroom-audio-primer-v4.md) (content is v5 despite filename). It covers all the technologies, evaluates them against our specific setup, and cites 2025-2026 research. Read at least the first 5 sections and the summary tables before diving into code.

Notable open-source tools to evaluate:
- **SE-DiCoW / TS-ASR-Whisper** — target-speaker ASR via conditioned Whisper, ICASSP 2026, SOTA results, [code available](https://github.com/BUTSpeechFIT/TS-ASR-Whisper). **Top priority.**
- **Silero VAD v6.2** — MIT, 2MB, browser-compatible
- **MeanFlow-TSE** — target-speaker extraction, flow-based, code available. We have deployment experience (see findings above).
- **SoloSpeech / USEF-TSE** — alternative TSE architectures
- **Moonshine v2** — lightweight streaming ASR, ONNX
- **Canary** (NVIDIA) — ASR with reported children's speech improvements
- **WeSpeaker** — speaker verification framework
- **pyannote.audio 4.0** — diarization (useful for multi-channel separation pipeline, also used by SE-DiCoW)

Children's speech datasets:
- **MyST** (My Science Tutor) — children reading/speaking
- **Samrómur Children** — Icelandic children's corpus
- **CSLU Kids** — Oregon Health & Science University

---

## Compliance Deadlines

| Deadline | What |
|---|---|
| April 22, 2026 | COPPA: voiceprints classified as biometric identifiers for children |
| August 2, 2026 | EU AI Act: high-risk education AI rules in effect |
| Various 2026 | State laws: IL SB 1920, CO SB 205, TX HB 149 |

These aren't hypothetical — they affect what we can deploy and how.

---

## Parallel Testing Plan (AI@UGA Undergrad Team)

This plan is designed so the undergrad team can make progress **independently** from the ArguAgent web app. Everything here runs in a standalone Python environment — no Next.js, no database, no web frontend needed. Results feed back into our deployment decisions.

### Setup (Week 1)

1. **Clone the audio pipeline repo** (separate from ArguAgent — link TBD, or work in `services/tse-service/` for now)
2. **Set up a Python 3.11+ environment** with PyTorch, transformers, speechbrain
3. **Download models:**
   - MeanFlow-TSE checkpoints (we have these, ~1.5GB)
   - SE-DiCoW / TS-ASR-Whisper (clone from [GitHub](https://github.com/BUTSpeechFIT/TS-ASR-Whisper))
   - Whisper base.en, large-v3-turbo (via HuggingFace)
   - ECAPA-TDNN (via SpeechBrain)
4. **Collect test audio:** Record yourselves! Each person records:
   - 10s enrollment clip (reading a paragraph)
   - Several 5-10s "clean" clips (just you talking)
   - Several 5-10s "overlap" clips (you talking while someone else talks nearby)
   - Label each clip with ground-truth transcript

### Track A: SE-DiCoW Evaluation (2 people, highest priority)

**Goal:** Get SE-DiCoW running and benchmark it against MeanFlow-TSE.

| Week | Deliverable |
|---|---|
| 1-2 | Get SE-DiCoW inference running locally (follow their README, use their pretrained checkpoint) |
| 2-3 | Run on our test audio: measure WER on clean speech, WER on overlapping speech, latency per chunk |
| 3-4 | Compare against MeanFlow-TSE on the same audio (we can provide scripts) |
| 4+ | Test on children's speech data if available (MyST corpus, or recruit younger volunteers) |

**Key questions to answer:**
- Does SE-DiCoW's self-enrollment work with short clips (2-5s)?
- How does it handle the case where the target speaker is silent? (MeanFlow outputs near-silence → Whisper hallucinates. Does SE-DiCoW handle this better?)
- What's the GPU requirement? Can it run on a laptop with a decent GPU?
- What's the latency per chunk on GPU vs CPU?

### Track B: ASR Model Comparison (1-2 people)

**Goal:** Find the best ASR model for children's math speech.

| Week | Deliverable |
|---|---|
| 1-2 | Set up evaluation harness: script that takes audio + ground-truth transcript → outputs WER |
| 2-3 | Benchmark Whisper (base.en, small.en, medium.en, large-v3-turbo) on adult test audio |
| 3-4 | Benchmark Canary-1B, Moonshine v2 on same audio |
| 4+ | Test all models on children's speech. Report which model handles math vocabulary best. |

**What to measure for each model:**
- WER (word error rate) on clean audio
- WER on noisy audio (with background speakers)
- Math vocabulary accuracy: does it transcribe "numerator" or "new marinator"?
- Latency: time to transcribe a 5s chunk
- Model size and memory usage

### Track C: Speaker Verification Benchmarking (1 person)

**Goal:** Quantify how well speaker verification works with different voices and enrollment lengths.

| Week | Deliverable |
|---|---|
| 1-2 | Set up ECAPA-TDNN (SpeechBrain) + cosine similarity scoring |
| 2-3 | Measure EER (equal error rate) with different enrollment lengths: 2s, 5s, 10s, 30s |
| 3-4 | Test with pairs of similar voices vs. different voices. Measure false accept/reject. |
| 4+ | If possible, test with children's voices. Compare ECAPA-TDNN vs. WavLM-based SV. |

### Track D: Test Data Creation (everyone contributes)

**Goal:** Build a reusable audio test suite.

**What we need:**
- **Clean speech clips** — one speaker, known transcript (ground truth)
- **Overlap clips** — two speakers talking simultaneously, both transcripts known
- **Noise clips** — speech with classroom-like background noise
- **Children's speech** — if we can get IRB approval or find existing datasets

**How to create overlap test data:**
1. Two people each record the same passage on separate devices
2. Mix the recordings at different signal-to-interference ratios (SIR): 0 dB, 10 dB, 20 dB, 40 dB
3. The 20-40 dB range simulates headset bleed; 0-10 dB simulates same-table conversation without headsets
4. Script for mixing: `python mix_audio.py --target speaker1.wav --interference speaker2.wav --sir 20`

### Reporting

**Weekly:** Post a short update (3-5 sentences) in the shared channel. What you tested, what you found, what's next.

**When you have results:** Create a markdown file in `docs/results/` with:
- What model/configuration you tested
- Test audio description (duration, speakers, noise conditions)
- Metrics (WER, SDR, latency, etc.)
- Your interpretation: does this work for our use case?

Negative results are just as valuable as positive ones. "We tried X and it failed because Y" saves everyone time.

---

## How to Contribute

1. Read the audio primer (really — it's long but it'll save you weeks)
2. **New team members:** Start with the Parallel Testing Plan above — pick a track (A, B, C, or D) and get set up in week 1
3. Ask questions. Nobody on this team has done all of this before.
4. Write tests before writing pipeline code
5. Document what you find, especially negative results ("we tried X and it didn't work because Y")

The goal is to learn and build something useful, not to ship a perfect system. We'll figure out what works by trying things and measuring them.

---

## Open Questions (add yours here)

- What Chromebook models will classrooms actually have? (Affects AEC, compute budget)
- Can we get IRB approval for classroom recordings this semester?
- What's the minimum viable pipeline that's better than what we have now?
- Should the test harness be Python-only or do we need browser-based tests too?
- Can we use the 20-40 dB headset advantage to simplify TSE? (Target voice is already dominant — maybe a lighter model suffices)
- How do we detect that overlap is happening in real-time so we know when to route to TSE?
