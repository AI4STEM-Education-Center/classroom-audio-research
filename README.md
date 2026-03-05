# Classroom Audio Research

Research pipeline for target-speaker processing, ASR, and speaker verification in classroom environments — focused on children's speech in educational settings.

**Team:** Jennifer Kleiman (PM/design), Arne Bewersdorff (hardware/architecture), AI@UGA undergraduates
**Parent project:** [GENIUS ArguAgent](https://github.com/AI4STEM-Education-Center/GENIUS_ArgueAgent)

---

## What This Repo Is For

We're building an AI co-teacher that joins small-group math discussions. For the AI to say anything useful, it needs to know what each student said — attributed to the right student, in near-real-time, in a noisy classroom.

This repo contains the **research and evaluation code** for the audio pipeline. It's separate from the main ArguAgent web app so that:
- You can work in Python without needing a Next.js/TypeScript environment
- We have a dedicated test suite with audio fixtures and ground-truth transcripts
- Multiple people can work on independent components in parallel

**Read the full context:** See [`docs/KICKOFF.md`](docs/KICKOFF.md) for the project kickoff document with problem statement, architecture, what we've tested, and the parallel testing plan.

---

## Quick Start

```bash
# Clone
git clone https://github.com/AI4STEM-Education-Center/classroom-audio-research.git
cd classroom-audio-research

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Download models (optional — needed for Track A/B/C)
python scripts/download_models.py
```

---

## Project Structure

```
classroom-audio-research/
├── docs/                     # Project docs, kickoff, audio primer
│   ├── KICKOFF.md            # Start here — project context + testing plan
│   └── results/              # Put your evaluation results here
├── src/
│   └── evaluation/           # Evaluation harness code
│       ├── metrics.py        # WER, SDR, EER computation
│       └── audio_utils.py    # Audio loading, mixing, resampling
├── scripts/
│   ├── download_models.py    # Download pretrained models
│   └── mix_audio.py          # Create overlap test data
├── tests/
│   └── fixtures/             # Test audio files + ground truth transcripts
├── models/                   # Downloaded model checkpoints (gitignored)
├── pyproject.toml
└── README.md
```

---

## Research Tracks

See the [Parallel Testing Plan](docs/KICKOFF.md#parallel-testing-plan-aiuga-undergrad-team) for full details.

| Track | Focus | People |
|-------|-------|--------|
| **A** | SE-DiCoW evaluation (top priority) | 2 |
| **B** | ASR model comparison | 1-2 |
| **C** | Speaker verification benchmarking | 1 |
| **D** | Test data creation | Everyone |

### Getting Started

1. Read [`docs/KICKOFF.md`](docs/KICKOFF.md) — the full project context
2. Pick a track
3. Set up your Python environment (see Quick Start above)
4. Start with Track D (record yourself!) while models download

---

## Recording Test Audio

Everyone should contribute test recordings. See Track D in the kickoff doc, but briefly:

1. **Enrollment clip** (10s): Read a paragraph aloud in a quiet room
2. **Clean speech clips** (5-10s each): Just you talking, record the ground-truth transcript
3. **Overlap clips** (5-10s each): You talking while someone else talks nearby

Put recordings in `tests/fixtures/` with a naming convention:
```
tests/fixtures/
├── speaker1/
│   ├── enrollment.wav
│   ├── clean_01.wav
│   ├── clean_01.txt          # ground truth transcript
│   ├── overlap_01.wav
│   └── overlap_01.txt
├── speaker2/
│   └── ...
└── mixed/                    # Programmatically mixed audio
    └── ...
```

Use `scripts/mix_audio.py` to create controlled overlap test data:
```bash
python scripts/mix_audio.py --target speaker1/clean_01.wav --interference speaker2/clean_01.wav --sir 20
```

---

## Reporting Results

When you have evaluation results, create a markdown file in `docs/results/`:

```markdown
# [Model Name] Evaluation — [Your Name], [Date]

## Setup
- Model: ...
- Hardware: ...
- Test audio: ...

## Results
| Metric | Clean | Noisy (20dB SIR) | Noisy (10dB SIR) |
|--------|-------|-------------------|-------------------|
| WER    |       |                   |                   |

## Notes
What worked, what didn't, surprises, next steps.
```

Negative results are valuable. "We tried X and it failed because Y" saves everyone time.

---

## Contributing

- Create a branch for your work: `git checkout -b track-a/your-name`
- Commit regularly, push daily
- Ask questions — nobody has done all of this before
- Write tests before writing pipeline code
