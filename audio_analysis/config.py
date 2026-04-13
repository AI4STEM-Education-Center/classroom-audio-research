"""
config.py — Shared constants, settings, and optional-dependency imports.

Every other module in audio_analysis imports from here.  Nothing in this
module imports from any sibling module, so there are no circular dependencies.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Path bootstrap ────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import openai as _openai_lib
    _OPENAI_AVAILABLE = True
except ImportError:
    _openai_lib = None  # type: ignore
    _OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv as _load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _load_dotenv = None  # type: ignore
    _DOTENV_AVAILABLE = False

try:
    import noisereduce as _noisereduce_lib
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    _noisereduce_lib = None  # type: ignore
    _NOISEREDUCE_AVAILABLE = False

try:
    import docx as _docx_lib
    _DOCX_AVAILABLE = True
except ImportError:
    _docx_lib = None  # type: ignore
    _DOCX_AVAILABLE = False

# ── audio_quality_check (sibling package) ────────────────────────────────────

try:
    from audio_quality_check import (
        TARGET_SR,
        ChannelReport,
        analyze_channel,
        compute_bandwidth,
    )
    _AQC_AVAILABLE = True
    _AQC_ERROR = ""
except ImportError as _imp_err:
    _AQC_AVAILABLE = False
    _AQC_ERROR = str(_imp_err)
    TARGET_SR = 22050
    ChannelReport = None  # type: ignore

    def compute_bandwidth(S_db, freqs, peak_db, drop_db):  # type: ignore
        """Fallback: highest frequency bin within drop_db of peak."""
        mean_spectrum = np.mean(S_db, axis=1)
        above = np.where(mean_spectrum >= peak_db - drop_db)[0]
        return float(freqs[above[-1]]) if len(above) > 0 else 0.0

    def analyze_channel(y, sr, ch_index):  # type: ignore
        return None

# ── Server defaults ───────────────────────────────────────────────────────────

DEFAULT_PORT = 5050
DEFAULT_RECORDINGS_DIR = _HERE.parent / "audio-capture-server" / "recordings"

# ── Audio analysis constants ──────────────────────────────────────────────────

ANALYSIS_SR = TARGET_SR          # 22050 Hz — matches audio_quality_check
FRAME_MS    = 30                 # diarization frame size (ms)
HOP_MS      = 10                 # diarization hop (ms)
SILENCE_DB_OFFSET = 10           # dB above noise floor → frame is "active speech"

# ── Diarization settings ──────────────────────────────────────────────────────

DIARIZATION_SMOOTH_K = 5         # median filter kernel (frames)
MIN_SEGMENT_FRAMES   = 3         # minimum run before a segment is emitted
DOMINANT_RATIO       = 0.65      # ratio threshold for "dominant" speaker (legacy)
BLEED_FACTOR         = 0.30      # M3/M2+M3: per-dB threshold raise when other mic is loud
SILERO_THRESHOLD     = 0.50      # M4/M2+M4: Silero speech-probability threshold (0–1)
SILERO_SR            = 16000     # Silero VAD requires 16 kHz input

# ── Diarization label constants ───────────────────────────────────────────────

L_SILENCE = "SILENCE"
L_A       = "SPEAKER_A"
L_B       = "SPEAKER_B"
L_A_DOM   = "SPEAKER_A_DOMINANT"
L_B_DOM   = "SPEAKER_B_DOMINANT"
L_OVERLAP = "OVERLAP"

LABEL_COLORS = {
    L_SILENCE: "#9E9E9E",
    L_A:       "#1976D2",
    L_B:       "#E65100",
    L_A_DOM:   "#42A5F5",
    L_B_DOM:   "#FF8A65",
    L_OVERLAP: "#7B1FA2",
}

# ── Transcription settings ────────────────────────────────────────────────────

MIN_TRANSCRIBE_SEC = 0.5    # skip segments shorter than this (too short for Whisper)
MIN_MERGE_GAP_SEC  = 0.6    # merge same-speaker segments with gap smaller than this
CROSSTALK_ALPHA      = 0.3    # fraction of the other mic to subtract (simple subtraction)
WHISPER_MODEL        = "whisper-1"

# ── Multi-channel enhancement settings ───────────────────────────────────────

ENHANCE_BETA          = 0.9    # Wiener suppression aggressiveness (0=none, 1=full)
ENHANCE_SMOOTH_FRAMES = 10     # temporal smoothing of power estimates (reduces musical noise)
ENHANCE_N_FFT         = 2048   # STFT window size for enhancement
ENHANCE_HOP           = 512    # STFT hop size for enhancement
