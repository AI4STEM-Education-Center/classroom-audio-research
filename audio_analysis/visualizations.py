"""
visualizations.py — Matplotlib plot generation for the analysis dashboard.

All functions return base64-encoded PNG data-URIs suitable for embedding in HTML.

Functions:
  _fig_to_base64(fig) -> str
  plot_waveform_overlay(y_a, y_b, sr) -> str
  plot_spectrograms(y_a, y_b, sr) -> str
  plot_enhancement_comparison(y_a, y_b, y_a_clean, y_b_clean, sr, name_a, name_b) -> str
  plot_diarization_timeline(diarization, duration_sec, name_a, name_b) -> str
  plot_energy_ratio(diarization) -> str
"""

import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

from config import (
    ENHANCE_N_FFT,
    ENHANCE_HOP,
    LABEL_COLORS,
    L_A, L_B, L_OVERLAP,
)
from vad import _vad_label


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG data-URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return "data:image/png;base64," + encoded


def plot_waveform_overlay(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(12, 3))
    t = np.arange(len(y_a)) / sr
    ax.plot(t, y_a, linewidth=0.4, color="#1976D2", alpha=0.85, label="Speaker A")
    ax.plot(t, y_b, linewidth=0.4, color="#E65100", alpha=0.75, label="Speaker B")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform Overlay — Both Microphones")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_spectrograms(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    img = None
    for ax, y, title, col in zip(
        axes,
        [y_a, y_b],
        ["Spectrogram — Speaker A", "Spectrogram — Speaker B"],
        ["#1976D2", "#E65100"],
    ):
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max
        )
        img = librosa.display.specshow(
            S, sr=sr, hop_length=512, x_axis="time", y_axis="hz",
            ax=ax, cmap="magma",
        )
        ax.set_title(title, color=col, fontweight="bold")
        ax.set_ylim(0, min(sr / 2, 10000))
    if img is not None:
        fig.colorbar(img, ax=axes, format="%+2.0f dB", pad=0.01)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_enhancement_comparison(
    y_a: np.ndarray, y_b: np.ndarray,
    y_a_clean: np.ndarray, y_b_clean: np.ndarray,
    sr: int,
    name_a: str = "Speaker A",
    name_b: str = "Speaker B",
) -> str:
    """
    2×2 spectrogram grid showing original vs. Wiener-enhanced audio for each speaker.
    Shared colorbar and consistent dB range so bleed reduction is visually apparent.
    """
    n_fft, hop = ENHANCE_N_FFT, ENHANCE_HOP
    cols = ["#1976D2", "#E65100"]

    def _stft_db(y):
        return librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)), ref=np.max
        )

    panels = [
        (_stft_db(y_a),       f"{name_a} — original", cols[0]),
        (_stft_db(y_a_clean), f"{name_a} — enhanced", cols[0]),
        (_stft_db(y_b),       f"{name_b} — original", cols[1]),
        (_stft_db(y_b_clean), f"{name_b} — enhanced", cols[1]),
    ]

    all_vals = np.concatenate([S.ravel() for S, _, _ in panels])
    vmin, vmax = float(np.percentile(all_vals, 5)), float(np.max(all_vals))

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True, sharey=True)
    img = None
    for ax, (S, title, col) in zip(axes.ravel(), panels):
        img = librosa.display.specshow(
            S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz",
            ax=ax, cmap="magma", vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, color=col, fontweight="bold", fontsize=9)
        ax.set_ylim(0, min(sr / 2, 8000))

    if img is not None:
        fig.colorbar(img, ax=axes, format="%+2.0f dB", pad=0.01)

    fig.suptitle(
        "Audio Enhancement — Wiener Cross-Channel Bleed Suppression",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_diarization_timeline(diarization: dict, duration_sec: float,
                               name_a: str = "Speaker A",
                               name_b: str = "Speaker B") -> str:
    """
    Two-lane timeline: one horizontal bar per speaker (A on top, B on bottom).
    OVERLAP segments are drawn on both lanes in purple.
    """
    from matplotlib.patches import Patch
    segs    = diarization.get("segments", [])
    summary = diarization.get("summary",  {})
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(12, 3.4), sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    for seg in segs:
        lbl   = seg["label"]
        start = seg["start_sec"]
        dur   = seg["end_sec"] - seg["start_sec"]
        if lbl in (L_A, L_OVERLAP):
            col = LABEL_COLORS[L_A] if lbl == L_A else LABEL_COLORS[L_OVERLAP]
            ax_a.barh(0, dur, left=start, height=0.6, color=col,
                      alpha=0.88, edgecolor="white", linewidth=0.3)
        if lbl in (L_B, L_OVERLAP):
            col = LABEL_COLORS[L_B] if lbl == L_B else LABEL_COLORS[L_OVERLAP]
            ax_b.barh(0, dur, left=start, height=0.6, color=col,
                      alpha=0.88, edgecolor="white", linewidth=0.3)

    for ax, name, col in [(ax_a, name_a, LABEL_COLORS[L_A]),
                           (ax_b, name_b, LABEL_COLORS[L_B])]:
        ax.set_xlim(0, max(duration_sec, 1))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_facecolor("#F5F5F5")
        ax.set_ylabel(name, color=col, fontsize=9, fontweight="bold",
                      rotation=0, labelpad=65, va="center")

    ax_a.set_title("Speaker Activity — Two-Lane View")
    ax_b.set_xlabel("Time (s)")

    handles = [
        Patch(color=LABEL_COLORS[L_A],       label=f"{name_a} only"),
        Patch(color=LABEL_COLORS[L_B],       label=f"{name_b} only"),
        Patch(color=LABEL_COLORS[L_OVERLAP], label="Overlap (both)"),
    ]
    ax_a.legend(handles=handles, loc="upper right", fontsize=7.5, ncol=3)

    txt = (
        f"{name_a}: {summary.get('a_total_pct', 0):.1f}% total  "
        f"{name_b}: {summary.get('b_total_pct', 0):.1f}% total  "
        f"Overlap: {summary.get('both_pct', 0):.1f}%  "
        f"Overlap ratio: {summary.get('overlap_ratio', 0):.1f}%  "
        f"Silence: {summary.get('silence_pct', 0):.1f}%"
    )
    ax_b.text(0.0, -0.38, txt, transform=ax_b.transAxes, fontsize=8, color="#333",
              verticalalignment="bottom")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_energy_ratio(diarization: dict) -> str:
    frames = diarization.get("frames", [])
    if not frames:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, "No frames available", ha="center", transform=ax.transAxes)
        return _fig_to_base64(fig)

    times  = np.array([f["time_sec"]    for f in frames])
    ea     = np.array([f["energy_a_db"] for f in frames])
    eb     = np.array([f["energy_b_db"] for f in frames])
    vad_a  = np.array([float(f["vad_a"]) for f in frames])
    vad_b  = np.array([float(f["vad_b"]) for f in frames])
    labels = [_vad_label(bool(f["vad_a"]), bool(f["vad_b"])) for f in frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for i in range(len(frames) - 1):
        ax1.axvspan(times[i], times[i + 1], color=LABEL_COLORS.get(labels[i], "#888"),
                    alpha=0.12, linewidth=0)

    ax1.plot(times, ea, linewidth=0.8, color="#1976D2", label="Speaker A")
    ax1.plot(times, eb, linewidth=0.8, color="#E65100", label="Speaker B")
    ax1.set_ylabel("Energy (dB)")
    ax1.set_title("Frame-wise Energy & VAD Activity")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.fill_between(times, vad_a * 0.45 + 0.55, 0.55, alpha=0.75, color=LABEL_COLORS[L_A])
    ax2.fill_between(times, vad_b * 0.45,         0.0,  alpha=0.75, color=LABEL_COLORS[L_B])
    ax2.axhline(0.5, color="#CCC", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("VAD")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.28, 0.78])
    ax2.set_yticklabels(["B", "A"], fontsize=8)
    ax2.set_title("VAD Activity (per speaker)")

    fig.tight_layout()
    return _fig_to_base64(fig)
