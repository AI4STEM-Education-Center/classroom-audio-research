#!/usr/bin/env python3
"""
ArguAgent Audio Analysis Server
================================
Local Flask server for analyzing parallel microphone recordings from classroom
argumentation sessions.

Provides:
  - SNR estimation per microphone (percentile-energy method)
  - Bandwidth analysis (3 dB / 10 dB / band distribution)
  - Energy-based speaker diarization across two parallel mic streams
  - Browser dashboard with inline visualizations

Usage:
    python analysis_server.py [--port 5050] [--recordings-dir PATH] [--debug]

Requires: flask, flask-cors, numpy, scipy, librosa, soundfile, matplotlib
"""

import argparse
import datetime
import os
import shutil
import traceback
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ── Internal modules ──────────────────────────────────────────────────────────

from config import (
    DEFAULT_PORT,
    DEFAULT_RECORDINGS_DIR,
    ANALYSIS_SR,
    _HERE,
    _AQC_AVAILABLE,
    _AQC_ERROR,
    _DOTENV_AVAILABLE,
    _load_dotenv,
)
from sessions import list_sessions, find_session_pair
from orchestrator import run_full_analysis, _json_safe
from pipeline_full import run_transcription
from pipeline_raw import run_transcription_raw
from pipeline_adaptive import run_transcription_adaptive
from pipeline_aec import run_transcription_aec
from pipeline_vad import run_transcription_vad
from pipeline_ratiovat import run_transcription_ratiovat
from pipeline_mvp import run_transcription_mvp
from pipeline_reftext import run_transcription_reftext
from pipeline_vad_chunked import run_transcription_vad_chunked

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

_config: dict = {
    "recordings_dir": DEFAULT_RECORDINGS_DIR,
    "port":           DEFAULT_PORT,
}
_openai_key: str = ""   # stored in memory only, never logged or persisted

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = Path(__file__).parent / "dashboard.html"
    return Response(html_path.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/api/sessions")
def api_sessions():
    rdir = Path(_config["recordings_dir"])
    return jsonify({
        "sessions":       list_sessions(rdir),
        "recordings_dir": str(rdir),
    })


@app.route("/api/analyze/session", methods=["POST"])
def api_analyze_session():
    body  = request.get_json(force=True) or {}
    spath = body.get("session_path", "").strip()
    if not spath:
        return jsonify({"error": "session_path is required"}), 400
    tag_a = body.get("tag_a", "").strip()
    tag_b = body.get("tag_b", "").strip()
    return _run_session(spath, tag_a=tag_a, tag_b=tag_b)


@app.route("/api/analyze/session/<session_name>")
def api_analyze_session_name(session_name: str):
    rdir = Path(_config["recordings_dir"])
    return _run_session(str(rdir / session_name))


def _run_session(session_path: str, tag_a: str = "", tag_b: str = ""):
    d = Path(session_path)
    if not d.exists():
        return jsonify({"error": f"Session directory not found: {session_path}"}), 404
    pair = find_session_pair(d, tag_a=tag_a, tag_b=tag_b)
    if not pair["headset_wav"] or not pair["array_wav"]:
        return jsonify({
            "error":    "Could not find a WAV pair in this session.",
            "warnings": pair["warnings"],
            "hint":     "The session needs WAVs from two microphones (matched by deviceLabel "
                        "'Microphone Array'/'Mikrofon', or by speaker 'tag'/'source' in the meta files).",
        }), 422
    try:
        result = run_full_analysis(
            pair["headset_wav"], pair["array_wav"],
            pair["headset_meta"], pair["array_meta"],
        )
        result["warnings"] = pair["warnings"] + result.get("warnings", [])
        return jsonify(_json_safe(result))
    except Exception as exc:
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


@app.route("/api/analyze/files", methods=["POST"])
def api_analyze_files():
    body = request.get_json(force=True) or {}
    fa = body.get("file_a", "").strip()
    fb = body.get("file_b", "").strip()
    if not fa or not fb:
        return jsonify({"error": "file_a and file_b are required"}), 400
    pa, pb = Path(fa), Path(fb)
    if not pa.exists():
        return jsonify({"error": f"file_a not found: {fa}"}), 404
    if not pb.exists():
        return jsonify({"error": f"file_b not found: {fb}"}), 404
    try:
        meta_a = {"wav_file": pa.name, "deviceLabel": "Headset (file_a)"}
        meta_b = {"wav_file": pb.name, "deviceLabel": "Microphone Array (file_b)"}
        result = run_full_analysis(pa, pb, meta_a, meta_b)
        return jsonify(_json_safe(result))
    except Exception as exc:
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


# ── Transcription routes ──────────────────────────────────────────────────────

@app.route("/api/config/key", methods=["POST"])
def api_set_key():
    global _openai_key
    body = request.get_json(force=True) or {}
    key  = body.get("api_key", "").strip()
    if not key:
        return jsonify({"error": "api_key required"}), 400
    _openai_key = key
    return jsonify({"ok": True, "hint": f"Key stored (ends: …{key[-4:]})"})


@app.route("/api/transcribe/session", methods=["POST"])
def api_transcribe_session():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({
            "error": "No OpenAI API key. "
                     "Set it via /api/config/key or include api_key in the request body."
        }), 400
    spath = body.get("session_path", "").strip()
    if not spath:
        return jsonify({"error": "session_path required"}), 400
    tag_a = body.get("tag_a", "").strip()
    tag_b = body.get("tag_b", "").strip()
    return _run_transcription_session(spath, api_key, tag_a=tag_a, tag_b=tag_b)


@app.route("/api/transcribe/files", methods=["POST"])
def api_transcribe_files():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    fa, fb = body.get("file_a", "").strip(), body.get("file_b", "").strip()
    if not fa or not fb:
        return jsonify({"error": "file_a and file_b required"}), 400
    pa, pb = Path(fa), Path(fb)
    if not pa.exists(): return jsonify({"error": f"file_a not found: {fa}"}), 404
    if not pb.exists(): return jsonify({"error": f"file_b not found: {fb}"}), 404
    try:
        meta_a = {"wav_file": pa.name, "deviceLabel": "Headset (file_a)"}
        meta_b = {"wav_file": pb.name, "deviceLabel": "Microphone Array (file_b)"}
        result = run_transcription(pa, pb, meta_a, meta_b, api_key)
        return jsonify(_json_safe(result))
    except Exception as exc:
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


def _transcribe_pair(body: dict, pipeline_fn, api_key: str):
    """Shared helper: resolve session pair + call a pipeline function."""
    spath = body.get("session_path", "").strip()
    if not spath:
        return jsonify({"error": "session_path required"}), 400
    d = Path(spath)
    if not d.exists():
        return jsonify({"error": f"Session directory not found: {spath}"}), 404
    tag_a = body.get("tag_a", "").strip()
    tag_b = body.get("tag_b", "").strip()
    pair  = find_session_pair(d, tag_a=tag_a, tag_b=tag_b)
    if not pair["headset_wav"] or not pair["array_wav"]:
        return jsonify({"error": "Could not find WAV pair.", "warnings": pair["warnings"]}), 422
    try:
        result = pipeline_fn(
            pair["headset_wav"], pair["array_wav"],
            pair["headset_meta"], pair["array_meta"],
            api_key,
        )
        return jsonify(_json_safe(result))
    except Exception as exc:
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


@app.route("/api/transcribe/session/raw", methods=["POST"])
def api_transcribe_session_raw():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key. Set it via /api/config/key."}), 400
    return _transcribe_pair(body, run_transcription_raw, api_key)


@app.route("/api/transcribe/session/adaptive", methods=["POST"])
def api_transcribe_session_adaptive():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key. Set it via /api/config/key."}), 400
    return _transcribe_pair(body, run_transcription_adaptive, api_key)


@app.route("/api/transcribe/session/aec", methods=["POST"])
def api_transcribe_session_aec():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    return _transcribe_pair(body, run_transcription_aec, api_key)


@app.route("/api/transcribe/session/vad", methods=["POST"])
def api_transcribe_session_vad():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    return _transcribe_pair(body, run_transcription_vad, api_key)


@app.route("/api/transcribe/session/ratiovat", methods=["POST"])
def api_transcribe_session_ratiovat():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    return _transcribe_pair(body, run_transcription_ratiovat, api_key)


@app.route("/api/transcribe/session/mvp", methods=["POST"])
def api_transcribe_session_mvp():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    return _transcribe_pair(body, run_transcription_mvp, api_key)


@app.route("/api/transcribe/session/reftext", methods=["POST"])
def api_transcribe_session_reftext():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    return _transcribe_pair(body, run_transcription_reftext, api_key)


@app.route("/api/transcribe/session/vadchunked", methods=["POST"])
def api_transcribe_session_vadchunked():
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key."}), 400
    return _transcribe_pair(body, run_transcription_vad_chunked, api_key)


PIPELINE_ORDER  = ["raw", "full", "adaptive", "aec", "vad", "ratiovat", "mvp", "reftext", "vadchunked"]
PIPELINE_LABELS = {
    "raw":        "Aligned only",
    "full":       "Full pipeline",
    "adaptive":   "Adaptive",
    "aec":        "AEC",
    "vad":        "VAD-gated",
    "ratiovat":   "Ratio-VAD+AEC",
    "mvp":        "MVP",
    "reftext":    "Ref-Text",
    "vadchunked": "VAD-chunked",
}


def _build_markdown_report(results: dict, session_dir: Path,
                            pair: dict, ts: str) -> str:
    """Build a comprehensive LLM-readable markdown comparison report."""
    import re
    lines = []
    now = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"
    def cnt(v):
        return str(v) if v is not None else "—"

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Pipeline Comparison Report", "",
        f"**Session:** `{session_dir.name}`  ",
        f"**Generated:** {now}  ",
    ]
    wav_a = pair.get("headset_wav")
    wav_b = pair.get("array_wav")
    if wav_a: lines.append(f"**File A (headset):** `{Path(wav_a).name}`  ")
    if wav_b: lines.append(f"**File B (array):**   `{Path(wav_b).name}`  ")
    lines.append("")

    # ── Derive shared metadata ────────────────────────────────────────────────
    s1name, s2name, has_truth = "Speaker A", "Speaker B", False
    for k in PIPELINE_ORDER:
        r = results.get(k)
        if r and not r.get("error") and r.get("evaluation"):
            ev = r["evaluation"]
            has_truth = bool(ev.get("ground_truth_available"))
            if has_truth:
                s1name = ev.get("s1_speaker", ev.get("name_a", s1name))
                s2name = ev.get("s2_speaker", ev.get("name_b", s2name))
            break

    # ── Bleed measurements (structured, from insights if available) ───────────
    lines += ["## Recording Characteristics", ""]
    bleed_found = False
    for k in ("reftext", "mvp", "ratiovat", "adaptive"):
        r = results.get(k)
        if not r or r.get("error"):
            continue
        ins = r.get("insights") or {}
        if ins.get("clean_channel") or ins.get("clean_bleed") is not None:
            clean_ch   = ins.get("clean_channel", "?")
            noisy_ch   = ins.get("noisy_channel", "?")
            clean_bl   = ins.get("clean_bleed")
            noisy_bl   = ins.get("noisy_bleed")
            asymmetry  = ins.get("asymmetry")
            td_applied = ins.get("text_diff_applied")
            lines.append(f"- **Clean channel:** {clean_ch} (bleed into it: {pct(clean_bl)})")
            lines.append(f"- **Noisy channel:** {noisy_ch} (bleed into it: {pct(noisy_bl)})")
            if asymmetry is not None:
                lines.append(f"- **Bleed asymmetry:** {pct(asymmetry)}"
                             + (" — text-diff active" if td_applied else " — text-diff SKIPPED (symmetric)"))
            bleed_found = True
            break
    if not bleed_found:
        # Fall back to regex on pipeline string
        for k in ("adaptive", "ratiovat"):
            r = results.get(k)
            if r and not r.get("error"):
                hits = re.findall(r"bleed[^|,\n]{0,80}", r.get("pipeline", ""))
                if hits:
                    for h in hits:
                        lines.append(f"- {h.strip()}")
                    bleed_found = True
                    break
    if not bleed_found:
        lines.append("_Bleed measurements not available._")

    # VAD frame breakdown from ratiovat or mvp
    for k in ("ratiovat", "mvp"):
        r = results.get(k)
        if not r or r.get("error"):
            continue
        pl = r.get("pipeline", "")
        m_overlap  = re.search(r"overlap: (\d+) frames", pl)
        m_silence  = re.search(r"silence: (\d+) frames", pl)
        if m_overlap:
            lines.append("")
            lines.append(f"**VAD frame breakdown** (from {PIPELINE_LABELS.get(k, k)}):")
            # Extract all solo-X: N frames
            solos = re.findall(r"solo-([^:]+): (\d+) frames", pl)
            for spk, n in solos:
                lines.append(f"- solo {spk}: {n} frames")
            lines.append(f"- overlap: {m_overlap.group(1)} frames")
            if m_silence:
                lines.append(f"- silence: {m_silence.group(1)} frames")
        break
    lines.append("")

    # ── WER summary table ─────────────────────────────────────────────────────
    lines += ["## WER Summary", ""]
    if has_truth:
        lines.append(f"| Pipeline | Overall WER | Harmonic WER | WER {s1name} | WER {s2name} | CER | Segs | Sim. segs |")
        lines.append( "| --- | --- | --- | --- | --- | --- | --- | --- |")
    else:
        lines.append("| Pipeline | Segs | Note |")
        lines.append("| --- | --- | --- |")

    for k in PIPELINE_ORDER:
        r = results.get(k)
        label = PIPELINE_LABELS.get(k, k)
        if r is None:
            continue
        if r.get("error"):
            cols = "| ERROR | — | — | — | — | — | — |" if has_truth else "| — | ERROR |"
            lines.append(f"| {label} {cols}")
            continue
        ev   = r.get("evaluation") or {}
        ov   = ev.get("overall")   or {}
        sa   = ev.get("speaker_a") or {}
        sb   = ev.get("speaker_b") or {}
        segs = r.get("segments_transcribed", "—")
        transcript = r.get("transcript") or []
        n_sim = sum(1 for seg in transcript if seg.get("simultaneous"))
        if has_truth:
            lines.append(
                f"| {label} | {pct(ov.get('wer'))} | {pct(ev.get('wer_harmonic'))} "
                f"| {pct(sa.get('wer'))} | {pct(sb.get('wer'))} "
                f"| {pct(ev.get('cer_overall'))} | {segs} | {n_sim} |"
            )
        else:
            lines.append(f"| {label} | {segs} | no ground truth |")
    lines.append("")

    if has_truth:
        lines += [
            "**Error type legend:**",
            "- **Insertions** = extra words said that aren't in ground truth (bleed from other speaker is a common cause)",
            "- **Deletions** = words in ground truth missing from transcript (over-filtering, gating, or Whisper skip)",
            "- **Substitutions** = wrong word transcribed",
            "",
        ]

    # ── Per-pipeline detail ───────────────────────────────────────────────────
    lines += ["## Per-Pipeline Details", ""]

    for k in PIPELINE_ORDER:
        r = results.get(k)
        label = PIPELINE_LABELS.get(k, k)
        lines.append(f"### {label}")
        lines.append("")
        if r is None:
            lines += ["_Not run._", ""]
            continue
        if r.get("error"):
            lines.append(f"**ERROR:** {r['error']}")
            if r.get("detail"):
                lines += ["", "```", r["detail"], "```"]
            lines.append("")
            continue

        ev     = r.get("evaluation") or {}
        ov     = ev.get("overall")   or {}
        sa     = ev.get("speaker_a") or {}
        sb     = ev.get("speaker_b") or {}
        segs   = r.get("segments_transcribed", "—")
        offset = r.get("sync_offset_sec")
        pl     = r.get("pipeline", "")
        transcript = r.get("transcript") or []
        n_sim  = sum(1 for seg in transcript if seg.get("simultaneous"))

        lines.append(f"**Pipeline note:** `{pl}`  ")
        lines.append(f"**Segments transcribed:** {segs} ({n_sim} flagged simultaneous)  ")
        if offset is not None:
            lines.append(f"**Sync offset:** {offset} s  ")
        lines.append("")

        if has_truth:
            # Speaker mapping note
            s1spk = ev.get("s1_speaker", "?")
            s2spk = ev.get("s2_speaker", "?")
            lines.append(f"_Ground truth mapping: S1 → {s1spk}, S2 → {s2spk}_")
            lines.append("")

            lines.append(f"| Metric | Overall | {s1name} | {s2name} |")
            lines.append( "| --- | --- | --- | --- |")
            lines.append(f"| WER            | {pct(ov.get('wer'))}            | {pct(sa.get('wer'))}            | {pct(sb.get('wer'))}            |")
            lines.append(f"| CER            | {pct(ev.get('cer_overall'))}    | —                               | —                               |")
            lines.append(f"| Ref words      | {cnt(ov.get('ref_words'))}      | {cnt(sa.get('ref_words'))}      | {cnt(sb.get('ref_words'))}      |")
            lines.append(f"| Hyp words      | {cnt(ov.get('hyp_words'))}      | {cnt(sa.get('hyp_words'))}      | {cnt(sb.get('hyp_words'))}      |")
            lines.append(f"| Insertions     | {cnt(ov.get('insertions'))}     | {cnt(sa.get('insertions'))}     | {cnt(sb.get('insertions'))}     |")
            lines.append(f"| Deletions      | {cnt(ov.get('deletions'))}      | {cnt(sa.get('deletions'))}      | {cnt(sb.get('deletions'))}      |")
            lines.append(f"| Substitutions  | {cnt(ov.get('substitutions'))}  | {cnt(sa.get('substitutions'))}  | {cnt(sb.get('substitutions'))}  |")
            lines.append("")

            # Word-level diffs — most important for LLM analysis
            for stats, title in [
                (ov, "Overall word-level diff"),
                (sa, f"{s1name} word-level diff"),
                (sb, f"{s2name} word-level diff"),
            ]:
                diff = (stats.get("diff_plain") or "").strip()
                if diff:
                    lines.append(f"<details>")
                    lines.append(f"<summary>{title} (correct / {{EXTRA}} / [MISSING])</summary>")
                    lines.append("")
                    lines.append("```")
                    lines.append(diff)
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

        # Pipeline-specific insights
        ins = r.get("insights")
        if ins:
            lines.append("**Pipeline insights:**")
            lines.append("")
            for ik, iv in ins.items():
                lines.append(f"- `{ik}`: {iv}")
            lines.append("")

        # Full transcript
        formatted = r.get("formatted", "").strip()
        if formatted:
            lines.append("<details>")
            lines.append("<summary>Full transcript</summary>")
            lines.append("")
            lines.append("```")
            lines.append(formatted)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
        lines.append("")

    # ── Diagnostic summary for LLM ────────────────────────────────────────────
    if has_truth:
        lines += ["## Diagnostic Summary", "",
                  "The following observations are intended to help an LLM diagnose pipeline failures.", ""]

        # Rank pipelines by harmonic-mean WER (falls back to overall if unavailable)
        def _rank_wer(k):
            ev = (results[k].get("evaluation") or {})
            h = ev.get("wer_harmonic")
            return h if h is not None else (ev.get("overall") or {}).get("wer", 999)

        ranked = sorted(
            [(k, _rank_wer(k))
             for k in PIPELINE_ORDER
             if results.get(k) and not results[k].get("error")
             and (results[k].get("evaluation") or {}).get("ground_truth_available")],
            key=lambda x: x[1],
        )
        if ranked:
            lines.append("**Pipeline ranking by harmonic-mean WER (best first):**")
            lines.append("_(Harmonic mean of per-speaker WERs. Prevents pipelines that drop one speaker from appearing best.)_")
            lines.append("")
            for rank, (k, wer) in enumerate(ranked, 1):
                lines.append(f"{rank}. {PIPELINE_LABELS.get(k, k)}: {pct(wer)}")
            lines.append("")

        # Per-speaker best
        for spk_key, spk_name in [("speaker_a", s1name), ("speaker_b", s2name)]:
            best_k, best_wer = None, float("inf")
            for k in PIPELINE_ORDER:
                r = results.get(k)
                if not r or r.get("error"): continue
                w = (r.get("evaluation") or {}).get(spk_key, {}).get("wer")
                if w is not None and w < best_wer:
                    best_wer, best_k = w, k
            if best_k:
                lines.append(f"**Best for {spk_name}:** {PIPELINE_LABELS.get(best_k, best_k)} — {pct(best_wer)}")
        lines.append("")

        # Insertion pattern analysis
        lines.append("**Insertion counts per pipeline (high insertions = bleed not removed):**")
        lines.append("")
        for k in PIPELINE_ORDER:
            r = results.get(k)
            if not r or r.get("error"): continue
            ev = r.get("evaluation") or {}
            if not ev.get("ground_truth_available"): continue
            ins_a = (ev.get("speaker_a") or {}).get("insertions")
            ins_b = (ev.get("speaker_b") or {}).get("insertions")
            lines.append(f"- {PIPELINE_LABELS.get(k, k)}: {s1name}={cnt(ins_a)} insertions, {s2name}={cnt(ins_b)} insertions")
        lines.append("")

    return "\n".join(lines)


@app.route("/api/transcribe/session/all", methods=["POST"])
def api_transcribe_session_all():
    """Run all pipelines in parallel and return results keyed by pipeline name."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    body    = request.get_json(force=True) or {}
    api_key = body.get("api_key", "").strip() or _openai_key
    if not api_key:
        return jsonify({"error": "No OpenAI API key. Set it via /api/config/key."}), 400
    spath = body.get("session_path", "").strip()
    if not spath:
        return jsonify({"error": "session_path required"}), 400
    d = Path(spath)
    if not d.exists():
        return jsonify({"error": f"Session directory not found: {spath}"}), 404
    tag_a = body.get("tag_a", "").strip()
    tag_b = body.get("tag_b", "").strip()
    pair  = find_session_pair(d, tag_a=tag_a, tag_b=tag_b)
    if not pair["headset_wav"] or not pair["array_wav"]:
        return jsonify({"error": "Could not find WAV pair.", "warnings": pair["warnings"]}), 422

    pipelines = {
        "raw":        run_transcription_raw,
        "full":       run_transcription,
        "adaptive":   run_transcription_adaptive,
        "aec":        run_transcription_aec,
        "vad":        run_transcription_vad,
        "ratiovat":   run_transcription_ratiovat,
        "mvp":        run_transcription_mvp,
        "reftext":    run_transcription_reftext,
        "vadchunked": run_transcription_vad_chunked,
    }

    def _run(name, fn):
        try:
            return name, fn(
                pair["headset_wav"], pair["array_wav"],
                pair["headset_meta"], pair["array_meta"],
                api_key,
            )
        except Exception as exc:
            return name, {"error": str(exc), "detail": traceback.format_exc()}

    results = {}
    with ThreadPoolExecutor(max_workers=9) as ex:
        futures = {ex.submit(_run, name, fn): name for name, fn in pipelines.items()}
        for f in as_completed(futures):
            name, res = f.result()
            results[name] = res

    # ── Auto-export markdown report ───────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = d / f"pipeline_comparison_{ts}.md"
    try:
        report_path.write_text(
            _build_markdown_report(results, d, pair, ts),
            encoding="utf-8",
        )
        results["_report_path"] = str(report_path)
    except Exception as exc:
        results["_report_warning"] = f"Could not write report: {exc}"

    return jsonify(_json_safe(results))


def _run_transcription_session(session_path: str, api_key: str,
                                tag_a: str = "", tag_b: str = ""):
    d = Path(session_path)
    if not d.exists():
        return jsonify({"error": f"Session directory not found: {session_path}"}), 404
    pair = find_session_pair(d, tag_a=tag_a, tag_b=tag_b)
    if not pair["headset_wav"] or not pair["array_wav"]:
        return jsonify({
            "error":    "Could not find WAV pair.",
            "warnings": pair["warnings"],
        }), 422
    try:
        result = run_transcription(
            pair["headset_wav"], pair["array_wav"],
            pair["headset_meta"], pair["array_meta"],
            api_key,
        )
        return jsonify(_json_safe(result))
    except Exception as exc:
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ArguAgent Audio Analysis Server")
    parser.add_argument("--port",           type=int, default=DEFAULT_PORT)
    parser.add_argument("--recordings-dir", type=str, default=str(DEFAULT_RECORDINGS_DIR))
    parser.add_argument("--env-file",       type=str, default="",
                        help="Path to .env file containing OPENAI_API_KEY")
    parser.add_argument("--debug",          action="store_true")
    args = parser.parse_args()

    _config["port"]           = args.port
    _config["recordings_dir"] = Path(args.recordings_dir)

    # ── Load API key from .env ────────────────────────────────────────────────
    global _openai_key
    env_candidates = [
        Path(args.env_file) if args.env_file else None,
        _HERE / ".env",
        _HERE.parent / ".env",
    ]
    for env_path in env_candidates:
        if env_path and env_path.exists():
            if _DOTENV_AVAILABLE:
                _load_dotenv(env_path, override=False)
            else:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break

    if not _openai_key:
        _openai_key = os.environ.get("OPENAI_API_KEY", "")

    print("=" * 62)
    print("  ArguAgent Audio Analysis Server")
    print("=" * 62)

    rdir = Path(args.recordings_dir)
    if rdir.exists():
        n = len(list_sessions(rdir))
        print(f"  Recordings dir : {rdir}")
        print(f"  Sessions found : {n}")
    else:
        print(f"  WARNING: recordings dir not found: {rdir}")

    if _openai_key:
        print(f"  OpenAI API key      : loaded (…{_openai_key[-4:]})")
    else:
        print("  OpenAI API key      : not set — enter it in the Transcription card")

    if _AQC_AVAILABLE:
        print(f"  audio_quality_check : OK  (ANALYSIS_SR = {ANALYSIS_SR} Hz)")
    else:
        print(f"  audio_quality_check : MISSING — {_AQC_ERROR}")
        print("                         Quality metrics will not be shown.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"  ffmpeg : {ffmpeg}")
    else:
        print("  ffmpeg : not found  (WAV files only; .webm decoding unavailable)")

    print(f"\n  Open in browser : http://localhost:{args.port}")
    print("=" * 62 + "\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
