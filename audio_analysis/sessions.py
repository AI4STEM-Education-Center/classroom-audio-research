"""
sessions.py — Session directory discovery and WAV-pair matching.

Functions:
  list_sessions(recordings_dir) -> list
  find_session_pair(session_dir, tag_a, tag_b) -> dict
"""

import json
from pathlib import Path
from typing import Optional


def list_sessions(recordings_dir: Path) -> list:
    sessions = []
    if not recordings_dir.exists():
        return sessions
    for d in sorted(recordings_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        sj = d / "session.json"
        if not sj.exists():
            continue
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        pair = find_session_pair(d)
        has_pair = bool(pair.get("headset_wav") and pair.get("array_wav")) \
                   or len(pair.get("available_tags", [])) >= 2
        sessions.append({
            "name":      d.name,
            "path":      str(d),
            "started":   meta.get("started", ""),
            "label":     meta.get("label", ""),
            "has_pair":  has_pair,
            "wav_count": len(list(d.glob("*.wav"))),
            "tags":      pair.get("available_tags", []),
        })
    return sessions


def find_session_pair(session_dir: Path,
                      tag_a: str = "", tag_b: str = "") -> dict:
    """
    Scan *_meta.json and pair WAV files.

    When tag_a / tag_b are provided the function picks the WAV whose 'tag' or
    'source' meta field matches exactly (case-insensitive), ignoring deviceLabel
    heuristics.  This lets the caller select a specific recording phase from a
    session that contains multiple phases (e.g. 'Arne_messy1' + 'Jennifer_messy1').

    Without explicit tags the original deviceLabel heuristic is used:
      'Microphone Array' → array (built-in laptop)
      'Mikrofon' / 'Microphone' → headset / single mic
    with a tag-alphabetical fallback if labels don't yield a pair.

    Returns a dict with keys:
      headset_wav, headset_meta, array_wav, array_meta, warnings, available_tags
    available_tags is an ordered list of unique tag strings found in this session.
    """
    warns: list = []

    # ── Single pass: collect all candidates ──────────────────────────────────
    candidates: list = []   # (tag, wav_path, meta, label_lower)
    seen_tags: set = set()
    available_tags: list = []

    for mf in sorted(session_dir.glob("*_meta.json")):
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        wav_name = meta.get("wav_file")
        if not wav_name:
            continue
        wav_path = session_dir / wav_name
        if not wav_path.exists():
            continue
        tag   = meta.get("tag") or meta.get("source") or ""
        label = (meta.get("deviceLabel") or meta.get("device_label") or "").lower()
        candidates.append((tag, wav_path, meta, label))
        if tag and tag not in seen_tags:
            seen_tags.add(tag)
            available_tags.append(tag)

    headset_wav: Optional[Path] = None
    headset_meta: Optional[dict] = None
    array_wav: Optional[Path] = None
    array_meta: Optional[dict] = None

    # ── Explicit tag selection ────────────────────────────────────────────────
    if tag_a or tag_b:
        tag_a_l = tag_a.lower()
        tag_b_l = tag_b.lower()
        for tag, wav_path, meta, label in candidates:
            t = tag.lower()
            if tag_a and t == tag_a_l and headset_wav is None:
                headset_wav, headset_meta = wav_path, meta
            elif tag_b and t == tag_b_l and array_wav is None:
                array_wav, array_meta = wav_path, meta
        if tag_a and headset_wav is None:
            warns.append(f"Tag '{tag_a}' (Speaker A) not found in this session.")
        if tag_b and array_wav is None:
            warns.append(f"Tag '{tag_b}' (Speaker B) not found in this session.")

    # ── Auto selection: deviceLabel heuristic ────────────────────────────────
    else:
        for tag, wav_path, meta, label in candidates:
            # Check "microphone array" FIRST (it also contains "microphone")
            if "microphone array" in label:
                if array_wav is None:
                    array_wav, array_meta = wav_path, meta
            elif "mikrofon" in label or "microphone" in label:
                if headset_wav is None:
                    headset_wav, headset_meta = wav_path, meta
                elif array_wav is None:
                    # Second microphone-class device — assign as speaker B.
                    array_wav, array_meta = wav_path, meta

        # Tag/source alphabetical fallback
        if not (headset_wav and array_wav):
            tag_cands = sorted(
                [(t.lower(), wp, m) for t, wp, m, _ in candidates if t],
                key=lambda x: x[0],
            )
            if len(tag_cands) >= 2:
                headset_wav,  headset_meta  = tag_cands[0][1], tag_cands[0][2]
                array_wav,    array_meta    = tag_cands[1][1], tag_cands[1][2]

    if not headset_wav and not array_wav:
        warns.append("No WAV files with recognized deviceLabel or speaker tag found in this session.")
    elif not headset_wav:
        warns.append("No speaker-A WAV found; only speaker-B available.")
    elif not array_wav:
        warns.append("No speaker-B WAV found; only speaker-A available.")

    return {
        "headset_wav":    headset_wav,
        "headset_meta":   headset_meta or {},
        "array_wav":      array_wav,
        "array_meta":     array_meta or {},
        "warnings":       warns,
        "available_tags": available_tags,
    }
