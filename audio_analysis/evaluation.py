"""
evaluation.py — Ground-truth WER/CER evaluation and word-level diff rendering.

Functions:
  evaluate_transcript(transcript_lines, name_a, name_b) -> dict
  _wer_stats(ref, hyp) -> dict
  _cer(ref, hyp) -> float
  _wer_align(ref_words, hyp_words) -> list
  _diff_html(alignment) -> str
  _diff_plain(alignment) -> str
"""

import html as _html_mod
import re
from typing import Optional

from config import (
    _HERE,
    _DOCX_AVAILABLE,
    _docx_lib,
)

# ── Ground-truth file ─────────────────────────────────────────────────────────

_GT_FILE = _HERE / "speaker_ground_truth.docx"


def _load_ground_truth_text() -> str:
    if _DOCX_AVAILABLE and _GT_FILE.exists():
        doc = _docx_lib.Document(str(_GT_FILE))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""


_GT_TEXT: str = _load_ground_truth_text()


# ── Parsing ───────────────────────────────────────────────────────────────────

def _clean_utterance(text: str) -> str:
    """Normalize a raw utterance string from the ground-truth DOCX.

    Strips mojibake em-dashes (â€" = U+2014 read as Latin-1), proper Unicode
    dashes, curly quotes, and other non-ASCII artifacts so they don't show up
    as spurious [MISSING: â] tokens in the WER diff.
    """
    # Proper Unicode dashes → space
    text = re.sub(r'[\u2013\u2014\u2015\u2012\u2011]', ' ', text)
    # Mojibake: UTF-8 em-dash (E2 80 94) decoded as Latin-1 → â followed by
    # two control/non-printable bytes.  A leading â in this context is noise.
    text = re.sub(r'â\S*', ' ', text)
    # Curly / typographic quotes → straight
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201c\u201d]', '"', text)
    # Strip inline speaker labels (e.g. "S1:" or "S2:" inside an utterance)
    text = re.sub(r'\bS\d+\s*:', ' ', text, flags=re.IGNORECASE)
    # Strip remaining non-ASCII artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text


def _parse_ground_truth(text: str) -> dict:
    """Extract per-speaker text blobs from ground truth. Returns s1, s2, all."""
    s1_parts, s2_parts, all_parts = [], [], []
    for m in re.finditer(r'S(1|2)[^:]*:\s*"([^"]+)"', text):
        spk = m.group(1)
        utt = _clean_utterance(m.group(2))
        all_parts.append(utt)
        if spk == "1":
            s1_parts.append(utt)
        else:
            s2_parts.append(utt)
    return {
        "s1":  " ".join(s1_parts),
        "s2":  " ".join(s2_parts),
        "all": " ".join(all_parts),
    }


# ── Alignment ─────────────────────────────────────────────────────────────────

def _wer_align(ref_words: list, hyp_words: list) -> list:
    """Levenshtein alignment. Returns list of (ref, hyp, tag) where tag ∈ ok/sub/del/ins."""
    r, h = len(ref_words), len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): d[i][0] = i
    for j in range(h + 1): d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1])
    alignment = []
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1].lower() == hyp_words[j-1].lower():
            alignment.append((ref_words[i-1], hyp_words[j-1], "ok"))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
            alignment.append((ref_words[i-1], hyp_words[j-1], "sub"))
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            alignment.append((ref_words[i-1], None, "del"))
            i -= 1
        else:
            alignment.append((None, hyp_words[j-1], "ins"))
            j -= 1
    return list(reversed(alignment))


# ── Diff rendering ────────────────────────────────────────────────────────────

def _diff_html(alignment: list) -> str:
    """HTML colour-coded diff.
    - ok  : green
    - sub : italic orange (said word) + red [expected word]
    - del : red [expected word]
    - ins : green {extra word}
    """
    parts = []
    for ref, hyp, tag in alignment:
        if tag == "ok":
            parts.append(f'<span class="ev-ok">{_html_mod.escape(hyp)}</span>')
        elif tag == "sub":
            parts.append(
                f'<span class="ev-sub" title="expected: {_html_mod.escape(ref)}">'
                f'{_html_mod.escape(hyp)}</span>'
                f'<span class="ev-del" title="missing">[{_html_mod.escape(ref)}]</span>'
            )
        elif tag == "del":
            parts.append(f'<span class="ev-del">[{_html_mod.escape(ref)}]</span>')
        else:  # ins
            parts.append(f'<span class="ev-ins">{{{_html_mod.escape(hyp)}}}</span>')
    return " ".join(parts)


def _diff_plain(alignment: list) -> str:
    """Plain-text diff with clear markers:
      - correct words: shown as-is
      - deletions (expected but missing): [MISSING: word]
      - insertions (present but not expected): {EXTRA: word}
      - substitutions: said_word [MISSING: expected_word]
    """
    parts = []
    for ref, hyp, tag in alignment:
        if tag == "ok":
            parts.append(hyp)
        elif tag == "sub":
            parts.append(f"{hyp} [MISSING: {ref}]")
        elif tag == "del":
            parts.append(f"[MISSING: {ref}]")
        else:  # ins
            parts.append(f"{{EXTRA: {hyp}}}")
    return " ".join(parts)


# ── WER / CER ─────────────────────────────────────────────────────────────────

def _wer_stats(ref: str, hyp: str) -> Optional[dict]:
    # Strip punctuation + non-alphabetic chars so em-dashes, speaker labels, etc.
    # don't appear as spurious tokens.  Mirrors the tokenisation used in text-diff.
    ref_words = re.findall(r"[a-z']+", ref.lower())
    hyp_words = re.findall(r"[a-z']+", hyp.lower())
    if not ref_words:
        return None
    al   = _wer_align(ref_words, hyp_words)
    subs = sum(1 for _, _, t in al if t == "sub")
    dels = sum(1 for _, _, t in al if t == "del")
    ins  = sum(1 for _, _, t in al if t == "ins")
    wer  = (subs + dels + ins) / len(ref_words)
    return {
        "wer":           round(wer, 4),
        "ref_words":     len(ref_words),
        "hyp_words":     len(hyp_words),
        "substitutions": subs,
        "deletions":     dels,
        "insertions":    ins,
        "diff_html":     _diff_html(al),
        "diff_plain":    _diff_plain(al),
    }


def _cer(ref: str, hyp: str) -> Optional[float]:
    ref_chars = list(ref.lower().replace(" ", ""))
    hyp_chars = list(hyp.lower().replace(" ", ""))
    if not ref_chars:
        return None
    al     = _wer_align(ref_chars, hyp_chars)
    errors = sum(1 for _, _, t in al if t != "ok")
    return round(errors / len(ref_chars), 4)


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate_transcript(transcript_lines: list, name_a: str, name_b: str) -> dict:
    if not _GT_TEXT:
        return {"ground_truth_available": False}
    gt    = _parse_ground_truth(_GT_TEXT)
    hyp_a = " ".join(l["text"] for l in transcript_lines if l["speaker"] == name_a)
    hyp_b = " ".join(l["text"] for l in transcript_lines if l["speaker"] == name_b)
    hyp_all = " ".join(l["text"] for l in sorted(transcript_lines, key=lambda x: x["start"]))

    # Try both S1/S2 → speaker assignments; use whichever gives lower combined WER
    stats_ab = (_wer_stats(gt["s1"], hyp_a), _wer_stats(gt["s2"], hyp_b))
    stats_ba = (_wer_stats(gt["s1"], hyp_b), _wer_stats(gt["s2"], hyp_a))

    def _combined_wer(pair):
        wers = [s["wer"] for s in pair if s]
        return sum(wers) / len(wers) if wers else float("inf")

    if _combined_wer(stats_ba) < _combined_wer(stats_ab):
        s1_name, s2_name = name_b, name_a
        sa, sb = stats_ba
    else:
        s1_name, s2_name = name_a, name_b
        sa, sb = stats_ab

    wer_a = sa["wer"] if sa else None
    wer_b = sb["wer"] if sb else None
    if wer_a is not None and wer_b is not None and (wer_a + wer_b) > 0:
        wer_harmonic = round(2 * wer_a * wer_b / (wer_a + wer_b), 4)
    else:
        wer_harmonic = None

    return {
        "ground_truth_available": True,
        "name_a":        name_a,
        "name_b":        name_b,
        "s1_speaker":    s1_name,
        "s2_speaker":    s2_name,
        "overall":       _wer_stats(gt["all"], hyp_all),
        "speaker_a":     sa,
        "speaker_b":     sb,
        "wer_harmonic":  wer_harmonic,
        "cer_overall":   _cer(gt["all"], hyp_all),
    }
