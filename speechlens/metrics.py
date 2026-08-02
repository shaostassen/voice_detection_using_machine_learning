"""Evaluation metrics: WER and CER via Levenshtein alignment.

WER = (substitutions + deletions + insertions) / reference_length, computed
on words. For unspaced scripts (zh/ja/ko/...) whitespace tokenization is
meaningless, so CER over characters is the honest metric.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)

CJK_LANGS = {"zh", "yue", "ja", "ko", "th", "lo", "my", "km"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(ref: List, hyp: List) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + (r != h))  # substitution
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = list(normalize_text(ref).replace(" ", ""))
    h = list(normalize_text(hyp).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def error_rate(ref: str, hyp: str, language: str = "en") -> Tuple[str, float]:
    """Pick the right metric for the language; returns (metric_name, value)."""
    if language in CJK_LANGS:
        return "cer", cer(ref, hyp)
    return "wer", wer(ref, hyp)
