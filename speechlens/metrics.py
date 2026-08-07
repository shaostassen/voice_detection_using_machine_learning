"""Evaluation metrics: WER and CER via Levenshtein alignment.

WER = (substitutions + deletions + insertions) / reference_length, computed
on words. For unspaced scripts (zh/ja/ko/...) whitespace tokenization is
meaningless, so CER over characters is the honest metric.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

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


# Alignment op codes. "del" has no hypothesis token and so never carries a
# confidence score — a confidence signal is structurally blind to deletions,
# which is why deletion-aware confidence is its own line of research.
EQUAL, SUB, DEL, INS = "eq", "sub", "del", "ins"


def align(ref: List, hyp: List) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Levenshtein alignment path as ``(op, ref_index, hyp_index)`` triples.

    ``levenshtein`` above keeps one row and so cannot reconstruct the path;
    this keeps the full matrix. Only used for labelling, never in a hot loop.

    Indices are ``None`` on the side the op does not consume: insertions have
    no ``ref_index``, deletions no ``hyp_index``. Ops are returned in reading
    order, so every hypothesis token appears exactly once as either ``eq``,
    ``sub`` or ``ins`` — which is what makes per-token correct/incorrect
    labelling well defined.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))

    ops: List[Tuple[str, Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        # Diagonal first so an equal pair is never spelled as ins+del.
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((EQUAL if ref[i - 1] == hyp[j - 1] else SUB, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((DEL, i - 1, None))
            i -= 1
        else:
            ops.append((INS, None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def token_labels(ref_tokens: List[str], hyp_tokens: List[str]) -> List[bool]:
    """Per-hypothesis-token correctness, one label per token, in order.

    ``True`` means the token survived alignment unchanged. Substitutions and
    insertions both count as errors: the token is on the page and it is wrong.
    Deletions produce no hypothesis token and so contribute no label — they
    are invisible to any per-token confidence signal, by construction.
    """
    labels = [False] * len(hyp_tokens)
    for op, _ri, hi in align(ref_tokens, hyp_tokens):
        if hi is not None:
            labels[hi] = op == EQUAL
    return labels


def word_labels(ref: str, hyp: str) -> List[bool]:
    """Per-hypothesis-word correctness, aligned to ``normalize_text(hyp).split()``."""
    return token_labels(normalize_text(ref).split(), normalize_text(hyp).split())


def labels_for_words(ref: str, words: List[str]) -> List[Optional[bool]]:
    """Correctness per *original* word, given the words as the ASR emitted them.

    Normalization is not one-to-one — "don't" becomes two tokens, "1999" may
    become one, and stray punctuation becomes none — so joining words into a
    string and splitting again silently desynchronizes scores from labels.
    Here each word is normalized independently and its tokens tracked back to
    it, which keeps the alignment honest and the indices exact.

    A word is correct only if every token it produced aligned as equal.
    Returns ``None`` for words that normalize away entirely (pure punctuation);
    those carry no evidence and must be dropped, not guessed.
    """
    hyp_tokens: List[str] = []
    owner: List[int] = []
    for wi, w in enumerate(words):
        toks = normalize_text(w).split()
        hyp_tokens.extend(toks)
        owner.extend([wi] * len(toks))

    tok_ok = token_labels(normalize_text(ref).split(), hyp_tokens)

    out: List[Optional[bool]] = [None] * len(words)
    for ti, ok in enumerate(tok_ok):
        wi = owner[ti]
        out[wi] = ok if out[wi] is None else (out[wi] and ok)
    return out


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
