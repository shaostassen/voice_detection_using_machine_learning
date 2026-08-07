"""Does a confidence score actually predict whether the word is wrong?

Two properties get conflated and they are not the same thing:

**Discrimination** — can the score *rank* errors below correct tokens? Measured
by AUROC. Invariant to any monotone rescaling, so calibration cannot fix a bad
AUROC and temperature scaling cannot improve it.

**Calibration** — does a reported 0.8 mean 80% correct? Measured by ECE and
NCE. Fixable post hoc, and worthless without discrimination.

A signal that tracks SNR between conditions can still have AUROC ~0.5 *within*
a condition, which would make it a noise meter rather than an error detector.
Telling those apart is the whole point of this module.

Everything here is pure: plain sequences in, floats out. No model, no weights,
no torch — so it is unit-testable against hand-computable cases, which matters
because these numbers decide what gets built next.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def _check(scores: Sequence[float], correct: Sequence[bool]) -> None:
    if len(scores) != len(correct):
        raise ValueError(f"length mismatch: {len(scores)} scores, "
                         f"{len(correct)} labels")


def auroc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """P(random correct token scores above a random incorrect one).

    Computed via the rank-sum identity rather than by sweeping thresholds, so
    ties are handled exactly (each tie contributes 0.5) instead of depending on
    sort order. 0.5 is chance; below 0.5 means the score is anti-correlated.

    Returns nan when one class is absent — AUROC is undefined there, and
    returning 0.5 would silently look like a real chance-level result.
    """
    _check(scores, correct)
    pos = [s for s, c in zip(scores, correct) if c]
    neg = [s for s, c in zip(scores, correct) if not c]
    if not pos or not neg:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):                      # average ranks within ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    rank_sum = sum(r for r, c in zip(ranks, correct) if c)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_nt(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the NPV-vs-TNR curve: precision/recall for *errors*.

    Not a restatement of AUROC. Flipping both the score sign and the labels
    leaves AUROC unchanged (it is symmetric under that double flip), so
    "AUROC for the error class" is the same number and tells you nothing new.
    This instead sweeps a threshold, predicting "error" for scores at or below
    it, and integrates

        NPV = TN / (TN + FN)   — of the tokens flagged, how many were wrong
        TNR = TN / (TN + FP)   — of the wrong tokens, how many were flagged

    computed as average precision over the error class. It is the metric that
    matters when errors are rare, because AUROC stays flattering while
    precision collapses. Chance level is the corpus error rate, not 0.5.
    """
    _check(scores, correct)
    n_err = sum(1 for c in correct if not c)
    if n_err == 0 or n_err == len(correct):
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i])  # worst first
    found = 0
    total = 0.0
    for k, i in enumerate(order, 1):
        if not correct[i]:
            found += 1
            total += found / k        # precision at this recall step
    return total / n_err


def ece(scores: Sequence[float], correct: Sequence[bool],
        bins: int = 10) -> float:
    """Expected calibration error: mean |confidence - accuracy| over bins.

    Equal-width bins over [0, 1], weighted by occupancy. Note the known
    weakness: with skewed scores most mass lands in one or two bins, so a low
    ECE can hide poor calibration elsewhere. Read it next to the reliability
    curve, not alone.
    """
    _check(scores, correct)
    if not scores:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores)
               if (s > lo or (b == 0 and s >= lo)) and s <= hi]
        if not idx:
            continue
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        conf = sum(scores[i] for i in idx) / len(idx)
        total += (len(idx) / len(scores)) * abs(conf - acc)
    return total


def nce(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Normalized cross entropy: how much the score beats the base rate.

    1.0 is perfect, 0.0 is no better than always predicting the corpus
    accuracy, negative means actively worse than that constant baseline.
    Unlike ECE this is a proper scoring rule, so it punishes confident errors
    rather than averaging them away.
    """
    _check(scores, correct)
    n = len(scores)
    if n == 0:
        return float("nan")
    base = sum(1 for c in correct if c) / n
    if base in (0.0, 1.0):
        return float("nan")

    eps = 1e-15
    h_base = -(base * math.log(base) + (1 - base) * math.log(1 - base))
    h_model = 0.0
    for s, c in zip(scores, correct):
        p = min(max(s, eps), 1 - eps)
        h_model -= math.log(p) if c else math.log(1 - p)
    h_model /= n
    return (h_base - h_model) / h_base


def risk_coverage(scores: Sequence[float],
                  correct: Sequence[bool]) -> List[Tuple[float, float]]:
    """Sweep an abstention threshold: ``[(coverage, risk), ...]``.

    Keep the highest-scoring tokens, abstain on the rest. Risk is the error
    rate among those kept. This is the operational curve — it answers "if I
    accept 80% of the output automatically, how wrong is it?", which is the
    question a deployment actually asks.
    """
    _check(scores, correct)
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out, errors = [], 0
    for rank, i in enumerate(order, 1):
        if not correct[i]:
            errors += 1
        out.append((rank / len(scores), errors / rank))
    return out


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve. Lower is better.

    Single-number summary of selective prediction quality. The floor is set by
    the corpus error rate, so compare it against the same corpus's
    ``aurc`` under a random score, never against an absolute target.
    """
    curve = risk_coverage(scores, correct)
    if not curve:
        return float("nan")
    return sum(r for _c, r in curve) / len(curve)


def summarize(scores: Sequence[float], correct: Sequence[bool],
              bins: int = 10) -> Dict[str, float]:
    """Every metric at once, for one condition. Keys are stable for tables."""
    _check(scores, correct)
    n = len(scores)
    return {
        "n": float(n),
        "accuracy": sum(1 for c in correct if c) / n if n else float("nan"),
        "mean_score": sum(scores) / n if n else float("nan"),
        "auroc": auroc(scores, correct),
        "auc_nt": auc_nt(scores, correct),
        "ece": ece(scores, correct, bins=bins),
        "nce": nce(scores, correct),
        "aurc": aurc(scores, correct),
    }
