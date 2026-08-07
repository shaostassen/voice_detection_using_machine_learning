"""Turn a confidence score into a probability you can actually act on.

Phase 1 measured the two properties separately and they came apart. Per-word
probability *ranks* errors well at every noise level (AUROC 0.76-0.89), and it
is well *calibrated* only down to about 0 dB SNR — at -5 dB its NCE goes
negative, meaning the numbers it reports are worse than just quoting the
corpus accuracy, even though the ranking is still informative.

So a raw score is fine for sorting and unfit for thresholding, and a single
global calibration would be wrong at both ends of the ladder. This module
provides the post-hoc fix and, more importantly, the operational question:
*if I accept everything above threshold t, how wrong is what I accepted?*

Pure Python — no sklearn, no torch, no model. Fits on ``(score, correct)``
pairs produced by ``scripts/reliability.py`` and serializes to JSON, so a
calibrator is a small data file rather than a dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _pava_blocks(values: List[float]) -> List[Tuple[float, int]]:
    """Pool Adjacent Violators, returning ``(mean, count)`` blocks.

    The standard isotonic solver: walk left to right merging any block that
    dips below its predecessor into their weighted mean, re-checking backwards
    because a merge can create a new dip.

    Blocks rather than per-point values because the block *size* is needed
    downstream — a plateau resting on four observations deserves less
    confidence than one resting on four hundred, and plain isotonic throws
    that away.
    """
    blocks: List[Tuple[float, int]] = []
    for v in values:
        blocks.append((v, 1))
        # `>=`, not `>`: equal adjacent blocks must pool too. Pooling equals
        # cannot change the isotonic fit (the mean of two equal means is the
        # same value), but it does produce the true plateau *size*, and the
        # smoothing above is a function of that size. With strict `>`, a run
        # of 400 identical labels stays 400 singleton blocks and every one of
        # them gets smoothed as though it rested on a single observation.
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            (v2, n2), (v1, n1) = blocks.pop(), blocks.pop()
            blocks.append(((v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2))
    return blocks


@dataclass
class Calibrator:
    """Monotone map from raw score to P(word is correct).

    Isotonic rather than Platt/temperature scaling: the relationship between
    Whisper's word probability and actual correctness is not logistic, and
    isotonic makes no shape assumption beyond monotonicity — which is exactly
    the property AUROC already established holds.

    Being monotone it cannot change the ranking, so AUROC is identical before
    and after. That is the point: calibration fixes the numbers, never the
    discrimination.
    """
    knots_x: List[float] = field(default_factory=list)
    knots_y: List[float] = field(default_factory=list)
    n_fit: int = 0
    condition: str = ""          # e.g. "clean", "0dB" — provenance, not logic

    @classmethod
    def fit(cls, scores: Sequence[float], correct: Sequence[bool],
            condition: str = "", smooth: bool = True) -> "Calibrator":
        """Fit by isotonic regression, Laplace-smoothed by default.

        Raw isotonic emits plateaus of exactly 0.0 and 1.0 — it saw only
        correct or only incorrect words there. On held-out data that is a
        claim of certainty, and one counterexample inside such a plateau
        destroys any log-loss score: measured on this project's own data,
        unsmoothed isotonic improved ECE at every noise level while driving
        NCE from +0.148 to -0.502 at 10 dB.

        Smoothing replaces each plateau's mean with ``(k + 1) / (n + 2)``, the
        Laplace estimate, so a plateau supported by four observations is
        pulled toward 0.5 far more than one supported by four hundred. Pass
        ``smooth=False`` for the textbook fit.
        """
        if len(scores) != len(correct):
            raise ValueError("scores and correct must be the same length")
        if not scores:
            raise ValueError("cannot fit a calibrator on no data")

        order = sorted(range(len(scores)), key=lambda i: scores[i])
        xs = [scores[i] for i in order]
        ys = [1.0 if correct[i] else 0.0 for i in order]

        fitted: List[float] = []
        for mean, count in _pava_blocks(ys):
            value = ((mean * count + 1.0) / (count + 2.0)) if smooth else mean
            fitted.extend([value] * count)

        # Collapse runs of equal fitted value into knots, keeping the last x
        # of each run so interpolation spans the plateau.
        kx: List[float] = []
        ky: List[float] = []
        for x, y in zip(xs, fitted):
            if ky and abs(ky[-1] - y) < 1e-12:
                kx[-1] = x
            else:
                kx.append(x)
                ky.append(y)
        return cls(knots_x=kx, knots_y=ky, n_fit=len(xs), condition=condition)

    def predict(self, score: float) -> float:
        """Calibrated P(correct). Clamped to the fitted range at both ends."""
        if not self.knots_x:
            return score
        if score <= self.knots_x[0]:
            return self.knots_y[0]
        if score >= self.knots_x[-1]:
            return self.knots_y[-1]
        lo, hi = 0, len(self.knots_x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.knots_x[mid] <= score:
                lo = mid
            else:
                hi = mid
        x0, x1 = self.knots_x[lo], self.knots_x[hi]
        y0, y1 = self.knots_y[lo], self.knots_y[hi]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (score - x0) / (x1 - x0)

    def to_dict(self) -> dict:
        return {"knots_x": self.knots_x, "knots_y": self.knots_y,
                "n_fit": self.n_fit, "condition": self.condition}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(knots_x=list(d["knots_x"]), knots_y=list(d["knots_y"]),
                   n_fit=int(d.get("n_fit", 0)),
                   condition=str(d.get("condition", "")))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "Calibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


def threshold_for_risk(scores: Sequence[float], correct: Sequence[bool],
                       max_risk: float) -> Tuple[Optional[float], float, float]:
    """Largest coverage whose error rate stays within ``max_risk``.

    Returns ``(threshold, coverage, realized_risk)``. Accept a word when its
    score is **>= threshold**.

    This is the question deployments actually ask — "I can tolerate 2% error
    in what I auto-accept; how much can I auto-accept?" — rather than "what
    confidence value feels safe?". Air traffic control ASR already works this
    way, with a mandated error ceiling and everything below it routed to a
    human.

    ``(None, 0.0, 0.0)`` means no threshold achieves the target: even the
    single highest-scoring word is not reliable enough, so nothing should be
    auto-accepted.
    """
    if len(scores) != len(correct):
        raise ValueError("scores and correct must be the same length")
    if not scores:
        return None, 0.0, 0.0

    # Candidate thresholds are *distinct* score values, and each is scored on
    # the whole set it would admit. Walking rank by rank instead looks right
    # and silently lies whenever scores tie: the reported risk describes a
    # prefix, but `score >= threshold` admits every tied word as well. After
    # isotonic calibration ties are the norm, not an edge case — plateaus are
    # exactly what it produces.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best: Tuple[Optional[float], float, float] = (None, 0.0, 0.0)
    errors = 0
    for rank, i in enumerate(order, 1):
        if not correct[i]:
            errors += 1
        # Only evaluate at a boundary: the next word has a strictly lower
        # score, so this prefix is exactly what `>= scores[i]` admits.
        if rank < len(order) and scores[order[rank]] == scores[i]:
            continue
        risk = errors / rank
        if risk <= max_risk:
            best = (scores[i], rank / len(scores), risk)
    return best


@dataclass
class ReliabilityPolicy:
    """A fitted calibrator plus the threshold it should be read against.

    Bundled deliberately: a threshold without the calibration that produced it
    is a magic number, which is how the original 0.55 came to be inherited and
    never questioned.
    """
    calibrator: Calibrator
    threshold: float
    target_risk: float
    coverage: float
    realized_risk: float
    notes: str = ""

    def accepts(self, raw_score: float) -> bool:
        return self.calibrator.predict(raw_score) >= self.threshold

    def to_dict(self) -> dict:
        return {"calibrator": self.calibrator.to_dict(),
                "threshold": self.threshold, "target_risk": self.target_risk,
                "coverage": self.coverage, "realized_risk": self.realized_risk,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "ReliabilityPolicy":
        return cls(calibrator=Calibrator.from_dict(d["calibrator"]),
                   threshold=float(d["threshold"]),
                   target_risk=float(d["target_risk"]),
                   coverage=float(d["coverage"]),
                   realized_risk=float(d["realized_risk"]),
                   notes=str(d.get("notes", "")))

    @classmethod
    def fit(cls, scores: Sequence[float], correct: Sequence[bool],
            target_risk: float = 0.02, condition: str = "",
            notes: str = "") -> "ReliabilityPolicy":
        cal = Calibrator.fit(scores, correct, condition=condition)
        calibrated = [cal.predict(s) for s in scores]
        thr, cov, risk = threshold_for_risk(calibrated, correct, target_risk)
        return cls(calibrator=cal,
                   threshold=1.1 if thr is None else thr,   # 1.1 accepts nothing
                   target_risk=target_risk, coverage=cov,
                   realized_risk=risk, notes=notes)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "ReliabilityPolicy":
        return cls.from_dict(json.loads(Path(path).read_text()))


POLICY_DIR = Path(__file__).parent / "policies"


def available_policies() -> List[str]:
    """Condition names with a bundled policy, e.g. ``['clean', '0', '5', ...]``."""
    if not POLICY_DIR.is_dir():
        return []
    return sorted(p.stem.replace("word_prob_", "").replace("neg", "-")
                  for p in POLICY_DIR.glob("word_prob_*.json"))


def load_bundled_policy(condition: str) -> ReliabilityPolicy:
    """Load a shipped policy by noise condition. No default, on purpose.

    The right operating point depends on how noisy the audio is — at 2%
    tolerated error the fitted coverage runs from 90% on clean speech to
    literally nothing at 0 dB SNR. Silently defaulting to the clean policy
    would auto-accept most of a transcript that is 29% wrong, which is worse
    than having no policy at all.

    Choosing well means knowing your conditions. Until the pipeline can
    estimate SNR itself, that judgement stays with the caller.
    """
    name = f"word_prob_{condition.replace('-', 'neg')}.json"
    path = POLICY_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"no bundled policy for condition {condition!r}; "
            f"available: {available_policies()}")
    return ReliabilityPolicy.load(path)
