"""Isotonic calibration and risk-targeted thresholds.

A calibrator that is subtly wrong produces confident, plausible numbers that
are not true — the exact failure this whole line of work exists to catch — so
these check invariants rather than just exercising the code.
"""
import json
import math

import pytest

from speechlens.calibration import (Calibrator, ReliabilityPolicy,
                                    threshold_for_risk)
from speechlens.confidence import auroc, ece


def _separable(n=200):
    """Scores that predict correctness well but are badly scaled."""
    scores, correct = [], []
    for i in range(n):
        s = i / (n - 1)
        scores.append(s * 0.5)          # squashed into [0, 0.5]
        correct.append(i > n * 0.3)     # 70% correct, concentrated high
    return scores, correct


# --- isotonic fit -----------------------------------------------------------

def test_output_is_monotone_non_decreasing():
    scores, correct = _separable()
    cal = Calibrator.fit(scores, correct)
    xs = [i / 100 for i in range(101)]
    ys = [cal.predict(x) for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))


def test_predictions_stay_in_unit_interval():
    scores, correct = _separable()
    cal = Calibrator.fit(scores, correct)
    for x in (-5.0, 0.0, 0.25, 0.5, 1.0, 99.0):
        assert 0.0 <= cal.predict(x) <= 1.0


def test_calibration_cannot_change_the_ranking():
    # Monotone maps preserve AUROC exactly. If calibration ever moved AUROC,
    # it would mean the map is not monotone and the fit is broken.
    scores, correct = _separable()
    cal = Calibrator.fit(scores, correct)
    after = [cal.predict(s) for s in scores]
    assert auroc(after, correct) == pytest.approx(auroc(scores, correct))


def test_calibration_improves_ece_on_miscalibrated_scores():
    scores, correct = _separable()
    cal = Calibrator.fit(scores, correct)
    after = [cal.predict(s) for s in scores]
    assert ece(after, correct) < ece(scores, correct)


def test_unsmoothed_separable_data_maps_to_the_extremes():
    scores = [0.1, 0.2, 0.8, 0.9]
    correct = [False, False, True, True]
    cal = Calibrator.fit(scores, correct, smooth=False)
    assert cal.predict(0.1) == pytest.approx(0.0)
    assert cal.predict(0.9) == pytest.approx(1.0)


def test_smoothing_never_claims_certainty():
    # The default must not emit 0.0 or 1.0. A plateau of exactly 1.0 asserts
    # "this word cannot be wrong"; one counterexample then destroys any
    # log-loss score. Measured here: unsmoothed isotonic drove NCE from
    # +0.148 to -0.502 at 10 dB while improving ECE.
    scores = [0.1, 0.2, 0.8, 0.9]
    correct = [False, False, True, True]
    cal = Calibrator.fit(scores, correct)
    assert 0.0 < cal.predict(0.1) < cal.predict(0.9) < 1.0


def test_smoothing_strength_scales_with_plateau_support():
    # A plateau resting on few observations should be pulled toward 0.5 much
    # harder than one resting on many.
    few = Calibrator.fit([0.5] * 4, [True] * 4)
    many = Calibrator.fit([0.5] * 400, [True] * 400)
    assert few.predict(0.5) < many.predict(0.5) < 1.0
    assert few.predict(0.5) == pytest.approx(5 / 6)      # (4+1)/(4+2)


def test_constant_score_yields_the_base_rate():
    # No information in the score: the honest answer is the corpus accuracy.
    scores = [0.7] * 10
    correct = [True] * 6 + [False] * 4
    assert Calibrator.fit(scores, correct, smooth=False).predict(0.7) == pytest.approx(0.6)
    # Smoothed sits near it, pulled slightly toward 0.5.
    smoothed = Calibrator.fit(scores, correct).predict(0.7)
    assert 0.5 < smoothed < 0.6
    assert smoothed == pytest.approx(7 / 12)


def test_fit_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError):
        Calibrator.fit([], [])
    with pytest.raises(ValueError):
        Calibrator.fit([0.1, 0.2], [True])


def test_round_trips_through_json():
    scores, correct = _separable()
    cal = Calibrator.fit(scores, correct, condition="clean")
    back = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert back.condition == "clean"
    for x in (0.0, 0.13, 0.37, 0.5, 1.0):
        assert back.predict(x) == pytest.approx(cal.predict(x))


# --- risk-targeted thresholds -----------------------------------------------

def test_threshold_meets_the_risk_target():
    scores = [0.99, 0.95, 0.9, 0.4, 0.3, 0.2]
    correct = [True, True, True, False, True, False]
    thr, cov, risk = threshold_for_risk(scores, correct, max_risk=0.0)
    assert risk == pytest.approx(0.0)
    assert cov == pytest.approx(0.5)      # the three correct top-ranked words
    assert thr == pytest.approx(0.9)


def test_looser_risk_target_buys_more_coverage():
    scores = [0.99, 0.95, 0.9, 0.4, 0.3, 0.2]
    correct = [True, True, True, False, True, False]
    _t0, cov_strict, _r0 = threshold_for_risk(scores, correct, 0.0)
    _t1, cov_loose, _r1 = threshold_for_risk(scores, correct, 0.5)
    assert cov_loose > cov_strict


def test_impossible_target_accepts_nothing():
    # Highest-scoring word is wrong, so no threshold reaches zero risk.
    scores = [0.9, 0.8, 0.7]
    correct = [False, True, True]
    thr, cov, _risk = threshold_for_risk(scores, correct, max_risk=0.0)
    assert thr is None
    assert cov == 0.0


def test_threshold_rejects_mismatched_input():
    with pytest.raises(ValueError):
        threshold_for_risk([0.1], [True, False], 0.1)


def test_reported_risk_is_what_the_threshold_actually_admits():
    # Regression: evaluating rank by rank reports the risk of a *prefix*, but
    # `score >= threshold` admits every tied word too. Isotonic calibration
    # produces plateaus, so ties are normal. Here the 0.5 block is half wrong;
    # a prefix-based search would return threshold=0.5 claiming low risk.
    scores = [0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.1]
    correct = [True, True, True, True, False, False, False]
    thr, cov, risk = threshold_for_risk(scores, correct, max_risk=0.1)

    admitted = [c for s, c in zip(scores, correct) if s >= thr]
    actual_risk = sum(1 for c in admitted if not c) / len(admitted)
    assert actual_risk == pytest.approx(risk)
    assert actual_risk <= 0.1
    assert cov == pytest.approx(len(admitted) / len(scores))


def test_threshold_is_consistent_with_its_own_report_on_random_data():
    import random
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(3, 40)
        # A tiny value set to force heavy ties.
        scores = [rng.choice([0.1, 0.4, 0.7, 1.0]) for _ in range(n)]
        correct = [rng.random() < 0.75 for _ in range(n)]
        target = rng.choice([0.0, 0.05, 0.2, 0.5])
        thr, cov, risk = threshold_for_risk(scores, correct, target)
        if thr is None:
            continue
        admitted = [c for s, c in zip(scores, correct) if s >= thr]
        assert admitted, "a returned threshold must admit something"
        assert sum(1 for c in admitted if not c) / len(admitted) == pytest.approx(risk)
        assert len(admitted) / len(scores) == pytest.approx(cov)
        assert risk <= target + 1e-12


# --- policy -----------------------------------------------------------------

def test_policy_accepts_high_and_rejects_low():
    scores, correct = _separable()
    pol = ReliabilityPolicy.fit(scores, correct, target_risk=0.05)
    assert pol.accepts(0.5) is True
    assert pol.accepts(0.0) is False


def test_policy_realized_risk_respects_its_target():
    scores, correct = _separable()
    pol = ReliabilityPolicy.fit(scores, correct, target_risk=0.05)
    assert pol.realized_risk <= 0.05 + 1e-12
    assert 0.0 < pol.coverage <= 1.0


def test_policy_that_cannot_meet_its_target_accepts_nothing():
    # A threshold above 1.0 is the explicit "auto-accept nothing" state, which
    # is the safe answer -- silently accepting everything would not be.
    scores = [0.9, 0.8, 0.7]
    correct = [False, False, False]
    pol = ReliabilityPolicy.fit(scores, correct, target_risk=0.0)
    assert pol.accepts(0.9) is False
    assert pol.coverage == 0.0


def test_policy_round_trips_through_json():
    scores, correct = _separable()
    pol = ReliabilityPolicy.fit(scores, correct, target_risk=0.05,
                                condition="clean", notes="unit test")
    back = ReliabilityPolicy.from_dict(json.loads(json.dumps(pol.to_dict())))
    assert back.threshold == pytest.approx(pol.threshold)
    assert back.notes == "unit test"
    assert back.accepts(0.5) == pol.accepts(0.5)


def test_bundled_policies_are_shipped_and_loadable():
    from speechlens.calibration import available_policies, load_bundled_policy
    names = available_policies()
    assert "clean" in names and "-5" in names
    pol = load_bundled_policy("clean")
    assert 0.0 < pol.threshold <= 1.0
    assert pol.coverage > 0.5          # clean audio is mostly auto-acceptable
    assert pol.realized_risk <= pol.target_risk + 1e-9


def test_bundled_policy_for_heavy_noise_accepts_nothing():
    # At -5 dB no threshold reaches 2% error, and the honest encoding of that
    # is a policy that refuses everything rather than one that lowers the bar.
    from speechlens.calibration import load_bundled_policy
    pol = load_bundled_policy("-5")
    assert pol.coverage == 0.0
    assert pol.accepts(0.99) is False


def test_unknown_condition_names_the_available_ones():
    from speechlens.calibration import load_bundled_policy
    with pytest.raises(FileNotFoundError, match="available"):
        load_bundled_policy("42")


def test_pava_handles_a_backwards_cascade():
    # Descending labels force repeated backward merges in the solver; the
    # correct answer is one pooled block at the mean, 3/5.
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    correct = [True, True, True, False, False]
    cal = Calibrator.fit(scores, correct, smooth=False)
    ys = [cal.predict(s) for s in scores]
    assert all(abs(y - 0.6) < 1e-9 for y in ys)
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))
