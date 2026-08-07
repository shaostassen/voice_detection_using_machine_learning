"""Confidence metrics, checked against hand-computable cases.

These numbers decide whether the reliability layer gets built at all, so a
metric that is subtly wrong would send the whole project down a false path.
Every test here has an answer derivable on paper.
"""
import math

import pytest

from speechlens.confidence import (aurc, auroc, auc_nt, ece, nce,
                                   risk_coverage, summarize)


# --- AUROC ------------------------------------------------------------------

def test_perfect_separation_is_one():
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    assert auroc(scores, correct) == pytest.approx(1.0)


def test_perfectly_inverted_is_zero():
    scores = [0.1, 0.2, 0.8, 0.9]
    correct = [True, True, False, False]
    assert auroc(scores, correct) == pytest.approx(0.0)


def test_all_ties_is_exactly_chance():
    # The case that matters most: a signal with no within-condition variance
    # must score 0.5, not 1.0 by accident of sort order. This is precisely
    # the situation for per-window avg_logprob inside a single window.
    scores = [0.83] * 6
    correct = [True, False, True, False, True, False]
    assert auroc(scores, correct) == pytest.approx(0.5)


def test_known_rank_sum_case():
    # pos={3,1}, neg={2,0} -> pairs (3>2,3>0,1<2,1>0) = 3/4
    scores = [3.0, 1.0, 2.0, 0.0]
    correct = [True, True, False, False]
    assert auroc(scores, correct) == pytest.approx(0.75)


def test_single_class_is_nan_not_a_fake_half():
    assert math.isnan(auroc([0.1, 0.2], [True, True]))
    assert math.isnan(auroc([0.1, 0.2], [False, False]))


def test_auroc_is_invariant_to_monotone_rescaling():
    # Why calibration cannot rescue discrimination.
    scores = [0.9, 0.7, 0.4, 0.1]
    correct = [True, True, False, False]
    squashed = [s ** 3 for s in scores]
    assert auroc(squashed, correct) == pytest.approx(auroc(scores, correct))


def test_auc_nt_is_one_when_errors_rank_lowest():
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    assert auc_nt(scores, correct) == pytest.approx(1.0)


def test_auc_nt_is_not_merely_auroc_flipped():
    # Guards the bug this replaced: flipping both score sign and labels leaves
    # AUROC unchanged, so an "AUROC for errors" implementation is a duplicate.
    # AUC-NT must differ from AUROC on an imperfect ranking.
    scores = [0.1, 0.2, 0.8, 0.9]
    correct = [True, True, False, False]
    # errors sit at ranks 3 and 4 of 4 when sorted worst-first
    assert auc_nt(scores, correct) == pytest.approx((1 / 3 + 2 / 4) / 2)
    assert auc_nt(scores, correct) != pytest.approx(auroc(scores, correct))


def test_auc_nt_undefined_without_both_classes():
    assert math.isnan(auc_nt([0.1, 0.2], [True, True]))
    assert math.isnan(auc_nt([0.1, 0.2], [False, False]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [True])


# --- calibration ------------------------------------------------------------

def test_ece_zero_when_confidence_matches_accuracy():
    # Ten tokens at 0.9 with exactly 9 correct: perfectly calibrated bin.
    scores = [0.9] * 10
    correct = [True] * 9 + [False]
    assert ece(scores, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_catches_total_overconfidence():
    scores = [1.0] * 4
    correct = [False] * 4
    assert ece(scores, correct) == pytest.approx(1.0)


def test_nce_positive_when_score_is_informative():
    scores = [0.95] * 8 + [0.05] * 8
    correct = [True] * 8 + [False] * 8
    assert nce(scores, correct) > 0.5


def test_nce_negative_when_score_is_actively_misleading():
    scores = [0.05] * 8 + [0.95] * 8
    correct = [True] * 8 + [False] * 8
    assert nce(scores, correct) < 0.0


def test_nce_undefined_without_both_classes():
    assert math.isnan(nce([0.5, 0.5], [True, True]))


# --- selective prediction ---------------------------------------------------

def test_risk_coverage_ends_at_full_coverage_and_corpus_error():
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    curve = risk_coverage(scores, correct)
    assert curve[-1][0] == pytest.approx(1.0)
    assert curve[-1][1] == pytest.approx(0.5)


def test_risk_stays_zero_while_only_correct_tokens_are_kept():
    scores = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    curve = risk_coverage(scores, correct)
    assert curve[0][1] == pytest.approx(0.0)
    assert curve[1][1] == pytest.approx(0.0)


def test_aurc_rewards_the_better_ranking():
    correct = [True, True, False, False]
    good = aurc([0.9, 0.8, 0.2, 0.1], correct)
    bad = aurc([0.1, 0.2, 0.8, 0.9], correct)
    assert good < bad


# --- summary ----------------------------------------------------------------

def test_summarize_reports_every_key():
    scores = [0.9, 0.8, 0.3, 0.2]
    correct = [True, True, False, False]
    s = summarize(scores, correct)
    assert set(s) == {"n", "accuracy", "mean_score", "auroc", "auc_nt",
                      "ece", "nce", "aurc"}
    assert s["n"] == 4
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["auroc"] == pytest.approx(1.0)
