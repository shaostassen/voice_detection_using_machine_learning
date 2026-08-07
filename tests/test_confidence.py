"""Confidence metrics, checked against hand-computable cases.

These numbers decide whether the reliability layer gets built at all, so a
metric that is subtly wrong would send the whole project down a false path.
Every test here has an answer derivable on paper.
"""
import math

import pytest

from speechlens.confidence import (aurc, auroc, auc_nt, ece, nce,
                                   normalized_entropy, renyi_entropy,
                                   risk_coverage, shannon_entropy, summarize)


# --- entropy ----------------------------------------------------------------

def _uniform(n):
    return [math.log(1.0 / n)] * n


def _onehot(n):
    return [0.0] + [-800.0] * (n - 1)   # log(1) and effectively log(0)


def test_uniform_distribution_has_maximum_entropy():
    for n in (2, 8, 64):
        assert shannon_entropy(_uniform(n)) == pytest.approx(math.log(n))


def test_certain_distribution_has_zero_entropy():
    assert shannon_entropy(_onehot(50)) == pytest.approx(0.0, abs=1e-9)


def test_underflow_guard_does_not_produce_nan():
    # log-probs from a 51k vocab routinely go below -700; exp() underflows and
    # 0 * -inf would be nan without the guard.
    lp = [0.0] + [-1e9] * 100
    assert shannon_entropy(lp) == pytest.approx(0.0, abs=1e-9)
    assert not math.isnan(renyi_entropy(lp, 2.0))


def test_renyi_recovers_shannon_at_alpha_one():
    lp = [math.log(p) for p in (0.5, 0.25, 0.15, 0.10)]
    assert renyi_entropy(lp, 1.0) == pytest.approx(shannon_entropy(lp))


def test_renyi_alpha_above_one_discounts_the_tail():
    # A peaked head with a long flat tail: Renyi-2 should read as more
    # confident (lower entropy) than Shannon, which spends range on the tail.
    lp = [math.log(0.7)] + [math.log(0.3 / 200)] * 200
    assert renyi_entropy(lp, 2.0) < shannon_entropy(lp)


def test_renyi_rejects_nonpositive_alpha():
    with pytest.raises(ValueError):
        renyi_entropy(_uniform(4), 0.0)


def test_normalized_entropy_is_confidence_oriented():
    # Higher must mean MORE confident, matching every other estimator here.
    # A mixed orientation shows up as AUROC < 0.5 and reads as a finding.
    assert normalized_entropy(_onehot(64)) == pytest.approx(1.0, abs=1e-9)
    assert normalized_entropy(_uniform(64)) == pytest.approx(0.0, abs=1e-9)


def test_normalized_entropy_stays_in_unit_range():
    lp = [math.log(p) for p in (0.9, 0.05, 0.03, 0.02)]
    v = normalized_entropy(lp)
    assert 0.0 <= v <= 1.0


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


def _auroc_brute(scores, correct):
    """Definition, straight from the probability statement. O(n^2)."""
    pos = [s for s, c in zip(scores, correct) if c]
    neg = [s for s, c in zip(scores, correct) if not c]
    if not pos or not neg:
        return float("nan")
    total = sum(1.0 if p > n else (0.5 if p == n else 0.0)
                for p in pos for n in neg)
    return total / (len(pos) * len(neg))


def test_rank_sum_matches_brute_force_including_ties():
    # The fast path uses the rank-sum identity with tie-averaged ranks. Ties
    # are where that goes wrong silently, so the value set is deliberately
    # tiny to force collisions.
    import random
    rng = random.Random(0)
    for _ in range(400):
        n = rng.randint(2, 30)
        scores = [rng.choice([0.1, 0.2, 0.3, 0.5, 0.9]) for _ in range(n)]
        correct = [rng.random() < 0.7 for _ in range(n)]
        fast, slow = auroc(scores, correct), _auroc_brute(scores, correct)
        if math.isnan(fast) and math.isnan(slow):
            continue
        assert fast == pytest.approx(slow, abs=1e-12)


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
