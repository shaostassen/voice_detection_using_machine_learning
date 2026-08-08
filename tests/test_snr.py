"""SNR estimation, checked against signals whose true SNR is known exactly.

The estimate selects which calibrated policy is applied, so being wrong here
silently picks the wrong accept threshold. These construct the signal, so the
right answer is arithmetic rather than opinion.
"""
import numpy as np
import pytest

from speechlens.snr import (MAX_SNR_DB, estimate_snr, nearest_condition)

SR = 16000


def _clip(snr_db=None, speech_s=2.0, silence_s=1.0, seed=0):
    """Speech-like tone between two silences, with white noise everywhere.

    Noise covers the whole clip, so the silent regions are a clean sample of
    it — exactly the situation the estimator assumes.
    """
    rng = np.random.default_rng(seed)
    n_sp, n_si = int(speech_s * SR), int(silence_s * SR)
    t = np.arange(n_sp) / SR
    speech = 0.4 * np.sin(2 * np.pi * 180 * t)

    y = np.concatenate([np.zeros(n_si), speech, np.zeros(n_si)])
    spans = [(silence_s, silence_s + speech_s)]

    if snr_db is not None:
        p_sig = float(np.mean(speech ** 2))
        p_noise = p_sig / (10 ** (snr_db / 10))
        y = y + rng.normal(0, np.sqrt(p_noise), y.size)
    return y.astype(np.float32), spans


@pytest.mark.parametrize("true_snr", [20.0, 10.0, 5.0, 0.0, -5.0])
def test_recovers_a_known_snr(true_snr):
    y, spans = _clip(snr_db=true_snr)
    est = estimate_snr(y, SR, spans)
    assert est == pytest.approx(true_snr, abs=1.0)


def test_ordering_is_monotone_across_the_ladder():
    ests = [estimate_snr(*(_clip(snr_db=s)[0], SR), _clip(snr_db=s)[1])
            for s in (20.0, 10.0, 5.0, 0.0, -5.0)]
    assert all(a > b for a, b in zip(ests, ests[1:]))


def test_noise_free_audio_reads_as_clean():
    y, spans = _clip(snr_db=None)
    assert estimate_snr(y, SR, spans) == pytest.approx(MAX_SNR_DB)


def test_subtracts_the_noise_floor_from_speech_regions():
    # Speech regions hold signal+noise. Without subtracting the floor the
    # estimate reads ~3 dB high at 0 dB SNR — a full rung of the ladder.
    y, spans = _clip(snr_db=0.0)
    assert estimate_snr(y, SR, spans) == pytest.approx(0.0, abs=1.0)


def test_returns_none_without_a_noise_sample():
    # All speech, no silence: nothing to measure the noise from.
    y, _ = _clip(snr_db=10.0, silence_s=0.0)
    assert estimate_snr(y, SR, [(0.0, len(y) / SR)]) is None


def test_returns_none_without_speech():
    y, _ = _clip(snr_db=10.0)
    assert estimate_snr(y, SR, []) is None


def test_returns_none_on_empty_input():
    assert estimate_snr(np.array([], dtype=np.float32), SR, []) is None


def test_accepts_speech_segment_objects_as_well_as_tuples():
    from speechlens.vad import SpeechSegment
    y, spans = _clip(snr_db=10.0)
    objs = [SpeechSegment(spans[0][0], spans[0][1])]
    assert estimate_snr(y, SR, objs) == pytest.approx(
        estimate_snr(y, SR, spans))


# --- condition selection ----------------------------------------------------

CONDS = ["clean", "20", "10", "5", "0", "-5"]


@pytest.mark.parametrize("snr,expected", [
    (38.0, "clean"),
    (19.0, "20"),
    (11.0, "10"),
    (4.5, "5"),
    (0.4, "0"),
    (-6.0, "-5"),
    (-30.0, "-5"),
])
def test_picks_the_nearest_condition(snr, expected):
    assert nearest_condition(snr, CONDS) == expected


def test_ties_resolve_to_the_noisier_policy():
    # 7.5 is equidistant from 5 and 10. Prefer 5: too cautious costs coverage,
    # too optimistic ships wrong words as trustworthy.
    assert nearest_condition(7.5, CONDS) == "5"
    assert nearest_condition(2.5, CONDS) == "0"


def test_no_estimate_means_no_automatic_choice():
    assert nearest_condition(None, CONDS) is None
    assert nearest_condition(10.0, []) is None


def test_nan_abstains_instead_of_picking_a_policy():
    # NaN arrives from averaging over an empty set of estimates, which is what
    # babble noise produces: it fills the silences, the VAD finds no
    # noise-only region, and every per-utterance estimate declines. NaN
    # compares false against everything, so an unguarded sort keeps insertion
    # order and silently returns a policy. Abstaining is the only honest
    # answer, and the noisiest policy is not a safe default either — it
    # accepts nothing, which looks like "audio is terrible" rather than
    # "SNR is unknown".
    assert nearest_condition(float("nan"), CONDS) is None
