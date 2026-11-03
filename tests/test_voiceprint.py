import numpy as np

from speechlens.synth import synth_utterance
from speechlens.voiceprint import distance, identify, mfcc, voiceprint

SPEAKERS = {
    "low_male": (105.0, 0.95),
    "mid_male": (135.0, 1.05),
    "female": (210.0, 1.18),
    "child": (290.0, 1.40),
}


def test_synth_pitch_is_the_requested_f0():
    sr = 16000
    y = synth_utterance(150.0, 1.0, n_vowels=1, seed=1, sr=sr)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    lo = (freqs > 80) & (freqs < 400)
    assert abs(freqs[lo][np.argmax(spec[lo])] - 150.0) < 25.0


def test_vocal_tract_scale_moves_the_formants():
    """Same f0, longer vs shorter tract: spectral energy shifts upward."""
    sr = 16000
    centroid = []
    for scale in (0.9, 1.3):
        y = synth_utterance(120.0, scale, n_vowels=4, seed=3, sr=sr)
        spec = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / sr)
        centroid.append((f * spec).sum() / spec.sum())
    assert centroid[1] > centroid[0]


def test_mfcc_shape():
    y = synth_utterance(120.0, 1.0, n_vowels=2, seed=0)
    C = mfcc(y, n_mfcc=20)
    assert C.shape[0] == 20
    assert C.shape[1] > 50  # ~100 frames per second of audio


def test_voiceprint_length_is_fixed_regardless_of_duration():
    a = voiceprint(synth_utterance(120.0, 1.0, n_vowels=3, seed=0))
    b = voiceprint(synth_utterance(120.0, 1.0, n_vowels=12, seed=1))
    assert a.shape == b.shape == (2 * 19,)


def test_same_speaker_is_closer_than_different_speaker():
    n = 16
    enrolled = voiceprint(synth_utterance(*SPEAKERS["low_male"], n_vowels=n, seed=0))
    same = voiceprint(synth_utterance(*SPEAKERS["low_male"], n_vowels=n, seed=1))
    other = voiceprint(synth_utterance(*SPEAKERS["female"], n_vowels=n, seed=1))
    assert distance(enrolled, same) < distance(enrolled, other)


def test_closed_set_identification_on_new_sentences():
    """Enrol on two sentences, identify on a third the model never saw."""
    n = 20
    gallery = {
        name: np.mean([voiceprint(synth_utterance(f0, sc, n_vowels=n, seed=s))
                       for s in (0, 1)], axis=0)
        for name, (f0, sc) in SPEAKERS.items()
    }
    hits = 0
    for name, (f0, sc) in SPEAKERS.items():
        for s in (100, 101):
            probe = voiceprint(synth_utterance(f0, sc, n_vowels=n, seed=s))
            hits += identify(probe, gallery)[0] == name
    assert hits >= 7  # of 8; the toy is not perfect and the test says so


def test_identify_rejects_empty_gallery():
    import pytest
    with pytest.raises(ValueError):
        identify(np.zeros(4), {})
