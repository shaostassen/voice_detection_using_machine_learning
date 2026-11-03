import numpy as np
import pytest

from speechlens.speaker import (Gallery, cosine, equal_error_rate,
                                threshold_for_far)
from speechlens.synth import synth_utterance
from speechlens.voiceprint import voiceprint

SR = 16000
SPEAKERS = {
    "low_male": (105.0, 0.95),
    "mid_male": (135.0, 1.05),
    "female": (210.0, 1.18),
    "child": (290.0, 1.40),
}
N = 20


@pytest.fixture(scope="module")
def embedder():
    """Toy encoder: voiceprint, standardised against unrelated reference voices.

    Raw MFCC means are all similar (cosine ~1 for everyone); centring on a
    population is what makes cosine discriminative. Real encoders learn this.
    """
    ref = np.array([voiceprint(synth_utterance(f, s, n_vowels=12, seed=900 + i))
                    for i, (f, s) in enumerate([(120, 1.0), (180, 1.1),
                                                (240, 1.25), (150, 0.9),
                                                (160, 1.1)])])
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    return lambda y, sr: (voiceprint(y, sr=sr) - mu) / sd


@pytest.fixture(scope="module")
def clips():
    return {n: [synth_utterance(f, s, n_vowels=N, seed=k) for k in range(4)]
            for n, (f, s) in SPEAKERS.items()}


def make_gallery(embedder, clips):
    g = Gallery(embedder, encoder_id="toy-v1")
    for n, cs in clips.items():
        for c in cs[:2]:
            g.enroll(n, c, SR)
    return g


def test_closed_set_identification(embedder, clips):
    g = make_gallery(embedder, clips)
    hits = sum(g.identify(c, SR)[0] == n for n, cs in clips.items() for c in cs[2:])
    assert hits >= 7  # of 8; the toy is imperfect and the test says so


def test_open_set_threshold_can_reject(embedder, clips):
    g = make_gallery(embedder, clips)
    stranger = synth_utterance(165.0, 1.1, n_vowels=N, seed=77)
    name, score, ranking = g.identify(stranger, SR, threshold=None)
    assert name is not None                       # closed-set always answers
    name2, _, _ = g.identify(stranger, SR, threshold=score + 1e-6)
    assert name2 is None                          # a threshold allows "none"
    assert ranking == sorted(ranking, key=lambda t: -t[1])


def test_verify_requires_enrolled_claim_and_explicit_threshold(embedder, clips):
    g = make_gallery(embedder, clips)
    with pytest.raises(KeyError):
        g.verify(clips["female"][2], SR, "nobody", threshold=0.5)
    with pytest.raises(TypeError):
        g.verify(clips["female"][2], SR, "female")  # no default threshold
    ok, s = g.verify(clips["female"][2], SR, "female", threshold=0.0)
    assert ok and s > 0.0
    ok_wrong, s_wrong = g.verify(clips["female"][2], SR, "child", threshold=0.0)
    assert s_wrong < s


def test_save_load_roundtrip_and_encoder_mismatch(tmp_path, embedder, clips):
    g = make_gallery(embedder, clips)
    path = tmp_path / "g.npz"
    g.save(path)
    g2 = Gallery.load(path, embedder, encoder_id="toy-v1")
    probe = clips["child"][3]
    assert g2.identify(probe, SR)[:2] == g.identify(probe, SR)[:2]
    with pytest.raises(ValueError, match="different models"):
        Gallery.load(path, embedder, encoder_id="other-encoder")


def test_empty_gallery_and_missing_encoder_id(embedder):
    with pytest.raises(ValueError):
        Gallery(embedder, encoder_id="")
    with pytest.raises(ValueError):
        Gallery(embedder, "x").scores(np.zeros(SR, np.float32), SR)


def test_zero_embedding_is_an_error_not_a_score():
    g = Gallery(lambda y, sr: np.zeros(4), "zero")
    with pytest.raises(ValueError, match="zero"):
        g.enroll("a", np.zeros(100, np.float32), SR)
    assert np.isnan(cosine(np.zeros(3), np.ones(3)))


def test_threshold_for_far_controls_false_accepts():
    rng = np.random.default_rng(0)
    imp = rng.normal(0.2, 0.15, size=2000)
    for far in (0.1, 0.01):
        t = threshold_for_far(imp, far)
        # Continuous scores, no ties: the achieved FAR must equal the target
        # exactly, i.e. neither too lax nor wastefully strict by even one trial.
        assert np.mean(imp >= t) == pytest.approx(far, abs=0.5 / len(imp))


def test_threshold_for_far_refuses_unresolvable_target():
    with pytest.raises(ValueError, match="cannot resolve"):
        threshold_for_far(np.linspace(0, 1, 50), 0.001)


def test_equal_error_rate_known_cases():
    perfect, _ = equal_error_rate([0.9, 0.8, 0.95], [0.1, 0.2, 0.0])
    assert perfect == 0.0
    chance, _ = equal_error_rate([0.5] * 10, [0.5] * 10)
    assert chance == pytest.approx(0.5, abs=0.01)


def test_calibrated_threshold_on_toy_scores(embedder, clips):
    """End to end: score same/different pairs, derive a threshold, apply it."""
    emb = {n: [embedder(c, SR) for c in cs] for n, cs in clips.items()}
    gen = [cosine(emb[n][i], emb[n][j]) for n in emb
           for i in range(4) for j in range(i + 1, 4)]
    imp = [cosine(emb[a][i], emb[b][j]) for a in emb for b in emb if a != b
           for i in range(4) for j in range(4)]
    eer, _ = equal_error_rate(gen, imp)
    assert eer < 0.15            # separable, but not perfect: report, don't hide
    t = threshold_for_far(imp, 0.05)
    assert np.mean(np.array(imp) >= t) <= 0.05
