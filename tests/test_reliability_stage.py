"""The per-word reliability stage, exercised through the DI seam.

No model weights: a fake transcriber returns segments carrying words, exactly
as faster-whisper would with word_timestamps on.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from speechlens.asr import TranscriptSegment
from speechlens.calibration import ReliabilityPolicy
from speechlens.lid import LIDResult
from speechlens.pipeline import SpeechLens


def _policy(target_risk=0.05):
    # High scores mostly correct, low scores mostly wrong.
    scores = [i / 99 for i in range(100)]
    correct = [i > 30 for i in range(100)]
    return ReliabilityPolicy.fit(scores, correct, target_risk=target_risk,
                                 condition="test")


class FakeDetector:
    def detect(self, y, sr=16000, speech_segments=None, max_chunks=3):
        return LIDResult("en", "English", 0.97, [("en", 0.97)], 0.1, 3, False)


class WordTranscriber:
    """Two words: one confident, one not.

    Probabilities are ``np.float64`` because that is what faster-whisper
    actually emits (``np.mean`` over token probabilities). It matters: a
    Python float here made an earlier version of this test pass while
    ``--json`` crashed in reality, since comparing a numpy float yields
    ``numpy.bool_``, which is not JSON-serializable.
    """
    def __init__(self, probs=(0.98, 0.20)):
        self.probs = probs

    def transcribe(self, y, language=None, cfg=None):
        words = [{"word": f" w{i}", "start": float(i), "end": float(i) + 0.5,
                  "prob": np.float64(p)} for i, p in enumerate(self.probs)]
        seg = TranscriptSegment(0, 0.0, 1.0, "w0 w1", -0.2, 0.01,
                                math.exp(-0.2), False, words)
        return [seg], SimpleNamespace(language="en", language_probability=0.99)


class NoWordTranscriber:
    def transcribe(self, y, language=None, cfg=None):
        seg = TranscriptSegment(0, 0.0, 1.0, "hello", -0.2, 0.01,
                                math.exp(-0.2), False)
        return [seg], SimpleNamespace(language="en", language_probability=0.99)


def _tone(seconds=3.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _lens(transcriber, policy=None):
    return SpeechLens(transcriber=transcriber, detector=FakeDetector(),
                      model_size="fake", policy=policy)


def test_no_policy_means_no_reliability_block():
    # Absent, not invented: the default install ships no fitted policy.
    r = _lens(WordTranscriber()).analyze((_tone(), 16000))
    assert r.reliability == {}
    words = r.transcript["segments"][0]["words"]
    assert "reliability" not in words[0]


def test_policy_annotates_every_word():
    r = _lens(WordTranscriber(), _policy()).analyze((_tone(), 16000))
    words = r.transcript["segments"][0]["words"]
    assert len(words) == 2
    for w in words:
        assert 0.0 <= w["reliability"] <= 1.0
        assert isinstance(w["accept"], bool)


def test_confident_word_accepted_and_weak_word_flagged_for_review():
    r = _lens(WordTranscriber(probs=(0.98, 0.05)), _policy()).analyze((_tone(), 16000))
    w_hi, w_lo = r.transcript["segments"][0]["words"]
    assert w_hi["reliability"] > w_lo["reliability"]
    assert w_hi["accept"] is True
    assert w_lo["accept"] is False
    assert r.reliability["accepted"] == 1
    assert r.reliability["review"] == 1
    assert r.reliability["coverage"] == pytest.approx(0.5)


def test_reliability_block_reports_its_operating_point():
    pol = _policy(target_risk=0.05)
    r = _lens(WordTranscriber(), pol).analyze((_tone(), 16000))
    rel = r.reliability
    assert rel["target_risk"] == pytest.approx(0.05)
    assert rel["threshold"] == pytest.approx(round(pol.threshold, 3))
    assert rel["condition"] == "test"
    assert rel["words"] == 2


def test_missing_word_timestamps_warns_instead_of_guessing():
    r = _lens(NoWordTranscriber(), _policy()).analyze((_tone(), 16000))
    assert r.reliability == {}
    assert any("word timestamps" in w for w in r.warnings)


def test_result_stays_json_serializable_with_reliability():
    r = _lens(WordTranscriber(), _policy()).analyze((_tone(), 16000))
    d = r.to_dict()
    json.dumps(d)          # the whole document, as `--json` writes it
    assert "reliability" in d
    assert d["reliability"]["estimator"] == "word_prob"


def test_annotations_are_plain_python_types():
    # Guards the numpy leak directly: json.dump fails *mid-write* on
    # numpy.bool_, leaving a half-written file rather than a clean error.
    r = _lens(WordTranscriber(), _policy()).analyze((_tone(), 16000))
    for w in r.transcript["segments"][0]["words"]:
        assert type(w["accept"]) is bool
        assert type(w["reliability"]) is float
    assert type(r.reliability["accepted"]) is int
    assert type(r.reliability["coverage"]) is float


def _stub_vad(monkeypatch, spans):
    """Force the VAD result so these test policy resolution, not Silero.

    Silero legitimately rejects synthetic tones as non-speech — that is the
    anti-hallucination property it exists for — so real VAD on a sine wave
    yields no speech and the SNR estimate correctly declines. Stubbing keeps
    the subject of the test the resolution path.
    """
    import speechlens.pipeline as pl
    from speechlens.vad import SpeechSegment
    segs = [SpeechSegment(a, b) for a, b in spans]
    monkeypatch.setattr(pl, "vad_detect", lambda y, sr: (segs, "stub"))


def test_auto_selects_a_policy_from_the_audio(monkeypatch):
    # Loud tone between two silences: a clean, high-SNR signal.
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    tone = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    y = np.concatenate([np.zeros(sr, np.float32), tone,
                        np.zeros(sr, np.float32)])
    _stub_vad(monkeypatch, [(1.0, 3.0)])

    r = _lens(WordTranscriber()).analyze((y, sr), policy="auto")
    assert r.reliability["selected_by"].startswith("auto")
    assert r.reliability["condition"] == "clean"


def test_auto_picks_a_noisier_policy_for_noisier_audio(monkeypatch):
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(int(2.0 * sr)) / sr
    tone = 0.4 * np.sin(2 * np.pi * 180 * t)
    y = np.concatenate([np.zeros(2 * sr), tone, np.zeros(2 * sr)])
    p_sig = float(np.mean(tone ** 2))
    y = (y + rng.normal(0, np.sqrt(p_sig), y.size)).astype(np.float32)  # 0 dB
    _stub_vad(monkeypatch, [(2.0, 4.0)])

    r = _lens(WordTranscriber()).analyze((y, sr), policy="auto")
    assert r.reliability["condition"] == "0"


def test_auto_declines_rather_than_defaulting_when_it_cannot_estimate(monkeypatch):
    # All speech, no noise sample. Guessing `clean` here would auto-accept a
    # transcript that might be 29% wrong, so the honest result is no policy.
    sr = 16000
    t = np.arange(int(3.0 * sr)) / sr
    y = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    _stub_vad(monkeypatch, [(0.0, 3.0)])

    r = _lens(WordTranscriber()).analyze((y, sr), policy="auto")
    assert r.reliability == {}
    assert any("estimate SNR" in w for w in r.warnings)


def test_per_call_policy_overrides_without_touching_shared_state():
    # The HTTP server shares one pipeline across requests. Passing the policy
    # per call must not leak it into the next request, which assigning to
    # `lens.policy` would.
    lens = _lens(WordTranscriber())
    assert lens.analyze((_tone(), 16000), policy=_policy()).reliability != {}
    assert lens.policy is None
    assert lens.analyze((_tone(), 16000)).reliability == {}


def test_reliability_key_present_even_when_unused():
    # The HTTP API returns to_dict() wholesale, so the key must exist for
    # clients rather than appearing only sometimes.
    d = _lens(WordTranscriber()).analyze((_tone(), 16000)).to_dict()
    assert d["reliability"] == {}
