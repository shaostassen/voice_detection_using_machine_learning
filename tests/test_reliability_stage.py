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
    """Two words: one confident, one not."""
    def __init__(self, probs=(0.98, 0.20)):
        self.probs = probs

    def transcribe(self, y, language=None, cfg=None):
        words = [{"word": f" w{i}", "start": float(i), "end": float(i) + 0.5,
                  "prob": p} for i, p in enumerate(self.probs)]
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
    json.dumps(d)
    assert "reliability" in d
    assert d["reliability"]["estimator"] == "word_prob"


def test_reliability_key_present_even_when_unused():
    # The HTTP API returns to_dict() wholesale, so the key must exist for
    # clients rather than appearing only sometimes.
    d = _lens(WordTranscriber()).analyze((_tone(), 16000)).to_dict()
    assert d["reliability"] == {}
