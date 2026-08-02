"""Pipeline orchestration tests.

The transcriber and detector are injected fakes, so these verify the actual
application logic — routing, forced-language bypass, warnings, JSON schema —
without needing model weights or a GPU.
"""
import json
import math
from types import SimpleNamespace

import numpy as np

from speechlens.asr import TranscriptSegment
from speechlens.lid import LIDResult
from speechlens.pipeline import SpeechLens


class FakeTranscriber:
    def transcribe(self, y, language=None, cfg=None):
        seg = TranscriptSegment(0, 0.0, 1.0, "hello world", -0.2, 0.01,
                                math.exp(-0.2), False)
        info = SimpleNamespace(language=language or "en",
                               language_probability=0.99)
        return [seg], info


class FakeDetector:
    def __init__(self):
        self.called = False

    def detect(self, y, sr=16000, speech_segments=None, max_chunks=3):
        self.called = True
        return LIDResult("en", "English", 0.97,
                         [("en", 0.97), ("de", 0.02)], 0.15, 3, False)


def _tone(seconds=3.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _lens(det=None):
    return SpeechLens(transcriber=FakeTranscriber(),
                      detector=det or FakeDetector(), model_size="fake")


def test_orchestration_and_json_schema():
    result = _lens().analyze((_tone(), 16000))
    d = result.to_dict()
    json.dumps(d)  # must be JSON-serializable end to end
    assert d["language"]["code"] == "en"
    assert d["transcript"]["text"] == "hello world"
    assert d["performance"]["rtf"] is not None
    assert 0.0 <= d["audio"]["speech_ratio"] <= 1.0
    assert d["audio"]["vad_backend"] in ("silero", "energy")


def test_forced_language_skips_lid():
    det = FakeDetector()
    result = _lens(det).analyze((_tone(), 16000), language="zh")
    assert det.called is False
    assert result.language["code"] == "zh"
    assert result.language["method"] == "forced"
    assert result.language["probability"] == 1.0


def test_short_clip_warning():
    result = _lens().analyze((_tone(seconds=1.0), 16000))
    assert any("under 2" in w for w in result.warnings)
