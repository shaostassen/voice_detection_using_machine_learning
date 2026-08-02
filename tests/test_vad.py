import numpy as np

from speechlens.vad import SpeechSegment, detect, energy_segments, speech_ratio


def _synthetic(sr=16000):
    """1 s near-silence, 1 s broadband burst, 1 s near-silence."""
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1e-4, sr).astype(np.float32)
    burst = rng.normal(0, 0.3, sr).astype(np.float32)
    return np.concatenate([quiet, burst, quiet.copy()])


def test_energy_vad_finds_burst():
    segs = energy_segments(_synthetic(), 16000)
    assert len(segs) == 1
    assert abs(segs[0].start - 1.0) < 0.1
    assert abs(segs[0].end - 2.0) < 0.2


def test_energy_vad_silence_yields_nothing():
    assert energy_segments(np.zeros(16000, dtype=np.float32), 16000) == []


def test_speech_ratio():
    segs = [SpeechSegment(0.0, 1.0), SpeechSegment(2.0, 3.0)]
    assert abs(speech_ratio(segs, 4.0) - 0.5) < 1e-9


def test_detect_energy_backend():
    segs, backend = detect(_synthetic(), 16000, backend="energy")
    assert backend == "energy" and len(segs) == 1
