"""Component tests for the production VAD path.

These use the actual Silero ONNX model bundled in the faster-whisper wheel —
no network, no GPU. Positive case uses espeak-ng TTS as a stand-in for real
speech; negative cases assert that silence and white noise are rejected,
which is precisely the anti-hallucination property the pipeline relies on.
"""
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("faster_whisper")

from speechlens.audio import load_audio
from speechlens.vad import silero_segments

ESPEAK = shutil.which("espeak-ng")


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory):
    if ESPEAK is None:
        pytest.skip("espeak-ng not installed")
    path = tmp_path_factory.mktemp("audio") / "speech.wav"
    subprocess.run(
        [ESPEAK, "-s", "150", "-w", str(path),
         "The quick brown fox jumps over the lazy dog, "
         "while the calibration microphone records everything."],
        check=True)
    return str(path)


def test_silero_rejects_silence():
    y = np.zeros(16000 * 3, dtype=np.float32)
    assert silero_segments(y) == []


def test_silero_rejects_white_noise():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.1, 16000 * 3).astype(np.float32)
    assert silero_segments(y) == []


def test_silero_accepts_speech(speech_wav):
    y, sr = load_audio(speech_wav)
    segs = silero_segments(y, sr)
    assert len(segs) >= 1
    assert sum(s.duration for s in segs) > 0.5
