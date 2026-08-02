import numpy as np

from speechlens.audio import from_array, peak_normalize, resample, to_mono


def test_to_mono_stereo():
    x = np.random.default_rng(0).normal(size=(1000, 2)).astype(np.float32)
    m = to_mono(x)
    assert m.shape == (1000,)
    np.testing.assert_allclose(m, x.mean(axis=1), rtol=1e-6)


def test_resample_length():
    sr = 22050
    y = np.random.default_rng(0).normal(size=sr).astype(np.float32)  # 1 s
    out = resample(y, sr, 16000)
    assert abs(len(out) - 16000) <= 2


def test_resample_preserves_tone():
    """A 440 Hz tone must still be a 440 Hz tone after 44.1k -> 16k."""
    sr, f = 44100, 440.0
    t = np.arange(sr) / sr
    y = np.sin(2 * np.pi * f * t).astype(np.float32)
    out = resample(y, sr, 16000)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / 16000)[int(np.argmax(spec))]
    assert abs(peak - f) < 2.0


def test_peak_normalize():
    y = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(y, peak=0.95)
    assert abs(float(np.max(np.abs(out))) - 0.95) < 1e-6


def test_from_array_canonical():
    y = np.random.default_rng(0).normal(size=(44100, 2)).astype(np.float32)
    out, sr = from_array(y, 44100)
    assert sr == 16000 and out.dtype == np.float32 and out.ndim == 1
