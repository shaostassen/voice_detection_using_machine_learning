import numpy as np

from speechlens.features import filterbank_centers, log_mel, mel_filterbank


def test_shapes():
    y = np.random.default_rng(0).normal(size=16000).astype(np.float32)
    L = log_mel(y)
    # 1 s @ 16 kHz, hop 160, centered -> 101 frames of 80 mel bins
    assert L.shape == (80, 101)


def test_filterbank_rows_nonzero_and_centers_monotonic():
    fb = mel_filterbank()
    assert fb.shape == (80, 201)
    assert (fb.sum(axis=1) > 0).all()
    c = filterbank_centers()
    assert (np.diff(c) > 0).all()


def test_tone_lands_in_correct_mel_bin():
    """A 1 kHz sine must excite the mel filter centered nearest 1 kHz."""
    sr, f = 16000, 1000.0
    t = np.arange(sr) / sr
    y = np.sin(2 * np.pi * f * t).astype(np.float32)
    L = log_mel(y, sr=sr)
    hot = int(np.argmax(L.mean(axis=1)))
    expected = int(np.argmin(np.abs(filterbank_centers(sr=sr) - f)))
    assert abs(hot - expected) <= 1


def test_dynamic_range_clamp():
    y = np.random.default_rng(0).normal(size=16000).astype(np.float32)
    L = log_mel(y)
    assert float(L.max() - L.min()) <= 8.0 + 1e-9
