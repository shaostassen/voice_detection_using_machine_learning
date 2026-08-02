"""Whisper-style log-mel spectrogram in pure NumPy.

This module exists for analysis, education, and unit testing; production ASR
uses faster-whisper's internal frontend. Parameters mirror Whisper:
16 kHz input, n_fft=400 (25 ms), hop=160 (10 ms) -> 100 frames/s, 80 mel bins
(large-v3 uses 128). The mel scale here is HTK, m = 2595*log10(1 + f/700);
Whisper uses the slaney variant, which is numerically close and irrelevant for
the invariants tested here.
"""
from __future__ import annotations

import numpy as np


def hz_to_mel(f) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = 16000, n_fft: int = 400, n_mels: int = 80,
                   fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1), slaney-normalized."""
    fmax = fmax or sr / 2
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
        fb[i] *= 2.0 / (hi - lo)  # slaney area normalization
    return fb


def filterbank_centers(sr: int = 16000, n_mels: int = 80,
                       fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Center frequency (Hz) of each mel filter."""
    fmax = fmax or sr / 2
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    return mel_to_hz(mel_pts)[1:-1]


def stft_power(y: np.ndarray, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """Power spectrogram, shape (n_fft//2 + 1, n_frames). Hann window, centered."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    pad = n_fft // 2
    y = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (len(y) - n_fft) // hop
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, n_fft),
        strides=(y.strides[0] * hop, y.strides[0]),
    )
    spec = np.fft.rfft(frames * window, axis=1)
    return (np.abs(spec) ** 2).T


def log_mel(y: np.ndarray, sr: int = 16000, n_fft: int = 400,
            hop: int = 160, n_mels: int = 80) -> np.ndarray:
    """Log-mel spectrogram with Whisper's 8-decade dynamic-range clamp."""
    S = stft_power(y, n_fft=n_fft, hop=hop)
    M = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels) @ S
    L = np.log10(np.maximum(M, 1e-10))
    return np.maximum(L, L.max() - 8.0)
