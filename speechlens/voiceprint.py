"""A deliberately simple voiceprint: MFCC statistics + nearest-neighbour.

This demonstrates the *shape* of speaker recognition — turn a variable-length
recording into one fixed-length vector that is stable for a speaker and
different between speakers, then compare vectors — using only NumPy.

Production systems replace the hand-built vector with a neural *speaker
embedding* (x-vector, ECAPA-TDNN) trained on thousands of speakers with a loss
that pulls same-speaker utterances together. That is what makes them survive
different words, microphones and rooms. This toy has none of that robustness:
treat it as an explainer, not as a security control.
"""
from __future__ import annotations

import numpy as np

from speechlens.features import log_mel


def _dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II basis, shape (n_out, n_in)."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    m = np.cos(np.pi * (n + 0.5) * k / n_in) * np.sqrt(2.0 / n_in)
    m[0] /= np.sqrt(2.0)
    return m


def mfcc(y: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    """Mel-frequency cepstral coefficients, shape (n_mfcc, n_frames).

    The DCT of the log-mel spectrum separates the smooth spectral *envelope*
    (low coefficients: the vocal-tract filter, i.e. vowel and speaker shape)
    from the fine harmonic *ripple* (high coefficients: the pitch source).
    """
    L = log_mel(y, sr=sr)
    return _dct_matrix(n_mfcc, L.shape[0]) @ L


def voiceprint(y: np.ndarray, sr: int = 16000, n_mfcc: int = 20,
               energy_floor_db: float = 25.0) -> np.ndarray:
    """Fixed-length vector for a recording: per-coefficient mean and std.

    Frames more than `energy_floor_db` below the loudest are dropped so
    silence does not dilute the statistics (a crude stand-in for a VAD).
    c0 (overall loudness) is discarded; loudness says nothing about who spoke.
    """
    L = log_mel(y, sr=sr)
    C = _dct_matrix(n_mfcc, L.shape[0]) @ L
    level = L.mean(axis=0)  # per-frame loudness, log10 power
    keep = level >= level.max() - energy_floor_db / 10.0
    C = C[1:, keep]
    return np.concatenate([C.mean(axis=1), C.std(axis=1)])


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two voiceprints (smaller = more alike)."""
    return float(np.linalg.norm(a - b))


def identify(probe: np.ndarray, gallery: dict[str, np.ndarray]) -> tuple[str, float]:
    """Closed-set identification: nearest enrolled speaker and its distance."""
    if not gallery:
        raise ValueError("gallery is empty")
    name = min(gallery, key=lambda k: distance(probe, gallery[k]))
    return name, distance(probe, gallery[name])
