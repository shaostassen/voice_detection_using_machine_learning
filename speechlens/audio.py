"""Audio I/O and preprocessing.

Canonical internal format everywhere downstream: float32 mono PCM at 16 kHz,
values nominally in [-1, 1]. Speech energy lives almost entirely below 8 kHz,
so 16 kHz (Nyquist = 8 kHz) is the universal ASR convention.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np

TARGET_SR = 16000


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio to mono by averaging channels."""
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    # Channels sit on whichever axis is smaller (<= 8 channels assumed).
    ch_axis = 1 if x.shape[1] <= x.shape[0] else 0
    return x.mean(axis=ch_axis)


def resample(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Anti-aliased polyphase resampling with an exact rational rate change."""
    if sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly

    g = gcd(target_sr, sr)
    return resample_poly(x, target_sr // g, sr // g).astype(np.float32)


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-8:
        return x
    return (x * (peak / m)).astype(np.float32)


def from_array(x: np.ndarray, sr: int, normalize: bool = False) -> Tuple[np.ndarray, int]:
    """Convert an arbitrary PCM array into the canonical format."""
    y = to_mono(np.asarray(x, dtype=np.float32))
    y = resample(y, sr, TARGET_SR)
    if normalize:
        y = peak_normalize(y)
    return np.ascontiguousarray(y, dtype=np.float32), TARGET_SR


def load_audio(source: Union[str, Path, BinaryIO],
               target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load any container format (wav/mp3/m4a/webm/...) as 16 kHz mono float32.

    Primary path is faster-whisper's PyAV decoder (bundled FFmpeg), so the same
    formats work from the CLI, the HTTP API, and browser MediaRecorder blobs.
    Falls back to soundfile + scipy for plain PCM formats.
    """
    try:
        from faster_whisper.audio import decode_audio

        y = decode_audio(source, sampling_rate=target_sr)
        return np.ascontiguousarray(y, dtype=np.float32), target_sr
    except Exception:
        import soundfile as sf

        data, sr = sf.read(source, dtype="float32", always_2d=False)
        return from_array(data, sr)
