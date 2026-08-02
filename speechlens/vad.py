"""Voice activity detection.

Primary backend: Silero VAD via the ONNX assets bundled inside the
faster-whisper wheel (no network calls, no torch). Fallback: an
energy/hysteresis detector so the pipeline degrades gracefully anywhere.

VAD is the single highest-leverage anti-hallucination measure: Whisper
hallucinates fluent text on silence and music, so we never let it see any.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SpeechSegment:
    start: float  # seconds
    end: float    # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


def _merge(segments: List[SpeechSegment], min_gap_s: float = 0.2) -> List[SpeechSegment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    out = [segments[0]]
    for seg in segments[1:]:
        if seg.start - out[-1].end <= min_gap_s:
            out[-1] = SpeechSegment(out[-1].start, max(out[-1].end, seg.end))
        else:
            out.append(seg)
    return out


def silero_segments(y: np.ndarray, sr: int = 16000, threshold: float = 0.5,
                    min_speech_ms: int = 250, min_silence_ms: int = 100,
                    pad_ms: int = 30) -> List[SpeechSegment]:
    """Silero VAD using faster-whisper's bundled model (expects 16 kHz)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    try:
        opts = VadOptions(threshold=threshold,
                          min_speech_duration_ms=min_speech_ms,
                          min_silence_duration_ms=min_silence_ms,
                          speech_pad_ms=pad_ms)
    except TypeError:  # option names drifted between versions
        opts = VadOptions()
    try:
        raw = get_speech_timestamps(y.astype(np.float32), vad_options=opts)
    except TypeError:
        raw = get_speech_timestamps(y.astype(np.float32), opts)

    duration = len(y) / sr
    segs = []
    for t in raw:
        s, e = float(t["start"]), float(t["end"])
        if e > duration * 1.5:  # values are sample indices, not seconds
            s, e = s / sr, e / sr
        segs.append(SpeechSegment(s, e))
    return _merge(segs)


def energy_segments(y: np.ndarray, sr: int = 16000, frame_ms: int = 30,
                    min_speech_ms: int = 200, hang_ms: int = 120) -> List[SpeechSegment]:
    """Frame-RMS detector with an adaptive threshold and hangover.

    Not speech-specific (a loud tone counts), but dependency-free and good
    enough to bound speech regions when Silero is unavailable.
    """
    frame = max(1, int(sr * frame_ms / 1000))
    n = len(y) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1))
    db = 20.0 * np.log10(rms + 1e-10)
    thr = max(float(db.max()) - 30.0, float(np.percentile(db, 20)) + 8.0)
    mask = db > thr

    hang = int(np.ceil(hang_ms / frame_ms))
    segs: List[SpeechSegment] = []
    start = None
    quiet = 0
    for i, active in enumerate(mask):
        if active:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet > hang:
                segs.append(SpeechSegment(start * frame / sr,
                                          (i - quiet + 1) * frame / sr))
                start, quiet = None, 0
    if start is not None:
        segs.append(SpeechSegment(start * frame / sr, n * frame / sr))

    segs = _merge(segs)
    min_len = min_speech_ms / 1000.0
    return [s for s in segs if s.duration >= min_len]


def detect(y: np.ndarray, sr: int = 16000,
           backend: str = "auto") -> Tuple[List[SpeechSegment], str]:
    """Return (segments, backend_used). backend: auto | silero | energy."""
    if backend in ("auto", "silero"):
        try:
            return silero_segments(y, sr), "silero"
        except Exception:
            if backend == "silero":
                raise
    return energy_segments(y, sr), "energy"


def speech_ratio(segments: List[SpeechSegment], total_duration: float) -> float:
    if total_duration <= 0:
        return 0.0
    return min(1.0, sum(s.duration for s in segments) / total_duration)
