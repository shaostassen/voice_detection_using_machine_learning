"""Estimate SNR from the VAD partition, so a reliability policy can be chosen.

Phase 3 left one thing unautomated: the calibrated accept/review policy is
condition-dependent — at 2% tolerated error, coverage runs from 90% on clean
speech to zero at 0 dB SNR — and the caller had to name the condition. Naming
it wrong is not a small error: choosing `clean` for 0 dB audio auto-accepts
most of a transcript that is 29% wrong.

The VAD already partitions the signal into speech and non-speech. Non-speech
regions are noise alone; speech regions are speech plus that same noise. That
is enough for the standard energy-ratio estimate, at no extra model cost.

Pure NumPy, so it is testable offline against signals with a known true SNR.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Above this the estimate stops being meaningful — non-speech regions of a
# studio recording are room tone, not the noise the speech competes with, so
# the ratio runs away. Everything past it is reported as "clean".
MAX_SNR_DB = 40.0

# Below roughly this much non-speech there is no reliable noise sample.
MIN_NOISE_S = 0.20
MIN_SPEECH_S = 0.30


def _mask(n: int, sr: int, spans: Sequence[Tuple[float, float]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for start, end in spans:
        a = max(0, int(start * sr))
        b = min(n, int(end * sr))
        if b > a:
            m[a:b] = True
    return m


def estimate_snr(y: np.ndarray, sr: int = 16000,
                 segments: Optional[Sequence] = None) -> Optional[float]:
    """Estimated SNR in dB, or ``None`` when the signal cannot support one.

    ``segments`` are speech spans — either ``SpeechSegment`` objects or plain
    ``(start, end)`` pairs. Omit them and VAD runs here.

    Returns ``None`` rather than a number whenever the estimate would be
    guesswork: no detected speech, or too little non-speech to sample the
    noise from. A missing estimate is recoverable by the caller; a fabricated
    one silently selects the wrong policy, which is the failure this whole
    layer exists to prevent.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return None

    if segments is None:
        from speechlens.vad import detect
        segments, _backend = detect(y.astype(np.float32), sr)

    spans: List[Tuple[float, float]] = []
    for s in segments or []:
        if isinstance(s, (tuple, list)):
            spans.append((float(s[0]), float(s[1])))
        else:
            spans.append((float(s.start), float(s.end)))

    speech = _mask(y.size, sr, spans)
    noise = ~speech
    if speech.sum() < MIN_SPEECH_S * sr or noise.sum() < MIN_NOISE_S * sr:
        return None

    p_speech = float(np.mean(y[speech] ** 2))
    p_noise = float(np.mean(y[noise] ** 2))
    if p_noise <= 0.0:
        return MAX_SNR_DB

    # Speech regions carry signal *plus* noise; subtract the noise floor so
    # the ratio is signal-to-noise rather than (signal+noise)-to-noise. The
    # difference is ~3 dB at 0 dB SNR, which is a whole rung of the ladder.
    p_signal = p_speech - p_noise
    if p_signal <= 0.0:
        return -MAX_SNR_DB

    snr = 10.0 * np.log10(p_signal / p_noise)
    return float(np.clip(snr, -MAX_SNR_DB, MAX_SNR_DB))


def nearest_condition(snr_db: Optional[float],
                      conditions: Sequence[str]) -> Optional[str]:
    """Pick the policy condition closest to an estimated SNR.

    Conditions are the bundled policy names: ``clean`` plus numeric dB rungs.
    ``clean`` is treated as the top of the range, and ties resolve **downward**
    to the noisier policy — being too cautious costs coverage, being too
    optimistic ships wrong words as trustworthy.

    ``None`` in gives ``None`` out: no estimate means no automatic choice.
    """
    # NaN as well as None. NaN arrives from any aggregation over an empty set
    # of estimates — e.g. averaging per-utterance SNRs when every one declined,
    # which is exactly what babble noise causes. Left unguarded, NaN compares
    # false against everything, the sort keeps insertion order, and the
    # function silently returns a policy instead of abstaining.
    if snr_db is None or snr_db != snr_db or not conditions:
        return None

    scored = []
    for c in conditions:
        value = MAX_SNR_DB if c == "clean" else float(c)
        # Second key ascending on the *value*, so an exact tie takes the lower
        # SNR — the more conservative policy.
        scored.append((abs(value - snr_db), value, c))
    scored.sort()
    return scored[0][2]
