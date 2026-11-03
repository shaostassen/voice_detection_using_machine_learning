"""A toy source-filter voice in pure NumPy — for teaching and offline tests.

Human speech is well modelled as two independent parts:

* **source** — the vocal folds buzz at a fundamental frequency f0 (a pulse
  train, rich in harmonics). f0 is what you hear as pitch.
* **filter** — the throat/mouth/nose act as a chain of resonators. Each
  resonance (a *formant*) boosts a band of harmonics. Which vowel you say is
  decided by the first two or three formant frequencies; *who* is speaking is
  partly decided by how long their vocal tract is, which scales every formant.

So a "speaker" here is just (f0, vocal-tract scale) and an "utterance" is a
random sequence of vowels. That is enough structure to make a speaker
recognisable to a classifier while being obviously not real speech. It exists
so the explainer and tests can run with no audio files and no model weights.
This is **not** a speech synthesiser and does not clone anyone.
"""
from __future__ import annotations

import numpy as np

# (F1, F2, F3) in Hz for an average adult male vocal tract (Peterson & Barney
# style values, rounded).
VOWELS: dict[str, tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
}
_BANDWIDTHS = (80.0, 100.0, 140.0)


def _resonator(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    """Second-order IIR resonance at `freq` Hz with `bw` Hz bandwidth."""
    r = np.exp(-np.pi * bw / sr)
    a1 = 2.0 * r * np.cos(2.0 * np.pi * freq / sr)
    a2 = -r * r
    gain = 1.0 - r  # keeps the cascade at a sane level
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for n in range(len(x)):
        yn = gain * x[n] + a1 * y1 + a2 * y2
        y[n] = yn
        y2, y1 = y1, yn
    return y


def _vowel(f0: float, scale: float, vowel: str, dur: float, sr: int,
           rng: np.random.Generator) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    # Slow vibrato + jitter so the pitch is not a laboratory-perfect constant.
    inst_f0 = f0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 6.28))
                    + 0.003 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    phase = np.cumsum(inst_f0 / sr)
    pulses = (np.diff(np.floor(phase), prepend=0.0) > 0).astype(np.float64)
    # Glottal roll-off: a one-pole low-pass tilts the harmonics down ~6 dB/oct.
    src = np.zeros(n)
    prev = 0.0
    for i in range(n):
        prev = pulses[i] + 0.95 * prev
        src[i] = prev
    src += 0.002 * rng.standard_normal(n)  # breath noise
    y = src
    for f, bw in zip(VOWELS[vowel], _BANDWIDTHS):
        y = _resonator(y, min(f * scale, 0.45 * sr), bw, sr)
    return y * np.hanning(n) ** 0.5  # fade edges so segments can overlap-add


def synth_utterance(f0: float, scale: float, *, n_vowels: int = 8,
                    seed: int = 0, sr: int = 16000,
                    vowel_dur: float = 0.22) -> np.ndarray:
    """Random vowel sequence for one toy speaker, float32, peak-normalised.

    `f0` is the mean pitch in Hz (≈100–130 male, ≈190–230 female);
    `scale` multiplies every formant (≈1.0 male, ≈1.15–1.2 female, ≈1.3+ child).
    Different `seed`s give different "sentences" from the *same* speaker.
    """
    rng = np.random.default_rng(seed)
    keys = list(VOWELS)
    hop = int(vowel_dur * sr * 0.8)
    out = np.zeros(hop * n_vowels + int(vowel_dur * sr))
    for k in range(n_vowels):
        seg = _vowel(f0 * rng.uniform(0.95, 1.05), scale,
                     keys[int(rng.integers(len(keys)))], vowel_dur, sr, rng)
        out[k * hop:k * hop + len(seg)] += seg
    out /= max(float(np.max(np.abs(out))), 1e-9)
    return (0.9 * out).astype(np.float32)
