#!/usr/bin/env python3
"""Corpora and noise types for the reliability studies.

Everything measured so far — per-word confidence works, entropy does not help,
the SNR estimator recovers the ladder — comes from one corpus of read English
with additive white noise. That is a finding about one setup, not a general
result, so this module exists to vary the two axes that could break it:

**Corpus.** `librispeech` is read, scripted, studio-clean, one accent family.
`edacc` is spontaneous dyadic conversation from the Edinburgh International
Accents of English corpus — real recording conditions, disfluencies, L1 and
L2 varieties, with accent metadata per utterance.

**Noise.** White noise is spectrally flat and stationary, which is the easiest
case for both ASR and for an energy-based SNR estimate. `pink` tilts the
spectrum toward where speech energy actually lives, and `babble` is built from
other utterances in the corpus — real speech spectrum, non-stationary, and the
hardest case for the SNR estimator because the noise itself looks like speech
to a VAD.

No torch: audio is loaded with `decode=False` and read through soundfile.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

Utterance = Tuple[np.ndarray, int, str, dict]   # audio, sr, reference, meta

# EdAcc marks stretches that its own scoring protocol excludes. Leaving them
# in would compare a transcript against a placeholder string and report a
# catastrophic, meaningless WER.
EDACC_SKIP = "IGNORE_TIME_SEGMENT_IN_SCORING"

MIN_DURATION_S = 2.0     # the pipeline warns below this; LID is unreliable
MIN_WORDS = 3            # a 1-2 word reference gives a hopelessly coarse WER


def _read(row) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    a = row["audio"]
    raw = a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(sr)


def load_librispeech(limit: int) -> List[Utterance]:
    """LibriSpeech dummy validation split — read English, studio conditions."""
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    out: List[Utterance] = []
    for i in range(min(limit, len(ds))):
        row = ds[i]
        y, sr = _read(row)
        out.append((y, sr, row["text"].strip(), {"accent": "read/US"}))
    return out


def load_edacc(limit: int, scan: int = 4000) -> List[Utterance]:
    """EdAcc test split — spontaneous, accented, real recording conditions.

    Streamed rather than downloaded (the corpus is ~40 h). Utterances arrive
    grouped by conversation, so taking the first N would sample a single
    speaker; this scans further and keeps a round-robin across speakers so the
    subset spans accents instead of one voice.
    """
    from datasets import Audio, load_dataset
    ds = load_dataset("edinburghcstr/edacc", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    by_speaker: dict = {}
    seen = 0
    for row in ds:
        seen += 1
        if seen > scan:
            break
        text = (row.get("text") or "").strip()
        if not text or EDACC_SKIP in text or len(text.split()) < MIN_WORDS:
            continue
        y, sr = _read(row)
        if len(y) / sr < MIN_DURATION_S:
            continue
        spk = row.get("speaker") or "?"
        by_speaker.setdefault(spk, []).append(
            (y, sr, text, {"accent": row.get("accent") or "?",
                           "l1": row.get("l1") or "?", "speaker": spk}))

    # Round-robin so the subset is spread across speakers, not front-loaded.
    out: List[Utterance] = []
    queues = [list(v) for v in by_speaker.values()]
    while queues and len(out) < limit:
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
        queues = [q for q in queues if q]
    return out


CORPORA = {"librispeech": load_librispeech, "edacc": load_edacc}


def load_corpus(name: str, limit: int) -> List[Utterance]:
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; have {sorted(CORPORA)}")
    return CORPORA[name](limit)


# --- noise ------------------------------------------------------------------

def _scale_to_snr(clean: np.ndarray, noise: np.ndarray,
                  snr_db: float) -> np.ndarray:
    p_clean = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    target = p_clean / (10 ** (snr_db / 10))
    return clean + noise * np.sqrt(target / max(p_noise, 1e-12))


def white(clean: np.ndarray, snr_db: float, rng, **_) -> np.ndarray:
    n = rng.normal(0, 1, len(clean)).astype(np.float32)
    return _scale_to_snr(clean, n, snr_db)


def pink(clean: np.ndarray, snr_db: float, rng, **_) -> np.ndarray:
    """1/f noise: energy concentrated low, where speech energy also is.

    Built by shaping white noise in the frequency domain rather than with a
    filter cascade, so the spectrum is exact and there is no transient.
    """
    n = len(clean)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spec * scale, n=n).astype(np.float32)
    return _scale_to_snr(clean, shaped, snr_db)


def babble(clean: np.ndarray, snr_db: float, rng,
           pool: Optional[List[np.ndarray]] = None, **_) -> np.ndarray:
    """Many overlapping voices — the hardest realistic case.

    Summing several speakers gives real speech spectrum and real temporal
    modulation. It is also adversarial for the SNR estimator: babble in a
    silent stretch can read as speech to a VAD, so the "noise sample" the
    estimate depends on may not be noise-only. That is the point of testing it.
    """
    if not pool:
        return white(clean, snr_db, rng)
    n = len(clean)
    mix = np.zeros(n, dtype=np.float32)
    for _ in range(6):
        src = pool[rng.integers(len(pool))]
        if len(src) < n:
            reps = int(np.ceil(n / len(src)))
            src = np.tile(src, reps)
        start = int(rng.integers(0, max(1, len(src) - n)))
        mix += src[start:start + n]
    return _scale_to_snr(clean, mix, snr_db)


NOISES = {"white": white, "pink": pink, "babble": babble}


def mix(name: str, clean: np.ndarray, snr_db: float, rng,
        pool: Optional[List[np.ndarray]] = None) -> np.ndarray:
    if name not in NOISES:
        raise SystemExit(f"unknown noise {name!r}; have {sorted(NOISES)}")
    return NOISES[name](clean, snr_db, rng, pool=pool)
