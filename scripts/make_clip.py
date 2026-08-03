#!/usr/bin/env python3
"""Materialize the canonical validation clip.

Every recorded run in docs/VALIDATION.md uses the same audio: the first
three utterances of the LibriSpeech dummy validation split, concatenated
(23.2 s, 16 kHz mono) with their ground-truth transcript. Rebuilding it
identically is what makes numbers from different machines comparable —
a bench on a different clip is a different measurement.

Writes the wav, and the reference transcript next to it as <name>.ref.txt.

    python scripts/make_clip.py /tmp/validation_clip.wav
    python scripts/validate.py bench /tmp/validation_clip.wav \\
        --models base,small,distil-large-v3 --device cpu --compute-type int8

Needs `pip install datasets` (validation-only; not a package dependency).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

N_UTTERANCES = 3


def build(n: int = N_UTTERANCES):
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit("this script needs `pip install datasets` "
                 "(validation-only, not a package dependency)")

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                      "clean", split="validation")
    # decode=False keeps `datasets` from reaching for torchcodec, and so
    # torch — this project holds a no-torch ceiling. soundfile reads the
    # FLAC bytes directly.
    ds = ds.cast_column("audio", Audio(decode=False))

    parts, refs, sr = [], [], None
    for i in range(min(n, len(ds))):
        row = ds[i]
        a = row["audio"]
        raw = a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes()
        chunk, sr = sf.read(io.BytesIO(raw), dtype="float32")
        parts.append(chunk)
        refs.append(row["text"].strip())
    return np.concatenate(parts), int(sr), " ".join(refs)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "validation_clip.wav")
    y, sr, ref = build()
    sf.write(str(out), y, sr)
    out.with_suffix(".ref.txt").write_text(ref + "\n")
    print(f"{out}  duration={len(y) / sr:.1f}s  sr={sr}  words={len(ref.split())}")
    print(f"{out.with_suffix('.ref.txt')}  (reference transcript)")


if __name__ == "__main__":
    main()
