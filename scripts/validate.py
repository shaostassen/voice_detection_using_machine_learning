#!/usr/bin/env python3
"""SpeechLens validation harness. Run on a GPU box; downloads model weights.

Three studies:

  smoke   Multilingual TTS round-trip. espeak-ng synthesizes known text in
          several languages; checks LID hits and reports error rates.
          Functional check, not a WER benchmark (espeak audio is synthetic).

  snr     Noise-robustness sweep. Mixes white noise into clean speech at
          falling SNR; reports WER/CER, LID stability, and flagged-segment
          counts per level. Run with and without --denoise to A/B it.

  bench   Real-time factor across model sizes on one file.

Examples:
  python scripts/validate.py smoke --model large-v3
  python scripts/validate.py snr clip.wav --ref "reference transcript"
  python scripts/validate.py snr clip.wav --ref "..." --denoise
  python scripts/validate.py bench clip.wav --models small,distil-large-v3,large-v3
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechlens.audio import load_audio  # noqa: E402
from speechlens.metrics import error_rate  # noqa: E402
from speechlens.pipeline import SpeechLens  # noqa: E402

# (espeak voice, expected whisper code, reference text)
SMOKE_SET = [
    ("en", "en", "The quick brown fox jumps over the lazy dog"),
    ("de", "de", "Der schnelle braune Fuchs springt über den faulen Hund"),
    ("fr", "fr", "Le renard brun rapide saute par-dessus le chien paresseux"),
    ("es", "es", "El rápido zorro marrón salta sobre el perro perezoso"),
    ("cmn", "zh", "今天天气很好，我们一起去公园散步吧"),
]


def tts(voice: str, text: str, path: Path) -> None:
    espeak = shutil.which("espeak-ng")
    if espeak is None:
        sys.exit("espeak-ng is required for the smoke study "
                 "(sudo apt install espeak-ng)")
    subprocess.run([espeak, "-v", voice, "-s", "150", "-w", str(path), text],
                   check=True)


def mix_at_snr(clean: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Add white noise scaled so 10*log10(P_signal/P_noise) == snr_db."""
    noise = rng.normal(0, 1, len(clean)).astype(np.float32)
    p_clean = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    target = p_clean / (10 ** (snr_db / 10))
    return clean + noise * np.sqrt(target / max(p_noise, 1e-12))


def cmd_smoke(args) -> None:
    lens = SpeechLens(model_size=args.model)
    print(f"{'voice':>6} {'expect':>7} {'got':>5} {'p':>5} "
          f"{'metric':>7} {'err':>6}")
    hits = 0
    with tempfile.TemporaryDirectory() as td:
        for voice, lang, text in SMOKE_SET:
            wav = Path(td) / f"{voice}.wav"
            tts(voice, text, wav)
            r = lens.analyze(str(wav))
            metric, err = error_rate(text, r.text, language=lang)
            hit = r.language["code"] == lang
            hits += hit
            mark = "" if hit else "   <-- LID MISS"
            print(f"{voice:>6} {lang:>7} {r.language['code']:>5} "
                  f"{r.language['probability']:>5.2f} {metric:>7} "
                  f"{err:>6.2f}{mark}")
    print(f"\nLID hits: {hits}/{len(SMOKE_SET)}. espeak audio is robotic and "
          "synthetic;\nexpect nonzero error rates — this validates plumbing "
          "and LID, not peak accuracy.")


def cmd_snr(args) -> None:
    lens = SpeechLens(model_size=args.model, denoise=args.denoise)
    clean, sr = load_audio(args.audio)
    rng = np.random.default_rng(0)
    levels = [None, 20, 10, 5, 0, -5]
    tag = " (denoise ON)" if args.denoise else ""
    print(f"model={args.model}{tag}")
    print(f"{'SNR dB':>7} {'lang':>5} {'p':>5} {'metric':>7} {'err':>6} "
          f"{'flagged':>8}")
    for snr in levels:
        y = clean if snr is None else mix_at_snr(clean, snr, rng)
        r = lens.analyze((y, sr))
        if args.ref:
            metric, err = error_rate(args.ref, r.text,
                                     language=r.language["code"])
            err_s = f"{err:>6.2f}"
        else:
            metric, err_s = "-", "     -"
        label = "clean" if snr is None else str(snr)
        print(f"{label:>7} {r.language['code']:>5} "
              f"{r.language['probability']:>5.2f} {metric:>7} {err_s} "
              f"{r.transcript['flagged_segments']:>8}")


def cmd_bench(args) -> None:
    clean, sr = load_audio(args.audio)
    dur = len(clean) / sr
    print(f"audio: {dur:.1f}s")
    print(f"{'model':>20} {'rtf':>8} {'x realtime':>11}")
    for size in args.models.split(","):
        lens = SpeechLens(model_size=size.strip())
        r = lens.analyze((clean, sr))
        rtf = r.performance["rtf"]
        print(f"{size.strip():>20} {rtf:>8.3f} {1.0 / rtf:>11.1f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke", help="multilingual TTS round-trip")
    s.add_argument("--model", default="large-v3")
    s.set_defaults(func=cmd_smoke)

    n = sub.add_parser("snr", help="noise robustness sweep")
    n.add_argument("audio")
    n.add_argument("--ref", default=None,
                   help="reference transcript for WER/CER")
    n.add_argument("--model", default="large-v3")
    n.add_argument("--denoise", action="store_true")
    n.set_defaults(func=cmd_snr)

    b = sub.add_parser("bench", help="RTF across model sizes")
    b.add_argument("audio")
    b.add_argument("--models", default="small,large-v3")
    b.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
