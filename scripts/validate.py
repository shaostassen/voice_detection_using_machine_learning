#!/usr/bin/env python3
"""SpeechLens validation harness. Run on a GPU box; downloads model weights.

Three studies:

  smoke   Multilingual TTS round-trip. espeak-ng synthesizes known text in
          several languages; checks LID hits and reports error rates.
          Functional check, not a WER benchmark (espeak audio is synthetic).

  snr     Noise-robustness sweep. Mixes white noise into clean speech at
          falling SNR; reports WER/CER, LID stability, and flagged-segment
          counts per level. Run with and without --denoise to A/B it.

  gate    Confidence-gate diagnosis. Runs the same SNR ladder but prints the
          per-segment confidence distribution and the flag count at several
          candidate thresholds, to find one whose flags lead the errors.

  bench   Real-time factor across model sizes on one file.

Every run prints a provenance banner (hardware / model / device / compute_type
/ date / raw command) so its numbers can be pasted straight into
docs/VALIDATION.md. Nothing gets recorded without it.

Examples:
  python scripts/validate.py smoke --model large-v3
  python scripts/validate.py snr clip.wav --ref "reference transcript"
  python scripts/validate.py snr clip.wav --ref "..." --denoise
  python scripts/validate.py bench clip.wav --models small,distil-large-v3,large-v3

  # T4 (Colab/Kaggle), the authoritative-numbers configuration:
  python scripts/validate.py smoke --model large-v3 \\
      --device cuda --compute-type float16
"""
from __future__ import annotations

import argparse
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechlens.asr import resolve_device  # noqa: E402
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


def hardware_label() -> str:
    """Best-effort one-line hardware string for the VALIDATION.md record."""
    smi = shutil.which("nvidia-smi")
    if smi is not None:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True)
            gpus = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            if gpus:
                return gpus[0] if len(gpus) == 1 else f"{len(gpus)}x {gpus[0]}"
        except Exception:
            pass
    # platform.processor() returns "arm" on macOS and "x86_64" on Linux —
    # too generic to identify the box a number came from, which is the whole
    # point of recording it. Ask the OS for the actual CPU model.
    cpu = ""
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    cpu = cpu or platform.processor() or platform.machine()
    return f"{cpu} ({platform.system()} {platform.machine()})"


def banner(study: str, args, model: str) -> tuple:
    """Print the provenance header every run must be recorded with.

    No number leaves this script without the hardware / model / compute_type /
    date it was produced on — see the 'no fabricated metrics' rule.
    """
    device, compute_type = resolve_device(args.device, args.compute_type)
    print(f"study={study}  hardware={hardware_label()}  model={model}  "
          f"device={device}  compute_type={compute_type}  "
          f"date={date.today().isoformat()}")
    # shlex.join so a --ref containing spaces stays copy-pasteable: the
    # recorded command has to actually re-run.
    print(f"cmd: python {shlex.join(sys.argv)}\n")
    return device, compute_type


def cmd_smoke(args) -> None:
    banner("smoke", args, args.model)
    lens = SpeechLens(model_size=args.model, device=args.device,
                      compute_type=args.compute_type)
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
    banner("snr", args, args.model)
    lens = SpeechLens(model_size=args.model, device=args.device,
                      compute_type=args.compute_type, denoise=args.denoise)
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


CANDIDATE_THRESHOLDS = (0.55, 0.65, 0.75, 0.85, 0.95)


def cmd_gate(args) -> None:
    """Diagnose the confidence gate against the SNR ladder.

    The 2026-08-02 sweep flagged zero segments at every noise level, including
    -5 dB where a third of the words were wrong. The count alone cannot say
    whether the threshold is merely too low or whether avg_logprob is too flat
    to threshold at all, so this prints the underlying distribution.

    Flagging is applied post-hoc to avg_logprob, so every candidate threshold
    is evaluated on one decode pass rather than re-decoding per threshold.
    """
    banner("gate", args, args.model)
    lens = SpeechLens(model_size=args.model, device=args.device,
                      compute_type=args.compute_type)
    clean, sr = load_audio(args.audio)
    rng = np.random.default_rng(0)          # same stream as cmd_snr
    levels = [None, 20, 10, 5, 0, -5]

    print("per-level confidence distribution "
          "(conf = exp(avg_logprob), the flagging signal)")
    print(f"{'SNR dB':>7} {'segs':>5} {'conf min':>9} {'conf mean':>10} "
          f"{'conf max':>9} {'nospeech':>9} {'lid p':>6} {'err':>6}")

    per_level = []
    for snr in levels:
        y = clean if snr is None else mix_at_snr(clean, snr, rng)
        r = lens.analyze((y, sr))
        segs = r.transcript["segments"]
        confs = [s["confidence"] for s in segs] or [float("nan")]
        nosp = max((s["no_speech_prob"] for s in segs), default=float("nan"))
        if args.ref:
            _metric, err = error_rate(args.ref, r.text,
                                      language=r.language["code"])
        else:
            err = float("nan")
        label = "clean" if snr is None else str(snr)
        print(f"{label:>7} {len(segs):>5} {min(confs):>9.3f} "
              f"{sum(confs) / len(confs):>10.3f} {max(confs):>9.3f} "
              f"{nosp:>9.3f} {r.language['probability']:>6.2f} {err:>6.2f}")
        per_level.append((label, confs, err))

    header = "  ".join(f"t={t:.2f}" for t in CANDIDATE_THRESHOLDS)
    print(f"\nflagged segments / total, by candidate threshold\n"
          f"{'SNR dB':>7} {'err':>6}  {header}")
    for label, confs, err in per_level:
        cells = "  ".join(
            f"{sum(1 for c in confs if c < t):>2}/{len(confs):<3}"
            for t in CANDIDATE_THRESHOLDS)
        print(f"{label:>7} {err:>6.2f}  {cells}")

    print("\nRead this as: a usable threshold is one whose flag count climbs "
          "*before*\nthe error rate does. A column that is all-zero or "
          "all-flagged is useless —\nthe first cannot warn, the second cries "
          "wolf on clean audio.")


def cmd_bench(args) -> None:
    banner("bench", args, args.models)
    clean, sr = load_audio(args.audio)
    dur = len(clean) / sr
    print(f"audio: {dur:.1f}s")
    print(f"{'model':>20} {'rtf':>8} {'x realtime':>11}")
    for size in args.models.split(","):
        lens = SpeechLens(model_size=size.strip(), device=args.device,
                          compute_type=args.compute_type)
        r = lens.analyze((clean, sr))
        rtf = r.performance["rtf"]
        print(f"{size.strip():>20} {rtf:>8.3f} {1.0 / rtf:>11.1f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared placement flags. Default "auto" keeps the previous behavior
    # (cuda+float16 where a GPU exists, cpu+int8 otherwise); passing them
    # explicitly is what lets a run be *recorded* rather than inferred.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    common.add_argument("--compute-type", default="auto",
                        help="float16 / int8_float16 / int8 / float32")

    s = sub.add_parser("smoke", parents=[common],
                       help="multilingual TTS round-trip")
    s.add_argument("--model", default="large-v3")
    s.set_defaults(func=cmd_smoke)

    n = sub.add_parser("snr", parents=[common], help="noise robustness sweep")
    n.add_argument("audio")
    n.add_argument("--ref", default=None,
                   help="reference transcript for WER/CER")
    n.add_argument("--model", default="large-v3")
    n.add_argument("--denoise", action="store_true")
    n.set_defaults(func=cmd_snr)

    g = sub.add_parser("gate", parents=[common],
                       help="diagnose the confidence gate on the SNR ladder")
    g.add_argument("audio")
    g.add_argument("--ref", default=None,
                   help="reference transcript, so flags can be compared "
                        "against the error rate they are supposed to lead")
    g.add_argument("--model", default="large-v3")
    g.set_defaults(func=cmd_gate)

    b = sub.add_parser("bench", parents=[common],
                       help="RTF across model sizes")
    b.add_argument("audio")
    b.add_argument("--models", default="small,large-v3")
    b.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
