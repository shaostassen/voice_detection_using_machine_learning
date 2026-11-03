#!/usr/bin/env python3
"""Record your voice, clone it, and score how well the clone worked.

    record   read a few prompts into the microphone      (needs `pip install sounddevice`)
    import   ...or use recordings you already have
    check    quality-check the takes (clipping, level, noise, length)
    run      clone, then score intelligibility + speaker similarity

    python scripts/clone_my_voice.py record --dir my_voice
    python scripts/clone_my_voice.py check  --dir my_voice
    python scripts/clone_my_voice.py run    --dir my_voice --backend chatterbox

Protocol (the same one SpeechForge's scripts/eval_zeroshot.py uses, so the
numbers mean something):

  * Take 0 is a longer passage and becomes the **reference** the model hears.
  * Every other take is **held out**: the model never hears it. Its text is
    synthesised in your cloned voice and compared with your real recording of
    that sentence.
  * Intelligibility: WER of an ASR over the clone, against the text.
  * Similarity: speaker-encoder cosine between the clone and the held-out real
    recording (SIM-o, never against the reference clip).
  * **Your real recordings are scored the same way** and printed as the
    ground-truth row. Read every number against that row; without it neither
    column means anything.

This is an orchestration script and needs SpeechForge (a separate project, so
that this package stays free of torch):

    pip install -e .                                          # this repo (ASR for WER)
    git clone https://github.com/shaostassen/speechforge
    pip install -e "./speechforge[chatterbox,similarity]"     # or point --speechforge-path at the clone

Only clone a voice that is yours, or that belongs to someone who has agreed.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MANIFEST = "manifest.json"
MIN_HELD_OUT = 2          # ground-truth similarity needs two real held-out clips
RECOMMENDED_HELD_OUT = 5

# Own-wording prompts. Take 0 is the reference passage (~15 s read aloud: long
# enough to carry your voice, short of the caps measured for current backends);
# the rest are held-out sentences, kept varied so the test is not one phoneme mix.
PROMPTS: List[Tuple[str, str]] = [
    ("reference",
     "I'm recording this so that a computer can learn what my voice sounds "
     "like. I'll speak at my normal pace, in a quiet room, and try not to "
     "rush. Some sentences are short, and some wander a bit before they "
     "finally arrive at the point."),
    ("heldout", "The weather turned cold this morning, so I wore a heavier coat."),
    ("heldout", "Could you please tell me where the nearest train station is?"),
    ("heldout", "She carefully measured two cups of flour, then added a pinch of salt."),
    ("heldout", "Nobody expected the meeting to finish before lunch, but it did."),
    ("heldout", "We watched the boats drift slowly across the harbour at sunset."),
    ("heldout", "I'd like a window seat, if one is still available."),
    ("heldout", "The new library opens on Thursday, and the first fifty visitors get a free book."),
]


# ---------------------------------------------------------------------------
# Take quality checks (pure NumPy: testable with no audio device or model)
# ---------------------------------------------------------------------------
def _frame_db(y: np.ndarray, sr: int, frame_s: float = 0.03) -> np.ndarray:
    n = max(int(sr * frame_s), 1)
    k = len(y) // n
    if k == 0:
        return np.array([-120.0])
    f = y[:k * n].reshape(k, n).astype(np.float64)
    return 20.0 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-10)


def check_take(y: np.ndarray, sr: int, role: str = "heldout") -> List[str]:
    """Problems worth fixing before this take is used. Empty list = fine.

    Noise and room matter most: they are cloned along with the voice.
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    warns: List[str] = []
    dur = len(y) / sr if sr else 0.0
    if dur < (8.0 if role == "reference" else 1.5):
        warns.append(f"very short ({dur:.1f}s)" +
                     ("; a reference under ~8s is too little to carry a voice"
                      if role == "reference" else ""))
    if len(y) == 0:
        return warns
    peak = float(np.max(np.abs(y)))
    if (np.abs(y) >= 0.99).mean() > 0.001:
        warns.append("clipping: the input level was too high; lower the mic gain "
                     "(clipping is cloned too)")
    if peak < 0.1:
        warns.append(f"very quiet (peak {peak:.2f}); raise the input level or "
                     "move closer to the mic")
    db = _frame_db(y, sr)
    speech, floor = float(np.percentile(db, 90)), float(np.percentile(db, 10))
    if speech - floor < 15.0:
        warns.append(f"noisy or no pauses: only {speech - floor:.0f} dB between "
                     "speech and the quietest parts (want >= 15 dB). Room noise is "
                     "baked into a clone")
    return warns


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def load_manifest(d: Path) -> dict:
    p = d / MANIFEST
    if not p.exists():
        raise SystemExit(f"{p} not found; run `record` or `import` first")
    return json.loads(p.read_text())


def save_manifest(d: Path, m: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / MANIFEST).write_text(json.dumps(m, indent=2, ensure_ascii=False))


def new_manifest(speaker: str) -> dict:
    return {"version": 1, "speaker": speaker, "created": date.today().isoformat(),
            "own_voice_confirmed": True, "takes": []}


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:                      # reference prep flattens channels, so mix down here
        y = y.mean(axis=1)
    return y, sr


def load_takes(d: Path) -> List[dict]:
    """Manifest + audio -> list of {file, text, role, audio, sr}. Validates."""
    m = load_manifest(d)
    if not m.get("own_voice_confirmed"):
        raise SystemExit("manifest does not confirm this is your own voice (or "
                         "the speaker's consent); refusing to clone")
    takes = []
    for t in m["takes"]:
        y, sr = read_wav(d / t["file"])
        takes.append({**t, "audio": y, "sr": sr})
    if sum(t["role"] == "reference" for t in takes) != 1:
        raise SystemExit("need exactly one take with role 'reference'")
    held = [t for t in takes if t["role"] == "heldout"]
    if len(held) < MIN_HELD_OUT:
        raise SystemExit(f"need at least {MIN_HELD_OUT} held-out takes (have "
                         f"{len(held)}): the ground-truth similarity row compares "
                         "one real clip with another")
    return takes


# ---------------------------------------------------------------------------
# Recording (interactive; needs a microphone)
# ---------------------------------------------------------------------------
def record_one(device: Optional[int] = None) -> Tuple[np.ndarray, int]:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("recording needs `pip install sounddevice` (and a "
                         "microphone). Alternatively use `import` with existing "
                         "wav files.")
    sr = int(sd.query_devices(device, "input")["default_samplerate"])
    chunks: List[np.ndarray] = []
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=device,
                        callback=lambda indata, f, t, s: chunks.append(indata.copy())):
        input("   recording... press Enter to stop ")
    return (np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)), sr


def cmd_record(args) -> int:
    import soundfile as sf
    d = Path(args.dir)
    m = new_manifest(args.speaker)
    print("You are about to record YOUR OWN voice. Find a quiet room, keep the mic a\n"
          "hand's width from your mouth, and read each prompt at a natural pace.\n")
    for i, (role, text) in enumerate(PROMPTS[:args.n]):
        while True:
            print(f"[{i + 1}/{min(args.n, len(PROMPTS))}] ({role})\n   \"{text}\"")
            input("   press Enter, then read it aloud ")
            y, sr = record_one(args.device)
            warns = check_take(y, sr, role)
            for w in warns:
                print(f"   ! {w}")
            if not warns:
                print(f"   ok ({len(y) / sr:.1f}s)")
            if warns and input("   retake? [Y/n] ").strip().lower() != "n":
                continue
            break
        fname = f"take_{i:02d}.wav"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / fname), y, sr, subtype="PCM_16")
        m["takes"].append({"file": fname, "text": text, "role": role})
        save_manifest(d, m)
    print(f"\nsaved {len(m['takes'])} takes to {d}/ . Next: "
          f"python scripts/clone_my_voice.py run --dir {d}")
    return 0


def cmd_import(args) -> int:
    d = Path(args.dir)
    if not args.own_voice:
        raise SystemExit("pass --own-voice to confirm these recordings are of "
                         "your own voice, or of someone who has agreed to this")
    m = load_manifest(d) if (d / MANIFEST).exists() else new_manifest(args.speaker)
    for src, text in args.take:
        i = len(m["takes"])
        role = "reference" if i == 0 else "heldout"
        y, sr = read_wav(Path(src))
        import soundfile as sf
        fname = f"take_{i:02d}.wav"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / fname), y, sr, subtype="PCM_16")
        m["takes"].append({"file": fname, "text": text, "role": role})
        print(f"  {fname} <- {src} ({role})")
    save_manifest(d, m)
    return 0


def cmd_check(args) -> int:
    d = Path(args.dir)
    bad = 0
    for t in load_manifest(d)["takes"]:
        y, sr = read_wav(d / t["file"])
        warns = check_take(y, sr, t["role"])
        bad += bool(warns)
        print(f"{t['file']}  {t['role']:9s} {len(y) / sr:5.1f}s  "
              f"{'ok' if not warns else ''}")
        for w in warns:
            print(f"    ! {w}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Clone + score
# ---------------------------------------------------------------------------
def hardware_label() -> str:
    cpu = ""
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu or platform.processor() or platform.machine()} ({platform.system()})"


def run_pipeline(takes: Sequence[dict], *, backend, transcribe: Callable,
                 encoder: Callable, clone_fn: Callable, apply_case: Callable,
                 text_case: str = "sentence", seed: Optional[int] = 0,
                 ref_seconds: Optional[float] = None,
                 out_dir: Optional[Path] = None, backend_name: str = "clone",
                 provenance: Optional[dict] = None) -> dict:
    """Clone each held-out sentence from the reference, then score. Pure of I/O
    on models: backend, ASR and encoder are injected, so tests use fakes.

    Returns a dict with per-utterance rows, the rendered table, and provenance.
    Per-utterance rows are built BEFORE the table is rendered, so a surprising
    aggregate can always be decomposed.
    """
    from speechforge.evaluate import (EvalTable, SystemScores, sim_o,
                                      score_intelligibility)
    from speechlens.metrics import wer

    ref = next(t for t in takes if t["role"] == "reference")
    held = [t for t in takes if t["role"] == "heldout"]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows, synth_pairs, real_pairs, synth_sims, real_sims = [], [], [], [], []
    ref_report = None
    for j, ho in enumerate(held):
        # What the MODEL reads is case-adjusted; scoring uses the original text.
        wav, sr, report = clone_fn(
            backend, apply_case(ho["text"], text_case), ref["audio"], ref["sr"],
            seed=seed, ref_seconds=ref_seconds,
            reference_text=apply_case(ref["text"], text_case))
        ref_report = ref_report or report
        for w in report.get("warnings", []):
            print(f"  reference warning: {w}")
        if out_dir:
            import soundfile as sf
            sf.write(str(out_dir / f"clone_{j:02d}.wav"), wav, sr)
        synth_pairs.append((wav, sr, ho["text"]))
        real_pairs.append((ho["audio"], ho["sr"], ho["text"]))
        # Held-out real recording; reference passed so sim_o refuses if the two
        # are ever the same clip (that would be SIM-r, which inflates).
        s_clone = sim_o(encoder, wav, sr, ho["audio"], ho["sr"],
                        reference_audio=ref["audio"])
        other = held[(j + 1) % len(held)]
        s_real = sim_o(encoder, other["audio"], other["sr"], ho["audio"], ho["sr"])
        synth_sims.append(s_clone)
        real_sims.append(s_real)
        rows.append({
            "text": ho["text"], "held_out_file": ho["file"],
            "clone_file": f"clone_{j:02d}.wav" if out_dir else None,
            "clone_seconds": round(len(wav) / sr, 2),
            "real_seconds": round(len(ho["audio"]) / ho["sr"], 2),
            "wer_clone": round(wer(ho["text"], transcribe(wav, sr)), 4),
            "wer_real": round(wer(ho["text"], transcribe(ho["audio"], ho["sr"])), 4),
            "sim_clone_vs_real": round(s_clone, 4),
            "sim_real_vs_real": round(s_real, 4),
        })

    table = EvalTable(corpus=f"your voice, {len(held)} held-out sentences",
                      hardware=hardware_label())
    table.set_ground_truth(SystemScores(
        name="your real voice",
        wer=score_intelligibility(transcribe, real_pairs),
        sim_o=float(np.nanmean(real_sims)), n_utterances=len(real_pairs),
        notes="real recordings, scored identically"))
    table.add(SystemScores(
        name=backend_name,
        wer=score_intelligibility(transcribe, synth_pairs),
        sim_o=float(np.nanmean(synth_sims)), n_utterances=len(synth_pairs)))

    caveats = []
    if len(held) < RECOMMENDED_HELD_OUT:
        caveats.append(f"only {len(held)} held-out sentences "
                       f"(recommended >= {RECOMMENDED_HELD_OUT})")
    caveats.append(
        "small samples characterise YOUR voice and setup; they are not a "
        "benchmark. WER differences below ~0.05 are not resolvable at this size, "
        "and your ears are the real test: listen to the clones next to the "
        "originals.")
    return {"rows": rows, "table": table.render(),
            "table_rows": {"ground_truth": table.ground_truth.to_dict(),
                           "systems": [r.to_dict() for r in table.rows]},
            "reference_report": ref_report, "caveats": caveats,
            "provenance": provenance or {}}


def _require_speechforge(path: Optional[str]) -> None:
    if path:
        sys.path.insert(0, str(Path(path).resolve()))
    try:
        import speechforge  # noqa: F401
    except ImportError:
        raise SystemExit(
            "SpeechForge is required for cloning and scoring (kept separate so "
            "this package stays torch-free):\n"
            "  git clone https://github.com/shaostassen/speechforge\n"
            "  pip install -e './speechforge[chatterbox,similarity]'\n"
            "or pass --speechforge-path /path/to/speechforge")


def cmd_run(args) -> int:
    _require_speechforge(args.speechforge_path)
    from speechforge.evaluate import whisper_transcriber
    from speechforge.similarity import load_encoder
    from speechforge.synth import clone, load_backend
    from speechforge.text import apply_case

    d = Path(args.dir)
    takes = load_takes(d)
    for t in takes:
        for w in check_take(t["audio"], t["sr"], t["role"]):
            print(f"  ! {t['file']}: {w}")

    print(f"\nloading {args.backend} ({args.device}), ASR '{args.asr_model}', "
          "speaker encoder...  (first run downloads weights)")
    backend = load_backend(args.backend, device=args.device)
    transcribe = whisper_transcriber(model_size=args.asr_model, language=args.language)
    encoder = load_encoder(device="cpu")

    out_dir = Path(args.out) if args.out else d / "clones"
    prov = {"date": date.today().isoformat(), "command": shlex.join(sys.argv),
            "hardware": hardware_label(), "backend": backend.describe(),
            "speaker_encoder": encoder.describe(),
            "asr": f"faster-whisper {args.asr_model}", "text_case": args.text_case,
            "seed": args.seed, "ref_seconds": args.ref_seconds,
            "reference_take": next(t["file"] for t in takes if t["role"] == "reference")}
    result = run_pipeline(
        takes, backend=backend, transcribe=transcribe, encoder=encoder,
        clone_fn=clone, apply_case=apply_case, text_case=args.text_case,
        seed=None if args.seed < 0 else args.seed, ref_seconds=args.ref_seconds,
        out_dir=out_dir, backend_name=args.backend, provenance=prov)
    report(result, d / "results.json")
    print(f"\nclones written to {out_dir}/ . Listen to clone_NN.wav next to the "
          f"matching take_NN.wav.")
    return 0


def report(result: dict, json_path: Path) -> None:
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("\nper utterance (WER real / clone, similarity clone-vs-real / real-vs-real):")
    for r in result["rows"]:
        print(f"  {r['wer_real']:.2f} / {r['wer_clone']:.2f}   "
              f"{r['sim_clone_vs_real']:.3f} / {r['sim_real_vs_real']:.3f}   {r['text']}")
    print()
    print(result["table"])
    print()
    for c in result["caveats"]:
        print(f"- {c}")
    if "chatterbox" in str(result.get("provenance", {}).get("backend", "")).lower():
        print("- Chatterbox watermarks every output.")
    print("- Say that a clone is synthetic when you share it.")
    print(f"\nfull detail (provenance, per-utterance rows): {json_path}")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record the prompts with your microphone")
    r.add_argument("--dir", required=True)
    r.add_argument("--speaker", default="me")
    r.add_argument("--n", type=int, default=len(PROMPTS),
                   help=f"how many prompts (default all {len(PROMPTS)}; min 3)")
    r.add_argument("--device", type=int, default=None, help="sounddevice input index")
    r.set_defaults(fn=cmd_record)

    i = sub.add_parser("import", help="use existing recordings (first = reference)")
    i.add_argument("--dir", required=True)
    i.add_argument("--speaker", default="me")
    i.add_argument("--own-voice", action="store_true",
                   help="confirm the voice is yours, or its owner has agreed")
    i.add_argument("--take", nargs=2, action="append", metavar=("WAV", "TEXT"),
                   required=True, help="repeat: a wav file and exactly what is said")
    i.set_defaults(fn=cmd_import)

    c = sub.add_parser("check", help="quality-check the takes")
    c.add_argument("--dir", required=True)
    c.set_defaults(fn=cmd_check)

    u = sub.add_parser("run", help="clone the held-out sentences and score them")
    u.add_argument("--dir", required=True)
    u.add_argument("--backend", default="chatterbox", choices=["chatterbox", "cosyvoice2"])
    u.add_argument("--device", default="cpu")
    u.add_argument("--asr-model", default="small",
                   help="faster-whisper size for WER. 'small' is practical on CPU; "
                        "large-v3 is the published-comparable choice but downloads ~3 GB")
    u.add_argument("--language", default="en")
    u.add_argument("--text-case", default="sentence", choices=["sentence", "raw"],
                   help="sentence (default): SpeechForge measured raw ALL-CAPS-style "
                        "text costing autoregressive TTS up to 5x WER")
    u.add_argument("--seed", type=int, default=0, help="-1 for unseeded sampling")
    u.add_argument("--ref-seconds", type=float, default=None,
                   help="override the backend's measured reference cap")
    u.add_argument("--out", default=None, help="where to write clone wavs")
    u.add_argument("--speechforge-path", default=None)
    u.set_defaults(fn=cmd_run)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "record" and args.n < 1 + MIN_HELD_OUT:
        raise SystemExit(f"--n must be at least {1 + MIN_HELD_OUT} "
                         "(one reference + two held-out)")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
