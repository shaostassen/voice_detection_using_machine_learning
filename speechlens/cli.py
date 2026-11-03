"""Command-line interface: `speechlens analyze | serve | speaker`."""
from __future__ import annotations

import argparse
import json
import sys


def _fmt_ts(t: float) -> str:
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def cmd_analyze(args) -> int:
    from speechlens.asr import RobustnessConfig
    from speechlens.pipeline import SpeechLens

    policy = None
    if args.reliability == "auto":
        policy = "auto"          # resolved from an SNR estimate in the pipeline
    elif args.reliability:
        from speechlens.calibration import load_bundled_policy
        policy = load_bundled_policy(args.reliability)

    cfg = RobustnessConfig(
        beam_size=args.beam,
        condition_on_previous_text=args.context,
        vad_filter=not args.no_vad,
        # Per-word reliability needs word timestamps; turning them on here
        # rather than failing on a flag combination the user cannot guess.
        word_timestamps=args.word_timestamps or bool(policy),
        initial_prompt=args.prompt,
    )
    lens = SpeechLens(model_size=args.model, device=args.device,
                      compute_type=args.compute_type, denoise=args.denoise,
                      policy=policy)
    result = lens.analyze(args.audio, language=args.language, cfg=cfg)

    lang = result.language
    print(f"\nLanguage : {lang['name']} ({lang['code']})  "
          f"p={lang['probability']:.2f}  [{lang['method']}]")
    if len(lang.get("top", [])) > 1:
        alts = ", ".join(f"{c}={p:.2f}" for c, p in lang["top"][1:])
        print(f"Also     : {alts}")
    print(f"Audio    : {result.audio['duration_s']}s, "
          f"{result.audio['speech_ratio']:.0%} speech "
          f"(VAD: {result.audio['vad_backend']})")
    print(f"Compute  : {result.performance['model']} on "
          f"{result.performance['device']}, RTF={result.performance['rtf']}\n")
    for seg in result.transcript["segments"]:
        flag = "  [!]" if seg["flagged"] else ""
        print(f"[{_fmt_ts(seg['start'])} -> {_fmt_ts(seg['end'])}] "
              f"({seg['confidence']:.2f}){flag} {seg['text']}")

    rel = result.reliability
    if rel:
        print(f"\nReliability ({rel['condition']} policy via "
              f"{rel['selected_by']}, {rel['target_risk']:.0%} target error): "
              f"{rel['accepted']}/{rel['words']} words auto-accepted "
              f"({rel['coverage']:.0%}), {rel['review']} need review")
        flagged_words = [w for s in result.transcript["segments"]
                         for w in (s.get("words") or []) if not w["accept"]]
        if flagged_words:
            preview = "  ".join(
                f"{w['word'].strip()}({w['reliability']:.2f})"
                for w in flagged_words[:12])
            more = "" if len(flagged_words) <= 12 else f"  (+{len(flagged_words) - 12} more)"
            print(f"  review: {preview}{more}")

    for w in result.warnings:
        print(f"\n! {w}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nJSON written to {args.json}")
    if args.txt:
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write(result.text + "\n")
        print(f"Transcript written to {args.txt}")
    return 0


def _speaker_gallery(args, create: bool):
    """Open (or start) a gallery with the SpeechBrain ECAPA encoder."""
    import os

    from speechlens.speaker import SPEECHBRAIN_SOURCE, Gallery, speechbrain_encoder

    exists = os.path.exists(args.gallery)
    if not exists and not create:
        raise SystemExit(f"no gallery at {args.gallery}; enrol someone first")
    enc_id = f"speechbrain:{SPEECHBRAIN_SOURCE}"
    embed = speechbrain_encoder()          # heavy import: after the cheap checks
    if exists:
        return Gallery.load(args.gallery, embed, enc_id)
    return Gallery(embed, enc_id)


def cmd_speaker(args) -> int:
    from speechlens.audio import load_audio

    if args.action == "enroll":
        g = _speaker_gallery(args, create=True)
        for path in args.audio:
            y, sr = load_audio(path)
            g.enroll(args.name, y, sr)
        g.save(args.gallery)
        print(f"enrolled {args.name!r} from {len(args.audio)} clip(s); "
              f"gallery now has: {', '.join(g.names)}")
        print("note: the gallery file is biometric data; keep it private and "
              "only enrol people who have agreed.")
        return 0

    if args.action == "verify" and args.threshold is None:
        raise SystemExit("verify needs --threshold. There is no safe default: "
                         "scores depend on the encoder and your audio. Measure "
                         "impostor scores and use speechlens.speaker."
                         "threshold_for_far (see docs/HOW_IT_WORKS.md).")
    g = _speaker_gallery(args, create=False)
    y, sr = load_audio(args.audio[0])
    if args.action == "identify":
        name, score, ranking = g.identify(y, sr, threshold=args.threshold)
        for n, sc in ranking:
            print(f"  {n:20s} cosine={sc:+.3f}")
        if name is None:
            print(f"-> no match above threshold {args.threshold}")
        else:
            note = "" if args.threshold is not None else (
                "  (closed-set: nearest enrolled speaker, even for a stranger; "
                "pass --threshold to allow 'none')")
            print(f"-> {name}{note}")
        return 0
    ok, score = g.verify(y, sr, args.name, args.threshold)
    print(f"{'ACCEPT' if ok else 'REJECT'}  claimed={args.name!r}  "
          f"cosine={score:+.3f}  threshold={args.threshold}")
    return 0 if ok else 1


def cmd_serve(args) -> int:
    import os

    os.environ.setdefault("SPEECHLENS_MODEL", args.model)
    os.environ.setdefault("SPEECHLENS_DEVICE", args.device)
    import uvicorn

    from speechlens.server import app
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="speechlens",
        description="Language identification + robust transcription, locally.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze one audio file")
    a.add_argument("audio")
    a.add_argument("--model", default="large-v3")
    a.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a.add_argument("--compute-type", default="auto")
    a.add_argument("--language", default=None,
                   help="force language code (skip LID)")
    a.add_argument("--beam", type=int, default=5)
    a.add_argument("--denoise", action="store_true")
    a.add_argument("--no-vad", action="store_true")
    a.add_argument("--context", action="store_true",
                   help="condition on previous text (off by default for "
                        "robustness)")
    a.add_argument("--word-timestamps", action="store_true")
    a.add_argument("--reliability", default=None, metavar="CONDITION",
                   help="per-word calibrated reliability. 'auto' estimates "
                        "SNR from the VAD partition and picks the matching "
                        "policy; or name one: clean, 20, 10, 5, 0, -5 (dB). "
                        "The choice matters — at 2%% tolerated error coverage "
                        "runs from 90%% on clean speech to 0%% at 0 dB. "
                        "Implies --word-timestamps.")
    a.add_argument("--prompt", default=None,
                   help="initial prompt / domain vocabulary")
    a.add_argument("--json", default=None, help="write full result JSON here")
    a.add_argument("--txt", default=None, help="write plain transcript here")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="run the web UI / HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--model", default="large-v3")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_serve)

    sp = sub.add_parser(
        "speaker", help="enrol / identify / verify speakers (needs the "
                        "'speaker' extra: torch + speechbrain)")
    sp.add_argument("action", choices=["enroll", "identify", "verify"])
    sp.add_argument("--gallery", required=True, help="gallery .npz file")
    sp.add_argument("--name", help="speaker name (enroll / verify)")
    sp.add_argument("--threshold", type=float, default=None,
                    help="cosine threshold. identify: optional (enables "
                         "open-set). verify: required.")
    sp.add_argument("audio", nargs="+", help="audio file(s)")
    sp.set_defaults(func=cmd_speaker)

    args = p.parse_args(argv)
    if getattr(args, "func", None) is cmd_speaker and args.action in (
            "enroll", "verify") and not args.name:
        p.error(f"speaker {args.action} needs --name")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
