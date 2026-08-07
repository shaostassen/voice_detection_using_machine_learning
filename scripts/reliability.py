#!/usr/bin/env python3
"""The decisive experiment: does Whisper confidence predict *which* word is wrong?

Everything measured so far has been between-condition — mean confidence falls
as SNR falls. That does not show the signal is usable for flagging. A score can
track SNR perfectly and still rank errors no better than chance *within* a
condition, which would make it a noise meter rather than an error detector.

So this computes per-word AUROC inside each SNR bin, over a corpus rather than
one clip, for several competing estimators:

  word_prob      per-word mean token probability   (genuinely per-word)
  seg_conf       exp(avg_logprob)                  (PER-WINDOW, broadcast)
  seg_min_word   min word probability in the segment
  speech_prob    1 - no_speech_prob                (PER-WINDOW, broadcast)

seg_conf is the signal the shipped confidence gate uses. It is constant across
every word of a 30 s decode window, so its within-window AUROC is 0.5 by
construction; the corpus-level number is carried anyway to show exactly how
much is lost by flagging on it.

Entropy-based estimators need full per-token distributions, which CTranslate2
does not expose — those live in the Colab/torch arm.

    python scripts/reliability.py --model small --limit 40
    python scripts/reliability.py --model large-v3 --device cpu --compute-type int8
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import banner, mix_at_snr  # noqa: E402

from speechlens.asr import RobustnessConfig  # noqa: E402
from speechlens.confidence import summarize  # noqa: E402
from speechlens.metrics import labels_for_words, wer  # noqa: E402
from speechlens.pipeline import SpeechLens  # noqa: E402

LEVELS = [None, 20, 10, 5, 0, -5]
ESTIMATORS = ("word_prob", "seg_conf", "seg_min_word", "speech_prob")


def load_corpus(limit: int):
    """LibriSpeech dummy validation split as (audio, sr, reference) triples.

    decode=False keeps `datasets` from reaching for torchcodec, and so torch —
    the same trick scripts/make_clip.py uses to hold the no-torch ceiling.
    """
    try:
        import soundfile as sf
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit("needs `pip install datasets` (validation-only)")

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for i in range(min(limit, len(ds))):
        row = ds[i]
        a = row["audio"]
        raw = a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes()
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
        out.append((y, int(sr), row["text"].strip()))
    return out


def collect(lens, audio, sr, reference, cfg):
    """One utterance -> per-word (estimator scores, correctness) rows.

    Words come back in reading order across segments, which is what lets the
    alignment labels line up with the scores one-to-one.
    """
    r = lens.analyze((audio, sr), cfg=cfg)
    segments = r.transcript["segments"]

    raw = {e: [] for e in ESTIMATORS}
    words_out = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            continue
        probs = [w["prob"] for w in words]
        seg_min = min(probs)
        for w, p in zip(words, probs):
            words_out.append(w["word"])
            raw["word_prob"].append(p)
            raw["seg_conf"].append(seg["confidence"])
            raw["seg_min_word"].append(seg_min)
            raw["speech_prob"].append(1.0 - seg["no_speech_prob"])

    # Label each word as the ASR emitted it. Normalisation is not one-to-one,
    # so joining and re-splitting would desynchronize scores from labels.
    labels = labels_for_words(reference, words_out)

    keep = [i for i, lab in enumerate(labels) if lab is not None]
    scores = {e: [raw[e][i] for i in keep] for e in ESTIMATORS}
    kept_labels = [labels[i] for i in keep]
    return (scores, kept_labels, len(labels) - len(keep)), r.text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-type", default="auto")
    p.add_argument("--limit", type=int, default=40, help="utterances")
    p.add_argument("--json", default=None, help="write raw pairs here")
    args = p.parse_args()

    banner("reliability", args, args.model)
    corpus = load_corpus(args.limit)
    total_s = sum(len(y) / sr for y, sr, _ in corpus)
    print(f"corpus: {len(corpus)} utterances, {total_s / 60:.1f} min audio")
    print(f"ladder: {LEVELS}\n")

    lens = SpeechLens(model_size=args.model, device=args.device,
                      compute_type=args.compute_type)
    cfg = RobustnessConfig(word_timestamps=True)

    rng = np.random.default_rng(0)
    per_level, dropped = {}, 0

    for snr in LEVELS:
        label = "clean" if snr is None else str(snr)
        agg = {e: [] for e in ESTIMATORS}
        labels_all, refs, hyps = [], [], []
        for y, sr, ref in corpus:
            audio = y if snr is None else mix_at_snr(y, snr, rng)
            (scores, labels, skipped), text = collect(lens, audio, sr, ref, cfg)
            dropped += skipped
            for e in ESTIMATORS:
                agg[e].extend(scores[e])
            labels_all.extend(labels)
            refs.append(ref)
            hyps.append(text)

        corpus_wer = wer(" ".join(refs), " ".join(hyps)) if refs else float("nan")
        per_level[label] = {"wer": corpus_wer, "n_words": len(labels_all),
                            "estimators": {}}
        for e in ESTIMATORS:
            per_level[label]["estimators"][e] = summarize(agg[e], labels_all)
        if args.json:
            per_level[label]["_raw"] = {
                "labels": labels_all,
                **{e: agg[e] for e in ESTIMATORS},
            }
        print(f"  {label:>5} dB  wer={corpus_wer:.3f}  words={len(labels_all)}")

    if dropped:
        print(f"\n{dropped} words dropped (normalized to nothing, e.g. bare "
              f"punctuation) — no evidence to label them with")

    print("\n" + "=" * 78)
    print("WITHIN-CONDITION AUROC — can the score rank errors below correct words?")
    print("0.5 is chance. This is the number the project turns on.")
    print("=" * 78)
    head = "  ".join(f"{e:>13}" for e in ESTIMATORS)
    print(f"{'SNR dB':>7} {'wer':>6} {'acc':>6}  {head}")
    for label, d in per_level.items():
        cells = "  ".join(f"{d['estimators'][e]['auroc']:>13.3f}"
                          for e in ESTIMATORS)
        acc = d["estimators"][ESTIMATORS[0]]["accuracy"]
        print(f"{label:>7} {d['wer']:>6.3f} {acc:>6.3f}  {cells}")

    print("\nAUC-NT (precision/recall for errors; chance = corpus error rate)")
    print(f"{'SNR dB':>7} {'chance':>7}  {head}")
    for label, d in per_level.items():
        acc = d["estimators"][ESTIMATORS[0]]["accuracy"]
        cells = "  ".join(f"{d['estimators'][e]['auc_nt']:>13.3f}"
                          for e in ESTIMATORS)
        print(f"{label:>7} {1 - acc:>7.3f}  {cells}")

    print("\nCalibration of the per-word signal (ECE / NCE, word_prob)")
    print(f"{'SNR dB':>7} {'ECE':>8} {'NCE':>8} {'AURC':>8}")
    for label, d in per_level.items():
        s = d["estimators"]["word_prob"]
        print(f"{label:>7} {s['ece']:>8.3f} {s['nce']:>8.3f} {s['aurc']:>8.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(per_level, indent=2))
        print(f"\nraw pairs -> {args.json}")


if __name__ == "__main__":
    main()
