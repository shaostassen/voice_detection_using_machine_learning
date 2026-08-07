#!/usr/bin/env python3
"""Phase 2: does entropy beat the chosen-token probability at spotting errors?

The confidence literature reports entropy-based estimators detecting incorrect
words 1.5-4x better than probability-based ones, at no extra compute, for CTC
and transducer models. Nobody appears to have checked it on Whisper, whose
decoder mixes acoustic and language-model evidence in a way CTC's does not.

Two constraints, both established by spikes before this was written:

1. CTranslate2 exposes full per-step distributions via `return_logits_vocab`,
   so entropy needs no torch -- but **only under greedy decoding**. With
   `beam_size > 1` the field comes back NULL. Production SpeechLens uses
   beam-5, so every number here is measured on a greedy decode and is not
   directly comparable to the beam-5 figures in the reliability study. The
   comparison inside this script is still valid: every estimator sees the
   same decode.

2. `logits[i]` is the distribution that produced `tokens[i]` (offset 0),
   verified by recomputing the cumulative log-probability and matching
   faster-whisper's own reported score to ~0.1%.

    python scripts/entropy_study.py --model small --limit 30
"""
from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import banner, mix_at_snr  # noqa: E402

from speechlens.confidence import (normalized_entropy, summarize)  # noqa: E402
from speechlens.metrics import labels_for_words, wer  # noqa: E402

LEVELS = [None, 20, 10, 5, 0, -5]
ESTIMATORS = ("word_prob", "word_min_prob", "entropy_h1", "entropy_h2",
              "entropy_min_h2")


def log_softmax(row: np.ndarray) -> np.ndarray:
    row = row.astype(np.float64)
    mx = row.max()
    shifted = row - mx
    return shifted - math.log(float(np.exp(shifted).sum()))


def decode_words(model, tokenizer, audio, sr):
    """Greedy decode one utterance -> per-word estimator scores.

    Returns ``(words, scores)`` where every estimator is oriented so that
    higher means more confident.
    """
    # pad_or_trim to exactly 3000 frames: the encoder expects a fixed 30 s
    # window and silently produces garbage on a short one. faster-whisper does
    # this internally (transcribe.py, `segment = pad_or_trim(segment)`); doing
    # the slice by hand without it is how this study first produced WER > 2.
    from faster_whisper.audio import pad_or_trim

    features = model.feature_extractor(audio)
    seg = pad_or_trim(features[:, : model.feature_extractor.nb_max_frames])
    enc = model.encode(seg)

    # Timestamps ON. Whisper pads every window to 30 s, and a short utterance
    # is mostly silence; told not to emit timestamps it rambles over the
    # padding and produces WER > 1 from pure insertion. The timestamp tokens
    # are what let it terminate at end of speech. (Production avoids this a
    # second way, with VAD gating -- bypassed here to isolate the decoder.)
    prompt = model.get_prompt(tokenizer, [], without_timestamps=False)

    res = model.model.generate(enc, [prompt], beam_size=1, max_length=440,
                               return_scores=True,
                               return_logits_vocab=True)[0]
    tokens = res.sequences_ids[0]
    if res.logits is None or not len(tokens):
        return [], {e: [] for e in ESTIMATORS}
    logits = np.array(res.logits[0], dtype=np.float32)

    # Per-token quantities, indexed the same way as `tokens`.
    tok_prob, tok_h1, tok_h2 = [], [], []
    for i, t in enumerate(tokens):
        if i >= logits.shape[0]:
            break
        lp = log_softmax(logits[i])
        tok_prob.append(float(math.exp(lp[t])))
        tok_h1.append(normalized_entropy(lp, alpha=1.0))
        tok_h2.append(normalized_entropy(lp, alpha=2.0))

    n = len(tok_prob)
    text_idx = [i for i, t in enumerate(tokens[:n]) if t < tokenizer.eot]
    text_tokens = [tokens[i] for i in text_idx]
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens)

    scores = {e: [] for e in ESTIMATORS}
    out_words, cursor = [], 0
    for w, wt in zip(words, word_tokens):
        idx = text_idx[cursor:cursor + len(wt)]
        cursor += len(wt)
        if not idx:
            continue
        p = [tok_prob[i] for i in idx]
        h1 = [tok_h1[i] for i in idx]
        h2 = [tok_h2[i] for i in idx]
        out_words.append(w)
        scores["word_prob"].append(sum(p) / len(p))
        scores["word_min_prob"].append(min(p))
        scores["entropy_h1"].append(sum(h1) / len(h1))
        scores["entropy_h2"].append(sum(h2) / len(h2))
        scores["entropy_min_h2"].append(min(h2))
    return out_words, scores


def load_corpus(limit: int):
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-type", default="auto")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()

    banner("entropy", args, args.model)
    print("NOTE: greedy decode (beam_size=1). CTranslate2 returns no logits "
          "under beam\nsearch, so these are not comparable to the beam-5 "
          "reliability study; the\nestimators here are all measured on the "
          "same greedy decode.\n")

    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer

    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual,
                          task="transcribe", language="en")

    corpus = load_corpus(args.limit)
    print(f"corpus: {len(corpus)} utterances, "
          f"{sum(len(y) / sr for y, sr, _ in corpus) / 60:.1f} min\n")

    rng = np.random.default_rng(0)
    per_level = {}

    for snr in LEVELS:
        label = "clean" if snr is None else str(snr)
        agg = {e: [] for e in ESTIMATORS}
        labels_all, refs, hyps = [], [], []
        for y, sr, ref in corpus:
            audio = y if snr is None else mix_at_snr(y, snr, rng)
            words, scores = decode_words(model, tokenizer, audio, sr)
            if not words:
                continue
            labels = labels_for_words(ref, words)
            keep = [i for i, lab in enumerate(labels) if lab is not None]
            for e in ESTIMATORS:
                agg[e].extend(scores[e][i] for i in keep)
            labels_all.extend(labels[i] for i in keep)
            refs.append(ref)
            hyps.append("".join(words))
        per_level[label] = {
            "wer": wer(" ".join(refs), " ".join(hyps)) if refs else float("nan"),
            "estimators": {e: summarize(agg[e], labels_all) for e in ESTIMATORS},
        }
        print(f"  {label:>5} dB  wer={per_level[label]['wer']:.3f}  "
              f"words={len(labels_all)}")

    head = "  ".join(f"{e:>15}" for e in ESTIMATORS)
    print("\n" + "=" * 96)
    print("WITHIN-CONDITION AUROC — probability vs entropy, same greedy decode")
    print("=" * 96)
    print(f"{'SNR dB':>7} {'wer':>6}  {head}")
    for label, d in per_level.items():
        cells = "  ".join(f"{d['estimators'][e]['auroc']:>15.3f}"
                          for e in ESTIMATORS)
        print(f"{label:>7} {d['wer']:>6.3f}  {cells}")

    print("\nAUC-NT (precision/recall for errors)")
    print(f"{'SNR dB':>7} {'chance':>7}  {head}")
    for label, d in per_level.items():
        acc = d["estimators"][ESTIMATORS[0]]["accuracy"]
        cells = "  ".join(f"{d['estimators'][e]['auc_nt']:>15.3f}"
                          for e in ESTIMATORS)
        print(f"{label:>7} {1 - acc:>7.3f}  {cells}")


if __name__ == "__main__":
    main()
