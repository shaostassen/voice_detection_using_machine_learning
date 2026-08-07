#!/usr/bin/env python3
"""Fit the reliability policy from a reliability-study JSON.

Answers the deployment question rather than the research one: *at a tolerated
error rate, how much of the transcript can be auto-accepted, and above what
score?* Fits an isotonic calibrator and a risk-targeted threshold per noise
condition, because Phase 1 showed a single global calibration cannot work —
per-word probability is well calibrated down to 0 dB and its NCE goes negative
at -5 dB while its AUROC stays 0.76.

Also reports held-out numbers via a split, since a calibrator scored on its
own fitting data will always look good.

    python scripts/fit_policy.py rel_large.json --target-risk 0.02
    python scripts/fit_policy.py rel_large.json --out speechlens/policies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechlens.calibration import (Calibrator, ReliabilityPolicy,  # noqa: E402
                                    threshold_for_risk)
from speechlens.confidence import auroc, ece, nce  # noqa: E402


def split(scores, labels, frac=0.5):
    """Deterministic interleaved split — every other word to each half.

    Interleaving rather than slicing keeps both halves drawn from the same
    utterances; a contiguous split would put different speakers in each side
    and confound calibration quality with speaker difficulty.
    """
    a = [(s, c) for i, (s, c) in enumerate(zip(scores, labels)) if i % 2 == 0]
    b = [(s, c) for i, (s, c) in enumerate(zip(scores, labels)) if i % 2 == 1]
    return ([x[0] for x in a], [x[1] for x in a],
            [x[0] for x in b], [x[1] for x in b])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path")
    p.add_argument("--estimator", default="word_prob")
    p.add_argument("--target-risk", type=float, default=0.02)
    p.add_argument("--out", default=None, help="directory to write policies")
    args = p.parse_args()

    data = json.loads(Path(args.json_path).read_text())
    print(f"estimator={args.estimator}  target_risk={args.target_risk}\n")

    print("Calibration, fit on half and scored on the held-out half")
    print(f"{'cond':>6} {'n':>6} {'acc':>6} {'AUROC':>7} "
          f"{'ECE raw':>8} {'ECE cal':>8} {'NCE raw':>8} {'NCE cal':>8}")

    policies = {}
    for cond, d in data.items():
        raw = d.get("_raw")
        if not raw:
            sys.exit("JSON has no _raw block; re-run reliability.py with --json")
        scores, labels = raw[args.estimator], raw["labels"]
        fs, fl, hs, hl = split(scores, labels)

        cal = Calibrator.fit(fs, fl, condition=cond)
        cal_h = [cal.predict(s) for s in hs]

        print(f"{cond:>6} {len(scores):>6} "
              f"{sum(hl)/len(hl):>6.3f} {auroc(hs, hl):>7.3f} "
              f"{ece(hs, hl):>8.3f} {ece(cal_h, hl):>8.3f} "
              f"{nce(hs, hl):>8.3f} {nce(cal_h, hl):>8.3f}")

        pol = ReliabilityPolicy.fit(scores, labels,
                                    target_risk=args.target_risk,
                                    condition=cond,
                                    notes=f"{args.estimator}, large-v3 int8, "
                                          f"73 LibriSpeech utts, {cond}")
        policies[cond] = pol

    print(f"\nOperating points at target risk {args.target_risk:.0%} "
          f"(accept when calibrated score >= threshold)")
    print(f"{'cond':>6} {'WER':>7} {'threshold':>10} {'coverage':>9} "
          f"{'risk':>7}  interpretation")
    for cond, pol in policies.items():
        wer = data[cond]["wer"]
        if pol.coverage == 0.0:
            note = "nothing is safe to auto-accept"
        else:
            note = (f"auto-accept {pol.coverage:.0%} of words, "
                    f"{pol.realized_risk:.1%} of those wrong")
        print(f"{cond:>6} {wer:>7.3f} {pol.threshold:>10.3f} "
              f"{pol.coverage:>9.1%} {pol.realized_risk:>7.1%}  {note}")

    print("\nCoverage at other risk tolerances (fraction auto-accepted)")
    targets = [0.01, 0.02, 0.05, 0.10]
    print(f"{'cond':>6} " + "  ".join(f"{t:>7.0%}" for t in targets))
    for cond, d in data.items():
        raw = d["_raw"]
        cal = policies[cond].calibrator
        cs = [cal.predict(s) for s in raw[args.estimator]]
        cells = []
        for t in targets:
            _thr, cov, _r = threshold_for_risk(cs, raw["labels"], t)
            cells.append(f"{cov:>7.1%}")
        print(f"{cond:>6} " + "  ".join(cells))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for cond, pol in policies.items():
            name = f"word_prob_{cond.replace('-', 'neg')}.json"
            pol.save(out / name)
        print(f"\nwrote {len(policies)} policies -> {out}/")


if __name__ == "__main__":
    main()
