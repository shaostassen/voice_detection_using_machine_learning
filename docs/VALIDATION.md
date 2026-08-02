# Validation log

Every `scripts/validate.py` run gets recorded here. A number is only allowed
to exist in this project alongside the run that produced it — hardware,
model, `compute_type`, date, and the raw command. `validate.py` prints all
five as a provenance banner on its first two lines, so a record is a paste,
not a reconstruction.

**Nothing below is filled in yet.** The tables are the shape the results
take; `—` means not-yet-measured, and it stays `—` until a real run replaces
it. No estimated, interpolated, or remembered values.

## How results get produced

| Environment | Role | Entry point |
|---|---|---|
| Free cloud T4 (Colab/Kaggle), `large-v3`, `float16` | **Authoritative numbers.** No local NVIDIA GPU exists on any dev machine. | `notebooks/validate_t4.ipynb` |
| Ryzen 9 9950X, 32 threads, `int8` | CPU baseline for the same clip | `scripts/validate.py bench --device cpu --compute-type int8` |
| MacBook Pro M2, `int8` | Local iteration only (`base` / `small`) | same, smaller models |
| Jetson Orin Nano | Edge target; GPU path unverified (needs a ctranslate2 CUDA build for JetPack) | pending |

The T4 notebook runs all three studies and emits a ready-to-paste block for
this file.

## Study 1 — smoke (multilingual LID + plumbing)

espeak-ng synthesizes a known sentence in 5 languages; checks the detected
language code and reports error rates. espeak audio is robotic and synthetic,
so **nonzero error rates are expected and are not an accuracy claim** — the
figure that matters is LID hits out of 5.

| date | hardware | model | compute_type | LID hits | command |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

## Study 2 — SNR sweep (noise robustness)

White noise mixed into labeled real speech at a falling SNR ladder (clean,
20, 10, 5, 0, −5 dB), seeded for reproducibility. Run twice, with and
without `--denoise`, to A/B the spectral-gating stage. Prior expectation:
denoise loses above ~5 dB SNR — record which arm actually wins.

Three things to watch:

1. **LID stability** as noise rises — chunk voting should hold the correct
   language further down the ladder than a single window would.
2. **Graceful WER degradation** rather than a cliff into hallucinated text.
3. **Flagged-segment counts climbing _before_ the transcript goes bad.**
   Flags leading errors is the confidence gate working; flags trailing
   errors means the gate is decorative.

WER is only meaningful against real speech with a ground-truth transcript
(the notebook defaults to labeled LibriSpeech utterances). A sweep run on
espeak audio validates plumbing only and must not be published.

**Denoise OFF** — hardware `—`, model `—`, compute_type `—`, date `—`

```
command: —
```

| SNR dB | lang | p | metric | err | flagged |
|---|---|---|---|---|---|
| clean | — | — | — | — | — |
| 20 | — | — | — | — | — |
| 10 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 0 | — | — | — | — | — |
| −5 | — | — | — | — | — |

**Denoise ON** — hardware `—`, model `—`, compute_type `—`, date `—`

```
command: —
```

| SNR dB | lang | p | metric | err | flagged |
|---|---|---|---|---|---|
| clean | — | — | — | — | — |
| 20 | — | — | — | — | — |
| 10 | — | — | — | — | — |
| 5 | — | — | — | — | — |
| 0 | — | — | — | — | — |
| −5 | — | — | — | — | — |

A/B verdict: **—**

## Study 3 — bench (real-time factor)

RTF = processing time / audio duration; lower is better. Model load happens
in the `SpeechLens` constructor and is excluded, so this measures decode
throughput, not cold start. Source of the portfolio RTF chip.

| date | hardware | compute_type | model | RTF | × realtime | command |
|---|---|---|---|---|---|---|
| — | — | — | base | — | — | — |
| — | — | — | small | — | — | — |
| — | — | — | distil-large-v3 | — | — | — |
| — | — | — | large-v3 | — | — | — |

## Config-change A/Bs

`RobustnessConfig` defaults are load-bearing (VAD on, context carry off,
temp ladder 0→1.0, CR gate 2.4, logprob gate −1.0, no-speech 0.6, flag
0.55). Changing one requires an SNR-sweep A/B recorded here.

| date | knob | from → to | effect on WER / flags | verdict |
|---|---|---|---|---|
| — | — | — | — | — |
