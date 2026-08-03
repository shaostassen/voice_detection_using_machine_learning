# Validation log

Every `scripts/validate.py` run gets recorded here. A number is only allowed
to exist in this project alongside the run that produced it — hardware,
model, `compute_type`, date, and the raw command. `validate.py` prints all
five as a provenance banner on its first two lines, so a record is a paste,
not a reconstruction. Raw stdout for each run is kept verbatim in
[`validation_runs/`](validation_runs/).

## How results get produced

| Environment | Role | Entry point |
|---|---|---|
| Free cloud T4 (Colab/Kaggle), `large-v3`, `float16` | **Authoritative numbers.** No local NVIDIA GPU exists on any dev machine. | `notebooks/validate_t4.ipynb` |
| MacBook Pro M2, `int8` | **CPU baseline** — measured 2026-08-02, same clip as the T4 run | `scripts/validate.py bench --device cpu --compute-type int8` |
| Ryzen 9 9950X, 32 threads, `int8` | Server-CPU baseline — **not yet run** | same command |
| Jetson Orin Nano | Edge target; GPU path unverified (needs a ctranslate2 CUDA build for JetPack) | pending |

`scripts/make_clip.py` rebuilds the canonical 23.2 s clip identically on any
machine. Use it before benching new hardware — a bench on a different clip
is a different measurement, not a comparison.

---

# Run — 2026-08-02 — Tesla T4

| field | value |
|---|---|
| hardware | Tesla T4 (Colab) |
| model | `large-v3` |
| compute_type | `float16` |
| device | `cuda` |
| audio source | 3 concatenated LibriSpeech utterances, 23.2 s, ground-truth transcript |
| publishable WER | yes |
| notebook | `notebooks/validate_t4.ipynb` |
| raw logs | [`validation_runs/`](validation_runs/) |

**Statistical power caveat, applies to everything below:** this is a single
23.2 s clip from one speaker. WER differences below roughly 0.05 are not
resolvable here, and the SNR ladder is non-monotonic in places (0.05 at
20 dB vs 0.03 at 10 dB) which is measurement noise, not a real effect. These
numbers characterize behavior and order-of-magnitude cost. They are not a
benchmark result on a labeled corpus.

## Study 1 — smoke (multilingual LID + plumbing)

```
python scripts/validate.py smoke --model large-v3 --device cuda --compute-type float16
```

| voice | expected | got | p | metric | err |
|---|---|---|---|---|---|
| en | en | en | 0.90 | wer | 0.00 |
| de | de | de | 0.99 | wer | 0.11 |
| fr | fr | fr | 0.32 | wer | 0.70 |
| es | es | es | 0.91 | wer | 0.00 |
| cmn | zh | zh | 0.80 | cer | 0.25 |

**LID hits: 5/5.** Chunk-voted LID identified every language correctly,
including Mandarin, and the CJK branch correctly dispatched to CER.

The error rates are *not* an accuracy claim — espeak-ng audio is robotic and
synthetic. The French row is the interesting one: LID stayed correct but at
p=0.32, and WER was 0.70. Low LID confidence tracked genuinely degraded
recognition rather than being noise, which is the intended behavior of the
probability output.

## Study 2 — SNR sweep (noise robustness)

White noise mixed at a falling SNR ladder, seeded (`default_rng(0)`) so the
ladder is reproducible.

**Denoise OFF** — `large-v3`, `float16`, Tesla T4, 2026-08-02

| SNR dB | lang | p | metric | err | flagged |
|---|---|---|---|---|---|
| clean | en | 1.00 | wer | 0.03 | 0 |
| 20 | en | 1.00 | wer | 0.05 | 0 |
| 10 | en | 1.00 | wer | 0.03 | 0 |
| 5 | en | 1.00 | wer | 0.05 | 0 |
| 0 | en | 0.97 | wer | 0.15 | 0 |
| −5 | en | 0.77 | wer | 0.33 | 0 |

**Denoise ON** — same run configuration, `--denoise`

| SNR dB | lang | p | metric | err | flagged |
|---|---|---|---|---|---|
| clean | en | 1.00 | wer | 0.05 | 0 |
| 20 | en | 1.00 | wer | 0.05 | 0 |
| 10 | en | 1.00 | wer | 0.05 | 0 |
| 5 | en | 1.00 | wer | 0.12 | 0 |
| 0 | en | 0.98 | wer | 0.28 | 0 |
| −5 | en | 0.63 | wer | 0.48 | 0 |

### What the sweep says

**1. LID stability — holds.** The language stayed `en` at every level down to
−5 dB, at p=1.00 through 5 dB. Confidence decayed smoothly (1.00 → 0.97 →
0.77) rather than collapsing, so the probability degrades in step with the
audio instead of staying falsely pegged.

**2. WER degradation — graceful.** 0.03 clean → 0.33 at −5 dB, with no cliff
and no runaway hallucination: an 11× error increase across a 25 dB swing in
conditions, monotone once past the noise floor of the measurement.

**3. Confidence flags — DID NOT FIRE. This is a negative result.**
`flagged` is 0 at every single level, including −5 dB where one word in
three is wrong. The stated design goal was that flag counts climb *before*
the transcript degrades; on this clip they never climbed at all. The
confidence gate contributed nothing here.

The mechanism is visible in the definition: a segment is flagged when
`exp(avg_logprob) < 0.55`, i.e. `avg_logprob < −0.598`. `large-v3` stays
confident on this material even when it is wrong — well-documented Whisper
overconfidence — so the 0.55 threshold is simply never crossed. **Do not
claim the confidence gate is validated.** It is unfalsified in the sense
that it was never exercised, which is the weaker and less interesting state.

Follow-up before this can be claimed to work: sweep the `low_confidence`
threshold against this ladder and find the value where flag counts lead WER
growth, or establish that `avg_logprob` on `large-v3` is too flat to
threshold usefully and that a different signal (`no_speech_prob`, entropy,
or LID probability, which *did* track degradation) is the right one.
Per the hard rules, any change to `low_confidence` needs its own A/B here.

### A/B verdict — denoise LOSES, and loses everywhere

| SNR dB | WER off | WER on | delta |
|---|---|---|---|
| clean | 0.03 | 0.05 | +0.02 |
| 20 | 0.05 | 0.05 | 0.00 |
| 10 | 0.03 | 0.05 | +0.02 |
| 5 | 0.05 | 0.12 | +0.07 |
| 0 | 0.15 | 0.28 | +0.13 |
| −5 | 0.33 | 0.48 | +0.15 |

Denoise is neutral-to-slightly-harmful in clean conditions and **clearly
harmful exactly where it was supposed to help** — the gap widens as SNR
falls, reaching +0.15 WER at −5 dB. It also degraded LID confidence at −5 dB
(0.63 with, 0.77 without).

This is stronger than the prior recorded in CLAUDE.md ("denoise loses above
~5 dB SNR"). The prior expected a crossover where spectral gating starts
paying off at low SNR; there is no crossover in this data. Spectral gating
smears the formants Whisper relies on, and Whisper's noise robustness is
already better than the filter's. **Keep `denoise=False` as the default;
`--denoise` stays opt-in and is not recommended at any SNR tested.**

## Study 3 — bench (real-time factor)

```
python scripts/validate.py bench /content/validation_clip.wav \
    --models base,small,distil-large-v3,large-v3 --device cuda --compute-type float16
```

23.2 s of audio, Tesla T4, `float16`, 2026-08-02. Model load happens in the
`SpeechLens` constructor and is excluded, so this is decode throughput, not
cold start. Timing covers the full pipeline: VAD, chunk-voted LID, and decode.

| model | RTF | × realtime |
|---|---|---|
| base | 0.038 | 26.3 |
| small | 0.036 | 27.8 |
| distil-large-v3 | 0.046 | 21.7 |
| large-v3 | 0.082 | 12.2 |

`large-v3` runs **12.2× faster than realtime** on a free-tier T4. `base` and
`small` are indistinguishable here (0.038 vs 0.036) — at ~27× realtime on a
23 s clip the per-run fixed costs dominate, so this bench cannot separate
them; it is not evidence that `small` is faster than `base`.
`distil-large-v3` gives 1.8× the throughput of `large-v3`.

---

# Run — 2026-08-02 — Apple M2 (CPU baseline)

| field | value |
|---|---|
| hardware | Apple M2 (Darwin arm64) |
| model | `base`, `small`, `distil-large-v3` |
| compute_type | `int8` |
| device | `cpu` |
| audio source | **the same 23.2 s LibriSpeech clip as the T4 run** |
| raw log | [`validation_runs/bench_cpu_m2.txt`](validation_runs/bench_cpu_m2.txt) |

Same clip, same command shape, so these rows are directly comparable to the
T4 table above rather than approximately so.

```
python scripts/validate.py bench clip.wav \
    --models base,small,distil-large-v3 --device cpu --compute-type int8
```

| model | RTF (M2, int8) | × realtime | RTF (T4, float16) | CPU slowdown |
|---|---|---|---|---|
| base | 0.083 | 12.0 | 0.038 | 2.2× |
| small | 0.205 | 4.9 | 0.036 | 5.7× |
| distil-large-v3 | 0.605 | 1.7 | 0.046 | 13.2× |

Run-to-run variance is ~2% (an earlier identical run gave 0.085 / 0.203 /
0.588), so these are stable at the precision shown.

### What the CPU baseline says

**The local-deployment claim holds.** `small` at int8 runs at **4.9×
realtime** on a laptop CPU, and even `distil-large-v3` clears realtime at
1.7×. A CPU-only box can run this pipeline live, which is the claim the
project's "no cloud calls" framing depends on and which no GPU number could
support.

**The GPU advantage widens sharply with model size** — 2.2× for `base`,
13.2× for `distil-large-v3`. The reason is visible in the T4 table: there,
all three models cost about the same (0.036–0.046), because a 23 s clip
does not saturate a T4 and fixed per-run overhead dominates. CPU has no such
headroom, so cost tracks parameter count. Two consequences:

- The T4 bench understates the real cost difference between model sizes.
  Do not use it to reason about model selection.
- Model choice matters far more on CPU than on GPU. On the T4, `large-v3`
  costs 2.2× `base`; the CPU spread across a smaller set of models is
  already 7.3×.

`large-v3` was not benchmarked on CPU here — expected to be slow enough that
it is a single-clip spot check only, not a realtime option. Untested rather
than assumed.

## Config-change A/Bs

`RobustnessConfig` defaults are load-bearing (VAD on, context carry off,
temp ladder 0→1.0, CR gate 2.4, logprob gate −1.0, no-speech 0.6, flag
0.55). Changing one requires an SNR-sweep A/B recorded here.

| date | knob | from → to | effect on WER / flags | verdict |
|---|---|---|---|---|
| 2026-08-02 | `denoise` | off → on | WER +0.02 to +0.15, worst at low SNR; LID p 0.77 → 0.63 at −5 dB | **rejected**, keep off |
| — | `low_confidence` (0.55) | — | open: gate never fired across the whole ladder | needs a threshold sweep |

## Open items

- **CPU baseline** on the 9950X (task 4) — not yet run.
- **`low_confidence` threshold sweep** — the gate did not fire at any SNR;
  it cannot be described as working until this is resolved.
- **Jetson Orin Nano** — ctranslate2 CUDA build for JetPack unattempted.
- Broader corpus (LibriSpeech test-clean/test-other, FLEURS) before any of
  these WER figures are described as benchmark results rather than
  characterization of one clip.
