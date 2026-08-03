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
| Ryzen 9 9950X, 32 threads, `int8` | **Server-CPU baseline** — measured 2026-08-03, byte-identical clip | same command |
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

---

# Run — 2026-08-03 — Ryzen 9 9950X (server CPU baseline)

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X 16-Core (32 threads), Linux x86_64 |
| model | `base`, `small`, `distil-large-v3`, `large-v3` |
| compute_type | `int8` |
| device | `cpu` |
| audio source | **byte-identical to the T4 and M2 clip** (sha256 `7bbf2ddf3d0f767d…`, built by `scripts/make_clip.py`) |
| raw log | [`validation_runs/bench_cpu_9950x.txt`](validation_runs/bench_cpu_9950x.txt) |

| model | RTF | × realtime | vs M2 | vs T4 |
|---|---|---|---|---|
| base | 0.028 | 35.7 | 3.0× faster | **1.4× faster than the T4** |
| small | 0.063 | 15.9 | 3.3× faster | 1.8× slower |
| distil-large-v3 | 0.160 | 6.2 | 3.8× faster | 3.5× slower |
| large-v3 | 0.259 | 3.9 | not run on M2 | 3.2× slower |

### What the server baseline says

**1. `large-v3` on CPU is realtime-viable — 3.9× realtime.** This was
recorded as untested, with a standing assumption that it "runs but slowly,
single-clip spot checks only". That assumption is now refuted: a 23 s clip
decodes in about 6 s on the 9950X. The full sweep on CPU at `large-v3` is
practical, not just a spot check.

**2. The 9950X beats the T4 on `base` (0.028 vs 0.038).** A CPU beating a
GPU looks wrong until you look at the T4 column: every model there costs
0.036–0.082 regardless of size, because a 23 s clip never saturates the
device and fixed overhead dominates. The T4's real advantage shows up only
as the model grows — 1.8× at `small`, 3.2× at `large-v3`. For short clips
and small models, the GPU is mostly waiting.

**3. Thread count does scale.** 3.0–3.8× over the M2 across the three
shared models. CTranslate2 int8 is getting real work out of 32 threads
rather than being memory-bound, and the advantage grows with model size,
which is the expected shape if compute rather than bandwidth is the limit.

Practical consequence: for this workload the 9950X is the sensible default
target. The T4 is worth it for `large-v3` throughput; below that it buys
little, and for `base` it loses.

---

# Run — 2026-08-03 — confidence-gate diagnosis (`gate` study)

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X 16-Core (32 threads), Linux x86_64 |
| model | `large-v3` |
| compute_type | `int8` |
| audio source | the same byte-identical 23.2 s labeled clip |
| raw log | [`validation_runs/gate_9950x.txt`](validation_runs/gate_9950x.txt) |

The 2026-08-02 sweep flagged zero segments at every level. A count alone
cannot say whether the threshold was too low or the signal was flat, so this
study prints the underlying `exp(avg_logprob)` distribution. Flagging is
post-hoc, so all candidate thresholds are evaluated on one decode pass.

| SNR dB | segs | confidence | no-speech | LID p | WER |
|---|---|---|---|---|---|
| clean | 2 | 0.957 | 0.000 | 1.00 | 0.05 |
| 20 | 2 | 0.945 | 0.008 | 1.00 | 0.05 |
| 10 | 2 | 0.943 | 0.009 | 1.00 | 0.03 |
| 5 | 4 | 0.886 | 0.030 | 0.99 | 0.05 |
| 0 | 4 | 0.817 | 0.052 | 0.96 | 0.15 |
| −5 | 4 | 0.669 | 0.042 | 0.82 | 0.33 |

Flagged segments / total, by candidate threshold:

| SNR dB | WER | t=0.55 | t=0.65 | t=0.75 | t=0.85 | t=0.95 |
|---|---|---|---|---|---|---|
| clean | 0.05 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 20 | 0.05 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 |
| 10 | 0.03 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 |
| 5 | 0.05 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 |
| 0 | 0.15 | 0/4 | 0/4 | 0/4 | 4/4 | 4/4 |
| −5 | 0.33 | 0/4 | 0/4 | 4/4 | 4/4 | 4/4 |

### Verdict — the signal was fine, the threshold was wrong

**`avg_logprob` is not flat.** Confidence falls monotonically with noise,
0.957 → 0.669, tracking WER closely. The earlier worry that it might be
unusable as a signal is settled: it is usable.

**0.55 was below the worst case.** The lowest confidence observed anywhere,
at 33% WER, was 0.669. A threshold of 0.55 could not fire under any
condition in this sweep — it was not a conservative gate, it was an inert
one, which is worse than none because it implies a safety net that is not
there.

**0.85 is the value that works.** It stays silent through the whole region
where WER ≤ 0.05 (clean, 20, 10, 5 dB) and fires the moment WER climbs to
0.15 at 0 dB, staying on at −5 dB. t=0.75 only fires at −5 dB, by which
point a third of the words are already wrong. t=0.95 flags clean-ish audio
at 20 dB where WER is 0.05 — it cries wolf.

**Two honest limitations.** (1) Confidence min = mean = max at every level:
segments decoded from the same window share an `avg_logprob`, so this is
effectively one reading per level, not a distribution — the threshold is
chosen from 6 points. (2) One clip, one speaker, read English. A default
should hold across accents, spontaneous speech, and other languages;
this shows 0.85 is right *here* and that 0.55 is wrong *everywhere*.

Also worth noting: **LID probability tracked degradation too** (1.00 →
0.82), independently confirming it as the fallback signal proposed when the
gate looked broken. It is no longer needed for this purpose, but it is a
second, cheaper indicator of the same thing.

### Confirmation — the gate working in the live pipeline

Re-ran the full SNR sweep with `low_confidence = 0.85` in place, same clip,
same seeded noise ladder. Raw log:
[`validation_runs/snr_9950x_t085.txt`](validation_runs/snr_9950x_t085.txt).

| SNR dB | lang | p | WER | flagged |
|---|---|---|---|---|
| clean | en | 1.00 | 0.05 | 0 |
| 20 | en | 1.00 | 0.05 | 0 |
| 10 | en | 1.00 | 0.03 | 0 |
| 5 | en | 0.99 | 0.05 | 0 |
| 0 | en | 0.96 | 0.15 | **4** |
| −5 | en | 0.82 | 0.33 | **4** |

The flagged column now separates the usable transcripts from the degraded
ones exactly where the error rate steps up, which is what the gate was
supposed to do from the beginning. This supersedes the flagged column in
the 2026-08-02 T4 sweep, which ran under the inert threshold.

(WER here is marginally different from the T4 run — 0.05 vs 0.03 on clean —
because this is `int8` on CPU rather than `float16`. A ~0.02 difference on
one clip is inside the noise floor of this measurement, not a quantization
finding.)

## Config-change A/Bs

`RobustnessConfig` defaults are load-bearing (VAD on, context carry off,
temp ladder 0→1.0, CR gate 2.4, logprob gate −1.0, no-speech 0.6, flag
0.55). Changing one requires an SNR-sweep A/B recorded here.

| date | knob | from → to | effect on WER / flags | verdict |
|---|---|---|---|---|
| 2026-08-02 | `denoise` | off → on | WER +0.02 to +0.15, worst at low SNR; LID p 0.77 → 0.63 at −5 dB | **rejected**, keep off |
| 2026-08-03 | `low_confidence` | 0.55 → **0.85** | 0.55 flagged 0/18 segments across the ladder incl. 33% WER; 0.85 flags 0 while WER ≤ 0.05 and 8/8 once WER ≥ 0.15 | **adopted** |

## Open items

- **Confirm 0.85 beyond one clip** — accents, spontaneous speech, non-English.
  0.55 is refuted everywhere; 0.85 is only established here.
- Broader corpus (LibriSpeech test-clean/test-other, FLEURS) before any of
  these WER figures are described as benchmark results rather than
  characterization of one clip.
- ~~Jetson Orin Nano~~ — **descoped by Shao 2026-08-03.**
