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

---

# Finding — 2026-08-07 — `avg_logprob` is per-window, not per-segment

| field | value |
|---|---|
| probe | [`scripts/probe_window_scope.py`](../scripts/probe_window_scope.py) |
| raw log | [`validation_runs/window_scope_probe.txt`](validation_runs/window_scope_probe.txt) |
| model | `small`, int8, CPU (the effect is structural, not model-specific) |
| audio | 130.5 s — 12 concatenated LibriSpeech utterances, forcing 5 decode windows |

faster-whisper computes `avg_logprob` **once per 30-second decode window**
(`transcribe.py:1466`, from `result.scores[0]` over the whole generated
sequence) and then assigns that one value to **every segment** the window
produced (`transcribe.py:1362`). `no_speech_prob` and `compression_ratio` are
broadcast the same way.

Measured on the 130 s clip:

| quantity | count |
|---|---|
| segments | 22 |
| distinct decode windows (`seek`) | 5 |
| **distinct `avg_logprob` values** | **5** |
| distinct `no_speech_prob` values | 5 |
| distinct `compression_ratio` values | 5 |
| per-word probability range | 0.150 – 0.999 |

Distinct values track the window count, not the segment count.

### Why this matters

`TranscriptSegment.confidence` is `exp(avg_logprob)`, so **the confidence this
project flags on is a per-window figure wearing a per-segment label.** Within
a window it is constant, which means it cannot — even in principle — pick the
bad segment out of a good window. That is the job a reliability flag exists to
do.

It also explains the `conf min == mean == max` rows in the 2026-08-03 gate
study, which at the time looked like a quirk of a short clip. It was not: it
is the data model.

**Consequence for the 0.85 threshold adopted on 2026-08-03: it is provisional.**
The A/B that justified it compared six *window-level* points across noise
conditions, where the signal does vary and does separate clean from degraded
audio. Nothing in that data speaks to within-window discrimination, because
there was none to measure. 0.55 remains refuted — it sat below the model's
observed floor and could never fire. 0.85 is a working default, not a
validated one, pending the word-level study below.

The genuinely per-unit signal was already in the output the whole time:
`words[].prob` is a mean over that word's own token probabilities
(`transcribe.py:1747`) and spanned 0.150–0.999 on the same clip while the
window value stayed flat. It is built at `speechlens/asr.py`, serialized to
JSON, and read by nothing — not the CLI, not the web UI, and not reachable
through the HTTP API at all.

---

# Run — 2026-08-07 — word-level confidence: does it predict *which* word is wrong?

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X 16-Core (32 threads), Linux x86_64 |
| model | `large-v3`, `int8`, beam-5 (production decode) |
| corpus | 73 LibriSpeech utterances, 8.0 min, ~1,160 words **per condition** |
| language | forced `en` — skips chunk-voted LID, removing LID drift as a confound |
| script | [`scripts/reliability.py`](../scripts/reliability.py) |
| raw log | [`validation_runs/reliability_9950x.txt`](validation_runs/reliability_9950x.txt) |

Every previous measurement here was **between-condition**: mean confidence
falls as SNR falls. That never established the property a flag actually needs
— **within-condition discrimination**, i.e. at a fixed noise level, does the
score rank wrong words below right ones? A signal can track SNR perfectly and
still be useless for flagging.

Word-level correctness comes from a Levenshtein backtrace against the
reference (`speechlens.metrics.labels_for_words`), with each word normalized
independently so multi-token words like "don't" cannot desynchronize scores
from labels.

## Within-condition AUROC (0.5 = chance)

| SNR dB | WER | accuracy | **word_prob** | seg_conf | seg_min_word | speech_prob |
|---|---|---|---|---|---|---|
| clean | 0.047 | 0.957 | **0.894** | 0.581 | 0.642 | 0.533 |
| 20 | 0.064 | 0.942 | **0.883** | 0.645 | 0.645 | 0.636 |
| 10 | 0.076 | 0.931 | **0.856** | 0.693 | 0.666 | 0.621 |
| 5 | 0.117 | 0.892 | **0.850** | 0.683 | 0.632 | 0.603 |
| 0 | 0.288 | 0.737 | **0.826** | 0.701 | 0.600 | 0.606 |
| −5 | 0.670 | 0.426 | **0.762** | 0.681 | 0.564 | 0.625 |

`word_prob` is the per-word mean token probability. `seg_conf` is
`exp(avg_logprob)` — the signal the shipped confidence gate actually uses.

## Verdict — the signal works, but it is not the one being used

**Per-word probability discriminates errors well at every noise level**
(AUROC 0.76–0.89), clearing the ≳0.75 bar set before the experiment ran. The
feared outcome — that confidence is merely an SNR meter with no within-
condition signal — is refuted.

**The gate is flagging on the weaker signal.** `seg_conf` trails `word_prob`
everywhere, and the gap is widest exactly where it matters most: on clean
audio, 0.894 vs **0.581**. That is close to chance, and clean audio is
precisely where errors are rare and a reliable flag is most valuable. The
cause is structural, not tuning: `seg_conf` is constant across a 30 s decode
window (see the per-window finding above), so within a window it cannot rank
anything at all.

**Error-detection precision tells the same story.** AUC-NT against a chance
level equal to the corpus error rate:

| SNR dB | chance | word_prob | seg_conf |
|---|---|---|---|
| clean | 0.043 | **0.316** (7.3× chance) | 0.227 |
| 10 | 0.069 | **0.421** (6.1×) | 0.307 |
| 0 | 0.263 | **0.610** (2.3×) | 0.454 |
| −5 | 0.574 | **0.776** (1.4×) | 0.718 |

**Calibration is better than expected.** For `word_prob`, ECE stays at
0.018–0.085 from clean through 0 dB — a reported 0.9 really does mean about
90% correct — and NCE stays positive (0.15–0.27), so the score beats the base
rate as a predictor. AURC of 0.006 on clean audio means selective prediction
works very well there.

**It breaks down at −5 dB**: ECE 0.218 and **NCE −0.130**, i.e. below zero.
At that point the score is worse than simply predicting the corpus accuracy
for every word, even though its AUROC is still 0.762. That is the
discrimination/calibration split in one row: the ranking is still informative
while the numbers attached to it have stopped meaning anything. Any
calibration layer must be conditioned on noise level, not fitted globally.

### Consequences

1. **Phase 3 is justified.** Build the calibrated reliability layer on
   `words[].prob`, not on `avg_logprob`.
2. **The 0.85 segment threshold should be superseded, not re-tuned.** No
   threshold on a per-window constant can do the job; the granularity is
   wrong, not the value.
3. Deletions remain invisible — a dropped word leaves no token to score. That
   is a structural blind spot of every per-word confidence signal, not a
   defect of this one.

---

# Run — 2026-08-07 — entropy vs probability: does the better estimator exist?

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X 16-Core (32 threads), Linux x86_64 |
| model | `large-v3`, `int8`, **greedy** (see constraint below) |
| corpus | the same 73 utterances, ~1,150 words per condition |
| script | [`scripts/entropy_study.py`](../scripts/entropy_study.py) |
| raw log | [`validation_runs/entropy_9950x.txt`](validation_runs/entropy_9950x.txt) |

The confidence literature reports entropy-based estimators detecting incorrect
words **1.5–4× better** than probability-based ones at no extra compute
([arXiv:2212.08703](https://arxiv.org/abs/2212.08703), SLT 2022) — but for CTC
and transducer models. Whisper's decoder mixes acoustic and language-model
evidence in a way CTC's does not, so it does not follow.

**Constraint, established by spike before the study was written:** CTranslate2
exposes full per-step distributions via `return_logits_vocab`, so entropy
needs no torch — but only under **greedy** decoding. With `beam_size > 1` the
field returns NULL. Production uses beam-5, so these numbers are not
comparable to the beam-5 reliability study above; within this table every
estimator sees the same greedy decode, so the comparison is internally valid.

Token↔logit alignment was verified rather than assumed: recomputing the
cumulative log-probability from the returned distributions reproduced
faster-whisper's own reported score to within 0.15%.

## Within-condition AUROC

| SNR dB | WER | word_prob | word_min_prob | entropy_h1 | entropy_h2 | entropy_min_h2 |
|---|---|---|---|---|---|---|
| clean | 0.038 | 0.882 | 0.889 | 0.887 | 0.883 | **0.890** |
| 20 | 0.046 | 0.867 | **0.875** | 0.868 | 0.868 | 0.874 |
| 10 | 0.068 | 0.855 | 0.861 | 0.857 | 0.857 | **0.863** |
| 5 | 0.121 | 0.859 | **0.866** | 0.852 | 0.859 | 0.864 |
| 0 | 0.293 | 0.834 | **0.835** | 0.833 | **0.835** | **0.835** |
| −5 | 0.684 | 0.811 | 0.815 | 0.813 | 0.815 | **0.816** |

## Verdict — the reported entropy advantage does not replicate on Whisper

**All five estimators are within 0.008 AUROC of each other at every noise
level.** There is no 1.5–4× advantage; there is no advantage at all worth
measuring. This is a negative replication, and a useful one: it says the
result is architecture-specific rather than general.

**What does help is the aggregation, not the estimator family.** Taking the
**minimum** over a word's tokens beats the mean, consistently, and the effect
is much clearer in error-detection precision than in AUROC:

| SNR dB | chance | word_prob (mean) | word_min_prob | entropy_min_h2 |
|---|---|---|---|---|
| clean | 0.036 | 0.249 | 0.274 | **0.291** |
| 20 | 0.043 | 0.294 | **0.374** | 0.357 |
| 10 | 0.063 | 0.356 | 0.426 | **0.444** |
| 5 | 0.111 | 0.464 | **0.502** | 0.490 |

That is a 10–25% relative gain in AUC-NT for free. The mechanism is plain: a
word is wrong if *any* of its tokens went wrong, so the weakest token carries
the evidence and averaging dilutes it.

### The engineering conclusion

**Entropy costs beam search and buys nothing.** Adopting it would mean giving
up beam-5 decoding — logits are greedy-only — plus carrying a 51,865-wide
distribution per decode step, in exchange for an AUROC difference inside the
noise floor. Use **`min` over a word's token probabilities**: it is the best
or joint-best estimator in almost every cell, needs no logits, and works
under the beam search already in production.

Caveat on scope: one corpus, read English, additive white noise. The negative
result is about *this* comparison on Whisper, not a claim that entropy is
useless in general.

---

# Phase 3 — 2026-08-07 — the calibrated reliability layer

| field | value |
|---|---|
| module | [`speechlens/calibration.py`](../speechlens/calibration.py) |
| fitter | [`scripts/fit_policy.py`](../scripts/fit_policy.py) |
| policies | [`speechlens/policies/`](../speechlens/policies/) (6, one per condition) |
| raw output | [`validation_runs/policy_fit.txt`](validation_runs/policy_fit.txt) |
| data | the Phase 1 pairs — `large-v3` int8, 73 utts, ~1,160 words per condition |

Isotonic regression rather than Platt/temperature scaling: the relation
between word probability and correctness is not logistic, and isotonic assumes
only monotonicity — the property AUROC already established. Being monotone it
cannot change the ranking, so AUROC is identical before and after. Calibration
fixes the numbers; it never fixes discrimination.

## Held-out calibration (fit on half, scored on the other half)

| cond | accuracy | AUROC | ECE raw | ECE cal | NCE raw | NCE cal |
|---|---|---|---|---|---|---|
| clean | 0.955 | 0.872 | 0.023 | 0.026 | 0.198 | **0.220** |
| 20 | 0.942 | 0.857 | 0.028 | 0.038 | 0.220 | 0.205 |
| 10 | 0.929 | 0.822 | 0.037 | **0.032** | 0.148 | 0.133 |
| 5 | 0.890 | 0.842 | 0.053 | **0.050** | 0.174 | **0.189** |
| 0 | 0.734 | 0.834 | 0.076 | **0.041** | 0.198 | **0.237** |
| −5 | 0.439 | 0.750 | 0.222 | **0.033** | −0.164 | **+0.149** |

Calibration is roughly neutral where the raw score was already well behaved
(clean through 10 dB) and transformative where it was not. At −5 dB it takes
ECE from 0.222 to 0.033 and flips NCE from **−0.164 to +0.149** — from "worse
than quoting the corpus average" to genuinely informative, without touching
the model. That is precisely the gap Phase 1 identified.

## Operating points at 2% tolerated error

Accept a word when its calibrated score clears the threshold; route the rest
to review.

| cond | WER | threshold | coverage | realized risk |
|---|---|---|---|---|
| clean | 0.047 | 0.765 | **90.4%** | 1.9% |
| 20 | 0.064 | 0.789 | 86.8% | 2.0% |
| 10 | 0.076 | 0.883 | 77.8% | 2.0% |
| 5 | 0.117 | 0.931 | 56.9% | 2.0% |
| 0 | 0.288 | — | **0%** | — |
| −5 | 0.670 | — | **0%** | — |

Coverage at other tolerances:

| cond | 1% | 2% | 5% | 10% |
|---|---|---|---|---|
| clean | 63.5% | 90.4% | 100% | 100% |
| 10 | 26.0% | 77.8% | 95.2% | 100% |
| 5 | 37.9% | 56.9% | 81.5% | 98.2% |
| 0 | 0% | 0% | 43.0% | 59.1% |
| −5 | 0% | 0% | 0% | 0% |

**The refusals are the most useful rows.** At 0 dB and below, no threshold
reaches 2% error: the honest answer is that none of the transcript can be
auto-accepted, and the fitted policy encodes that by refusing everything
rather than quietly lowering the bar. Compare with the old behaviour, where a
single global threshold would have passed most of a transcript that is 29%
wrong.

## Two bugs worth recording, both caught by tests

**Threshold selection was tie-unsafe.** Choosing the marginal item's score as
the threshold reports the risk of a *prefix*, but `score >= threshold` admits
every tied word too. Isotonic calibration produces plateaus, so ties are the
norm rather than an edge case. Fixed by evaluating only at distinct score
boundaries, over the whole admitted set; pinned by a property test that
re-derives risk and coverage from the returned threshold on random tie-heavy
data.

**Unsmoothed isotonic destroys NCE while improving ECE.** Plain isotonic emits
plateaus of exactly 0.0 and 1.0 — claims of certainty — and one held-out
counterexample inside such a plateau craters any log-loss score. Measured
here: NCE went from +0.148 to **−0.502** at 10 dB while ECE improved. Fixed
with Laplace smoothing by plateau support, `(k+1)/(n+2)`, so a plateau resting
on four observations is pulled toward 0.5 far harder than one resting on four
hundred. That in turn exposed a third bug: the PAVA solver only merged on
strict violation, leaving runs of equal values as singleton blocks, so every
one was smoothed as though supported by a single observation.

## Automatic policy selection — 2026-08-07

The limitation above (the caller had to name the noise condition) is closed.
The VAD already partitions the signal: non-speech regions are noise alone,
speech regions are speech plus that same noise, which is enough for an
energy-ratio SNR estimate at no extra model cost
([`speechlens/snr.py`](../speechlens/snr.py)).

Validated against the real ladder — the same clip, the same seeded noise, real
Silero VAD:

| true SNR | estimated | error | policy picked |
|---|---|---|---|
| clean | 38.5 | — | clean ✓ |
| 20 | 20.5 | +0.5 | 20 ✓ |
| 10 | 9.4 | −0.6 | 10 ✓ |
| 5 | 5.5 | +0.5 | 5 ✓ |
| 0 | 0.3 | +0.3 | 0 ✓ |
| −5 | −4.7 | +0.3 | −5 ✓ |

**6/6 correct selections, all within 0.6 dB.** `--reliability auto` now picks
the policy from the audio.

Two details that matter more than they look:

- **The noise floor is subtracted** from the speech-region power. Speech
  regions carry signal *plus* noise, so the naive ratio reads ~3 dB high at
  0 dB SNR — a full rung of the ladder, and therefore the wrong policy.
- **Ties resolve to the noisier policy.** Being too cautious costs coverage;
  being too optimistic ships wrong words as trustworthy.

When the estimate is impossible — no speech, or no non-speech to sample the
noise from — `auto` applies **no policy** and says so in the warnings, rather
than falling back to `clean`. A missing annotation is recoverable; a
fabricated one silently auto-accepts a transcript that might be 29% wrong.

---

# Run — 2026-08-07 — second corpus: does any of this generalize?

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X, `large-v3` int8, beam-5 |
| corpus | **EdAcc** (Edinburgh International Accents of English), test split |
| sample | 100 utterances, 26.9 min, ~3,470 words per condition |
| accents | Bulgarian, Catalan, Chinese, Eastern European, French, Scottish, … |
| loader | [`scripts/corpora.py`](../scripts/corpora.py) |
| raw log | [`validation_runs/reliability_edacc.txt`](validation_runs/reliability_edacc.txt) |

Everything before this came from one corpus of read, scripted, studio-clean
English. EdAcc changes every axis that could have been carrying the result:
**spontaneous** dyadic conversation, **real recording conditions**, and L1/L2
accent variety.

Three loader details that would otherwise have silently corrupted the run:
EdAcc marks non-scoring stretches with `IGNORE_TIME_SEGMENT_IN_SCORING` (left
in, they compare a transcript against a placeholder and report a meaningless
WER); streaming yields utterances grouped by conversation, so taking the first
100 samples *one speaker* and would have produced an "accent diversity" result
from a single Scottish talker; and the audio is 32 kHz with a 2.0 s median
duration, a third of it under one second.

## Within-condition AUROC — EdAcc vs LibriSpeech

| SNR dB | EdAcc WER | EdAcc `word_prob` | LibriSpeech `word_prob` | EdAcc `seg_conf` |
|---|---|---|---|---|
| clean | 0.153 | **0.824** | 0.894 | 0.621 |
| 20 | 0.167 | **0.845** | 0.883 | 0.644 |
| 10 | 0.191 | **0.858** | 0.856 | 0.661 |
| 5 | 0.212 | **0.855** | 0.850 | 0.696 |
| 0 | 0.276 | **0.838** | 0.826 | 0.739 |
| −5 | 0.379 | **0.819** | 0.762 | 0.755 |

**The core finding replicates.** Per-word probability ranks errors at
AUROC 0.82–0.86 on spontaneous accented speech, and the segment signal the old
gate used remains far worse everywhere — 0.824 vs 0.621 on clean audio, the
same shape of gap seen on LibriSpeech.

It is in fact **more stable** here: 0.82–0.86 across the whole ladder against
LibriSpeech's 0.76–0.89, and *better* at −5 dB (0.819 vs 0.762). Harder audio
to begin with leaves less headroom for added noise to destroy.

**A prediction that checked out.** EdAcc's clean WER is 0.153, three times
LibriSpeech's 0.047 — but word-level *accuracy* is 0.951 against 0.957,
essentially identical. Measured beforehand: 4.6% of EdAcc reference words are
fillers (`uh` 102, `um` 68 in the sample), and Whisper performs implicit
disfluency removal. Those become deletions, which inflate WER while producing
no hypothesis word to score. The WER gap is largely a transcription-convention
mismatch, not a confidence-quality difference — which is why it was worth
measuring before seeing the results rather than reaching for afterwards.

## The decisive test: does a fitted policy transfer?

AUROC replicating is necessary but not sufficient. The question that decides
whether the shipped policies are honest is: **at a promised 2% error rate,
what error does a LibriSpeech-fitted policy actually deliver on EdAcc?**

| cond | ECE, own fit | ECE, transferred | promised | **delivered** | coverage |
|---|---|---|---|---|---|
| clean | 0.009 | 0.028 | 2.0% | **2.8%** | 87.8% |
| 20 | 0.017 | 0.051 | 2.0% | **2.4%** | 80.8% |
| 10 | 0.012 | 0.074 | 2.0% | **1.7%** | 58.1% |
| 5 | 0.017 | 0.067 | 2.0% | **1.2%** | 47.4% |
| 0 | 0.015 | 0.115 | 2.0% | — | 0% |
| −5 | 0.021 | 0.221 | 2.0% | — | 0% |

**The decision transfers; the probability does not.** Two different things
come apart here and conflating them would be the easy mistake:

- **The accept/reject gate holds.** Delivered risk is 1.2–2.8% against a 2.0%
  promise. The worst case overshoots by 0.8 points; the rest come in at or
  under target. Coverage drops (87.8% vs 90.4% on clean, 58.1% vs 77.8% at
  10 dB), so transfer costs throughput and errs conservative — the right
  direction to fail in.
- **The calibrated number does not hold.** Transferred ECE is 3–10× worse than
  an EdAcc-native fit, degrading as noise rises (0.009 → 0.028 clean;
  0.021 → 0.221 at −5 dB). So *"this word is 87% likely correct"* is
  materially wrong on an unseen corpus even while the gate built on it still
  works.

Practical consequence: **use the shipped policies as a gate, not as a
displayed probability.** If a downstream consumer needs the number itself to
mean something, refit on data from its own domain — which is cheap, since
`scripts/reliability.py` plus `scripts/fit_policy.py` is the whole pipeline.

Worth noting: EdAcc's own-corpus ECE (0.009–0.021) is *better* than
LibriSpeech's (0.022–0.032). The corpus is not intrinsically harder to
calibrate; it is the transfer that hurts.

---

# Run — 2026-08-07 — babble noise: where the method breaks

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X, `large-v3` int8, beam-5 |
| corpus | LibriSpeech, 73 utterances (identical to the white-noise run) |
| noise | **babble** — six overlapping voices drawn from the corpus itself |
| raw log | [`validation_runs/reliability_babble.txt`](validation_runs/reliability_babble.txt) |

White noise is flat and stationary, the easiest case for both the decoder and
an energy-based SNR estimate. Babble has real speech spectrum and real temporal
modulation, and is the standard realistic hard case.

| SNR dB | WER babble | WER white | `word_prob` AUROC | `seg_conf` AUROC |
|---|---|---|---|---|
| clean | 0.047 | 0.047 | 0.894 | 0.581 |
| 20 | **0.047** | 0.064 | 0.859 | 0.614 |
| 10 | **0.056** | 0.076 | 0.837 | 0.587 |
| 5 | 0.112 | 0.117 | 0.870 | 0.706 |
| 0 | **0.516** | 0.288 | 0.828 | 0.772 |
| −5 | **0.996** | 0.670 | **0.591** | 0.660 |

**Babble is gentler at high SNR and catastrophic at low SNR.** At 20 and 10 dB
it costs *less* WER than white noise (0.047 vs 0.064; 0.056 vs 0.076) — it has
temporal gaps the decoder can hear through, where white noise fills every one.
Below 5 dB that reverses hard: 1.8× the WER at 0 dB, and at −5 dB the
transcript is essentially destroyed (WER 0.996, word accuracy 0.118).

**The confidence signal fails in exactly one condition, and it is this one.**
`word_prob` AUROC holds at 0.83–0.89 down through 0 dB, then collapses to
**0.591** at −5 dB babble — near chance, the only such reading in any run. The
mechanism is legible: with six voices at equal power the model is confidently
transcribing *a* speaker, just not the target one. From the decoder's point of
view nothing went wrong, so its confidence carries no signal about an error
defined against a different talker. This is also the only cell where
`seg_conf` (0.660) beats `word_prob`.

**Boundary condition, stated plainly:** per-word confidence predicts errors
across every condition tested *except* speech-shaped interference severe
enough that the model is transcribing the wrong voice. That is a real limit of
the method, not a tuning problem.

## Babble also defeats automatic policy selection

Tested directly, mean estimate over 5 utterances:

| true SNR | white: estimate → policy | babble: estimate → policy |
|---|---|---|
| 20 | 20.1 → `20` ✓ | 21.4 → `20` ✓ |
| 10 | 10.5 → `10` ✓ | 12.9 → `10` ✓ |
| 5 | 5.3 → `5` ✓ | **no estimate** |
| 0 | 0.4 → `0` ✓ | **no estimate** |
| −5 | −4.5 → `−5` ✓ | **no estimate** |

At and below 5 dB, babble fills the silences, the VAD finds no noise-only
region, and `estimate_snr` returns `None` for every utterance — it declines
rather than inventing a number, which is the designed behaviour. But the
consequence is that `--reliability auto` applies **no policy at all** in
precisely the conditions where a policy matters most. Honest, and a real gap:
an estimator that does not depend on finding silence (spectral-subtraction or
a learned SNR head) is the fix.

**This test also found a latent bug.** Averaging per-utterance estimates when
all of them declined yields `NaN`, and `NaN` compares false against
everything, so the sort in `nearest_condition` kept insertion order and
silently returned a policy. Now guarded and pinned by a test — the noisiest
policy is not a safe default either, since it accepts nothing and so reads as
"this audio is terrible" rather than "the SNR is unknown".

## What still isn't established

- **Deletions remain invisible.** A dropped word leaves no token to score.
  Structural to per-word confidence, and EdAcc makes it more visible because
  Whisper deletes fillers wholesale.
- **SNR estimation without a silence sample** — needed for babble-like
  conditions, per the table above.
- Two corpora, both English; three synthetic noise types, no recorded noise.
  No claim beyond that.

## Config-change A/Bs

`RobustnessConfig` defaults are load-bearing (VAD on, context carry off,
temp ladder 0→1.0, CR gate 2.4, logprob gate −1.0, no-speech 0.6, flag
0.55). Changing one requires an SNR-sweep A/B recorded here.

| date | knob | from → to | effect on WER / flags | verdict |
|---|---|---|---|---|
| 2026-08-02 | `denoise` | off → on | WER +0.02 to +0.15, worst at low SNR; LID p 0.77 → 0.63 at −5 dB | **rejected**, keep off |
| 2026-08-03 | `low_confidence` | 0.55 → **0.85** | 0.55 flagged 0/18 segments across the ladder incl. 33% WER; 0.85 flags 0 while WER ≤ 0.05 and 8/8 once WER ≥ 0.15 | **adopted**, then **superseded** — see below |
| 2026-08-07 | flagging *signal* | `exp(avg_logprob)` per segment → **`min` word probability** | segment signal is a per-window constant, AUROC 0.581 on clean audio vs 0.894 for per-word; min-over-tokens beats mean by 10–25% AUC-NT | **pending Phase 3** |

## Open items

- **Phase 3: ship the calibrated reliability layer** on `min` word
  probability, conditioned on noise level (global calibration fails — NCE
  goes negative at −5 dB while AUROC stays 0.762). This supersedes the 0.85
  segment threshold rather than re-tuning it.
- **Second corpus.** Everything here is read English with additive white
  noise. Both the positive result (per-word works) and the negative one
  (entropy does not help) need a different corpus and real noise before
  either is stated as general.
- **Deletions stay invisible.** A dropped word leaves no token to score;
  structural to per-word confidence, not specific to this signal.
- Broader corpus (LibriSpeech test-clean/test-other, FLEURS) before any of
  these WER figures are described as benchmark results rather than
  characterization of one clip.
- ~~Jetson Orin Nano~~ — **descoped by Shao 2026-08-03.**
