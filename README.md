# SpeechLens

[![CI](https://github.com/shaostassen/voice_detection_using_machine_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/shaostassen/voice_detection_using_machine_learning/actions/workflows/ci.yml)

Local language identification + robust transcription. Point it at an audio
file (or a microphone), get back the language with a calibrated-ish
probability, a transcript with per-segment confidence, and an honest set of
warnings — all computed on your own hardware, no cloud calls.

The interesting engineering is the robustness harness around Whisper, not the
model itself: Silero VAD gating (the single biggest anti-hallucination
measure), chunk-voting language ID fused in log space, a beam-search decode
with a temperature fallback ladder, repetition and no-speech gates, and
confidence flagging so low-quality output is marked instead of
silently trusted.

## Install

```bash
cd speechlens
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # dev extra = pytest
# optional spectral-gating denoise stage:
pip install -e ".[denoise]"
```

GPU inference goes through CTranslate2, which needs cuBLAS and cuDNN 9 for
CUDA 12. If you hit `libcudnn` errors on first GPU run:

```bash
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9"
export LD_LIBRARY_PATH=$(python3 -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; \
  print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

First run of a given model size downloads weights from Hugging Face
(large-v3 is ~3 GB); after that everything is offline.

`espeak-ng` is only needed for the TTS-based tests and the `smoke`
validation study: `sudo apt install espeak-ng`.

## Quickstart

CLI:

```bash
speechlens analyze clip.mp3                          # full auto, large-v3
speechlens analyze clip.mp3 --model small --device cpu
speechlens analyze interview.m4a --language zh --word-timestamps --json out.json
```

Python:

```python
from speechlens import SpeechLens, RobustnessConfig

lens = SpeechLens(model_size="large-v3")     # load once, reuse
result = lens.analyze("clip.wav")
print(result.language["code"], result.language["probability"])
print(result.text)
for seg in result.transcript["segments"]:
    if seg["flagged"]:
        print("low confidence:", seg["start"], seg["text"])
```

Web UI / HTTP API:

```bash
speechlens serve --port 7860        # then open http://127.0.0.1:7860
curl -F "file=@clip.wav" http://127.0.0.1:7860/api/analyze | jq .language
```

The UI supports drag-drop, file pick, and in-browser microphone recording
(MediaRecorder webm — decoded server-side by the bundled FFmpeg).

## Robustness knobs

Everything lives in `RobustnessConfig` (speechlens/asr.py) with hardened
defaults. The ones that matter in practice: `vad_filter=True` keeps silence
and music away from the decoder, which is where most hallucinations come
from. `temperature=(0.0 ... 1.0)` is the fallback ladder — a segment is
re-decoded hotter whenever `compression_ratio > 2.4` (repetition loop) or
`avg_logprob < -1.0` (garbage). `condition_on_previous_text=False` trades a
little cross-segment consistency for zero error propagation on hard audio.
`no_speech_threshold=0.6` suppresses phantom segments. Segments whose
`exp(avg_logprob)` falls below `low_confidence=0.85` are flagged, not hidden.

**Scope caveat on that flag.** faster-whisper computes `avg_logprob`,
`no_speech_prob` and `compression_ratio` once per **30-second decode window**
and copies them onto every segment from that window — 22 segments over 5
windows yield exactly 5 distinct values
([probe](docs/validation_runs/window_scope_probe.txt)). So the flag is a
per-window judgement wearing a per-segment label, and it cannot tell a good
segment from a bad one inside the same window. `words[].prob` is the real
per-unit signal and discriminates far better; see
[`docs/VALIDATION.md`](docs/VALIDATION.md). Treat `0.85` as a working default,
not a validated one.
`initial_prompt` biases decoding toward domain vocabulary (product names,
jargon). The optional `--denoise` stage is deliberately off by default —
Whisper is trained on noisy audio and spectral gating can smear formants;
A/B it with the SNR study before trusting it.

## Results

Tesla T4 (Colab), `large-v3`, `float16`, 2026-08-02. One 23.2 s clip of
concatenated LibriSpeech utterances with a ground-truth transcript. Raw logs
and the full analysis are in [`docs/VALIDATION.md`](docs/VALIDATION.md);
reproduce with [`notebooks/validate_t4.ipynb`](notebooks/validate_t4.ipynb).

**Throughput** — RTF, decode only; model load is excluded. All three
columns ran on the byte-identical clip.

| model | T4 `float16` | Ryzen 9 9950X `int8` | Apple M2 `int8` |
|---|---|---|---|
| base | 0.038 (26.3×) | **0.028 (35.7×)** | 0.083 (12.0×) |
| small | 0.036 (27.8×) | 0.063 (15.9×) | 0.205 (4.9×) |
| distil-large-v3 | 0.046 (21.7×) | 0.160 (6.2×) | 0.605 (1.7×) |
| large-v3 | 0.082 (12.2×) | 0.259 (3.9×) | not benchmarked |

**No GPU is required.** `large-v3` runs at 3.9× realtime on a desktop CPU
and `small` at 15.9×; even a laptop clears realtime on everything through
`distil-large-v3`. For a tool whose premise is local execution, that is the
load-bearing result.

Two things worth reading off this table. The 9950X **beats the T4 on
`base`** — not because the CPU is faster, but because a 23 s clip never
saturates a T4, so every model there costs 0.036–0.082 regardless of size
and fixed overhead dominates. The GPU's advantage only appears as the model
grows: 1.8× at `small`, 3.2× at `large-v3`. Consequently the GPU bench
*understates* the cost difference between model sizes — reason about model
selection from the CPU columns, where the spread is 9× rather than 2×.

**Noise robustness** — white noise at a seeded SNR ladder, denoise off,
`large-v3` int8 on the 9950X at the current flag threshold.

| SNR dB | lang | p | WER | flagged |
|---|---|---|---|---|
| clean | en | 1.00 | 0.05 | 0 |
| 20 | en | 1.00 | 0.05 | 0 |
| 10 | en | 1.00 | 0.03 | 0 |
| 5 | en | 0.99 | 0.05 | 0 |
| 0 | en | 0.96 | 0.15 | 4 |
| −5 | en | 0.82 | 0.33 | 4 |

Language ID held `en` across the entire ladder (chunk-voted LID, 5/5 on the
multilingual smoke study including Mandarin), WER degraded smoothly rather
than collapsing into hallucinated text, and the confidence gate stays silent
while WER ≤ 0.05 then flags every segment once it steps to 0.15.

**Word-level reliability** — 73 utterances, ~1,160 words per condition. Can a
confidence score rank *which* word is wrong, within a single noise condition?
AUROC, 0.5 = chance:

| SNR dB | WER | `words[].prob` | segment `confidence` |
|---|---|---|---|
| clean | 0.047 | **0.894** | 0.581 |
| 10 | 0.076 | **0.856** | 0.693 |
| 0 | 0.288 | **0.826** | 0.701 |
| −5 | 0.670 | **0.762** | 0.681 |

Per-word probability predicts errors well everywhere. The segment-level
signal the flag currently uses does not — on clean audio it is near chance,
because it is a per-window constant and cannot rank anything inside a window.
Calibration of the per-word signal is good down to 0 dB (ECE 0.018–0.085) and
fails below that (NCE −0.130 at −5 dB, while AUROC stays 0.762 — the ranking
survives, the numbers stop meaning anything).

Three results went against expectations, and all are load-bearing:

- **The confidence gate was inert, and the obvious diagnosis was wrong.**
  The first sweep flagged nothing at any level, including −5 dB. That reads
  like `avg_logprob` being a useless signal; measuring the distribution
  showed the opposite — confidence falls 0.957 → 0.669, tracking WER
  closely. The threshold, 0.55, was simply *below the worst value the model
  ever produces*, so it could never fire. Raised to 0.85 with the A/B in
  [`docs/VALIDATION.md`](docs/VALIDATION.md); the table above is the
  re-run. An inert gate is worse than none — it implies a safety net that
  isn't there.
- **`--denoise` lost at every SNR**, by +0.02 to +0.15 WER, worst where it
  was supposed to help most. It stays off by default.
- **Entropy did not beat probability.** The confidence literature reports
  entropy-based estimators detecting errors 1.5–4× better for CTC and
  transducer models; on Whisper all five estimators tested landed within
  0.008 AUROC of each other. What helped instead was aggregation — taking the
  **min** over a word's tokens rather than the mean, worth 10–25% in
  error-detection precision, because a word is wrong if any one of its tokens
  is and averaging dilutes that. Entropy also requires greedy decoding
  (CTranslate2 returns no logits under beam search), so it costs beam-5 and
  buys nothing.

Single clip, single speaker: WER differences below ~0.05 are not resolvable
here (RTF is steadier, ~2% run to run). These characterize behavior and
cost; they are not corpus benchmarks. Jetson Orin Nano numbers are still
pending.

## Per-word reliability

The segment flag answers "is this chunk of audio bad?". The reliability layer
answers the question you actually have — "which words should I not trust?" —
by calibrating the per-word probability against measured correctness and
picking a threshold from a tolerated error rate.

```bash
speechlens analyze clip.wav --reliability clean
```

```
[00:16.82 -> 00:22.54] (0.83)  [!] before us, similarly he's drawn from eating and its results...

Reliability (clean policy, 2% target error): 52/60 words auto-accepted (87%), 8 need review
  review: Mr.(0.74) middle(0.75) is(0.69) matter.(0.74) roast(0.71) similarly(0.57) he's(0.74) and(0.75)
```

The reference reads *"similes drawn from eating"*; `base` heard *"similarly
he's drawn"*. Both wrong words are in the review list and `similarly` scores
lowest of all 60. Meanwhile the segment gate marked **all four segments**
`[!]` at an identical `0.83` — that repeated value is the per-window constant,
saying "everything is suspect" about a transcript that is 95% correct.

Pick the policy that matches your audio (`clean, 20, 10, 5, 0, -5` dB SNR).
There is deliberately no default: at 2% tolerated error, coverage runs from
90% on clean speech to **zero** at 0 dB, so defaulting to `clean` would
auto-accept most of a transcript that is 29% wrong. Full operating points and
the calibration method are in [`docs/VALIDATION.md`](docs/VALIDATION.md).

In Python, and over HTTP via a `reliability` form field:

```python
from speechlens.calibration import load_bundled_policy
lens = SpeechLens(model_size="small", policy=load_bundled_policy("clean"))
result = lens.analyze("clip.wav", cfg=RobustnessConfig(word_timestamps=True))
result.reliability          # coverage, threshold, words needing review
```

## Validation runbook

Unit and component tests (no weights, no GPU, run anywhere):

```bash
pytest -q
```

Model-level validation on a GPU box:

```bash
python scripts/validate.py smoke                       # multilingual TTS round-trip: LID + plumbing
python scripts/validate.py snr clip.wav --ref "known transcript"   # WER/CER vs SNR ladder
python scripts/validate.py snr clip.wav --ref "..." --denoise      # A/B the denoise stage
python scripts/validate.py gate clip.wav --ref "..."               # confidence distribution vs flag thresholds
python scripts/validate.py bench clip.wav --models small,distil-large-v3,large-v3

python scripts/make_clip.py clip.wav    # rebuild the canonical labeled clip first
```

Every run prints a provenance banner — hardware, model, device,
`compute_type`, date, and the raw command — because a number is only
recordable alongside the run that produced it. Placement is explicit via
`--device {auto,cuda,cpu}` and `--compute-type {float16,int8_float16,int8,…}`;
`auto` picks cuda+float16 where a GPU exists and cpu+int8 otherwise.

No local NVIDIA GPU is used for this project, so the authoritative numbers
come from a free cloud T4 running `large-v3` at `float16`:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaostassen/voice_detection_using_machine_learning/blob/main/notebooks/validate_t4.ipynb)

`notebooks/validate_t4.ipynb` clones this repo, installs the stack
(including the cuDNN 9 that CTranslate2's CUDA path needs), re-checks the
offline test invariant, runs all three studies against labeled LibriSpeech
audio, and emits a block ready to paste into
[`docs/VALIDATION.md`](docs/VALIDATION.md) — where every result lives.

For a real accuracy number, run WER on a labeled set you care about
(LibriSpeech test-clean/test-other, FLEURS for multilingual, or your own
recordings) using `speechlens.metrics.error_rate` — it picks WER for spaced
scripts and CER for zh/ja/ko/etc.

## Jetson / edge notes

The same code runs on a Jetson Orin Nano: use `--model small
--compute-type int8` (or `int8_float16`). CTranslate2 has aarch64 support;
Silero VAD runs on CPU via onnxruntime either way. Expect roughly real-time
with `small` on the Orin Nano, far faster on desktop GPUs.

## Extension seams

- **ECAPA-TDNN LID fusion**: `lid.fuse_distributions` already fuses arbitrary
  probability dicts — add a VoxLingua107 ECAPA head (speechbrain) as a second
  distribution source for extra robustness on very short clips.
- **distil-large-v3** for ~5x throughput at near-large quality on
  English-heavy workloads.
- **Streaming**: chunk the input at VAD boundaries and feed segments
  incrementally; the pipeline stages are already segment-oriented.
- **Diarization**: pyannote speaker turns compose cleanly with the
  per-segment schema.

## Layout

```
speechlens/
├── speechlens/
│   ├── audio.py       decode any container → 16 kHz mono float32
│   ├── features.py    NumPy log-mel frontend (analysis + tests)
│   ├── vad.py         Silero (bundled ONNX) + energy fallback
│   ├── lid.py         chunk-voting LID, log-space fusion
│   ├── asr.py         faster-whisper wrapper + RobustnessConfig
│   ├── pipeline.py    orchestrator → AnalysisResult (JSON-ready)
│   ├── metrics.py     WER / CER
│   ├── cli.py         speechlens analyze | serve
│   ├── server.py      FastAPI: /api/analyze, /api/health, web UI
│   └── static/index.html
├── tests/             offline unit + component tests
└── scripts/validate.py  GPU validation harness (smoke / snr / bench)
```
