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
per-segment confidence flagging so low-quality output is marked instead of
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
`exp(avg_logprob)` falls below `low_confidence=0.55` are flagged, not hidden.
`initial_prompt` biases decoding toward domain vocabulary (product names,
jargon). The optional `--denoise` stage is deliberately off by default —
Whisper is trained on noisy audio and spectral gating can smear formants;
A/B it with the SNR study before trusting it.

## Results

Tesla T4 (Colab), `large-v3`, `float16`, 2026-08-02. One 23.2 s clip of
concatenated LibriSpeech utterances with a ground-truth transcript. Raw logs
and the full analysis are in [`docs/VALIDATION.md`](docs/VALIDATION.md);
reproduce with [`notebooks/validate_t4.ipynb`](notebooks/validate_t4.ipynb).

**Throughput** — decode only; model load is excluded.

| model | RTF | × realtime |
|---|---|---|
| base | 0.038 | 26.3 |
| small | 0.036 | 27.8 |
| distil-large-v3 | 0.046 | 21.7 |
| large-v3 | 0.082 | 12.2 |

**Noise robustness** — white noise at a seeded SNR ladder, denoise off.

| SNR dB | lang | p | WER | flagged |
|---|---|---|---|---|
| clean | en | 1.00 | 0.03 | 0 |
| 20 | en | 1.00 | 0.05 | 0 |
| 10 | en | 1.00 | 0.03 | 0 |
| 5 | en | 1.00 | 0.05 | 0 |
| 0 | en | 0.97 | 0.15 | 0 |
| −5 | en | 0.77 | 0.33 | 0 |

Language ID held `en` across the entire ladder (chunk-voted LID, 5/5 on the
multilingual smoke study including Mandarin), and WER degraded smoothly
rather than collapsing into hallucinated text.

Two results went against expectations, and both are load-bearing:

- **The confidence gate never fired.** Zero flagged segments at every level,
  including −5 dB where a third of the words are wrong — `large-v3` remains
  confident while incorrect, so `exp(avg_logprob)` never crosses 0.55. The
  gate is unexercised, not validated. Threshold sweep is an open item.
- **`--denoise` lost at every SNR**, by +0.02 to +0.15 WER, worst where it
  was supposed to help most. It stays off by default.

Single clip, single speaker: WER differences below ~0.05 are not resolvable
here. These characterize behavior and cost; they are not corpus benchmarks.
CPU baseline on the 9950X and the Jetson Orin Nano numbers are still pending.

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
python scripts/validate.py bench clip.wav --models small,distil-large-v3,large-v3
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
