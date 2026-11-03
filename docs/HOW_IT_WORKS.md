# How speech recognition works — and what else a voice can do

Three questions, in order:

1. How does sound become text?
2. Can a voice identify a person?
3. Can my voice be cloned?

Short answers: (1) a stack of signal processing, then a neural network that
"reads" a picture of the sound; (2) yes, with a different kind of model, and
with real limits; (3) yes, from surprisingly little audio — which is exactly
why (2) stops being a safe password.

You can run the first two parts yourself, with NumPy only:

```bash
PYTHONPATH=. python scripts/explain.py stages      # sound -> spectrogram -> mel
PYTHONPATH=. python scripts/explain.py speakers    # toy speaker identification
```

---

## 1. From sound frequencies to text

```
air pressure ──► samples ──► frames ──► spectrum ──► mel + log ──► encoder ──► decoder ──► text
 (physics)       16,000/s    25 ms/10ms    FFT        80 bands     transformer  one token
                                                                                at a time
 └───────────── signal processing (speechlens/audio.py, features.py) ─┘ └── neural network (asr.py) ──┘
```

### 1.1 Sound is a list of numbers

A microphone measures air pressure thousands of times a second. At 16 kHz that
is 16,000 numbers per second; by the Nyquist limit this captures everything up
to 8 kHz, which holds nearly all speech information. Everything downstream of
`audio.py` assumes mono float32 at 16 kHz.

A raw waveform is a poor input for recognition: the same word spoken twice
gives two completely different-looking wiggles. What stays constant is *which
frequencies are present*.

### 1.2 Frequencies over time: the spectrogram

Speech changes every few tens of milliseconds, so we cut the signal into
**25 ms windows, one every 10 ms** (100 frames per second) and run an FFT on
each window. The FFT answers "how much energy is there at each frequency?" for
that instant. Stack the answers side by side and you have a **spectrogram**: an
image with time on x, frequency on y, brightness for energy.
`features.stft_power` does exactly this.

Why 25 ms? Long enough to contain several pitch periods (so frequencies are
resolvable), short enough that the mouth has not moved on to the next sound.

### 1.3 Why the picture contains the words

Voiced speech is a **source** (vocal folds buzzing at a pitch f0, a comb of
harmonics) passed through a **filter** (throat, mouth and nose, whose
resonances — *formants* — boost certain bands). The vowel in "beat" vs "boot"
differs almost entirely in where the formants sit (F2 ≈ 2300 Hz vs ≈ 870 Hz).
Consonants show up as bursts (p, t, k), noise (s, sh), or formant movements.
So a phoneme is a recognisable *shape* in the spectrogram, and pitch — the
source — is mostly irrelevant to *what* was said. (It is very relevant to
*who* said it; see section 2.)

`speechlens/synth.py` builds a toy voice from precisely this source–filter
model, so you can see the formants in `scripts/explain.py stages`.

### 1.4 Squeezing it to what an ear hears: mel + log

Two more tweaks, both in `features.log_mel`:

- **Mel scale.** Human hearing resolves low frequencies finely and high ones
  coarsely (the difference between 200 and 300 Hz is obvious; between 7,200
  and 7,300 Hz it is not). 201 linear FFT bins are pooled into **80 triangular
  bands** spaced accordingly (`large-v3` uses 128).
- **Log.** Perceived loudness is logarithmic, and speech spans many orders of
  magnitude of power. Taking logs (and clamping to an 8-decade range, as
  Whisper does) stops loud vowels from drowning quiet consonants.

The result — an 80 × T "image" — is the **log-mel spectrogram**, and it is the
input to essentially every modern recogniser.

### 1.5 The neural network: image in, tokens out

Everything above is fixed maths with no learning. Here the learning starts.
Whisper (the model faster-whisper runs in `asr.py`) is an
**encoder–decoder transformer**:

1. Audio is padded or cut to a **30-second window**: an 80 × 3000 image.
2. The **encoder** (a small convolutional stem, then transformer layers)
   produces ~1,500 vectors, one per 20 ms, each summarising "what sound is
   happening here, in context".
3. The **decoder** produces text **one token at a time**, like a language
   model. Each step it looks at the encoder vectors (attention) and at the
   tokens it has already written, and outputs a probability over ~50k tokens
   (sub-word pieces). Decoding starts with control tokens —
   `<|startoftranscript|>`, a **language** token (this is where language ID
   comes from, `lid.py`), `<|transcribe|>` — and timestamps are themselves
   tokens at 20 ms resolution.
4. **Beam search** keeps several candidate sentences alive and picks the one
   with the best total probability; the **temperature fallback ladder** in
   `RobustnessConfig` re-decodes hotter when a window looks like garbage.

The decoder is why recognition is not just "match sounds to letters". It is
also a language model: "recognise speech" and "wreck a nice beach" are nearly
the same sound, and the decoder picks the one that is probable *as English*.
That is the model's strength and also its characteristic failure — on silence
or noise it will confidently write fluent text that was never said
(hallucination), which is what the VAD gate in `vad.py` exists to prevent.

The probabilities the decoder emits are also where `words[].prob` and the
per-word reliability layer come from: low probability on a token is the model
saying "I'm not sure".

**Other designs.** Whisper's is called attention-based encoder–decoder.
Older and streaming systems use **CTC** (wav2vec 2.0 and friends: one label
per audio frame, collapse repeats) or **RNN-Transducer** (used for on-device
dictation because it works left-to-right with low latency). Same front-end,
different back-end.

**How it learned.** Whisper was trained on ~680,000 hours of audio paired with
transcripts scraped from the web. Nobody wrote rules for "ph sounds like f";
the network found the mapping because the same shapes kept recurring next to
the same words. This also explains the weak spots: accents and languages
under-represented in that data, and domain vocabulary (hence `initial_prompt`).

---

## 2. Can a voice identify a person?

**Yes.** It is a separate task from transcription, and Whisper does **not** do
it — a transcript does not say who spoke. Three related problems:

| task | question | output |
|---|---|---|
| **verification** | "Is this the person they claim to be?" (1:1) | accept/reject by threshold |
| **identification** | "Which of these N enrolled people is it?" (1:N) | a name, or "none" |
| **diarization** | "Who spoke when?" in one recording, speakers unknown | labelled time segments |

### How it works

Same front-end (frames → spectrum → mel), then a network trained for a
different goal: not "what was said" but **"who said it"**.

1. A neural network maps a recording of any length to a single fixed-size
   vector — a **speaker embedding** (192–512 numbers).
2. It is trained on thousands of speakers so that utterances by the same person
   land near each other and different people land far apart, regardless of the
   words. Well-known designs: **x-vectors**, **ECAPA-TDNN**.
3. To enrol someone you embed a few seconds of their speech and store the
   vector. To verify, embed the new audio and compare by cosine similarity
   against a threshold. Identification is "nearest enrolled vector".

What the embedding captures is mostly the *physical* voice — vocal-tract
length and shape (formant positions), and to a lesser degree pitch range and
speaking habits — not the content. That is the same source–filter split from
1.3, used in the opposite direction: recognition tries to ignore who is
speaking, speaker ID tries to ignore what is said.

### Try the idea

`speechlens/voiceprint.py` does a hand-built version: average the MFCCs
(a compact description of the spectral envelope, i.e. the vocal tract) over
the utterance, and match by nearest neighbour. On the toy voices in
`speechlens/synth.py`:

```
low_male   → low_male   (dist 1.04; next closest 2.79)
female     → female     (dist 1.17; next closest 3.83)
```

On 6 toy speakers, accuracy rose with utterance length — 77% at 8 vowels, 85%
at 16, 95% at 30 — and the errors were between speakers with similar pitch
and vocal-tract length. Both trends are real: **more audio is better**, and
**similar voices are the hard case**. The toy has none of the robustness of a
trained embedding and only ever heard synthetic vowels, so none of these
numbers say anything about real people.

### Real-world performance and limits

- Published ECAPA-TDNN results on VoxCeleb1 report an equal error rate of
  roughly 1% on clean, long-ish clips. Short clips (< 3 s), phone-quality
  audio, background noise, a cold, shouting, or a different microphone than
  at enrolment all degrade it, sometimes a lot.
- It is **probabilistic**, not proof. A threshold trades false accepts for
  false rejects; there is no setting that has neither.
- Twins, siblings and skilled impersonators raise the false-accept rate.
- It can be **spoofed** — by recordings, and increasingly by cloned voices
  (section 3). Systems that matter pair it with liveness/anti-spoofing
  checks; a voice alone should not be a password.
- **Legal and ethical:** a voiceprint is biometric data. Under GDPR it is a
  special category (Art. 9) needing a lawful basis such as explicit consent;
  several US states (e.g. Illinois' BIPA) require written consent and
  carry statutory damages. Identifying people from recordings of them
  without their knowledge is the use to avoid. Enrolling yourself, or
  people who have agreed, is fine.

### Doing it for real: `speechlens.speaker`

`speechlens/speaker.py` is the enrol/identify/verify logic around a pluggable
encoder (any callable `(audio, sr) -> vector`, the same interface SpeechForge's
`SpeakerEncoder` uses). It has no neural network in it, so it imports without
torch. For a real encoder, install the opt-in extra, which brings SpeechBrain's
pretrained ECAPA-TDNN:

```bash
pip install -e ".[speaker]"          # speechbrain + torch; weights download from Hugging Face
speechlens speaker enroll   --gallery people.npz --name me  me1.wav me2.wav me3.wav
speechlens speaker identify --gallery people.npz unknown.wav
speechlens speaker identify --gallery people.npz unknown.wav --threshold 0.55   # open-set: allow "none"
speechlens speaker verify   --gallery people.npz --name me --threshold 0.55 claim.wav
```

```python
from speechlens.speaker import Gallery, speechbrain_encoder, threshold_for_far, cosine

enc = speechbrain_encoder()
g = Gallery(enc, encoder_id="speechbrain:ecapa-voxceleb")
for clip, sr in my_clips:                 # 3-5 clips, varied conditions
    g.enroll("me", clip, sr)
name, score, ranking = g.identify(new_clip, sr)
```

Design decisions that are easy to get wrong, enforced in code:

- **No default threshold.** Cosine scores belong to an encoder and to your
  audio. SpeechForge measured same-speaker pairs at 0.89–0.98 on its encoder,
  against 0.69–0.76 reported in papers — a threshold copied from one would
  have fired on every valid pair. `verify` requires a threshold, and
  `threshold_for_far(impostor_scores, far)` derives one from impostor scores
  you collect: score recordings of *different* people against your gallery, then
  pick the false-accept rate you can live with. It refuses a target the sample
  is too small to resolve (FAR 0.1% needs ≥ 1,000 impostor trials).
  `equal_error_rate` summarises how separable your scores are.
- **`identify` is closed-set by default**: it names the nearest enrolled
  speaker even for a total stranger. Pass `--threshold` for "none of them".
- **A gallery is tied to its encoder** and `Gallery.load` refuses a mismatch.
- **Enrolling several clips** and comparing to their centroid beats one clip.

**What has and has not been tested.** The gallery, scoring, calibration and
persistence logic is covered by offline tests (`tests/test_speaker.py`) using a
toy encoder on synthetic voices; a mutation of the FAR threshold's off-by-one
makes them fail. The ECAPA adapter and the CLI paths that load it were **not**
run against real weights here — the sandbox this was written in cannot reach
Hugging Face. Treat the first real run as the validation, and measure your own
EER before trusting a threshold.

**Diarization** ("who spoke when") is a different tool:
`pyannote.audio`. Its speaker turns compose with the per-segment transcript
schema; that is not wired in.

## 3. Can my voice be cloned?

**Yes — and for your own voice, with your consent, it is a legitimate and
practical thing to do.** The difficulty has collapsed: a few years ago it took
hours of studio recording per voice; current zero-shot models need seconds.

### How it works

Text-to-speech runs the recognition pipeline in reverse:

```
text ──► acoustic model ──► mel spectrogram ──► vocoder ──► waveform
         (what to say,       (the same 80-band   (neural net that
          how, in whose       picture as in      fills in the phase
          voice)              section 1.4)       and fine detail)
```

Cloning adds one input: a description of **whose voice**. Two approaches:

- **Zero-shot / few-shot.** A speaker encoder (section 2's embedding, or the
  reference audio itself fed in as a prompt) conditions the acoustic model.
  No training on your voice; you hand it a few seconds to a few tens of
  seconds and it imitates timbre. It captures timbre better than speaking
  style, and results vary a lot by model and by reference.
- **Fine-tuning.** Train or adapt a model on 10 minutes to a few hours of your
  speech (VITS-family models, RVC for voice conversion). More work, more
  faithful, and much better at your rhythm and expressiveness.

**Licences live in the weights, not the code.** This matters for what you may
do with the output. From SpeechForge's audit (`docs/LICENSES.md` there, dated
2026-08-07 — re-check before relying on it): Chatterbox is MIT for code *and*
weights; CosyVoice 2 is Apache-2.0 for both; F5-TTS has MIT code but
`cc-by-nc-4.0` weights; XTTS-v2 is non-commercial for the model *and its
outputs*. Chatterbox also watermarks every output, unconditionally. For a
personal clone of your own voice almost any of these is fine; for anything you
sell or publish, read the weights licence.

### Doing it well (your voice, your consent)

All of this is measured or enforced in SpeechForge's `reference.py`, not just
folklore:

- **The reference clip decides quality, more than the model.** Noise, room
  tone and microphone are cloned along with your voice, because they land in
  the speaker embedding. A clean 10 s clip beats two noisy minutes.
- **Trim trailing silence.** Continuation-style models treat the reference as
  a literal prefix, so trailing room tone becomes the start of what they say.
- **Reference length is model-specific.** "About 10 s" is CosyVoice's advice;
  applied to Chatterbox it cost 24% relative WER on the clips it truncated
  (a 20 s cap was the measured optimum there). Do not assume.
- **Give the model ordinary prose.** Passing corpus-style `ALL CAPS` text to
  Chatterbox gave WER 0.112 vs 0.022 in sentence case on the same
  LibriSpeech utterances (n=60, SpeechForge's measurement). Punctuate your
  input.
- Record varied sentences and emotions, not a monotone read.

### Measuring a clone

The earlier sections are the evaluation harness, and SpeechForge already
implements it:

- **Intelligibility:** transcribe the synthesised audio with this package and
  compute WER against the input text (`speechlens.metrics`).
- **Similarity:** embed a *held-out real recording* of you and the clone
  with a speaker encoder and take the cosine. Use a recording the clone was
  **not** conditioned on — scoring against the reference clip itself (SIM-r)
  inflates the number, and SpeechForge's `sim_o` raises if you do it by mistake.
- **Always score the real recordings the same way** and read the clone against
  that ground-truth row. Similarity has no universal ceiling: it depends on the
  encoder and the speaker, and a WER *below* the real-speech row means the
  comparison has stopped measuring intelligibility. SpeechForge's reported
  LibriSpeech numbers, for orientation (their README, not re-run here):

  | | WER (Whisper) | speaker similarity |
  |---|---|---|
  | real speech (ground truth) | 0.021 | 0.964 |
  | Chatterbox, sentence case | 0.022 | 0.949 |
  | CosyVoice 2, no truncation | 0.036 | 0.958 |

The consequence cuts both ways: a good enough clone scores close to the real
you, which is exactly why it can also pass a speaker-verification check.

### A concrete workflow with SpeechForge

SpeechForge is a separate repository
([shaostassen/speechforge](https://github.com/shaostassen/speechforge)) so this
package keeps its no-torch ceiling. Its core API, as of its current `main`:

```python
import soundfile as sf
from speechforge.synth import load_backend, clone
from speechforge.text import sentence_case

ref, sr = sf.read("my_voice.wav")                 # your own clean recording, MONO
                                                   # (reference prep flattens channels)
backend = load_backend("chatterbox", device="cpu")  # pip install -e ".[chatterbox]"
audio, out_sr, report = clone(
    backend, sentence_case("Hello, this is a synthetic copy of my voice."),
    ref, sr, out_path="clone.wav", seed=0)
print(report)                                      # what was trimmed/capped, and warnings
```

then score it with `speechforge.evaluate.sim_o` (similarity against a held-out
recording of you) and this package's `analyze`/`metrics` (WER). The
end-to-end protocol, including the ground-truth row, is
`scripts/eval_zeroshot.py` there. I read these signatures from its source; I
did not run synthesis here — no GPU, no model weights, and Hugging Face is not
reachable from this sandbox. Chatterbox on CPU measured about 2× *slower* than
realtime in SpeechForge's own bench, so budget for that.

### One script: record, clone, score

`scripts/clone_my_voice.py` runs the whole protocol above on your own voice:

```bash
pip install -e .                                        # this repo
git clone https://github.com/shaostassen/speechforge
pip install -e "./speechforge[chatterbox,similarity]"   # torch; first run downloads weights

pip install sounddevice                                 # only for recording
python scripts/clone_my_voice.py record --dir my_voice  # 7 prompts; warns on clipping/noise/level
python scripts/clone_my_voice.py check  --dir my_voice
python scripts/clone_my_voice.py run    --dir my_voice --backend chatterbox
```

Already have recordings? `import --own-voice --take a.wav "exactly what is said" --take b.wav "..."`
(first take = the reference). `run` clones each held-out sentence from the
reference, writes `clone_NN.wav` next to your `take_NN.wav` so you can listen to
both, prints per-utterance and aggregate scores with your real voice as the
ground-truth row, and saves `results.json` with the provenance (backend,
encoder, ASR model, seed, hardware, command).

Things it does on purpose: take 0 is a longer passage used as the reference and
everything else is held out; clones are scored against held-out recordings,
never the reference; sentences are sentence-cased before the model sees them;
it refuses to run without confirmation that the voice is yours (or its owner
agreed); it stops with fewer than two held-out takes. The default ASR is
`small` so it is practical on a CPU, which means its WER is **not** comparable
to SpeechForge's published `large-v3` numbers — use `--asr-model large-v3` for
that. With about 6 sentences these results describe your voice and setup; they
are not a benchmark, and listening to the clones is the real test.

The pipeline logic is tested offline with fake models
(`tests/test_clone_my_voice.py`; set `SPEECHFORGE_PATH` to a checkout or the
pipeline tests skip). The recording step, and `run` with real Chatterbox,
Whisper and the speaker encoder, have **not** been run — the dev sandbox has no
microphone, no GPU and no route to Hugging Face. Expect to fix a rough edge or
two on your first real run, and tell me what breaks.

### Risks, and what to do about them

- **Never clone someone else's voice without their permission.** Impersonation
  for fraud ("it's me, I need you to transfer money") is the dominant misuse;
  use is regulated and being tightened in many jurisdictions.
- **Don't use your voice as a password.** Once a clone of you exists, any
  voice-based authentication of you is weaker. Prefer a second factor.
- **Synthetic speech is hard to detect from transcription.** SpeechForge
  measured that per-word confidence does not separate synthetic from real speech
  when the words are correct — the reliability layer here measures
  acoustic–lexical fit, not provenance, and is not a deepfake detector.
  Detection needs dedicated anti-spoofing models or provenance markers
  (watermarks, signed audio).
- If you publish cloned audio, say that it is synthetic.

---

## Where to go next

| want to… | do this |
|---|---|
| see sound → mel yourself | `python scripts/explain.py stages --audio your.wav` |
| transcribe | `speechlens analyze clip.wav --model small` |
| know which words to trust | `speechlens analyze clip.wav --reliability auto` |
| identify speakers properly | `pip install -e ".[speaker]"`, then `speechlens speaker enroll/identify/verify`; calibrate your own threshold |
| clone your voice | SpeechForge (section 3 workflow), then score WER + held-out similarity |

Further reading: Radford et al. 2022, *Robust Speech Recognition via
Large-Scale Weak Supervision* (Whisper); Desplanques et al. 2020, *ECAPA-TDNN*;
Snyder et al. 2018, *X-vectors*; Baevski et al. 2020, *wav2vec 2.0*.
