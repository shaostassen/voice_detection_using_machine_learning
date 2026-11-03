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

### Tools if you want the real thing

- **SpeechBrain** — pretrained ECAPA-TDNN speaker verification
  (`speechbrain/spkrec-ecapa-voxceleb`).
- **pyannote.audio** — speaker diarization and embeddings.
- **Resemblyzer** — a lightweight d-vector embedder.

All need PyTorch, so none are added here: this package keeps a deliberate
no-torch ceiling (see the SpeechForge note in the README). The clean seam is
the one the README names — diarization turns compose with the per-segment
schema, so "who said each segment" can be attached to `AnalysisResult`
without touching the ASR path.

---

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
  No training on your voice; you hand it 3–30 seconds and it imitates timbre.
  Examples: XTTS-v2, F5-TTS, OpenVoice. Quick, but it captures timbre better
  than speaking style, and results vary by model.
- **Fine-tuning.** Train or adapt a model on 10 minutes to a few hours of your
  speech (VITS-family models, RVC for voice conversion). More work, more
  faithful, and much better at your rhythm and expressiveness.

Check each model's licence — several popular ones restrict commercial use.

### Doing it well (your voice, your consent)

- Record in a quiet room, a consistent mic position, no music/reverb. Clean
  input is the biggest quality lever, ahead of model choice.
- Include varied sentences and emotions, not just a monotone read.
- Give the model *text with punctuation*. The SpeechForge project found
  unpunctuated `ALL CAPS` transcripts cost an autoregressive TTS up to 5× WER.

### Measuring a clone

The two earlier sections are the evaluation harness:

- **Intelligibility:** transcribe the synthesised audio with this package and
  compute WER against the input text (`speechlens.metrics`). This is the
  field-standard metric and is exactly what the companion project
  [SpeechForge](https://github.com/shaostassen/speechforge) does.
- **Similarity:** embed the real recording and the clone with a speaker
  encoder (section 2) and compare — the same cosine score that decides
  "is this the same person?" decides "is this a good clone?". Which is also
  why a good enough clone can pass speaker verification.

SpeechForge is the right place to build the cloning side: it already consumes
this package. This repo stays torch-free on purpose.

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
| identify speakers properly | add SpeechBrain ECAPA or pyannote (torch) beside, not inside, this package |
| clone your voice | SpeechForge, then score it with section 3's two metrics |

Further reading: Radford et al. 2022, *Robust Speech Recognition via
Large-Scale Weak Supervision* (Whisper); Desplanques et al. 2020, *ECAPA-TDNN*;
Snyder et al. 2018, *X-vectors*; Baevski et al. 2020, *wav2vec 2.0*.
