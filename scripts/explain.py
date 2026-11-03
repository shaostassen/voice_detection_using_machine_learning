"""Walk through what a speech recogniser does with sound, stage by stage.

NumPy-only, no weights, no network:

    python scripts/explain.py stages                 # toy vowel -> spectrogram -> mel
    python scripts/explain.py stages --audio my.wav  # same, on your own recording
    python scripts/explain.py speakers               # toy speaker identification

`stages` stops where the neural network begins and says so; the rest of the
path (encoder, token decoder) is what faster-whisper runs — see
docs/HOW_IT_WORKS.md.
"""
from __future__ import annotations

import argparse

import numpy as np

from speechlens.features import filterbank_centers, log_mel, stft_power
from speechlens.synth import synth_utterance
from speechlens.voiceprint import distance, identify, voiceprint

SR = 16000
SHADES = " .:-=+*#%@"


def ascii_heatmap(M: np.ndarray, rows: int = 16, cols: int = 72) -> str:
    """Downsample a (freq, time) matrix to a terminal picture, low freq at bottom."""
    fr = np.array_split(np.arange(M.shape[0]), rows)
    tc = np.array_split(np.arange(M.shape[1]), cols)
    small = np.array([[M[np.ix_(f, t)].mean() for t in tc] for f in fr])
    lo, hi = small.min(), small.max()
    idx = ((small - lo) / max(hi - lo, 1e-9) * (len(SHADES) - 1)).astype(int)
    return "\n".join("".join(SHADES[i] for i in row) for row in idx[::-1])


def cmd_stages(args: argparse.Namespace) -> None:
    if args.audio:
        from speechlens.audio import load_audio
        y = load_audio(args.audio)
        if isinstance(y, tuple):
            y = y[0]
        src = args.audio
    else:
        y = synth_utterance(120.0, 1.0, n_vowels=10, seed=4)
        src = "toy synthetic voice (f0=120 Hz, 10 random vowels)"
    y = np.asarray(y, dtype=np.float32)

    print(f"[1] sound -> numbers\n    {src}\n"
          f"    {len(y)} samples @ {SR} Hz = {len(y)/SR:.2f} s; "
          f"each sample is one air-pressure reading, range "
          f"[{y.min():+.2f}, {y.max():+.2f}]\n")

    S = stft_power(y)
    print(f"[2] slice into 25 ms windows every 10 ms, FFT each one\n"
          f"    power spectrogram {S.shape[0]} freq bins x {S.shape[1]} frames "
          f"(= {S.shape[1] / (len(y)/SR):.0f} frames/s)")
    mid = S[:, S.shape[1] // 2]
    freqs = np.fft.rfftfreq(400, 1 / SR)
    mid = np.where(freqs > 50, mid, 0.0)  # ignore the DC bin
    top = np.argsort(mid)[::-1][:3]
    print("    loudest bins in the middle frame: "
          + ", ".join(f"{freqs[i]:.0f} Hz" for i in sorted(top)) + "\n")

    L = log_mel(y)
    print(f"[3] warp to the mel scale + take logs\n"
          f"    {L.shape[0]} bands, spaced like the ear's resolution "
          f"({filterbank_centers()[0]:.0f} Hz ... {filterbank_centers()[-1]:.0f} Hz); "
          f"log compresses 8 decades of loudness into one range\n")
    print("    log-mel spectrogram (time ->, low freq at the bottom):\n")
    print("\n".join("    " + r for r in ascii_heatmap(L).splitlines()))
    print(f"\n[4] ---- the neural network starts here ----\n"
          f"    Whisper pads/cuts to 30 s, so its encoder sees an {L.shape[0]} x 3000 image\n    (80 mel bands; large-v3 uses 128).\n"
          f"    The encoder turns it into ~1500 vectors (one per 20 ms); the decoder then\n"
          f"    emits text *tokens* one at a time, each conditioned on the audio vectors\n"
          f"    and on the tokens so far. See docs/HOW_IT_WORKS.md, section 1.")


def cmd_speakers(_: argparse.Namespace) -> None:
    speakers = {
        "low_male": (105.0, 0.95), "mid_male": (135.0, 1.05),
        "female": (210.0, 1.18), "child": (290.0, 1.40),
    }
    n = 20
    print(f"Enrol: 2 toy sentences per speaker ({n} vowels each) -> one voiceprint.")
    gallery = {k: np.mean([voiceprint(synth_utterance(f0, sc, n_vowels=n, seed=s))
                           for s in (0, 1)], axis=0)
               for k, (f0, sc) in speakers.items()}
    print("Identify: a *new* sentence from each speaker (nearest voiceprint wins).\n")
    print(f"{'truth':10s} {'guess':10s} {'dist':>6s}   distances to all enrolled")
    right = 0
    for truth, (f0, sc) in speakers.items():
        probe = voiceprint(synth_utterance(f0, sc, n_vowels=n, seed=100))
        guess, d = identify(probe, gallery)
        right += guess == truth
        row = "  ".join(f"{k}={distance(probe, v):5.2f}" for k, v in gallery.items())
        print(f"{truth:10s} {guess:10s} {d:6.2f}   {row}")
    print(f"\n{right}/{len(speakers)} correct. This is a toy: real systems use neural "
          "embeddings (ECAPA-TDNN) — see docs/HOW_IT_WORKS.md, section 2.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stages", help="sound -> spectrogram -> mel, with numbers")
    s.add_argument("--audio", help="your own file (any format ffmpeg/soundfile reads)")
    s.set_defaults(fn=cmd_stages)
    sub.add_parser("speakers", help="toy speaker identification").set_defaults(fn=cmd_speakers)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
