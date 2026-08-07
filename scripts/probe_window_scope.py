"""Is faster-whisper's avg_logprob per-segment or per-30s-decode-window?

Concatenate enough LibriSpeech utterances to force several windows, decode,
and print seek (window id) alongside avg_logprob for every segment. If the
count of distinct avg_logprob values equals the count of distinct seeks --
not the count of segments -- the value is per-window and broadcast.
"""
import io
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation")
ds = ds.cast_column("audio", Audio(decode=False))

parts = []
for i in range(12):
    a = ds[i]["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    parts.append(y)

y = np.concatenate(parts)
sr = 16000
print(f"duration={len(y)/sr:.1f}s  -> expect ~{int(len(y)/sr // 30) + 1} windows")

m = WhisperModel("small", device="cpu", compute_type="int8")
segs, _info = m.transcribe(y, language="en", beam_size=5,
                           word_timestamps=True,
                           condition_on_previous_text=False)

rows = []
for s in segs:
    wprobs = [w.probability for w in (s.words or [])]
    rows.append((s.seek, s.avg_logprob, s.no_speech_prob, s.compression_ratio,
                 len(wprobs),
                 min(wprobs) if wprobs else float("nan"),
                 max(wprobs) if wprobs else float("nan"),
                 s.text.strip()[:34]))

print(f"\n{len(rows)} segments\n")
print(f"{'seek':>6} {'avg_logprob':>12} {'no_speech':>10} {'compr':>7} "
      f"{'nw':>3} {'wmin':>6} {'wmax':>6}  text")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>12.6f} {r[2]:>10.6f} {r[3]:>7.3f} "
          f"{r[4]:>3} {r[5]:>6.3f} {r[6]:>6.3f}  {r[7]}")

n_seg = len(rows)
n_alp = len(set(round(r[1], 9) for r in rows))
n_seek = len(set(r[0] for r in rows))
allw = [p for r in rows for p in ([r[5], r[6]] if r[4] else [])]

print(f"\nsegments                    : {n_seg}")
print(f"distinct seek (windows)     : {n_seek}")
print(f"distinct avg_logprob        : {n_alp}")
print(f"distinct no_speech_prob     : {len(set(round(r[2], 9) for r in rows))}")
print(f"distinct compression_ratio  : {len(set(round(r[3], 9) for r in rows))}")
if allw:
    print(f"word probability range      : {min(allw):.3f} .. {max(allw):.3f}")

print()
if n_alp == n_seek and n_seek < n_seg:
    print("CONFIRMED: avg_logprob is PER-WINDOW, broadcast to every segment "
          "in that window.")
elif n_alp == n_seg:
    print("REFUTED: avg_logprob is genuinely per-segment.")
else:
    print("INCONCLUSIVE: distinct values match neither windows nor segments.")
