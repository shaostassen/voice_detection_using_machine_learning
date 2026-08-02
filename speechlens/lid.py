"""Language identification via Whisper's implicit LID with chunk voting.

Whisper's decoder predicts a language token immediately after
<|startoftranscript|>; the softmax over the ~100 language tokens is
P(language | audio) for one 30 s window. A single window is fragile — a music
intro, a noise burst, or code switching can flip it — so we score several
windows sampled from detected speech regions and fuse the distributions in
log space (geometric mean). Log fusion penalizes any language that a chunk
strongly rejects, which is the behavior you want when one window is corrupted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

LANG_NAMES = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "de": "German",
    "es": "Spanish", "ru": "Russian", "ko": "Korean", "fr": "French",
    "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish",
    "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish",
    "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek",
    "ms": "Malay", "cs": "Czech", "ro": "Romanian", "da": "Danish",
    "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian", "th": "Thai",
    "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian",
}


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


@dataclass
class LIDResult:
    code: str
    name: str
    probability: float
    top: List = field(default_factory=list)   # [(code, prob), ...] best-first
    entropy: float = 0.0
    chunks_used: int = 0
    low_confidence: bool = False
    method: str = "whisper_chunk_vote"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "probability": round(self.probability, 4),
            "top": [(c, round(p, 4)) for c, p in self.top[:3]],
            "entropy": round(self.entropy, 3),
            "chunks_used": self.chunks_used,
            "low_confidence": self.low_confidence,
            "method": self.method,
        }


def fuse_distributions(dists: List[Dict[str, float]],
                       eps: float = 1e-6) -> Dict[str, float]:
    """Geometric-mean fusion: sum log-probs across chunks, renormalize."""
    keys = set().union(*dists)
    logp = {k: 0.0 for k in keys}
    for d in dists:
        for k in keys:
            logp[k] += math.log(d.get(k, 0.0) + eps)
    m = max(logp.values())
    exp = {k: math.exp(v - m) for k, v in logp.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


def entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy (nats). Low = confident, high = the model is torn."""
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


class LanguageDetector:
    """Chunk-voting LID over a shared faster-whisper model."""

    CHUNK_S = 30.0

    def __init__(self, model):
        self.model = model

    def _chunk_probs(self, chunk: np.ndarray) -> Dict[str, float]:
        # faster-whisper computes LID eagerly when transcribe() is called with
        # language=None, but the returned segments generator is lazy — leaving
        # it unconsumed makes this an encoder pass + LID only, i.e. cheap.
        _segments, info = self.model.transcribe(
            chunk, language=None, beam_size=1,
            condition_on_previous_text=False, vad_filter=False,
        )
        probs = getattr(info, "all_language_probs", None)
        if probs:
            return {code: float(p) for code, p in probs}
        return {info.language: float(info.language_probability)}

    def _pick_windows(self, n_samples: int, sr: int, speech_segments,
                      max_chunks: int):
        chunk = int(self.CHUNK_S * sr)
        if speech_segments:
            mid = speech_segments[len(speech_segments) // 2]
            anchors = [speech_segments[0].start, mid.start,
                       max(0.0, speech_segments[-1].end - self.CHUNK_S)]
        else:
            dur = n_samples / sr
            anchors = [0.0, max(0.0, dur / 2 - self.CHUNK_S / 2),
                       max(0.0, dur - self.CHUNK_S)]
        starts, seen = [], set()
        for a in anchors:
            s = min(max(0, int(a * sr)), max(0, n_samples - chunk))
            if s not in seen:
                seen.add(s)
                starts.append(s)
            if len(starts) >= max_chunks:
                break
        return [(s, min(s + chunk, n_samples)) for s in starts]

    def detect(self, y: np.ndarray, sr: int = 16000, speech_segments=None,
               max_chunks: int = 3) -> LIDResult:
        windows = self._pick_windows(len(y), sr, speech_segments or [],
                                     max_chunks)
        dists = []
        for s, e in windows:
            if (e - s) / sr < 1.0:
                continue
            dists.append(self._chunk_probs(y[s:e]))
        if not dists:
            return LIDResult("en", "English", 0.0, [], 0.0, 0, True)

        fused = fuse_distributions(dists)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        code, p = ranked[0]
        speech_s = sum(s.duration for s in (speech_segments or []))
        low = p < 0.5 or (speech_segments is not None and speech_s < 2.0)
        return LIDResult(code, language_name(code), p, ranked[:5],
                         entropy(fused), len(dists), low)
