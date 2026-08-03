"""Transcription via faster-whisper with an explicit robustness harness.

Every anti-hallucination / accuracy knob lives in RobustnessConfig with
hardened defaults, so the decode behavior is inspectable and testable rather
than a pile of implicit library defaults.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RobustnessConfig:
    beam_size: int = 5
    # Temperature fallback ladder: retry decoding hotter when quality gates
    # fail; stochastic sampling escapes autoregressive repetition attractors.
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4   # repetition-loop detector
    log_prob_threshold: float = -1.0           # low-confidence retry trigger
    no_speech_threshold: float = 0.6           # phantom-segment suppressor
    condition_on_previous_text: bool = False   # off = no error propagation
    vad_filter: bool = True                    # Silero gate inside the decode
    vad_min_silence_ms: int = 300
    word_timestamps: bool = False
    initial_prompt: Optional[str] = None       # domain-vocabulary biasing
    # Segment flagging threshold. Was 0.55, which could never fire: the
    # 2026-08-03 gate sweep (docs/VALIDATION.md) measured large-v3 confidence
    # bottoming out at 0.669 with 33% WER, so 0.55 sat below the worst case
    # and flagged nothing at any noise level. 0.85 is the value that stays
    # silent while WER <= 0.05 and fires once it climbs to 0.15.
    low_confidence: float = 0.85


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    confidence: float          # exp(avg_logprob): geometric-mean token prob
    flagged: bool
    words: Optional[list] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "avg_logprob": round(self.avg_logprob, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
            "confidence": round(self.confidence, 3),
            "flagged": self.flagged,
        }
        if self.words is not None:
            d["words"] = self.words
        return d


def resolve_device(device: str = "auto",
                   compute_type: str = "auto") -> Tuple[str, str]:
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def load_model(model_size: str = "large-v3", device: str = "auto",
               compute_type: str = "auto"):
    from faster_whisper import WhisperModel

    device, compute_type = resolve_device(device, compute_type)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


class Transcriber:
    def __init__(self, model):
        self.model = model

    def transcribe(self, y, language: Optional[str] = None,
                   cfg: Optional[RobustnessConfig] = None
                   ) -> Tuple[List[TranscriptSegment], object]:
        cfg = cfg or RobustnessConfig()
        segments, info = self.model.transcribe(
            y,
            language=language,
            beam_size=cfg.beam_size,
            temperature=list(cfg.temperature),
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            no_speech_threshold=cfg.no_speech_threshold,
            condition_on_previous_text=cfg.condition_on_previous_text,
            vad_filter=cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": cfg.vad_min_silence_ms},
            word_timestamps=cfg.word_timestamps,
            initial_prompt=cfg.initial_prompt,
        )
        out: List[TranscriptSegment] = []
        for i, s in enumerate(segments):
            conf = min(1.0, math.exp(s.avg_logprob))
            flagged = conf < cfg.low_confidence or s.no_speech_prob > 0.5
            words = None
            if cfg.word_timestamps and s.words:
                words = [{"word": w.word, "start": round(w.start, 2),
                          "end": round(w.end, 2),
                          "prob": round(w.probability, 3)} for w in s.words]
            out.append(TranscriptSegment(
                i, float(s.start), float(s.end), s.text.strip(),
                float(s.avg_logprob), float(s.no_speech_prob),
                conf, flagged, words))
        return out, info
