"""Pipeline orchestrator: load -> (denoise) -> VAD -> LID -> ASR -> gate -> result."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from speechlens.asr import RobustnessConfig, Transcriber, resolve_device
from speechlens.audio import from_array, load_audio
from speechlens.lid import LanguageDetector, LIDResult, language_name
from speechlens.vad import detect as vad_detect
from speechlens.vad import speech_ratio


@dataclass
class AnalysisResult:
    audio: dict
    language: dict
    transcript: dict
    performance: dict
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "audio": self.audio,
            "language": self.language,
            "transcript": self.transcript,
            "performance": self.performance,
            "warnings": self.warnings,
        }

    @property
    def text(self) -> str:
        return self.transcript["text"]


class SpeechLens:
    """End-to-end analyzer. Loads the model once; reuse across calls.

    For tests and dependency injection, pass prebuilt ``transcriber`` and
    ``detector`` and no model is loaded at all.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "auto",
                 compute_type: str = "auto", denoise: bool = False,
                 model=None, transcriber: Optional[Transcriber] = None,
                 detector: Optional[LanguageDetector] = None):
        if transcriber is None or detector is None:
            if model is None:
                from speechlens.asr import load_model
                model = load_model(model_size, device, compute_type)
            transcriber = transcriber or Transcriber(model)
            detector = detector or LanguageDetector(model)
        self.transcriber = transcriber
        self.detector = detector
        self.denoise = denoise
        self.model_size = model_size
        self.device, self.compute_type = resolve_device(device, compute_type)

    def analyze(self, source, language: Optional[str] = None,
                cfg: Optional[RobustnessConfig] = None) -> AnalysisResult:
        t0 = time.perf_counter()
        warnings: list = []

        # --- stage 1: canonical audio -------------------------------------
        if isinstance(source, tuple):
            y, sr = from_array(*source)
        else:
            y, sr = load_audio(source)
        duration = len(y) / sr

        # --- stage 2: optional denoise (off by default: Whisper is trained
        # on noisy audio, and spectral gating can smear formants; A/B it) ---
        if self.denoise:
            try:
                import noisereduce as nr
                y = nr.reduce_noise(y=y, sr=sr).astype(np.float32)
            except ImportError:
                warnings.append("denoise requested but noisereduce is not "
                                "installed (pip install 'speechlens[denoise]');"
                                " skipping")

        # --- stage 3: VAD ---------------------------------------------------
        segments, vad_backend = vad_detect(y, sr)
        ratio = speech_ratio(segments, duration)
        if duration > 3.0 and ratio < 0.25:
            warnings.append(f"only {ratio:.0%} of the clip is speech; results "
                            "reflect the speech regions only")
        if duration < 2.0:
            warnings.append("clip under 2 s: language ID is unreliable this "
                            "short")

        # --- stage 4: language ID (or forced) -------------------------------
        if language:
            lid = LIDResult(language, language_name(language), 1.0,
                            [(language, 1.0)], 0.0, 0, False, method="forced")
        else:
            lid = self.detector.detect(y, sr, speech_segments=segments)
            if lid.low_confidence:
                warnings.append("low language-ID confidence; consider a longer"
                                " sample or forcing --language")

        # --- stage 5: robust decode -----------------------------------------
        cfg = cfg or RobustnessConfig()
        tsegs, _info = self.transcriber.transcribe(y, language=lid.code,
                                                   cfg=cfg)

        # --- stage 6: confidence gate ----------------------------------------
        flagged = sum(1 for s in tsegs if s.flagged)
        if tsegs and flagged / len(tsegs) > 0.4:
            warnings.append(f"{flagged}/{len(tsegs)} segments are "
                            "low-confidence; try a larger model, --denoise, or"
                            " cleaner audio")

        elapsed = time.perf_counter() - t0
        return AnalysisResult(
            audio={
                "duration_s": round(duration, 2),
                "sample_rate": sr,
                "speech_ratio": round(ratio, 3),
                "vad_backend": vad_backend,
                "speech_segments": [(round(s.start, 2), round(s.end, 2))
                                    for s in segments],
            },
            language=lid.to_dict(),
            transcript={
                "text": " ".join(s.text for s in tsegs).strip(),
                "segments": [s.to_dict() for s in tsegs],
                "flagged_segments": flagged,
            },
            performance={
                "model": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "total_time_s": round(elapsed, 2),
                "rtf": round(elapsed / duration, 3) if duration > 0 else None,
            },
            warnings=warnings,
        )
