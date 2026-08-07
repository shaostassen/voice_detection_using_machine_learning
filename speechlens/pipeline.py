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
    # Empty unless a ReliabilityPolicy is configured and word timestamps are on.
    reliability: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audio": self.audio,
            "language": self.language,
            "transcript": self.transcript,
            "performance": self.performance,
            "warnings": self.warnings,
            "reliability": self.reliability,
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
                 detector: Optional[LanguageDetector] = None,
                 policy=None):
        if transcriber is None or detector is None:
            if model is None:
                from speechlens.asr import load_model
                model = load_model(model_size, device, compute_type)
            transcriber = transcriber or Transcriber(model)
            detector = detector or LanguageDetector(model)
        self.transcriber = transcriber
        self.detector = detector
        # Optional ReliabilityPolicy. Kept outside the block above so passing
        # one never triggers a model load, and absent means "skip the stage"
        # rather than "fall back to something unvalidated".
        self.policy = policy
        self.denoise = denoise
        self.model_size = model_size
        self.device, self.compute_type = resolve_device(device, compute_type)

    def _resolve_policy(self, policy, y, sr, segments, warnings: list):
        """Turn ``"auto"`` into a concrete policy using an SNR estimate.

        Returns ``(policy, selected_by)``. When the estimate is unavailable the
        policy is dropped rather than defaulted: guessing `clean` on unknown
        audio would auto-accept most of a transcript that might be 29% wrong,
        and no annotation at all is the safer failure.
        """
        if policy is None:
            policy = self.policy
        if policy is None:
            return None, ""
        if policy != "auto":
            return policy, "explicit"

        from speechlens.calibration import (available_policies,
                                            load_bundled_policy)
        from speechlens.snr import estimate_snr, nearest_condition

        snr = estimate_snr(y, sr, segments)
        cond = nearest_condition(snr, available_policies())
        if cond is None:
            warnings.append("could not estimate SNR (needs both speech and "
                            "non-speech audio), so no reliability policy was "
                            "applied; pass one explicitly")
            return None, ""
        return load_bundled_policy(cond), f"auto (SNR≈{snr:.0f} dB)"

    def _score_words(self, tsegs, warnings: list, policy=None,
                     selected_by: str = "explicit") -> dict:
        """Annotate each word with calibrated reliability; summarize coverage.

        Returns an empty dict when no policy is configured or when word
        timestamps are off, rather than inventing a score — an absent number
        is honest, a guessed one is the failure mode this project exists to
        avoid.
        """
        if policy is None:
            return {}

        words = [w for s in tsegs for w in (s.words or [])]
        if not words:
            if tsegs:
                warnings.append("reliability policy is set but word "
                                "timestamps are off; pass "
                                "RobustnessConfig(word_timestamps=True)")
            return {}

        accepted = 0
        for w in words:
            # float()/bool() are load-bearing, not cosmetic. faster-whisper's
            # word probability is a numpy float64, and while that serializes
            # (it subclasses float), comparing it yields numpy.bool_ — which
            # does NOT subclass bool and makes json.dump raise mid-write,
            # leaving a truncated file. Same for the numpy.int64 the
            # accumulator would otherwise become.
            r = float(policy.calibrator.predict(w["prob"]))
            ok = bool(r >= policy.threshold)
            w["reliability"] = round(r, 3)
            w["accept"] = ok
            accepted += int(ok)

        return {
            "estimator": "word_prob",
            "selected_by": selected_by,
            "target_risk": policy.target_risk,
            "threshold": round(policy.threshold, 3),
            "words": len(words),
            "accepted": accepted,
            "coverage": round(accepted / len(words), 3),
            "review": len(words) - accepted,
            "condition": policy.calibrator.condition,
        }

    def analyze(self, source, language: Optional[str] = None,
                cfg: Optional[RobustnessConfig] = None,
                policy=None) -> AnalysisResult:
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

        # --- stage 7: per-word reliability (opt-in) ---------------------------
        # The segment flag above reads `exp(avg_logprob)`, which faster-whisper
        # computes once per 30 s decode window and copies onto every segment in
        # it — so it cannot rank words inside a window at all (AUROC 0.58 on
        # clean audio vs 0.89 for the per-word signal; docs/VALIDATION.md).
        # This stage annotates each word with a calibrated P(correct) and an
        # accept/review decision instead.
        policy, selected_by = self._resolve_policy(policy, y, sr, segments,
                                                   warnings)
        reliability = self._score_words(tsegs, warnings, policy, selected_by)

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
            reliability=reliability,
        )
