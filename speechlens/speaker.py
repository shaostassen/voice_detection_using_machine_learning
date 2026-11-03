"""Speaker enrolment, identification and verification over pluggable embeddings.

This module is the *logic* around a speaker encoder; it deliberately contains no
neural network, so importing it never pulls in torch. The encoder is any
callable ``(audio, sr) -> vector`` — the same interface SpeechForge's
``SpeakerEncoder`` exposes, so that one plugs straight in:

    from speechforge.similarity import SpeakerEncoder
    gallery = Gallery(SpeakerEncoder(), encoder_id="wavlm-base-plus-sv")

or use ``speechbrain_encoder()`` below for a pretrained ECAPA-TDNN.

Three rules this module enforces, because each is an easy way to fool yourself:

* **There is no default threshold.** Cosine scores are specific to the encoder,
  the microphone and the audio length: SpeechForge measured same-speaker pairs
  at 0.89-0.98 on its encoder against 0.69-0.76 published for others. A number
  borrowed from a paper or another model is a guess. ``verify`` therefore
  requires a threshold, and ``threshold_for_far`` derives one from impostor
  scores *you* measure with *your* encoder and audio.
* **Galleries are tied to their encoder.** Embeddings from two encoders live in
  different spaces; comparing them silently gives noise. ``Gallery.load``
  refuses a mismatch.
* **Open-set vs closed-set is explicit.** ``identify`` with no threshold always
  returns the nearest enrolled speaker, even for a stranger. Pass a threshold to
  allow "none of them".

A saved gallery is biometric data. Treat the file accordingly: get consent from
the people in it, and see docs/HOW_IT_WORKS.md section 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Embedder = Callable[[np.ndarray, int], np.ndarray]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; NaN for a zero vector (a failed encode is not 0.0)."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"embedding shape mismatch: {x.shape} vs {y.shape}")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("encoder returned a zero or non-finite embedding")
    return v / n


class Gallery:
    """Enrolled speakers, each a set of unit-length embeddings."""

    def __init__(self, embedder: Embedder, encoder_id: str):
        if not encoder_id:
            raise ValueError("encoder_id is required: it is saved with the "
                             "gallery so embeddings are never mixed across models")
        self.embedder = embedder
        self.encoder_id = encoder_id
        self._emb: Dict[str, List[np.ndarray]] = {}

    # -- enrolment -----------------------------------------------------
    def enroll(self, name: str, audio: np.ndarray, sr: int) -> None:
        """Add one recording for `name`. Enrol several, in varied conditions:
        a speaker's centroid over 3-5 clips beats any single clip."""
        self._emb.setdefault(name, []).append(
            _unit(self.embedder(np.asarray(audio, dtype=np.float32), sr)))

    @property
    def names(self) -> List[str]:
        return sorted(self._emb)

    def centroid(self, name: str) -> np.ndarray:
        return _unit(np.mean(self._emb[name], axis=0))

    # -- scoring -------------------------------------------------------
    def scores(self, audio: np.ndarray, sr: int) -> List[Tuple[str, float]]:
        """Cosine to every enrolled speaker's centroid, best first."""
        if not self._emb:
            raise ValueError("gallery is empty")
        p = _unit(self.embedder(np.asarray(audio, dtype=np.float32), sr))
        out = [(n, float(p @ self.centroid(n))) for n in self._emb]
        return sorted(out, key=lambda t: -t[1])

    def identify(self, audio: np.ndarray, sr: int,
                 threshold: Optional[float] = None
                 ) -> Tuple[Optional[str], float, List[Tuple[str, float]]]:
        """(name, score, ranking). `name` is None if open-set and below threshold."""
        ranking = self.scores(audio, sr)
        name, score = ranking[0]
        if threshold is not None and score < threshold:
            return None, score, ranking
        return name, score, ranking

    def verify(self, audio: np.ndarray, sr: int, claimed: str,
               threshold: float) -> Tuple[bool, float]:
        """1:1 check of a claimed identity. The threshold is mandatory."""
        if claimed not in self._emb:
            raise KeyError(f"{claimed!r} is not enrolled")
        p = _unit(self.embedder(np.asarray(audio, dtype=np.float32), sr))
        s = float(p @ self.centroid(claimed))
        return s >= threshold, s

    # -- persistence (no pickle: plain arrays only) --------------------
    def save(self, path: Union[str, Path]) -> None:
        arrays = {"encoder_id": np.array(self.encoder_id),
                  "names": np.array(self.names)}
        for i, n in enumerate(self.names):
            arrays[f"emb_{i}"] = np.stack(self._emb[n])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path], embedder: Embedder,
             encoder_id: str) -> "Gallery":
        with np.load(path, allow_pickle=False) as z:
            saved = str(z["encoder_id"])
            if saved != encoder_id:
                raise ValueError(
                    f"gallery was built with encoder {saved!r} but {encoder_id!r} "
                    f"was supplied; embeddings from different models are not "
                    f"comparable. Re-enrol with the new encoder.")
            g = cls(embedder, encoder_id)
            for i, n in enumerate(z["names"]):
                g._emb[str(n)] = list(z[f"emb_{i}"])
        return g


# -- threshold calibration --------------------------------------------------
def threshold_for_far(impostor_scores: Sequence[float], far: float) -> float:
    """Lowest threshold whose false-accept rate on `impostor_scores` is <= far.

    The estimate is only as good as the sample: with N impostor trials you cannot
    resolve a FAR below ~1/N, and this raises rather than pretend to.
    """
    s = np.sort(np.asarray(impostor_scores, dtype=np.float64))
    s = s[np.isfinite(s)]
    if not 0.0 < far < 1.0:
        raise ValueError("far must be in (0, 1)")
    if len(s) * far < 1.0:
        raise ValueError(
            f"{len(s)} impostor trials cannot resolve FAR={far:g}; "
            f"need at least {int(np.ceil(1 / far))}")
    k = int(np.floor(far * len(s)))        # impostors allowed above the threshold
    # accepting score >= t; choose t just above the (k+1)-th highest impostor.
    cut = s[len(s) - k - 1]
    return float(np.nextafter(cut, np.inf))


def equal_error_rate(genuine: Sequence[float],
                     impostor: Sequence[float]) -> Tuple[float, float]:
    """(EER, threshold at EER). Where false-accept and false-reject cross."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    if g.size == 0 or i.size == 0:
        raise ValueError("need both genuine and impostor scores")
    best = (2.0, 0.0, 0.0)
    for t in np.unique(np.concatenate([g, i])):
        far = float(np.mean(i >= t))
        frr = float(np.mean(g < t))
        gap = abs(far - frr)
        if gap < best[0]:
            best = (gap, (far + frr) / 2.0, float(t))
    return best[1], best[2]


# -- real encoder (optional, imported lazily) ------------------------------
SPEECHBRAIN_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def speechbrain_encoder(source: str = SPEECHBRAIN_SOURCE,
                        device: str = "cpu") -> Embedder:
    """Pretrained ECAPA-TDNN via SpeechBrain (192-d embeddings, 16 kHz input).

    Requires ``pip install speechbrain torch`` (the ``speaker`` extra). Weights
    download from Hugging Face on first use and are cached afterwards.
    """
    try:
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:                       # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
    except ImportError as e:                      # pragma: no cover
        raise ImportError(
            "speaker encoding needs torch and speechbrain: "
            "pip install 'speechlens[speaker]'") from e

    clf = EncoderClassifier.from_hparams(source=source,
                                         run_opts={"device": device})

    def embed(audio: np.ndarray, sr: int) -> np.ndarray:
        y = np.asarray(audio, dtype=np.float32).ravel()
        if sr != 16000:
            from speechlens.audio import resample
            y = resample(y, sr, 16000)
        with torch.no_grad():
            e = clf.encode_batch(torch.from_numpy(y).unsqueeze(0).to(device))
        return e.squeeze().cpu().numpy().astype(np.float64)

    return embed
