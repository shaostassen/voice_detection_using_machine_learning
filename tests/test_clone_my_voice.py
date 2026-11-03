"""Offline tests for scripts/clone_my_voice.py.

The take checks, manifest handling and CLI guards need nothing. The pipeline
tests need SpeechForge's core (numpy/scipy/soundfile, no torch) on the path:
set SPEECHFORGE_PATH to a checkout, or `pip install speechforge`; otherwise
they are skipped.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speechlens.synth import synth_utterance
from speechlens.voiceprint import voiceprint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clone_my_voice.py"
SR = 16000


@pytest.fixture(scope="module")
def cmv():
    spec = importlib.util.spec_from_file_location("clone_my_voice", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clone_my_voice"] = mod
    spec.loader.exec_module(mod)
    return mod


def voice(seed=0, n=10):
    return synth_utterance(130.0, 1.05, n_vowels=n, seed=seed)


# ---- take quality ----------------------------------------------------------
def test_clean_take_passes(cmv):
    y = np.concatenate([np.zeros(SR // 2, np.float32), voice(n=14),
                        np.zeros(SR // 2, np.float32)])
    assert cmv.check_take(y, SR, "heldout") == []


def test_clipping_quiet_short_and_noisy_are_flagged(cmv):
    loud = np.clip(voice(n=14) * 5.0, -1, 1).astype(np.float32)
    assert any("clipping" in w for w in cmv.check_take(loud, SR))
    assert any("quiet" in w for w in cmv.check_take(voice(n=14) * 0.02, SR))
    assert any("short" in w for w in cmv.check_take(voice(n=1)[:SR // 2], SR))
    # a reference needs more audio than an ordinary held-out sentence
    short_ref = np.concatenate([np.zeros(SR // 4, np.float32), voice(n=10)])
    assert any("reference" in w for w in cmv.check_take(short_ref, SR, "reference"))
    rng = np.random.default_rng(0)
    noisy = (voice(n=14) + 0.3 * rng.standard_normal(len(voice(n=14)))).astype(np.float32)
    assert any("noisy" in w for w in cmv.check_take(noisy, SR))


# ---- manifest / import guards -----------------------------------------------
def write_takes(d, n=4):
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        p = d / f"src_{i}.wav"
        sf.write(str(p), voice(seed=i, n=12), SR)
        files.append(str(p))
    return files


def test_import_requires_own_voice_flag(cmv, tmp_path):
    f = write_takes(tmp_path / "src")
    with pytest.raises(SystemExit, match="own-voice"):
        cmv.main(["import", "--dir", str(tmp_path / "d"), "--take", f[0], "hello"])


def test_import_then_load_roundtrip(cmv, tmp_path):
    f = write_takes(tmp_path / "src", 4)
    args = ["import", "--dir", str(tmp_path / "d"), "--own-voice"]
    for i, p in enumerate(f):
        args += ["--take", p, f"sentence number {i}"]
    assert cmv.main(args) == 0
    takes = cmv.load_takes(tmp_path / "d")
    assert [t["role"] for t in takes] == ["reference", "heldout", "heldout", "heldout"]
    assert takes[1]["text"] == "sentence number 1"


def test_load_takes_refuses_too_few_heldout_and_missing_consent(cmv, tmp_path):
    f = write_takes(tmp_path / "src", 2)       # 1 reference + 1 held-out
    cmv.main(["import", "--dir", str(tmp_path / "d"), "--own-voice",
              "--take", f[0], "a", "--take", f[1], "b"])
    with pytest.raises(SystemExit, match="held-out"):
        cmv.load_takes(tmp_path / "d")
    m = cmv.load_manifest(tmp_path / "d")
    m["own_voice_confirmed"] = False
    cmv.save_manifest(tmp_path / "d", m)
    with pytest.raises(SystemExit, match="own voice"):
        cmv.load_takes(tmp_path / "d")


def test_record_needs_enough_prompts(cmv):
    with pytest.raises(SystemExit, match="at least"):
        cmv.main(["record", "--dir", "x", "--n", "2"])


def test_run_explains_missing_speechforge(cmv, monkeypatch):
    monkeypatch.setitem(sys.modules, "speechforge", None)   # force ImportError
    with pytest.raises(SystemExit, match="SpeechForge is required"):
        cmv._require_speechforge(None)


# ---- the pipeline, with fakes ----------------------------------------------
@pytest.fixture
def forge():
    p = os.environ.get("SPEECHFORGE_PATH")
    if p:
        sys.path.insert(0, p)
    pytest.importorskip("speechforge.evaluate")
    from speechforge.synth import clone
    from speechforge.text import apply_case
    return clone, apply_case


def _key(y):
    y = np.asarray(y, dtype=np.float64)
    return (len(y), round(float(np.sum(np.abs(y))), 2))


def make_fakes(takes, clone_quality="good"):
    """Backend that 'clones' with the toy voice; ASR/encoder as lookups."""
    registry = {_key(t["audio"]): t["text"] for t in takes}

    class Backend:
        sample_rate = SR
        ref_seconds = None
        align_ms = None
        needs_reference_text = False
        calls = []

        def describe(self):
            return "fake-backend"

        def synthesize(self, text, reference_path, seed=None, reference_text=None):
            self.calls.append((text, reference_path, seed))
            wav = synth_utterance(130.0, 1.05, n_vowels=12, seed=500 + len(self.calls))
            if clone_quality == "bad":
                wav = np.random.default_rng(1).standard_normal(len(wav)).astype(np.float32) * 0.1
            registry[_key(wav)] = text if clone_quality == "good" else "garbled words here"
            return wav, SR

    ref = np.array([voiceprint(synth_utterance(f, s, n_vowels=12, seed=900 + i))
                    for i, (f, s) in enumerate([(120, 1.0), (180, 1.1), (240, 1.25),
                                                (150, 0.9), (160, 1.1)])])
    mu, sd = ref.mean(0), ref.std(0) + 1e-6

    def encoder(y, sr):
        return (voiceprint(y, sr=sr) - mu) / sd

    def transcribe(y, sr):
        return registry.get(_key(y), "unknown")

    return Backend(), transcribe, encoder


def build_takes(n_held=3):
    texts = ["the weather turned cold this morning",
             "could you tell me where the station is",
             "she measured two cups of flour carefully",
             "nobody expected the meeting to finish early"]
    out = [{"file": "take_00.wav", "text": "i am recording this passage now",
            "role": "reference", "audio": voice(seed=50, n=20), "sr": SR}]
    for i in range(n_held):
        out.append({"file": f"take_{i + 1:02d}.wav", "text": texts[i],
                    "role": "heldout", "audio": voice(seed=60 + i, n=12), "sr": SR})
    return out


def test_pipeline_end_to_end_with_fakes(cmv, forge, tmp_path):
    clone, apply_case = forge
    takes = build_takes(3)
    backend, transcribe, encoder = make_fakes(takes)
    res = cmv.run_pipeline(takes, backend=backend, transcribe=transcribe,
                           encoder=encoder, clone_fn=clone, apply_case=apply_case,
                           out_dir=tmp_path / "clones", backend_name="fake",
                           provenance={"backend": "fake-backend"})
    # one clone per held-out sentence, none from the reference
    assert len(res["rows"]) == 3 and len(backend.calls) == 3
    assert all((tmp_path / "clones" / f"clone_{i:02d}.wav").exists() for i in range(3))
    # the model was handed sentence-cased text, scoring used the original
    assert backend.calls[0][0] == "The weather turned cold this morning."
    assert res["rows"][0]["text"] == "the weather turned cold this morning"
    # the ground-truth row is present and the table says so
    assert "your real voice" in res["table"] and "read every row against this" in res["table"]
    assert res["table_rows"]["ground_truth"]["wer"] == 0.0
    assert res["table_rows"]["systems"][0]["wer"] == 0.0
    # same-voice clone is more similar to the real voice than a stranger would be
    assert res["rows"][0]["sim_clone_vs_real"] > 0.0
    json.dumps(res, default=str)           # results are serialisable


def test_pipeline_surfaces_a_bad_clone(cmv, forge, tmp_path):
    clone, apply_case = forge
    takes = build_takes(3)
    backend, transcribe, encoder = make_fakes(takes, clone_quality="bad")
    res = cmv.run_pipeline(takes, backend=backend, transcribe=transcribe,
                           encoder=encoder, clone_fn=clone, apply_case=apply_case,
                           backend_name="fake")
    gt = res["table_rows"]["ground_truth"]
    clone_row = res["table_rows"]["systems"][0]
    assert clone_row["wer"] > gt["wer"] + 0.5          # garbled -> large WER
    assert clone_row["sim_o"] < gt["sim_o"]            # noise is not your voice


def test_small_sample_caveat_is_reported(cmv, forge):
    clone, apply_case = forge
    takes = build_takes(2)
    backend, transcribe, encoder = make_fakes(takes)
    res = cmv.run_pipeline(takes, backend=backend, transcribe=transcribe,
                           encoder=encoder, clone_fn=clone, apply_case=apply_case,
                           backend_name="fake")
    assert any("only 2 held-out" in c for c in res["caveats"])
    assert any("not a benchmark" in c for c in res["caveats"])
