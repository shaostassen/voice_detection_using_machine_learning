"""FastAPI server: JSON API + single-page web UI.

Env:
    SPEECHLENS_MODEL   model size or path (default: large-v3)
    SPEECHLENS_DEVICE  auto | cuda | cpu  (default: auto)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="SpeechLens", version="0.1.0")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from speechlens.pipeline import SpeechLens
        _pipeline = SpeechLens(
            model_size=os.environ.get("SPEECHLENS_MODEL", "large-v3"),
            device=os.environ.get("SPEECHLENS_DEVICE", "auto"),
        )
    return _pipeline


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": os.environ.get("SPEECHLENS_MODEL", "large-v3"),
        "device": os.environ.get("SPEECHLENS_DEVICE", "auto"),
        "loaded": _pipeline is not None,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  language: Optional[str] = Form(None),
                  reliability: Optional[str] = Form(None)):
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.close()
        cfg = policy = None
        if reliability:
            # Passed per call rather than assigned onto the shared pipeline
            # singleton: mutating it would leak the policy into every later
            # request, including ones that never asked for reliability.
            from speechlens.asr import RobustnessConfig
            from speechlens.calibration import load_bundled_policy
            policy = load_bundled_policy(reliability)
            cfg = RobustnessConfig(word_timestamps=True)
        result = get_pipeline().analyze(tmp.name, language=language or None,
                                        cfg=cfg, policy=policy)
        return JSONResponse(result.to_dict())
    except Exception as exc:
        return JSONResponse(
            {"error": f"Couldn't analyze that file: {exc}"}, status_code=500)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
