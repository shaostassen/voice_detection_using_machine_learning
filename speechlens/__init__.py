"""SpeechLens: local language identification + robust transcription.

Public API:

    from speechlens import SpeechLens
    lens = SpeechLens(model_size="large-v3")
    result = lens.analyze("clip.wav")
    print(result.language["code"], result.text)
"""
from speechlens.asr import RobustnessConfig
from speechlens.pipeline import AnalysisResult, SpeechLens

__version__ = "0.1.0"
__all__ = ["SpeechLens", "AnalysisResult", "RobustnessConfig", "__version__"]
