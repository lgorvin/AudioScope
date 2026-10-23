"""AudioScope analysis engine (UI-independent).

Typical use::

    from audioscope.engine import analyze_file
    result = analyze_file("loop.wav")
    print(result.tempo.display, result.key.display)
"""
from .analyzer import ENGINE_VERSION, analyze_audio, analyze_file, parse_filename_hints
from .io import SUPPORTED_EXTENSIONS, AudioLoadError, is_supported, load_audio
from .types import AnalysisResult, AudioType, Reliability

__all__ = [
    "ENGINE_VERSION", "analyze_audio", "analyze_file", "parse_filename_hints", "SUPPORTED_EXTENSIONS",
    "AudioLoadError", "is_supported", "load_audio", "AnalysisResult", "AudioType", "Reliability",
]
