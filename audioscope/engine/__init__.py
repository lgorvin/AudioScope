"""AudioScope analysis engine (UI-independent).

Typical use::

    from audioscope.engine import analyze_file
    result = analyze_file("loop.wav")
    print(result.tempo.display, result.key.display)
"""
from .analyzer import ENGINE_VERSION, analyze_audio, analyze_file, parse_filename_hints, transcribe_file
from .midi import notes_to_midi, write_midi
from .transcribe import BassTranscription, NoteEvent, quantize, transcribe_bass
from .io import SUPPORTED_EXTENSIONS, AudioLoadError, is_supported, load_audio
from .types import AnalysisResult, AudioType, Reliability

__all__ = [
    "ENGINE_VERSION", "analyze_audio", "analyze_file", "parse_filename_hints", "SUPPORTED_EXTENSIONS",
    "AudioLoadError", "is_supported", "load_audio", "AnalysisResult", "AudioType", "Reliability",
    "transcribe_file", "transcribe_bass", "quantize", "BassTranscription", "NoteEvent", "notes_to_midi", "write_midi",
]
