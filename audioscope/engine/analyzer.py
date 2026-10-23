"""Analysis orchestration: chooses a strategy per audio type.

=============  ===========================================  ==================================
Type           Tempo                                         Key / pitch
=============  ===========================================  ==================================
Full song      percussive+full onsets, section aggregation   per-section chroma, aggregated
Drum loop      transient onsets + exact loop length          key only from tuned drums (capped)
Melodic loop   note/chord onsets + exact loop length         chroma key; bass note
Bass loop      note onsets + exact loop length               tracked-note key; most frequent note
Vocal          note onsets (often no pulse -> N/A)           tracked-note key; most frequent note
One-shot       none (N/A)                                     fundamental or chord root
=============  ===========================================  ==================================

Filename tags such as "128bpm" or "F#m" are reported separately as hints and
are never substituted for a detection.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

import numpy as np

from . import music
from .classify import classify
from .features import Features
from .io import AudioData, load_audio
from .key import estimate_key
from .pitch import estimate_pitch
from .tempo import estimate_tempo
from .types import AnalysisResult, AudioType, Classification, KeyResult, Reliability

ENGINE_VERSION = "1.0.0"

Progress = Callable[[str], None]

_BPM_RE = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d{1,2})?)\s*(?:bpm)(?![a-z])", re.I)
_BPM_PREFIX_RE = re.compile(r"bpm\s*(\d{2,3}(?:\.\d{1,2})?)(?!\d)", re.I)
_KEY_RE = re.compile(r"(?<![A-Za-z])([A-G])([#b♯♭]|sharp|flat)?\s?(major|minor|maj|min|m|M)?(?![A-Za-z])")


def parse_filename_hints(path: str | None) -> dict:
    """BPM / key tags written into sample filenames (e.g. 'Loop_128bpm_F#min.wav')."""
    if not path:
        return {}
    name = os.path.splitext(os.path.basename(path))[0]
    flat = re.sub(r"[_\-.()\[\]]+", " ", name)
    out: dict = {}
    m = _BPM_RE.search(flat) or _BPM_PREFIX_RE.search(flat)
    if m:
        v = float(m.group(1))
        if 40 <= v <= 300:
            out["bpm"] = v
    for km in _KEY_RE.finditer(flat):
        letter, acc, mode = km.groups()
        if not acc and not mode:
            continue  # a bare letter is too ambiguous (take 'A', variation 'B', ...)
        acc = {"sharp": "#", "flat": "b", "♯": "#", "♭": "b"}.get(acc or "", acc or "")
        parsed = music.parse_key(f"{letter}{acc}{mode or ''}")
        if parsed:
            out["key"] = music.key_name(*parsed)
            out["camelot"] = music.camelot(*parsed)
            break
    return out


def analyze_file(path: str, audio_type: AudioType | None = None, progress: Progress | None = None) -> AnalysisResult:
    if progress:
        progress("Decoding")
    audio = load_audio(path)
    return analyze_audio(audio, audio_type=audio_type, progress=progress)


def analyze_audio(audio: AudioData, audio_type: AudioType | None = None, progress: Progress | None = None) -> AnalysisResult:
    t0 = time.perf_counter()
    say = progress or (lambda _s: None)
    feat = Features(audio.y, audio.sr)
    warnings: list[str] = []
    if feat.is_silent:
        warnings.append("File is silent")
    if feat.peak >= 0.999:
        warnings.append("Audio clips (peaks at full scale)")

    say("Classifying")
    if audio_type is None:
        cls = classify(feat, audio.path)
    else:
        cls = Classification(audio_type=audio_type, reliability=Reliability.HIGH, manual=True,
                             evidence=["Type set manually"])
    t = cls.audio_type
    lossy = audio.codec.upper() in ("MPEG", "MP3", "FFMPEG", "OGG", "VORBIS")

    say("Detecting tempo")
    tempo = estimate_tempo(feat, t, is_lossy=lossy)

    say("Detecting key and pitch")
    if t in (AudioType.ONE_SHOT, AudioType.BASS_LOOP, AudioType.VOCAL):
        pitch, mono = estimate_pitch(feat, t)
        if t == AudioType.ONE_SHOT:
            key = _one_shot_key(feat, pitch)
        elif mono is not None and mono.sum() > 0:
            key = estimate_key(feat, t, pcp_override=mono)
        else:
            key = estimate_key(feat, t)
    else:
        key = estimate_key(feat, t)
        pcp = np.asarray(key.pitch_class_profile) if key.pitch_class_profile else None
        pitch, _ = estimate_pitch(feat, t, pcp=pcp)

    if key.key is not None:
        pitch.prefer_flats = music.key_uses_flats(key.key.tonic, key.key.mode)
    if key.tuning_cents is not None and abs(key.tuning_cents) >= 10 and key.key is not None:
        key.evidence.append(f"Material is tuned {key.tuning_cents:+.0f}¢ from A440")

    return AnalysisResult(
        path=audio.path, duration=round(audio.duration, 4), sample_rate=audio.native_sr, channels=audio.channels,
        classification=cls, tempo=tempo, key=key, pitch=pitch, filename_hints=parse_filename_hints(audio.path),
        warnings=warnings, elapsed=round(time.perf_counter() - t0, 3), engine_version=ENGINE_VERSION,
    )


def _one_shot_key(feat: Features, pitch) -> KeyResult:
    """One-shots: a single note has a root, not a key. Chords get a (low-reliability) key reading."""
    if pitch.chord is None:
        res = KeyResult(tuning_cents=round(feat.tuning * 100.0, 1))
        if pitch.root_midi is not None:
            res.evidence.append("Single pitched note: the root note is reported; a key needs several pitches")
        else:
            res.evidence.append("No pitched content")
        return res
    res = estimate_key(feat, AudioType.ONE_SHOT, tonic_hint=pitch.root_midi % 12)
    if res.key is not None:
        res.evidence.append(f"Key implied by a single chord ({pitch.chord}) is ambiguous by nature")
    return res
