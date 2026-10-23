"""Result data types for the analysis engine.

All results are plain dataclasses that serialise to JSON-compatible dicts via
``to_dict()`` so the engine can be driven from a GUI, a CLI or (later) a plugin
host without any Qt dependency.

Confidence policy
-----------------
The engine never reports a "confidence percentage". Each detector reports a
qualitative :class:`Reliability` derived from measurable evidence (periodicity
strength, profile-correlation margins, agreement between song sections, loop
length consistency, pitch stability). The raw evidence values are kept in
``metrics`` so they can be inspected and the thresholds re-calibrated against
labelled audio with ``eval/evaluate.py``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from . import music


class AudioType(str, Enum):
    FULL_SONG = "full_song"
    DRUM_LOOP = "drum_loop"
    MELODIC_LOOP = "melodic_loop"
    BASS_LOOP = "bass_loop"
    VOCAL = "vocal"
    ONE_SHOT = "one_shot"

    @property
    def label(self) -> str:
        return {
            AudioType.FULL_SONG: "Full song",
            AudioType.DRUM_LOOP: "Drum loop",
            AudioType.MELODIC_LOOP: "Melodic loop",
            AudioType.BASS_LOOP: "Bass loop",
            AudioType.VOCAL: "Vocal",
            AudioType.ONE_SHOT: "One-shot",
        }[self]

    @property
    def is_loop(self) -> bool:
        return self in (AudioType.DRUM_LOOP, AudioType.MELODIC_LOOP, AudioType.BASS_LOOP, AudioType.VOCAL)


class Reliability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "n/a"

    @property
    def label(self) -> str:
        return {"high": "High", "medium": "Medium", "low": "Low", "n/a": "N/A"}[self.value]

    @property
    def rank(self) -> int:
        return {"high": 3, "medium": 2, "low": 1, "n/a": 0}[self.value]

    @staticmethod
    def from_rank(rank: int) -> "Reliability":
        return [Reliability.NONE, Reliability.LOW, Reliability.MEDIUM, Reliability.HIGH][max(0, min(3, rank))]


@dataclass
class TempoCandidate:
    bpm: float
    strength: float  # periodicity evidence relative to the best candidate (best = 1.0); not a probability
    relation: str = "alternative"  # primary | half-time | double-time | alternative | loop-length


@dataclass
class TempoResult:
    bpm: float | None = None
    reliability: Reliability = Reliability.NONE
    candidates: list[TempoCandidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)
    bars: int | None = None  # loop length in bars, when derived from duration
    octave_ambiguous: bool = False
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def display(self) -> str:
        if self.bpm is None:
            return "N/A"
        return format_bpm(self.bpm)


def format_bpm(bpm: float) -> str:
    if abs(bpm - round(bpm)) < 0.05:
        return f"{round(bpm):d}"
    return f"{bpm:.1f}"


@dataclass
class KeyCandidate:
    tonic: int
    mode: str
    score: float  # profile correlation (Pearson r), not a probability

    @property
    def name(self) -> str:
        return music.key_name(self.tonic, self.mode)

    @property
    def short_name(self) -> str:
        return music.key_short_name(self.tonic, self.mode)

    @property
    def camelot(self) -> str:
        return music.camelot(self.tonic, self.mode)


@dataclass
class KeyResult:
    key: KeyCandidate | None = None
    reliability: Reliability = Reliability.NONE
    candidates: list[KeyCandidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    tuning_cents: float | None = None
    pitch_class_profile: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.key.name if self.key else "N/A"

    @property
    def camelot_display(self) -> str:
        return self.key.camelot if self.key else "N/A"


@dataclass
class PitchResult:
    """Root note / pitch content. Deliberately separate from the musical key."""

    root_midi: int | None = None
    root_cents: float | None = None  # deviation of the measured pitch from the named note
    root_hz: float | None = None
    root_kind: str = ""  # fundamental | chord root | most frequent note | bass note
    chord: str | None = None
    dominant_pitches: list[tuple[str, float]] = field(default_factory=list)  # (name, relative strength)
    reliability: Reliability = Reliability.NONE
    evidence: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    prefer_flats: bool = False  # spell note names to match a flat key

    @property
    def root_name(self) -> str | None:
        return music.midi_to_name(self.root_midi, self.prefer_flats) if self.root_midi is not None else None

    @property
    def root_pitch_class(self) -> str | None:
        return music.pc_name(self.root_midi, self.prefer_flats) if self.root_midi is not None else None

    @property
    def display(self) -> str:
        if self.root_midi is None:
            return "N/A"
        text = self.root_name
        if self.root_cents is not None and abs(self.root_cents) >= 5:
            text += f" ({self.root_cents:+.0f}¢)"
        return text


@dataclass
class Classification:
    audio_type: AudioType
    reliability: Reliability
    detail: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    manual: bool = False


@dataclass
class AnalysisResult:
    path: str
    duration: float
    sample_rate: int
    channels: int
    classification: Classification
    tempo: TempoResult
    key: KeyResult
    pitch: PitchResult
    filename_hints: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    engine_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(dataclasses.asdict(self))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AnalysisResult":
        c = d["classification"]
        t = d["tempo"]
        k = d["key"]
        p = d["pitch"]
        tempo = TempoResult(**{**t, "reliability": Reliability(t["reliability"]),
                               "candidates": [TempoCandidate(**x) for x in t["candidates"]]})
        key = KeyResult(**{**k, "reliability": Reliability(k["reliability"]),
                           "key": KeyCandidate(**k["key"]) if k["key"] else None,
                           "candidates": [KeyCandidate(**x) for x in k["candidates"]]})
        pitch = PitchResult(**{**p, "reliability": Reliability(p["reliability"]),
                               "dominant_pitches": [tuple(x) for x in p["dominant_pitches"]]})
        cls = Classification(**{**c, "audio_type": AudioType(c["audio_type"]),
                                "reliability": Reliability(c["reliability"])})
        return AnalysisResult(**{**d, "classification": cls, "tempo": tempo, "key": key, "pitch": pitch})


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        return x.item()
    return x
