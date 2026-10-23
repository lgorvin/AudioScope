"""Music-theory helpers: note names, key names and Camelot notation.

Pitch classes are integers 0..11 with C = 0.
"""
from __future__ import annotations

import math

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Conventional spellings for key tonics (fewest accidentals / most common usage).
MAJOR_TONIC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONIC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

MAJOR = "major"
MINOR = "minor"


def pc_name(pc: int, prefer_flats: bool = False) -> str:
    names = NOTE_NAMES_FLAT if prefer_flats else NOTE_NAMES_SHARP
    return names[int(pc) % 12]


def midi_to_name(midi: int, prefer_flats: bool = False) -> str:
    """MIDI note number -> scientific pitch name (MIDI 60 = C4)."""
    midi = int(round(midi))
    names = NOTE_NAMES_FLAT if prefer_flats else NOTE_NAMES_SHARP
    return f"{names[midi % 12]}{midi // 12 - 1}"


def key_uses_flats(tonic: int, mode: str) -> bool:
    """True for keys conventionally written with flats (F, Bb, Eb ... and Dm, Gm, Cm, Fm ...)."""
    major_tonic = tonic % 12 if mode == MAJOR else (tonic + 3) % 12
    return major_tonic in (5, 10, 3, 8, 1)  # F, Bb, Eb, Ab, Db


def hz_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def key_name(tonic: int, mode: str) -> str:
    """Standard notation, e.g. 'A minor', 'Eb major'."""
    names = MAJOR_TONIC_NAMES if mode == MAJOR else MINOR_TONIC_NAMES
    return f"{names[tonic % 12]} {mode}"


def key_short_name(tonic: int, mode: str) -> str:
    """Compact notation, e.g. 'Am', 'Eb'."""
    names = MAJOR_TONIC_NAMES if mode == MAJOR else MINOR_TONIC_NAMES
    return names[tonic % 12] + ("m" if mode == MINOR else "")


def camelot(tonic: int, mode: str) -> str:
    """Camelot wheel code. C major = 8B, A minor = 8A."""
    major_tonic = tonic % 12 if mode == MAJOR else (tonic + 3) % 12
    number = (7 * major_tonic + 8) % 12 or 12
    return f"{number}{'B' if mode == MAJOR else 'A'}"


def relative_key(tonic: int, mode: str) -> tuple[int, str]:
    if mode == MAJOR:
        return (tonic + 9) % 12, MINOR
    return (tonic + 3) % 12, MAJOR


def parse_key(text: str) -> tuple[int, str] | None:
    """Parse 'Am', 'A minor', 'F#maj', 'Bbm', '8A' (Camelot) etc."""
    t = text.strip().replace("♯", "#").replace("♭", "b")
    if not t:
        return None
    # Camelot code
    if t[:-1].isdigit() and t[-1].upper() in "AB":
        number = int(t[:-1])
        if 1 <= number <= 12:
            mode = MAJOR if t[-1].upper() == "B" else MINOR
            for pc in range(12):
                if camelot(pc, mode) == f"{number}{t[-1].upper()}":
                    return pc, mode
        return None
    letter = t[0].upper()
    if letter not in "CDEFGAB":
        return None
    pc = NOTE_NAMES_SHARP.index(letter)
    rest = t[1:]
    if rest[:1] == "#":
        pc, rest = pc + 1, rest[1:]
    elif rest[:1] == "b":
        pc, rest = pc - 1, rest[1:]
    rest = rest.strip()
    if rest == "M":  # 'CM' = C major in some sample-pack naming
        return pc % 12, MAJOR
    rest = rest.lower()
    if rest in ("", "maj", "major"):
        mode = MAJOR
    elif rest in ("m", "min", "minor", "-"):
        mode = MINOR
    else:
        return None
    return pc % 12, mode


ALL_KEYS: list[tuple[int, str]] = [(pc, MAJOR) for pc in range(12)] + [(pc, MINOR) for pc in range(12)]


# Chord templates used to name chord one-shots: intervals above the root.
CHORD_TEMPLATES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "mMaj7": (0, 3, 7, 11),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "add9": (0, 2, 4, 7),
    "madd9": (0, 2, 3, 7),
    "5": (0, 7),
}


def chord_name(root: int, quality: str) -> str:
    return f"{NOTE_NAMES_SHARP[root % 12]}{quality}"
