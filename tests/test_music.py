import pytest

from audioscope.engine import music
from audioscope.engine.types import KeyCandidate, format_bpm


@pytest.mark.parametrize("tonic,mode,code", [
    (0, "major", "8B"), (9, "minor", "8A"), (7, "major", "9B"), (4, "minor", "9A"), (11, "major", "1B"),
    (8, "minor", "1A"), (6, "major", "2B"), (3, "minor", "2A"), (5, "major", "7B"), (2, "minor", "7A"),
    (4, "major", "12B"), (1, "minor", "12A"), (1, "major", "3B"), (10, "minor", "3A"), (0, "minor", "5A"),
])
def test_camelot(tonic, mode, code):
    assert music.camelot(tonic, mode) == code


def test_relative_keys_share_camelot_number():
    for pc in range(12):
        rel = music.relative_key(pc, "major")
        assert music.camelot(pc, "major")[:-1] == music.camelot(*rel)[:-1]


@pytest.mark.parametrize("text,expected", [
    ("Am", (9, "minor")), ("A minor", (9, "minor")), ("F#m", (6, "minor")), ("Bbmaj", (10, "major")),
    ("Eb", (3, "major")), ("C#min", (1, "minor")), ("8A", (9, "minor")), ("12B", (4, "major")),
    ("CM", (0, "major")), ("xyz", None), ("H", None),
])
def test_parse_key(text, expected):
    assert music.parse_key(text) == expected


def test_note_names():
    assert music.midi_to_name(60) == "C4"
    assert music.midi_to_name(69) == "A4"
    assert music.midi_to_name(61, prefer_flats=True) == "Db4"
    assert abs(music.midi_to_hz(69) - 440.0) < 1e-9
    assert abs(music.hz_to_midi(261.6256) - 60) < 1e-3


def test_key_display():
    k = KeyCandidate(tonic=3, mode="major", score=0.9)
    assert k.name == "Eb major" and k.camelot == "5B" and k.short_name == "Eb"
    k = KeyCandidate(tonic=6, mode="minor", score=0.9)
    assert k.name == "F# minor" and k.camelot == "11A" and k.short_name == "F#m"


def test_format_bpm():
    assert format_bpm(128.0) == "128"
    assert format_bpm(127.96) == "128"
    assert format_bpm(87.5) == "87.5"
