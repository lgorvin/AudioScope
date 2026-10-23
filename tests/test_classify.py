import pytest

from audioscope.engine.classify import classify, filename_type_hints
from audioscope.engine.types import AudioType
from tests import synth
from tests.conftest import to_features

A = AudioType


@pytest.mark.parametrize("name,make,expected", [
    ("house drums", lambda: synth.drum_loop(128, 4, "house"), A.DRUM_LOOP),
    ("trap drums", lambda: synth.drum_loop(140, 2, "trap"), A.DRUM_LOOP),
    ("chords", lambda: synth.chord_loop(95, 9, "minor", variant=1), A.MELODIC_LOOP),
    ("stabs", lambda: synth.chord_loop(120, 3, "major", rhythm="stabs"), A.MELODIC_LOOP),
    ("bass", lambda: synth.bass_loop(100, 7, "minor"), A.BASS_LOOP),
    ("vocal", lambda: synth.vocal_phrase(4, "major"), A.VOCAL),
    ("note", lambda: synth.one_shot_note(57), A.ONE_SHOT),
    ("808", lambda: synth.eight08(36), A.ONE_SHOT),
    ("kick", lambda: synth.kick()[:30000], A.ONE_SHOT),
    ("snare", lambda: synth.snare(), A.ONE_SHOT),
    ("chord stab", lambda: synth.chord_stab(50, "min7"), A.ONE_SHOT),
    ("song", lambda: synth.full_song(124, 9, "minor"), A.FULL_SONG),
])
def test_classification(name, make, expected):
    assert classify(to_features(make())).audio_type == expected, name


def test_filename_hints():
    h = filename_type_hints("C:/packs/Trap_Kick_03.wav")
    assert h.get(A.ONE_SHOT, 0) > 0
    h = filename_type_hints("Melodic Loop 140bpm Am.wav")
    assert h.get(A.MELODIC_LOOP, 0) > 0 and h.get(A.ONE_SHOT, 0) < 0
    assert filename_type_hints("vox_chop_hook.flac").get(A.VOCAL, 0) > 0
    assert filename_type_hints("") == {}


def test_evidence_is_explained():
    c = classify(to_features(synth.drum_loop(128, 4, "house")))
    assert c.evidence and c.metrics["strong_onsets"] >= 3
