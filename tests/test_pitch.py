"""Root-note detection for one-shots, 808s, chords and monophonic loops."""
import pytest

from audioscope.engine.pitch import estimate_pitch
from audioscope.engine.types import AudioType, Reliability
from tests import synth
from tests.conftest import to_features

A = AudioType


@pytest.mark.parametrize("midi", [28, 33, 40, 45, 52, 57, 64, 69, 76, 84])
def test_one_shot_fundamental(midi):
    r, _ = estimate_pitch(to_features(synth.one_shot_note(midi)), A.ONE_SHOT)
    assert r.root_midi == midi
    assert r.root_kind == "fundamental"
    assert abs(r.root_cents) < 8


@pytest.mark.parametrize("cents", [-40, -20, 25, 45])
def test_detuned_one_shot_reports_cents(cents):
    r, _ = estimate_pitch(to_features(synth.one_shot_note(50, detune_cents=cents)), A.ONE_SHOT)
    assert r.root_midi == 50
    assert abs(r.root_cents - cents) < 6


@pytest.mark.parametrize("midi", [29, 33, 36, 41])
def test_808_with_pitch_drop(midi):
    """The settled pitch, not the initial pitch drop, decides the note."""
    r, _ = estimate_pitch(to_features(synth.eight08(midi, glide_semitones=12)), A.ONE_SHOT)
    assert r.root_midi == midi


@pytest.mark.parametrize("root,quality,name", [(53, "maj", "F"), (57, "min", "Am"), (50, "min7", "Dm7"),
                                                (48, "maj7", "Cmaj7")])
def test_chord_stab(root, quality, name):
    r, _ = estimate_pitch(to_features(synth.chord_stab(root, quality)), A.ONE_SHOT)
    assert r.chord == name
    assert r.root_midi % 12 == root % 12
    assert r.root_kind == "chord root"


@pytest.mark.parametrize("make", [lambda: synth.hat(decay=0.05), lambda: synth.noise_burst()])
def test_noise_one_shots_have_no_pitch(make):
    r, _ = estimate_pitch(to_features(make()), A.ONE_SHOT)
    assert r.root_midi is None and r.reliability == Reliability.NONE


def test_bass_loop_most_frequent_note_and_pcp():
    r, pcp = estimate_pitch(to_features(synth.bass_loop(100, 7, "minor", variant=1)), A.BASS_LOOP)
    assert r.root_midi % 12 == 7
    assert pcp is not None and pcp.argmax() == 7


def test_melodic_loop_bass_note():
    r, _ = estimate_pitch(to_features(synth.chord_loop(100, 9, "minor", variant=1)), A.MELODIC_LOOP)
    assert r.root_midi % 12 == 9
    assert r.root_kind == "most prominent bass note"


def test_vocal_phrase_tonic_is_most_frequent():
    r, pcp = estimate_pitch(to_features(synth.vocal_phrase(4, "major")), A.VOCAL)
    assert r.root_midi % 12 == 4
