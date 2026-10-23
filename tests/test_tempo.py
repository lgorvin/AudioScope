"""Tempo detection on synthetic signals with exact ground truth."""
import pytest

from audioscope.engine.tempo import estimate_tempo
from audioscope.engine.types import AudioType, Reliability
from tests import synth
from tests.conftest import to_features

A = AudioType


@pytest.mark.parametrize("style,bpm,bars", [
    ("house", 124, 4), ("house", 128, 1), ("techno", 132, 4), ("trap", 140, 4), ("trap", 150, 1),
    ("boombap", 90, 4), ("boombap", 100, 2), ("reggaeton", 95, 4), ("house", 120, 8),
])
def test_drum_loops_exact(style, bpm, bars):
    r = estimate_tempo(to_features(synth.drum_loop(bpm, bars, style)), A.DRUM_LOOP)
    assert r.bpm == bpm
    assert r.bars == bars
    assert r.reliability in (Reliability.HIGH, Reliability.MEDIUM)


@pytest.mark.parametrize("style,bpm,bars", [("dnb", 174, 4), ("hiphop16", 80, 4), ("boombap", 85, 4)])
def test_octave_ambiguous_patterns_report_both_readings(style, bpm, bars):
    """Half/double-time ambiguous grooves: either octave is acceptable, but the other must be offered."""
    r = estimate_tempo(to_features(synth.drum_loop(bpm, bars, style)), A.DRUM_LOOP)
    assert r.bpm in (bpm, bpm / 2, bpm * 2)
    offered = {c.bpm for c in r.candidates}
    assert any(abs(b - bpm) < 0.01 for b in offered)
    if r.bpm != bpm:
        assert r.octave_ambiguous and r.reliability != Reliability.HIGH


def test_half_and_double_time_always_listed():
    r = estimate_tempo(to_features(synth.drum_loop(128, 4, "house")), A.DRUM_LOOP)
    relations = {c.relation: c.bpm for c in r.candidates}
    assert relations.get("half-time") == 64.0
    assert relations.get("double-time") == 256.0


def test_one_bar_dnb_uses_whole_bar_octave():
    r = estimate_tempo(to_features(synth.drum_loop(174, 1, "dnb")), A.DRUM_LOOP)
    assert r.bpm == 174 and r.bars == 1


@pytest.mark.parametrize("bpm,tonic,mode,rhythm", [(97, 9, "minor", "stabs"), (120, 0, "major", "sustain"),
                                                    (140, 2, "minor", "stabs")])
def test_melodic_loops(bpm, tonic, mode, rhythm):
    r = estimate_tempo(to_features(synth.chord_loop(bpm, tonic, mode, rhythm=rhythm)), A.MELODIC_LOOP)
    assert r.bpm == bpm


@pytest.mark.parametrize("bpm", [92, 128])
def test_bass_loops(bpm):
    r = estimate_tempo(to_features(synth.bass_loop(bpm, 5, "minor")), A.BASS_LOOP)
    assert r.bpm == bpm


def test_loop_with_tail_falls_back_to_onsets():
    import numpy as np

    y = synth.drum_loop(126, 4, "house")
    y = np.concatenate([y, np.zeros(int(0.37 * synth.SR), dtype=np.float32)])  # irregular cut
    r = estimate_tempo(to_features(y), A.DRUM_LOOP)
    assert r.bpm is not None and abs(r.bpm - 126) < 0.6
    assert r.bars is None


@pytest.mark.parametrize("bpm,style", [(124, "house"), (140, "trap"), (92, "boombap")])
def test_full_song_sections(bpm, style):
    r = estimate_tempo(to_features(synth.full_song(bpm, 9, "minor", style=style, variant=1)), A.FULL_SONG)
    assert r.bpm == bpm
    assert r.metrics["section_agreement"] >= 0.7


@pytest.mark.parametrize("make", [
    lambda: synth.pad_loop(0, "minor", 8.0),
    lambda: synth.pad_loop(5, "major", 7.31),
    lambda: synth.white_noise(6.0),
])
def test_no_pulse_is_na(make):
    r = estimate_tempo(to_features(make()), A.MELODIC_LOOP)
    assert r.bpm is None and r.reliability == Reliability.NONE


def test_no_pulse_loop_length_only_offered_as_alternative():
    r = estimate_tempo(to_features(synth.pad_loop(0, "minor", 8.0)), A.MELODIC_LOOP)
    assert r.bpm is None
    # 8.000 s = 2 bars at 60 or 4 bars at 120 BPM; offered, never chosen
    assert sorted(c.bpm for c in r.candidates if c.relation == "loop-length") == [60.0, 120.0]


def test_one_shot_has_no_tempo():
    r = estimate_tempo(to_features(synth.one_shot_note(60)), A.ONE_SHOT)
    assert r.bpm is None
