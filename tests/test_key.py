"""Key detection on synthetic harmonic material with exact ground truth."""
import numpy as np
import pytest

from audioscope.engine import music
from audioscope.engine.key import estimate_key, key_scores
from audioscope.engine.types import AudioType, Reliability
from tests import synth
from tests.conftest import to_features

A = AudioType


def test_profiles_recover_scale_tonic():
    for tonic in range(12):
        for mode, scale in (("major", synth.MAJOR_SCALE), ("minor", synth.MINOR_SCALE)):
            pcp = np.zeros(12)
            for i, deg in enumerate(scale):
                pcp[(tonic + deg) % 12] = 1.0 + (1.0 if i in (0, 4) else 0) + (0.5 if i == 2 else 0)
            best = key_scores(pcp)[0]
            assert (best.tonic, best.mode) == (tonic, mode)


@pytest.mark.parametrize("tonic", range(12))
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_cadential_chord_loops_all_keys(tonic, mode):
    """I-IV-V-I / i-iv-V-i loops: unambiguous tonal centre, detuned by up to +-30 cents."""
    y = synth.chord_loop(100 + 3 * tonic, tonic, mode, variant=1, seed=tonic, detune_cents=(tonic - 6) * 5)
    r = estimate_key(to_features(y), A.MELODIC_LOOP)
    assert r.key is not None
    assert (r.key.tonic, r.key.mode) == (tonic, mode)
    assert abs(r.tuning_cents - (tonic - 6) * 5) <= 6


@pytest.mark.parametrize("tonic,mode", [(9, "minor"), (2, "minor"), (0, "major"), (7, "major")])
def test_axis_progression_never_overconfident(tonic, mode):
    """vi-IV-I-V vs I-V-vi-IV use identical chords: the tonal centre is genuinely ambiguous.

    Required behaviour: the labelled key wins, or its relative wins while the labelled key is
    the first alternative and the result is not rated high.
    """
    y = synth.chord_loop(110, tonic, mode, variant=0, seed=1)
    r = estimate_key(to_features(y), A.MELODIC_LOOP)
    assert r.key is not None
    if (r.key.tonic, r.key.mode) != (tonic, mode):
        assert (r.key.tonic, r.key.mode) == music.relative_key(tonic, mode)
        assert (r.candidates[1].tonic, r.candidates[1].mode) == (tonic, mode)
        assert r.reliability != Reliability.HIGH


@pytest.mark.parametrize("make,atype", [
    (lambda: synth.white_noise(), A.MELODIC_LOOP),
    (lambda: synth.drum_loop(128, 4, "house"), A.DRUM_LOOP),
    (lambda: synth.one_shot_note(57), A.ONE_SHOT),
    (lambda: synth.one_shot_note(48) + synth.one_shot_note(55), A.ONE_SHOT),  # bare fifth
])
def test_key_na_when_undefined(make, atype):
    r = estimate_key(to_features(make()), atype)
    assert r.key is None and r.reliability == Reliability.NONE
    assert r.evidence


def test_pad_minor():
    r = estimate_key(to_features(synth.pad_loop(0, "minor")), A.MELODIC_LOOP)
    assert r.key is not None and r.key.name == "C minor"


@pytest.mark.parametrize("tonic,mode", [(9, "minor"), (2, "major")])
def test_full_song_key(tonic, mode):
    r = estimate_key(to_features(synth.full_song(110, tonic, mode, variant=1)), A.FULL_SONG)
    assert r.key is not None and (r.key.tonic, r.key.mode) == (tonic, mode)
    assert "section_agreement" in r.metrics


def test_alternatives_and_camelot_present():
    r = estimate_key(to_features(synth.chord_loop(100, 9, "minor", variant=1)), A.MELODIC_LOOP)
    assert len(r.candidates) >= 3
    assert r.key.camelot == "8A"
    assert r.candidates[0].score >= r.candidates[1].score - 0.11  # tie-break may promote a near-equal key
    assert music.camelot(r.candidates[1].tonic, r.candidates[1].mode)
