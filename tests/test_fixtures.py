"""End-to-end regression test on the rendered reference fixtures (WAV / FLAC / MP3).

Regenerate with ``python tests/make_fixtures.py``.
"""
import csv
import os

import librosa
import pytest

from audioscope.engine import analyze_file, music
from audioscope.engine.types import Reliability

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MANIFEST = os.path.join(HERE, "manifest.csv")


def _rows():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not _rows(), reason="fixtures not generated (python tests/make_fixtures.py)")
@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["file"])
def test_fixture(row):
    r = analyze_file(os.path.join(HERE, row["file"]))
    assert r.classification.audio_type.value == row["type"]

    if row["bpm"] == "none":
        assert r.tempo.bpm is None and r.tempo.reliability == Reliability.NONE
    elif row["bpm"]:
        assert r.tempo.bpm is not None and abs(r.tempo.bpm - float(row["bpm"])) <= 0.5

    if row["key"] == "none":
        assert r.key.key is None
    elif row["key"]:
        tonic, mode = music.parse_key(row["key"])
        assert r.key.key is not None and (r.key.key.tonic, r.key.key.mode) == (tonic, mode)
        assert r.key.key.camelot == music.camelot(tonic, mode)

    if row["root"] == "none":
        assert r.pitch.root_midi is None
    elif row["root"]:
        if row["root"][-1].isdigit():
            assert r.pitch.root_midi == int(round(librosa.note_to_midi(row["root"])))
        else:
            assert r.pitch.root_midi % 12 == int(librosa.note_to_midi(row["root"] + "4")) % 12
