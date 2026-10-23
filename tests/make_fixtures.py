"""Render the reference fixtures in tests/fixtures/ (deterministic) and their manifest.csv.

The manifest uses the same schema as eval/evaluate.py --labels, so the fixtures double
as a small labelled dataset:  python eval/evaluate.py --dataset tests/fixtures --labels tests/fixtures/manifest.csv

Run:  python tests/make_fixtures.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import synth  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# file, category, type, bpm, key, root, generator
FIXTURES = [
    ("drums_house_124_4bars.wav", "loop", "drum_loop", "124", "", "", lambda: synth.drum_loop(124, 4, "house", seed=1)),
    ("drums_trap_140_2bars.flac", "loop", "drum_loop", "140", "", "", lambda: synth.drum_loop(140, 2, "trap", seed=2)),
    ("drums_boombap_92_4bars.mp3", "loop", "drum_loop", "92", "", "", lambda: synth.drum_loop(92, 4, "boombap", seed=3)),
    ("keys_Am_100_4bars.wav", "loop", "melodic_loop", "100", "A minor", "",
     lambda: synth.chord_loop(100, 9, "minor", variant=1, seed=4)),
    ("stabs_Eb_128_4bars.flac", "loop", "melodic_loop", "128", "Eb major", "",
     lambda: synth.chord_loop(128, 3, "major", rhythm="stabs", variant=1, seed=5, detune_cents=-20)),
    ("keys_F#m_90_4bars.mp3", "loop", "melodic_loop", "90", "F# minor", "",
     lambda: synth.chord_loop(90, 6, "minor", variant=1, seed=6)),
    ("bass_Gm_100_4bars.wav", "loop", "bass_loop", "100", "G minor", "", lambda: synth.bass_loop(100, 7, "minor", variant=1)),
    ("pad_Cm_no_pulse.flac", "loop", "melodic_loop", "none", "C minor", "", lambda: synth.pad_loop(0, "minor", 8.0)),
    ("vocal_E_phrase.wav", "loop", "vocal", "none", "E major", "", lambda: synth.vocal_phrase(4, "major", seed=0)),
    ("oneshot_note_A2.wav", "oneshot", "one_shot", "none", "none", "A2", lambda: synth.one_shot_note(45)),
    ("oneshot_note_D4_plus30c.flac", "oneshot", "one_shot", "none", "none", "D4",
     lambda: synth.one_shot_note(62, detune_cents=30)),
    ("oneshot_808_F1.wav", "oneshot", "one_shot", "none", "none", "F1", lambda: synth.eight08(29, dur=2.5)),
    ("oneshot_stab_Dm7.mp3", "oneshot", "one_shot", "none", "", "D", lambda: synth.chord_stab(50, "min7")),
    ("oneshot_hat.wav", "oneshot", "one_shot", "none", "none", "none", lambda: synth.hat(decay=0.06)),
    ("song_house_124_Am.mp3", "song", "full_song", "124", "A minor", "",
     lambda: synth.full_song(124, 9, "minor", style="house", variant=1)),
    ("song_boombap_92_D.flac", "song", "full_song", "92", "D major", "",
     lambda: synth.full_song(92, 2, "major", style="boombap", variant=1)),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, cat, typ, bpm, key, root, make in FIXTURES:
        y = np.asarray(make(), dtype=np.float32)
        ext = os.path.splitext(name)[1][1:].upper()
        kw = {"format": ext}
        if ext == "WAV":
            kw["subtype"] = "PCM_16"
        sf.write(os.path.join(OUT, name), y, synth.SR, **kw)
        rows.append({"file": name, "category": cat, "type": typ, "bpm": bpm, "key": key, "root": root})
        print("wrote", name)
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "category", "type", "bpm", "key", "root"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
