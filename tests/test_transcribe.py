"""Bass transcription (note events) and MIDI export."""
import struct

import librosa
import numpy as np
import pytest

from audioscope.engine.midi import PPQ, notes_to_midi
from audioscope.engine.transcribe import NoteEvent, quantize, transcribe_bass
from audioscope.engine.types import AudioType, Reliability
from tests import synth
from tests.conftest import to_features

A = AudioType


def note_f1(notes, truth, tol=0.05):
    used, hits = set(), 0
    for start, _dur, midi in truth:
        for i, n in enumerate(notes):
            if i not in used and n.midi == midi and abs(n.start - start) <= tol:
                used.add(i)
                hits += 1
                break
    p, r = hits / max(len(notes), 1), hits / max(len(truth), 1)
    return 2 * p * r / max(p + r, 1e-9)


@pytest.mark.parametrize("bpm,seed", [(90, 1), (128, 2), (150, 3)])
def test_bass_loop_notes(bpm, seed):
    y, truth = synth.bass_loop(bpm, seed % 12, "minor", seed=seed, return_notes=True)
    r = transcribe_bass(to_features(y), A.BASS_LOOP)
    assert note_f1(r.notes, truth) >= 0.95
    assert r.reliability == Reliability.HIGH


def test_legato_line_and_repeated_notes():
    line = [(0, 1, 36), (1, 1, 39), (2, 0.5, 41), (2.5, 0.5, 43), (3, 1, 46), (4, 2, 34),
            (6, 0.25, 36), (6.25, 0.25, 36), (6.5, 1.5, 31)]
    y, truth = synth.bass_line(120, line)
    r = transcribe_bass(to_features(y), A.BASS_LOOP)
    assert note_f1(r.notes, truth) == 1.0
    for n, (s, d, _m) in zip(r.notes, truth):
        assert abs((n.end - n.start) - d) < 0.06  # durations survive


def test_808_pitch_drop_is_one_note_at_target_pitch():
    r = transcribe_bass(to_features(synth.eight08(31, dur=2.0, glide_semitones=12)), A.ONE_SHOT)
    assert [n.midi for n in r.notes] == [31]
    assert r.notes[0].start < 0.03 and r.notes[0].end > 1.9


def test_808_pattern_mostly_right():
    pat = [(0, 1.5, 29), (1.5, 0.5, 29), (2, 1, 32), (3, 0.75, 27), (4, 2, 34), (6, 2, 29)]
    y, truth = synth.bass_line(140, pat, style="808")
    r = transcribe_bass(to_features(y), A.BASS_LOOP)
    assert note_f1(r.notes, truth) >= 0.8


def test_detuned_line_snaps_to_notes_and_reports_tuning():
    y, truth = synth.bass_line(110, [(i * 0.5, 0.45, 40 + [0, 3, 5, 7][i % 4]) for i in range(16)])
    y = librosa.effects.pitch_shift(np.asarray(y, dtype=np.float32), sr=synth.SR, n_steps=0.3)
    r = transcribe_bass(to_features(y), A.BASS_LOOP)
    assert note_f1(r.notes, truth) >= 0.95
    assert 20 <= r.tuning_cents <= 40


def test_bass_inside_a_mix():
    yb, truth = synth.bass_loop(100, 9, "minor", seed=5, return_notes=True)
    yc = synth.chord_loop(100, 9, "minor", variant=1, with_bass=False)
    n = min(len(yb), len(yc))
    r = transcribe_bass(to_features(yb[:n] * 0.7 + yc[:n] * 0.6), A.MELODIC_LOOP)
    assert note_f1(r.notes, truth) >= 0.85
    assert r.reliability != Reliability.HIGH  # from a mix: never high


@pytest.mark.parametrize("make", [lambda: np.zeros(44100, dtype=np.float32), lambda: synth.hat(decay=0.05)])
def test_nothing_to_transcribe(make):
    r = transcribe_bass(to_features(make()), A.ONE_SHOT)
    assert r.notes == [] and r.reliability == Reliability.NONE


def test_quantize():
    notes = [NoteEvent(0.01, 0.24, 36, 100), NoteEvent(0.26, 0.49, 38, 90), NoteEvent(0.52, 0.53, 40, 80)]
    q = quantize(notes, 120)  # 16th = 0.125 s
    assert [(n.start, n.end) for n in q] == [(0.0, 0.25), (0.25, 0.5), (0.5, 0.625)]


def _parse_smf(data: bytes):
    """Tiny independent SMF reader for the test: returns (ppq, tempo_us, [(tick_on, tick_off, key, vel)])."""
    assert data[:4] == b"MThd"
    _len, fmt, ntrk, ppq = struct.unpack(">IHHH", data[4:14])
    assert (fmt, ntrk) == (0, 1) and data[14:18] == b"MTrk"
    end = 22 + struct.unpack(">I", data[18:22])[0]
    i, tick, tempo, open_, notes, status = 22, 0, None, {}, [], 0
    while i < end:
        delta = 0
        while True:
            b = data[i]
            i += 1
            delta = (delta << 7) | (b & 0x7F)
            if b < 0x80:
                break
        tick += delta
        if data[i] == 0xFF:
            kind, length = data[i + 1], data[i + 2]
            if kind == 0x51:
                tempo = int.from_bytes(data[i + 3:i + 6], "big")
            i += 3 + length
            continue
        status = data[i]
        if status & 0xF0 == 0xC0:
            i += 2
            continue
        key, vel = data[i + 1], data[i + 2]
        i += 3
        if status & 0xF0 == 0x90 and vel > 0:
            open_[key] = (tick, vel)
        else:
            on, v = open_.pop(key)
            notes.append((on, tick, key, v))
    return ppq, tempo, sorted(notes)


def test_midi_file_roundtrip():
    notes = [NoteEvent(0.0, 0.5, 36, 110), NoteEvent(0.5, 0.75, 36, 90), NoteEvent(1.0, 2.0, 43, 127)]
    ppq, tempo, parsed = _parse_smf(notes_to_midi(notes, bpm=120))
    assert ppq == PPQ and tempo == 500000
    assert parsed == [(0, 480, 36, 110), (480, 720, 36, 90), (960, 1920, 43, 127)]


def test_midi_timing_follows_tempo():
    ppq, tempo, parsed = _parse_smf(notes_to_midi([NoteEvent(0.5, 1.0, 40, 100)], bpm=90))
    assert tempo == round(60_000_000 / 90)
    assert parsed[0][:2] == (round(0.5 * PPQ * 1.5), round(1.0 * PPQ * 1.5))
