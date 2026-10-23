import json
import os

import numpy as np
import pytest
import soundfile as sf

from audioscope.engine import AnalysisResult, AudioType, analyze_file, parse_filename_hints
from audioscope.engine.io import AudioLoadError, load_audio
from tests import synth


@pytest.mark.parametrize("name,bpm,key", [
    ("Loop_128bpm_F#min.wav", 128.0, "F# minor"),
    ("140 BPM - Am - Dark Keys.wav", 140.0, "A minor"),
    ("PL_Bass_Eb_major_92bpm.flac", 92.0, "Eb major"),
    ("Kick_01.wav", None, None),
    ("Take A 2.wav", None, None),
    ("chord_C#m.wav", None, "C# minor"),
])
def test_filename_hints(name, bpm, key):
    h = parse_filename_hints(name)
    assert h.get("bpm") == bpm
    assert h.get("key") == key


def test_load_formats(tmp_path):
    y = synth.chord_loop(120, 0, "major", bars=1)
    for ext, kw in ((".wav", {}), (".flac", {}), (".mp3", {"format": "MP3"})):
        p = tmp_path / f"t{ext}"
        sf.write(p, y, synth.SR, **kw)
        a = load_audio(str(p))
        assert a.sr == 22050 and a.native_sr == synth.SR
        assert abs(a.duration - len(y) / synth.SR) < 0.1


def test_load_errors(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(str(tmp_path / "missing.wav"))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(AudioLoadError):
        load_audio(str(bad))


def test_analyze_and_roundtrip(tmp_path):
    p = tmp_path / "loop.wav"
    sf.write(p, synth.drum_loop(128, 4, "house"), synth.SR)
    r = analyze_file(str(p))
    assert r.tempo.bpm == 128
    d = r.to_dict()
    json.dumps(d)  # JSON-serialisable
    r2 = AnalysisResult.from_dict(d)
    assert r2.tempo.bpm == r.tempo.bpm and r2.classification.audio_type == r.classification.audio_type


def test_manual_override(tmp_path):
    p = tmp_path / "x.wav"
    sf.write(p, synth.chord_loop(100, 9, "minor", variant=1), synth.SR)
    r = analyze_file(str(p), audio_type=AudioType.ONE_SHOT)
    assert r.classification.manual and r.classification.audio_type == AudioType.ONE_SHOT
    assert r.tempo.bpm is None  # one-shot strategy: no tempo


def test_single_note_root_not_key(tmp_path):
    p = tmp_path / "note.wav"
    sf.write(p, synth.one_shot_note(45), synth.SR)
    r = analyze_file(str(p))
    assert r.pitch.root_name == "A2"
    assert r.key.key is None
    assert any("root note" in e for e in r.key.evidence)


def test_silence(tmp_path):
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros(44100 * 2, dtype=np.float32), synth.SR)
    r = analyze_file(str(p))
    assert r.tempo.bpm is None and r.key.key is None and r.pitch.root_midi is None
    assert "File is silent" in r.warnings
