"""Synthetic test-signal generators with known ground truth.

Everything is deterministic for a given ``seed``. Loops are rendered to an
exact number of bars and wrap their tails around to the start, like loops
exported from a DAW.
"""
from __future__ import annotations

import numpy as np

SR = 44100

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- instruments
def kick(sr=SR, f0=150.0, f1=48.0, decay=0.35, rng=None):
    n = int(sr * (decay * 2.5))
    t = np.arange(n) / sr
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.sin(phase) * np.exp(-t / decay)
    click = np.exp(-t / 0.002) * 0.5
    return (y + click) * 0.9


def snare(sr=SR, decay=0.15, rng=None):
    rng = rng or np.random.default_rng(0)
    n = int(sr * decay * 3)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0)  # brighter
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05)
    return (0.35 * noise * np.exp(-t / decay) + 0.5 * tone) * 0.7


def hat(sr=SR, decay=0.04, rng=None):
    rng = rng or np.random.default_rng(1)
    n = int(sr * decay * 4)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    for _ in range(2):
        noise = np.diff(noise, prepend=0)
    return 0.12 * noise * np.exp(-t / decay)


def tone(midi, dur, sr=SR, harmonics=8, attack=0.01, release=0.08, decay_rate=0.0, detune_cents=0.0, vibrato=0.0):
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = midi_hz(midi + detune_cents / 100.0)
    inst = f * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(inst) / sr
    y = np.zeros(n)
    for h in range(1, harmonics + 1):
        if f * h < sr / 2 * 0.9:
            y += np.sin(h * phase) / h
    env = np.ones(n)
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    env[:a] = np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r)
    if decay_rate:
        env *= np.exp(-t * decay_rate)
    return y * env


def eight08(midi, dur=2.0, sr=SR, glide_semitones=12.0, glide_time=0.04, harmonics=3):
    n = int(sr * dur)
    t = np.arange(n) / sr
    m = midi + glide_semitones * np.exp(-t / glide_time)
    f = midi_hz(0) * 2 ** (m / 12)
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    env = np.exp(-t / (dur / 3))
    env[: int(0.003 * sr)] *= np.linspace(0, 1, int(0.003 * sr))
    return 0.8 * np.tanh(1.5 * y * env)


# ---------------------------------------------------------------- helpers
def _place(buf, x, pos, wrap=True):
    n = len(buf)
    end = pos + len(x)
    if end <= n:
        buf[pos:end] += x
    else:
        k = n - pos
        buf[pos:] += x[:k]
        if wrap:
            rest = x[k:]
            while len(rest):
                m = min(len(rest), n)
                buf[:m] += rest[:m]
                rest = rest[m:]
    return buf


def _norm(y, peak=0.8):
    m = np.max(np.abs(y))
    return (y / m * peak).astype(np.float32) if m > 0 else y.astype(np.float32)


def loop_len(bpm, bars, sr=SR):
    return int(round(bars * 4 * 60.0 / bpm * sr))


# ---------------------------------------------------------------- drum patterns
# 16 steps per bar. K kick, S snare, H closed hat, O open hat.
DRUM_PATTERNS = {
    "house": {"K": [0, 4, 8, 12], "S": [4, 12], "H": [2, 6, 10, 14], "O": []},
    "trap": {"K": [0, 7, 10], "S": [8], "H": list(range(0, 16, 2)) + [13, 15], "O": []},
    "boombap": {"K": [0, 7, 10], "S": [4, 12], "H": list(range(0, 16, 2)), "O": []},
    "dnb": {"K": [0, 10], "S": [4, 12], "H": list(range(0, 16, 2)), "O": []},
    "techno": {"K": [0, 4, 8, 12], "S": [], "H": [2, 6, 10, 14], "O": [2, 6, 10, 14]},
    "reggaeton": {"K": [0, 4, 8, 12], "S": [3, 6, 11, 14], "H": [], "O": []},
    "hiphop16": {"K": [0, 3, 8, 11], "S": [4, 12], "H": list(range(16)), "O": []},
}


def drum_loop(bpm, bars=4, style="house", sr=SR, seed=0, swing=0.0, humanize_ms=0.0):
    rng = np.random.default_rng(seed)
    n = loop_len(bpm, bars, sr)
    buf = np.zeros(n)
    step = 60.0 / bpm / 4
    pat = DRUM_PATTERNS[style]
    k, s, h, o = kick(sr), snare(sr, rng=rng), hat(sr, rng=rng), hat(sr, decay=0.15, rng=rng)
    for bar in range(bars):
        for name, sample, gain in (("K", k, 1.0), ("S", s, 0.9), ("H", h, 0.8), ("O", o, 0.5)):
            for st in pat[name]:
                t = (bar * 16 + st) * step
                if swing and st % 2 == 1:
                    t += swing * step
                if humanize_ms:
                    t += rng.normal(0, humanize_ms / 1000)
                pos = int(round(t * sr)) % n
                _place(buf, sample * gain * (0.9 + 0.1 * rng.random()), pos)
    return _norm(buf)


# ---------------------------------------------------------------- harmonic material
PROGRESSIONS = {
    # (root offset from tonic, quality)
    "major": [[(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],      # I V vi IV
              [(0, "maj"), (5, "maj"), (7, "maj"), (0, "maj")]],     # I IV V I
    "minor": [[(0, "min"), (8, "maj"), (3, "maj"), (10, "maj")],     # i VI III VII (same chords as relative major)
              [(0, "min"), (5, "min"), (7, "maj"), (0, "min")]],     # i iv V i (harmonic minor)
}


def progression(tonic, mode, variant=0):
    """Diatonic progression as (root_pc_offset, quality) relative to tonic."""
    return PROGRESSIONS[mode][variant % 2]


def chord_notes(root_midi, quality):
    third = 4 if quality == "maj" else 3
    return [root_midi, root_midi + third, root_midi + 7]


def chord_loop(bpm, tonic, mode, bars=4, sr=SR, harmonics=6, with_bass=True, rhythm="sustain", seed=0,
               detune_cents=0.0, with_melody=True, variant=0):
    rng = np.random.default_rng(seed)
    n = loop_len(bpm, bars, sr)
    buf = np.zeros(n)
    beat = 60.0 / bpm
    prog = progression(tonic, mode, variant)
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    for bar in range(bars):
        off, q = prog[bar % len(prog)]
        root = 48 + (tonic + off) % 12  # C3..B3
        notes = chord_notes(root + 12, q)  # C4 octave voicing
        if rhythm == "sustain":
            hits = [(0.0, 4.0)]
        else:  # stabs on beats 1, 2&, 4 (syncopated)
            hits = [(0.0, 0.5), (1.5, 0.5), (3.0, 0.5)]
        for start, length in hits:
            t0 = (bar * 4 + start) * beat
            for m in notes:
                x = tone(m, length * beat, sr, harmonics=harmonics, detune_cents=detune_cents) * 0.25
                _place(buf, x, int(round(t0 * sr)) % n)
        if with_bass:
            for b8 in range(8):
                if b8 in (0, 3, 4, 6):
                    t0 = (bar * 4 + b8 * 0.5) * beat
                    x = tone(root - 12, 0.45 * beat, sr, harmonics=4, detune_cents=detune_cents) * 0.5
                    _place(buf, x, int(round(t0 * sr)) % n)
        if with_melody:
            for b in range(4):
                deg = int(rng.integers(0, 7)) if b else 0
                m = 72 + tonic % 12 + scale[deg] if b else 72 + (tonic + off) % 12
                t0 = (bar * 4 + b) * beat
                x = tone(m, 0.9 * beat, sr, harmonics=4, detune_cents=detune_cents) * 0.15
                _place(buf, x, int(round(t0 * sr)) % n)
    return _norm(buf)


def bass_loop(bpm, tonic, mode, bars=4, sr=SR, seed=0, variant=0):
    rng = np.random.default_rng(seed)
    n = loop_len(bpm, bars, sr)
    buf = np.zeros(n)
    beat = 60.0 / bpm
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    prog = progression(tonic, mode, variant)
    for bar in range(bars):
        off, _ = prog[bar % len(prog)]
        root = 36 + (tonic + off) % 12
        for e in range(8):
            if e in (0, 2, 3, 5, 6):
                m = root if e in (0, 3, 6) else 36 + (tonic + scale[int(rng.integers(0, 7))]) % 12
                if e == 0 and bar == 0:
                    m = 36 + tonic % 12
                x = tone(m, 0.48 * beat, sr, harmonics=5, attack=0.005, release=0.03) * 0.6
                _place(buf, x, int(round((bar * 4 + e * 0.5) * beat * sr)) % n)
    return _norm(buf)


def pad_loop(tonic, mode, seconds=8.0, sr=SR):
    """Slowly evolving pad: no rhythmic pulse at all."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    third = 4 if mode == "major" else 3
    y = np.zeros(n)
    for m in (48 + tonic, 60 + tonic, 60 + tonic + third, 60 + tonic + 7, 72 + tonic):
        for d in (-6, 6):
            y += tone(m, seconds, sr, harmonics=5, attack=1.5, release=1.5, detune_cents=d) * 0.2
    y *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.09 * t)
    return _norm(y)


def vocal_phrase(tonic, mode, seconds=6.0, sr=SR, seed=0):
    """Monophonic, vibrato-laden, glide-connected melody: vocal-like pitch behaviour."""
    rng = np.random.default_rng(seed)
    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    n = int(seconds * sr)
    notes = []
    t = 0.0
    seq = [0, 2, 4, 2, 0, 4, 3, 1, 0]
    while t < seconds:
        deg = seq[len(notes) % len(seq)] if rng.random() < 0.8 else int(rng.integers(0, 7))
        d = float(rng.uniform(0.35, 0.8))
        notes.append((t, d, 60 + tonic % 12 + scale[deg]))
        t += d
    pitch = np.zeros(n)
    amp = np.zeros(n)
    for start, d, m in notes:
        a, b = int(start * sr), min(n, int((start + d) * sr))
        pitch[a:b] = m
        amp[a:b] = 1.0
    # portamento + vibrato + slight scoop
    k = int(0.06 * sr)
    pitch = np.convolve(np.pad(pitch, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    tt = np.arange(n) / sr
    pitch = pitch + 0.35 * np.sin(2 * np.pi * 5.2 * tt) + rng.normal(0, 0.05, n).cumsum() * 0.0005
    f = 440 * 2 ** ((pitch - 69) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros(n)
    for h, g in zip(range(1, 12), (1, .6, .5, .35, .3, .2, .15, .1, .08, .05, .04)):
        y += g * np.sin(h * ph)
    gap = np.convolve(amp, np.ones(int(0.03 * sr)) / int(0.03 * sr), mode="same")
    noise = rng.standard_normal(n) * 0.01
    return _norm(y * gap + noise)


def one_shot_note(midi, dur=1.5, sr=SR, harmonics=8, detune_cents=0.0, decay_rate=2.0):
    y = tone(midi, dur, sr, harmonics=harmonics, attack=0.003, release=0.2, detune_cents=detune_cents, decay_rate=decay_rate)
    return _norm(y)


def chord_stab(root_midi, quality="maj", dur=1.2, sr=SR):
    ivs = {"maj": (0, 4, 7), "min": (0, 3, 7), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10)}[quality]
    y = sum(tone(root_midi + i, dur, sr, harmonics=6, attack=0.003, release=0.3, decay_rate=2.5) for i in ivs)
    return _norm(y)


def noise_burst(dur=0.3, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    return _norm(rng.standard_normal(n) * np.exp(-t / (dur / 5)))


def white_noise(seconds=6.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    return _norm(rng.standard_normal(int(seconds * sr)) * 0.3)


def full_song(bpm, tonic, mode, sr=SR, style="house", seed=0, variant=0):
    """~ intro (chords) / verse (drums+chords) / break (chords) / drop (all)."""
    sections = [("intro", 8, False), ("main", 16, True), ("break", 8, False), ("drop", 16, True), ("outro", 4, True)]
    parts = []
    for i, (_, bars, drums) in enumerate(sections):
        harm = chord_loop(bpm, tonic, mode, bars=bars, sr=sr, seed=seed + i, rhythm="stabs" if drums else "sustain",
                          with_bass=drums, variant=variant)
        if drums:
            d = drum_loop(bpm, bars=bars, style=style, sr=sr, seed=seed + i)
            n = min(len(d), len(harm))
            harm = harm[:n] * 0.6 + d[:n] * 0.7
        parts.append(harm)
    return _norm(np.concatenate(parts))
