"""Bass-line transcription: note events (start, end, pitch, velocity) for MIDI export.

Method
------
1. Low-pass the signal to the bass register so the lowest line dominates. For
   mixes (songs, melodic and drum loops) the harmonic component is used, the
   cut-off is lower and quiet frames are ignored, so chords and drums leak less.
2. pYIN f0 tracking (30-420 Hz, ~12 ms frames), corrected by the file's tuning
   offset so detuned samples still land on the right notes.
3. Note boundaries:
   * re-attacks: rises in loudness of >= 5 dB (energy onsets). Loudness rather
     than spectral change is used, because an 808's pitch drop changes the
     spectrum without being a new note;
   * legato pitch changes: a new *steady* pitch held for >= 45 ms without an attack.
4. Segments that contain no steady pitch (pitch drops, slides, scoops) are folded
   into the note they lead into; fragments shorter than 60 ms are merged into a
   neighbour. Each note's pitch is the loudness/voicing-weighted vote of its
   steady frames; velocity follows the attack level (relative dB -> 30-127).

Best on isolated bass, 808s and bass loops. In mixes the lowest voice is usually
the bass, but kicks and low pads can intrude; reliability says so.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import music
from .features import Features
from .types import AudioType, Reliability

HOP = 256
PYIN_FRAME = 2048  # 93 ms: >2 periods of a 30 Hz sub, short enough not to smear 808 pitch drops
FMIN = 30.0
MIN_NOTE_S = 0.06
STEADY_RUN_S = 0.045
ONSET_RISE_DB = 5.0
SILENCE_GAP_S = 0.04
MIX_TYPES = (AudioType.FULL_SONG, AudioType.DRUM_LOOP, AudioType.MELODIC_LOOP)
# (low-pass Hz, pYIN fmax Hz, loudness gate dB re: loudest frame)
ISOLATED = (400.0, 420.0, -30.0)
MIXED = (250.0, 280.0, -22.0)


@dataclass
class NoteEvent:
    start: float  # seconds
    end: float
    midi: int
    velocity: int  # 1..127
    cents: float = 0.0  # mean deviation of the played pitch from the note (after tuning correction)

    @property
    def name(self) -> str:
        return music.midi_to_name(self.midi)


@dataclass
class BassTranscription:
    notes: list[NoteEvent] = field(default_factory=list)
    tuning_cents: float = 0.0
    reliability: Reliability = Reliability.NONE
    evidence: list[str] = field(default_factory=list)
    duration: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reliability"] = self.reliability.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "BassTranscription":
        return BassTranscription(notes=[NoteEvent(**n) for n in d["notes"]], tuning_cents=d["tuning_cents"],
                                 reliability=Reliability(d["reliability"]), evidence=list(d["evidence"]),
                                 duration=d["duration"], metrics=dict(d.get("metrics", {})))

    @property
    def range_text(self) -> str:
        if not self.notes:
            return ""
        lo = min(n.midi for n in self.notes)
        hi = max(n.midi for n in self.notes)
        return music.midi_to_name(lo) if lo == hi else f"{music.midi_to_name(lo)}-{music.midi_to_name(hi)}"


def _lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(6, cutoff, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _tuning_from_frames(midi: np.ndarray, weight: np.ndarray) -> float:
    """Weighted median offset (semitones) of steady pitch frames from the equal-tempered grid."""
    dev = midi - np.round(midi)
    order = np.argsort(dev)
    cdf = np.cumsum(weight[order]) / (weight.sum() + 1e-12)
    return float(dev[order][min(np.searchsorted(cdf, 0.5), len(dev) - 1)])


def _glide_target(track: np.ndarray) -> int | None:
    """Target note of a decaying (exponential) glide such as an 808 pitch drop, else None."""
    p = track[np.isfinite(track)][-6:]
    if p.size < 4:
        return None
    if np.ptp(p) < 0.5:  # not gliding, just short
        return int(round(float(np.median(p))))
    d = np.diff(p)
    if not (np.all(d < 0) or np.all(d > 0)):
        return None
    r = float(np.median(d[1:] / d[:-1]))
    if not 0.3 < r < 0.95:
        return None
    return int(round(p[-1] + d[-1] * r / (1 - r)))


def _energy_onsets(db: np.ndarray, gate_db: float) -> list[int]:
    """Frames where loudness rises >= ONSET_RISE_DB within 3 frames; placed at the start of the rise."""
    n = len(db)
    rise = np.zeros(n)
    for k in range(1, n):
        rise[k] = db[k] - db[max(0, k - 3):k].min()
    peaks = [k for k in range(1, n - 1) if rise[k] >= ONSET_RISE_DB and rise[k] >= rise[k - 1]
             and rise[k] > rise[k + 1] and db[k] > gate_db]
    out: list[int] = []
    for k in peaks:
        j = k
        while j > 0 and k - j < 3 and db[j - 1] < db[j]:
            j -= 1
        if out and j - out[-1] < 3:
            continue
        out.append(j)
    return out


def transcribe_bass(feat: Features, audio_type: AudioType | None = None) -> BassTranscription:
    res = BassTranscription(duration=feat.duration)
    if feat.is_silent or feat.duration < 0.08:
        res.evidence.append("Silent or too short")
        return res
    mix = audio_type in MIX_TYPES
    cutoff, fmax, gate_db = MIXED if mix else ISOLATED
    y = _lowpass(np.asarray(feat.y_harmonic if mix else feat.y, dtype=np.float32), feat.sr, cutoff)
    if len(y) < 4096:
        y = np.pad(y, (0, 4096 - len(y)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, prob = librosa.pyin(y, fmin=FMIN, fmax=fmax, sr=feat.sr, frame_length=PYIN_FRAME, hop_length=HOP,
                                        fill_na=np.nan)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, prob, rms = f0[:n], voiced[:n], np.nan_to_num(prob[:n]), rms[:n]
    if rms.max() <= 1e-6:
        res.evidence.append("No energy in the bass register")
        return res
    db = np.maximum(20 * np.log10(rms / rms.max() + 1e-9), -80.0)
    fps = feat.sr / HOP
    raw = np.where(voiced, librosa.hz_to_midi(np.where(voiced, f0, 1.0)), np.nan)
    active = voiced & (db > gate_db) & np.isfinite(raw)
    if active.sum() < 3:
        res.evidence.append("No stable pitch in the bass register")
        return res
    # steady = pitch moving less than 0.15 semitone per 10 ms (glides and drops are not steady)
    dm = np.abs(np.gradient(np.nan_to_num(raw, nan=0.0))) * (0.01 * fps)
    steady = active & (dm < 0.15)
    # tuning from the line itself (a generic estimate is biased by 808 pitch drops)
    tuning = _tuning_from_frames(raw[steady], (rms * prob)[steady]) if steady.sum() >= 5 else 0.0
    res.tuning_cents = round(tuning * 100.0, 1)
    midi = raw - tuning
    # settled = steady and close to a semitone: the slow tail of a glide (still ~0.5 st off) is excluded
    steady &= np.abs(midi - np.round(np.nan_to_num(midi))) < 0.35
    label = np.where(steady, np.round(np.nan_to_num(midi)), -1).astype(int)

    # ---- boundaries ----------------------------------------------------------------
    onsets = set(_energy_onsets(db, gate_db))
    run_min = max(2, int(round(STEADY_RUN_S * fps)))
    first_active = int(np.argmax(active))
    if not any(abs(o - first_active) <= run_min for o in onsets):
        onsets.add(first_active)
    boundaries = set(onsets)
    prev_label, prev_end, i = None, 0, 0
    while i < n:
        if label[i] < 0:
            i += 1
            continue
        j = i
        while j < n and label[j] == label[i]:
            j += 1
        if j - i >= run_min:
            # legato change: a new steady pitch with no attack anywhere since the previous steady pitch
            if prev_label is not None and label[i] != prev_label \
                    and not any(prev_end - run_min <= o <= i + run_min for o in onsets):
                k = i  # the new note starts where the pitch left the old one
                while k > prev_end and label[k - 1] != prev_label and active[k - 1]:
                    k -= 1
                boundaries.add(k)
            prev_label, prev_end = label[i], j
        i = j
    edges = sorted(b for b in boundaries if 0 <= b < n) + [n]

    # ---- segments -> notes ------------------------------------------------------------
    weight = rms * prob
    gap = max(2, int(round(SILENCE_GAP_S * fps)))
    segs = []  # [start_frame, end_frame(excl), pitch | None, attack_level, is_onset]
    for a, b in zip(edges[:-1], edges[1:]):
        act = np.arange(a, b)[active[a:b]]
        if act.size == 0:
            continue
        first, last = int(act[0]), int(act[-1])
        run = 0
        for g in range(first, last + 1):  # >= 40 ms of silence ends the note
            run = 0 if active[g] else run + 1
            if run >= gap:
                last = g - run
                break
        st = [k for k in range(first, last + 1) if steady[k]]
        pitch = None
        if len(st) >= 3:
            votes: dict[int, float] = {}
            for k in st:
                votes[label[k]] = votes.get(label[k], 0.0) + weight[k]
            pitch = max(votes, key=votes.get)
        else:  # a short note that never settles (808 drop): extrapolate the glide to its target
            pitch = _glide_target(midi[first:last + 1])
        start = a if a in onsets else first
        segs.append([start, last + 1, pitch, float(rms[first:min(first + int(0.06 * fps) + 1, n)].max()), a in onsets])

    # glide-only segments lead into the next note (808 drops, scoops); otherwise they are dropped
    out: list[list] = []
    for idx, s in enumerate(segs):
        if s[2] is None:
            nxt = segs[idx + 1] if idx + 1 < len(segs) else None
            if nxt is not None and nxt[0] - s[1] <= gap:
                nxt[0], nxt[3], nxt[4] = s[0], max(s[3], nxt[3]), s[4] or nxt[4]
            continue
        out.append(s)
    # merge very short fragments and same-pitch continuations without a new attack
    merged: list[list] = []
    for s in out:
        if merged:
            p = merged[-1]
            touching = s[0] - p[1] <= gap
            if touching and (s[2] == p[2] and not s[4] or (s[1] - s[0]) / fps < MIN_NOTE_S and s[2] == p[2]):
                p[1] = s[1]
                continue
            if touching and (p[1] - p[0]) / fps < MIN_NOTE_S:  # previous fragment too short: absorb into this note
                s[0], s[3] = p[0], max(p[3], s[3])
                merged[-1] = s
                continue
        merged.append(s)
    merged = [s for s in merged if (s[1] - s[0]) / fps >= MIN_NOTE_S]

    if merged:
        peak = max(s[3] for s in merged)
        for s in merged:
            on = [k for k in range(s[0], s[1]) if steady[k] and label[k] == s[2]]
            cents = float(np.mean(midi[on] - s[2]) * 100.0) if on else 0.0
            vel = int(np.clip(round(127 + 2.5 * 20 * np.log10(s[3] / peak + 1e-9)), 30, 127))
            res.notes.append(NoteEvent(start=round(s[0] / fps, 4), end=round(min(s[1] / fps, feat.duration), 4),
                                       midi=int(s[2]), velocity=vel, cents=round(cents, 1)))

    # ---- reliability ---------------------------------------------------------------------
    loud = db > gate_db
    voiced_share = float(active.sum() / max(loud.sum(), 1))
    steady_share = float(steady.sum() / max(active.sum(), 1))
    res.metrics = {"voiced_share": round(voiced_share, 3), "steady_share": round(steady_share, 3),
                   "notes": len(res.notes)}
    if not res.notes:
        res.evidence.append("No note long enough to transcribe")
        return res
    rank = 3 if voiced_share >= 0.75 and steady_share >= 0.6 else 2 if voiced_share >= 0.5 else 1
    if mix:
        rank = min(rank, 2)
        res.evidence.append("Bass extracted from a mix (low-passed harmonic part): check notes by ear")
    if audio_type == AudioType.DRUM_LOOP:
        res.evidence.append("Drum loop: the notes are the lowest tuned element, usually the kick or 808")
    res.reliability = Reliability.from_rank(rank)
    res.evidence.append(f"{len(res.notes)} note{'s' if len(res.notes) != 1 else ''}, {res.range_text}; "
                        f"{voiced_share * 100:.0f}% of the audible bass is pitched")
    if abs(res.tuning_cents) >= 10:
        res.evidence.append(f"Pitches corrected for a tuning offset of {res.tuning_cents:+.0f}¢")
    return res


def quantize(notes: list[NoteEvent], bpm: float, division: int = 4) -> list[NoteEvent]:
    """Snap starts/ends to a grid of ``division`` steps per beat (4 = 16th notes); notes keep >= 1 step."""
    step = 60.0 / bpm / division
    out = []
    for nt in notes:
        s = round(nt.start / step) * step
        e = max(s + step, round(nt.end / step) * step)
        out.append(NoteEvent(start=round(s, 6), end=round(e, 6), midi=nt.midi, velocity=nt.velocity, cents=nt.cents))
    for a, b in zip(out, out[1:]):  # a bass line is monophonic: snapping must not create overlaps
        if a.end > b.start:
            a.end = max(a.start + step / 2, b.start)
    return out
