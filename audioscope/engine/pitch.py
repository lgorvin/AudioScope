"""Root note and dominant-pitch detection (distinct from musical key).

* Monophonic material (one-shots, 808s, bass lines, vocals): probabilistic YIN
  (pYIN) f0 tracking. Frames are weighted by loudness and voicing probability,
  and fast pitch movement (808 pitch drops, scoops, glides) is down-weighted so
  the settled pitch decides the note. The measured offset from equal
  temperament (A = 440 Hz) is reported in cents.
* Polyphonic material (chord stabs, melodic loops): harmonic pitch salience
  from the constant-Q transform; chord one-shots are named with chord
  templates, loops report their most prominent bass note.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import librosa
import numpy as np

from . import music
from .features import Features
from .key import normalise_pcp, pcp_entropy, present_pitch_classes
from .types import AudioType, PitchResult, Reliability

PYIN_FMIN = 30.0  # below B0: covers sub-bass 808s
PYIN_FMAX = 1600.0
MAX_ONESHOT_SECONDS = 4.0
MAX_LOOP_SECONDS = 30.0


@dataclass
class MonoPitch:
    midi: np.ndarray  # per-frame fractional MIDI (nan where unvoiced)
    weight: np.ndarray  # per-frame weight (loudness x voicing, glides suppressed)
    voiced_fraction: float  # of non-silent frames


def track_monophonic(feat: Features, max_seconds: float, hop: int) -> MonoPitch | None:
    y = feat.y[: int(max_seconds * feat.sr)]
    if len(y) < 4096 or np.max(np.abs(y)) < 1e-4:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, prob = librosa.pyin(y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=feat.sr, frame_length=4096,
                                        hop_length=hop, fill_na=np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, voiced, prob, rms = f0[:n], voiced[:n], prob[:n], rms[:n]
    loud = rms >= 0.05 * (rms.max() + 1e-12)
    if loud.sum() == 0:
        return None
    midi = np.where(voiced, librosa.hz_to_midi(np.where(voiced, f0, 1.0)), np.nan)
    # suppress glides: weight by pitch steadiness over neighbouring frames
    d = np.abs(np.gradient(np.nan_to_num(midi, nan=0.0)))
    frames_per_10ms = max(1e-3, 0.01 * feat.sr / hop)
    steady = np.exp(-(d / frames_per_10ms) / 0.1)  # 0.1 semitone per 10 ms -> weight 1/e
    weight = np.where(voiced & loud, rms * prob * steady, 0.0)
    voiced_fraction = float((voiced & loud).sum() / loud.sum())
    return MonoPitch(midi=midi, weight=weight, voiced_fraction=voiced_fraction)


def note_histogram(mp: MonoPitch, tuning_semitones: float = 0.0) -> dict[int, float]:
    hist: dict[int, float] = {}
    ok = mp.weight > 0
    for m, w in zip(mp.midi[ok] - tuning_semitones, mp.weight[ok]):
        k = int(round(m))
        hist[k] = hist.get(k, 0.0) + float(w)
    return hist


def mono_pcp(mp: MonoPitch, tuning_semitones: float) -> np.ndarray:
    pcp = np.zeros(12)
    for note, w in note_histogram(mp, tuning_semitones).items():
        pcp[note % 12] += w
    return pcp


def _salience_mean(feat: Features, min_midi: int = 24, max_midi: int = 107) -> np.ndarray:
    sal = feat.pitch_salience
    e = feat.harmonic_frame_energy
    act = e >= 0.05 * (e.max() + 1e-12)
    if not act.any():
        return np.zeros(sal.shape[0])
    lo, hi = max(0, min_midi - 24), min(sal.shape[0], max_midi - 24 + 1)
    out = np.zeros(sal.shape[0])
    out[lo:hi] = sal[lo:hi][:, act].mean(axis=1)
    return out


def _match_chord(pcs: list[int], pcp: np.ndarray) -> tuple[int, str] | None:
    """Best chord template for the present pitch classes (all template notes must be present)."""
    present = set(pcs)
    best = None
    for root in range(12):
        for quality, ivs in music.CHORD_TEMPLATES.items():
            notes = {(root + i) % 12 for i in ivs}
            if not notes <= present:
                continue
            extra = len(present - notes)
            score = len(notes) * 2 - extra * 1.5 + 0.5 * pcp[root]
            if best is None or score > best[0]:
                best = (score, root, quality)
    if best is None:
        return None
    return best[1], best[2]


def dominant_pitches(pcp: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    p = normalise_pcp(pcp)
    order = [int(i) for i in np.argsort(-p) if p[i] >= 0.3][:n]
    return [(music.pc_name(i), round(float(p[i]), 3)) for i in order]


def estimate_pitch(feat: Features, audio_type: AudioType, pcp: np.ndarray | None = None) -> tuple[PitchResult, np.ndarray | None]:
    """Returns (pitch result, monophonic PCP or None). The PCP lets key detection use tracked notes."""
    res = PitchResult()
    if feat.is_silent:
        res.evidence.append("Silent")
        return res, None
    tuning = feat.tuning
    harmonic = 1.0 - feat.percussive_ratio

    if audio_type in (AudioType.ONE_SHOT, AudioType.BASS_LOOP, AudioType.VOCAL):
        oneshot = audio_type == AudioType.ONE_SHOT
        mp = track_monophonic(feat, MAX_ONESHOT_SECONDS if oneshot else MAX_LOOP_SECONDS, 256 if oneshot else 512)
        if mp is not None:
            res.metrics["voiced_fraction"] = round(mp.voiced_fraction, 3)
        if oneshot and harmonic < 0.15:
            res.evidence.append("Noise-like percussive sound – no pitch")
            return res, None
        poly_pcs = present_pitch_classes(_fold(_salience_mean(feat)), rel=0.5) if harmonic > 0.2 else []
        if oneshot and len(poly_pcs) >= 3:
            # chords: pYIN would report the implied (missing) common fundamental instead
            return _polyphonic_oneshot(feat, res), None
        if mp is not None and mp.voiced_fraction >= 0.25 and mp.weight.sum() > 0:
            return _monophonic(mp, res, oneshot, tuning), (None if oneshot else mono_pcp(mp, tuning))
        if oneshot and len(poly_pcs) >= 2:
            return _polyphonic_oneshot(feat, res), None
        res.evidence.append("No stable pitch (noise-like or percussive sound)")
        return res, None

    # loops / songs: bass note + dominant pitches
    if harmonic < 0.1:
        res.evidence.append("Too little pitched content")
        return res, None
    if pcp is not None:
        res.dominant_pitches = dominant_pitches(pcp)
    if audio_type == AudioType.FULL_SONG:
        res.evidence.append("Root note applies to samples; for full songs see the key")
        return res, None
    bass = _salience_mean(feat, 24, 59)
    if bass.max() <= 0:
        return res, None
    bass_pcp = _fold(bass)
    order = np.argsort(-bass_pcp)
    top = int(order[0])
    dominance = float(bass_pcp[top] / (bass_pcp.sum() + 1e-12))
    res.metrics["bass_dominance"] = round(dominance, 3)
    if pcp_entropy(bass_pcp) >= 0.9 or dominance < 0.12:
        res.evidence.append("Bass register has no dominant pitch (noise-like)")
        return res, None
    # lowest octave in which the winning pitch class is strong
    octaves = [i for i in range(len(bass)) if i % 12 == top and bass[i] >= 0.3 * bass.max()]
    res.root_midi = 24 + (octaves[0] if octaves else top)
    res.root_kind = "most prominent bass note"
    res.root_cents = round(tuning * 100.0, 1) if abs(tuning) > 0.05 else None
    res.reliability = Reliability.MEDIUM if dominance >= 0.25 else Reliability.LOW
    if audio_type == AudioType.DRUM_LOOP:
        res.reliability = Reliability.LOW if dominance >= 0.25 else Reliability.NONE
        if res.reliability == Reliability.NONE:
            res.root_midi = None
            res.evidence.append("No dominant tuned element (kick/808) found")
            return res, None
        res.evidence.append("Pitch of the dominant tuned low element (kick/808)")
    else:
        res.evidence.append(f"Bass register is centred on {music.pc_name(top)} ({dominance * 100:.0f}% of bass pitch energy)")
    return res, None


def _fold(sal: np.ndarray) -> np.ndarray:
    pcp = np.zeros(12)
    for i, v in enumerate(sal):
        pcp[i % 12] += v
    return pcp


def _monophonic(mp: MonoPitch, res: PitchResult, oneshot: bool, tuning: float) -> PitchResult:
    hist = note_histogram(mp, 0.0)  # absolute (A440) note names for the root
    total = sum(hist.values())
    note = max(hist, key=hist.get)
    share = hist[note] / total
    ok = (mp.weight > 0) & (np.round(mp.midi) == note)
    dev = (mp.midi[ok] - note) * 100.0
    w = mp.weight[ok]
    order = np.argsort(dev)
    cdf = np.cumsum(w[order]) / w.sum()
    cents = float(dev[order][np.searchsorted(cdf, 0.5)])
    spread = float(np.sqrt(np.average((dev - cents) ** 2, weights=w)))
    res.root_midi = int(note)
    res.root_cents = round(cents, 1)
    res.root_hz = round(music.midi_to_hz(note + cents / 100.0), 2)
    res.root_kind = "fundamental" if oneshot else "most frequent note"
    res.metrics.update({"note_share": round(share, 3), "cents_spread": round(spread, 1)})
    # dominant pitches: strongest notes by weighted duration
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:5]
    best_w = top[0][1]
    res.dominant_pitches = [(music.midi_to_name(k), round(v / best_w, 3)) for k, v in top if v / best_w >= 0.15]
    if oneshot:
        if mp.voiced_fraction >= 0.6 and share >= 0.6 and spread <= 25:
            res.reliability = Reliability.HIGH
        elif mp.voiced_fraction >= 0.4 and share >= 0.45:
            res.reliability = Reliability.MEDIUM
        else:
            res.reliability = Reliability.LOW
        res.evidence.append(f"Stable pitch {music.midi_to_name(note)} {cents:+.0f}Â¢ ({res.root_hz:.1f} Hz), "
                            f"{share * 100:.0f}% of pitched energy")
        if share < 0.6:
            res.evidence.append("Pitch moves noticeably (glide, vibrato or several notes)")
    else:
        res.reliability = Reliability.MEDIUM if share >= 0.3 and mp.voiced_fraction >= 0.5 else Reliability.LOW
        res.evidence.append(f"{music.midi_to_name(note)} is the most frequent note ({share * 100:.0f}% of pitched time)")
    return res


def _polyphonic_oneshot(feat: Features, res: PitchResult) -> PitchResult:
    sal = _salience_mean(feat)
    pcp = _fold(sal)
    pcs = present_pitch_classes(pcp, rel=0.5)
    res.dominant_pitches = dominant_pitches(pcp)
    chord = _match_chord(pcs, normalise_pcp(pcp))
    strong = [i for i in range(len(sal)) if sal[i] >= 0.3 * sal.max()]
    if chord is not None:
        root, quality = chord
        # place the root in the lowest octave where it is strong
        octs = [i for i in strong if i % 12 == root] or [i for i in range(len(sal)) if i % 12 == root]
        res.root_midi = 24 + octs[0]
        res.chord = music.chord_name(root, quality)
        res.root_kind = "chord root"
        res.reliability = Reliability.MEDIUM if len(pcs) <= 4 else Reliability.LOW
        res.evidence.append(f"Chord {res.chord} matched from pitch classes " + ", ".join(music.pc_name(p) for p in pcs))
    elif strong:
        res.root_midi = 24 + strong[0]
        res.root_kind = "lowest strong note"
        res.reliability = Reliability.LOW
        res.evidence.append("Several pitches without a recognised chord shape; lowest strong note reported")
    tuning = feat.tuning
    if res.root_midi is not None and abs(tuning) > 0.05:
        res.root_cents = round(tuning * 100.0, 1)
    return res
