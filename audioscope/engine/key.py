"""Musical key estimation.

Method
------
* Pitch-class profile (PCP): log-compressed, tuning-corrected constant-Q
  chroma of the harmonic component, energy-weighted over frames. A bass-register
  chroma (C1-B3) is blended in because the tonic is over-represented in the bass.
* Key templates: the Krumhansl-Kessler probe-tone profiles and Temperley's
  Kostka-Payne corpus profiles. Each of the 24 keys is scored by the mean
  Pearson correlation of the PCP with both rotated profiles.
* Songs: the PCP is computed per section as well; section agreement feeds the
  reliability rating and reveals modulations.

N/A policy
----------
A key is only reported when the material supports one: enough harmonic
energy, a non-flat PCP, at least three clearly present pitch classes and a
template correlation above a floor. A single note or a bare fifth defines a
root, not a key; the analyzer reports those as root notes instead.
"""
from __future__ import annotations

import numpy as np

from . import music
from .features import Features
from .types import AudioType, KeyCandidate, KeyResult, Reliability

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPERLEY_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
TEMPERLEY_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

BASS_WEIGHT = 0.0  # bass blend evaluated (0, 0.5): no consistent gain
MIN_SCORE = 0.55  # below this no key template fits
MIN_PITCH_CLASSES = 3
FLAT_ENTROPY = 0.95  # normalised entropy above which the PCP is effectively flat (noise)
HIGH_SCORE, HIGH_MARGIN = 0.80, 0.08
MEDIUM_SCORE, MEDIUM_MARGIN = 0.70, 0.04
RELATIVE_HIGH_GAP = 0.25  # "high" needs the relative major/minor (the commonest confusion) this far behind
RELATIVE_TIE = 0.20  # max correlation gap for the opening-bass tie-break (relative key: same notes)
FIFTH_TIE = 0.05  # ... and for keys a fifth apart
SALIENCE_PRESENCE = 0.14  # on the overtone-suppressed profile
# Key chroma starts at C2 (65 Hz): the octave below is dominated by kicks and sub-bass
# whose pitch rarely reflects the harmony (bass lines still count via their harmonics).
KEY_MIN_MIDI = 36
MONO_PRESENCE = 0.15  # tracked-note histograms are sparse: a note at 15% of the top note is present


def _templates() -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    out = {}
    for tonic in range(12):
        out[(tonic, music.MAJOR)] = (np.roll(KK_MAJOR, tonic), np.roll(TEMPERLEY_MAJOR, tonic))
        out[(tonic, music.MINOR)] = (np.roll(KK_MINOR, tonic), np.roll(TEMPERLEY_MINOR, tonic))
    return out


_TEMPLATES = _templates()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def key_scores(pcp: np.ndarray) -> list[KeyCandidate]:
    """All 24 keys sorted by mean template correlation (best first)."""
    out = []
    for (tonic, mode), (kk, tp) in _TEMPLATES.items():
        out.append(KeyCandidate(tonic=tonic, mode=mode, score=round(0.5 * (_pearson(pcp, kk) + _pearson(pcp, tp)), 4)))
    out.sort(key=lambda k: -k.score)
    return out


def normalise_pcp(pcp: np.ndarray) -> np.ndarray:
    """Scale to 0..1 after removing the floor shared by all pitch classes."""
    p = np.asarray(pcp, dtype=float)
    p = p - p.min()
    return p / p.max() if p.max() > 0 else p


def pcp_entropy(pcp: np.ndarray) -> float:
    p = normalise_pcp(pcp)
    if p.sum() <= 0:
        return 1.0
    q = p / p.sum()
    q = q[q > 0]
    return float(-(q * np.log(q)).sum() / np.log(12))


def present_pitch_classes(pcp: np.ndarray, rel: float = 0.4) -> list[int]:
    p = normalise_pcp(pcp)
    return [int(i) for i in np.argsort(-p) if p[i] >= rel]


def weighted_pcp(feat: Features, frames: slice | None = None) -> tuple[np.ndarray, float]:
    """Energy-weighted PCP (full range + bass blend). Returns (pcp, active_fraction)."""
    full = feat.chroma(KEY_MIN_MIDI, 107)
    bass = feat.chroma(KEY_MIN_MIDI, 59)
    energy = feat.harmonic_frame_energy
    if frames is not None:
        full, bass, energy = full[:, frames], bass[:, frames], energy[frames]
    if energy.size == 0 or energy.max() <= 0:
        return np.zeros(12), 0.0
    active = energy >= 0.05 * energy.max()
    w = energy * active
    pf = (full * w).sum(axis=1) / (w.sum() + 1e-12)
    pb = (bass * w).sum(axis=1) / (w.sum() + 1e-12)
    pcp = normalise_pcp(pf) + BASS_WEIGHT * normalise_pcp(pb)
    return pcp, float(active.mean())


def salience_pcp(feat: Features) -> np.ndarray:
    """Energy-weighted pitch-class profile of the harmonic-salience representation (overtones suppressed)."""
    sal = feat.pitch_salience
    energy = feat.harmonic_frame_energy
    if energy.size == 0 or energy.max() <= 0:
        return np.zeros(12)
    w = energy * (energy >= 0.05 * energy.max())
    per_semi = (sal * w).sum(axis=1) / (w.sum() + 1e-12)
    pcp = np.zeros(12)
    for i, v in enumerate(per_semi):
        pcp[i % 12] += v
    return pcp


def _reliability(cands: list[KeyCandidate]) -> tuple[int, float, bool]:
    """Rank (0..3), margin to the runner-up, and whether the runner-up is the relative key."""
    best, second = cands[0], cands[1]
    margin = best.score - second.score
    rel = music.relative_key(best.tonic, best.mode)
    relative_close = (second.tonic, second.mode) == rel
    rel_score = next(c.score for c in cands if (c.tonic, c.mode) == rel)
    if best.score >= HIGH_SCORE and margin >= HIGH_MARGIN and best.score - rel_score >= RELATIVE_HIGH_GAP:
        rank = 3
    elif best.score >= MEDIUM_SCORE and margin >= MEDIUM_MARGIN:
        rank = 2
    else:
        rank = 1
    return rank, margin, relative_close


def opening_bass_pc(feat: Features) -> int | None:
    """Clearly dominant bass pitch class at the start of a loop (first 1/8, at most 2 s), else None."""
    bass = feat.chroma(KEY_MIN_MIDI, 59)
    if bass.shape[1] < 4:
        return None
    n = max(2, min(int(bass.shape[1] / 8), int(2.0 * feat.sr / 512)))
    p = bass[:, :n].sum(axis=1)
    order = np.argsort(-p)
    if p[order[0]] <= 0 or p[order[0]] < 1.25 * p[order[1]]:
        return None
    return int(order[0])


def key_from_pcp(pcp: np.ndarray, res: KeyResult, max_rank: int = 3, tonic_hint: int | None = None,
                 presence: float = 0.4, hint_source: str = "the loop's opening bass note",
                 count_pcp: np.ndarray | None = None) -> KeyResult:
    """Fill ``res`` from a pitch-class profile, applying the N/A rules.

    ``tonic_hint`` (a loop's opening bass note) only decides between a key and its
    relative major/minor when the two fit almost equally well.
    """
    res.pitch_class_profile = [round(float(x), 4) for x in normalise_pcp(pcp)]
    ent = pcp_entropy(pcp)
    # pitch classes are counted on an overtone-suppressed profile when one is given, so a
    # single note's harmonics or a power chord's fifth cannot pass as extra notes
    present = present_pitch_classes(count_pcp if count_pcp is not None else pcp, presence)
    res.metrics["pcp_entropy"] = round(ent, 4)
    res.metrics["pitch_classes"] = len(present)
    cands = key_scores(pcp)
    res.candidates = cands[:5]
    if ent >= FLAT_ENTROPY:
        res.evidence.append("Pitch content is spread evenly (noise-like) – no key")
        return res
    if len(present) < MIN_PITCH_CLASSES:
        names = ", ".join(music.pc_name(p) for p in present) or "none"
        res.evidence.append(f"Only {len(present)} clear pitch class(es) ({names}) – a key is not defined by this")
        return res
    best = cands[0]
    res.metrics["best_correlation"] = best.score
    if best.score < MIN_SCORE:
        res.evidence.append(f"Pitch content does not fit any major/minor key (best r = {best.score:.2f})")
        return res
    rank, margin, relative_close = _reliability(cands)
    if tonic_hint is not None and best.tonic != tonic_hint:
        # near-ties only: the relative key (same notes) or a key a fifth away (one note differs)
        rel = music.relative_key(best.tonic, best.mode)
        for cand in cands[1:4]:
            close = (cand.tonic, cand.mode) == rel and best.score - cand.score < RELATIVE_TIE
            close = close or (cand.mode == best.mode and (cand.tonic - best.tonic) % 12 in (5, 7)
                              and best.score - cand.score < FIFTH_TIE)
            if close and cand.tonic == tonic_hint:
                cands.remove(cand)
                cands.insert(0, cand)
                res.candidates = cands[:5]
                best = cand
                margin = best.score - cands[1].score  # negative: chosen against a slightly better fit
                rank = min(rank, 2)
                relative_close = (cands[1].tonic, cands[1].mode) == music.relative_key(best.tonic, best.mode)
                res.evidence.append(f"Near-tie with {cands[1].name} decided by {hint_source} "
                                    f"({music.pc_name(tonic_hint)})")
                break
    res.metrics["margin"] = round(margin, 4)
    res.key = best
    res.reliability = Reliability.from_rank(min(rank, max_rank))
    res.evidence.append(f"Template fit r = {best.score:.2f}; next best {cands[1].name} (r = {cands[1].score:.2f})")
    if relative_close and abs(margin) < HIGH_MARGIN:
        res.evidence.append(f"Relative {cands[1].mode} ({cands[1].name}) is nearly as likely – same notes, different tonal centre")
    elif len(present) <= 4:
        res.evidence.append(f"Only {len(present)} pitch classes present – key inferred from limited material")
    return res


def estimate_key(feat: Features, audio_type: AudioType, pcp_override: np.ndarray | None = None,
                 tonic_hint: int | None = None) -> KeyResult:
    """``tonic_hint`` overrides the loop opening-bass hint (e.g. a chord one-shot's root)."""
    res = KeyResult()
    if feat.is_silent:
        res.evidence.append("Silent")
        return res
    res.tuning_cents = round(feat.tuning * 100.0, 1)

    if pcp_override is not None:
        res.evidence.append("Pitch-class profile from tracked monophonic pitch")
        hint = opening_bass_pc(feat) if audio_type.is_loop else None
        # a single melodic/bass line rarely pins the key down: never rate it "high"
        key_from_pcp(pcp_override, res, max_rank=2, tonic_hint=hint, presence=MONO_PRESENCE)
        if res.key is not None:
            res.evidence.append("Key inferred from a single melodic line – reliability capped at medium")
        return res

    pcp, active = weighted_pcp(feat)
    count_pcp = salience_pcp(feat)
    res.metrics["harmonic_ratio"] = round(1 - feat.percussive_ratio, 4)
    if active < 0.02 or (1 - feat.percussive_ratio) < 0.08:
        res.evidence.append("Too little harmonic (pitched) content")
        return res

    max_rank = 3
    if audio_type == AudioType.DRUM_LOOP:
        max_rank = 1
        res.evidence.append("Drum loop: key reflects tuned drums/808s only")
    elif audio_type == AudioType.ONE_SHOT:
        max_rank = 1

    if audio_type == AudioType.FULL_SONG and feat.duration > 50:
        return _song_key(feat, pcp, res, count_pcp)
    if tonic_hint is not None:
        return key_from_pcp(pcp, res, max_rank, tonic_hint=tonic_hint, hint_source="the chord root",
                            count_pcp=count_pcp, presence=SALIENCE_PRESENCE)
    hint = opening_bass_pc(feat) if audio_type.is_loop else None
    return key_from_pcp(pcp, res, max_rank, tonic_hint=hint, count_pcp=count_pcp, presence=SALIENCE_PRESENCE)


def _song_key(feat: Features, pcp: np.ndarray, res: KeyResult, count_pcp: np.ndarray | None = None,
              win_s: float = 30.0) -> KeyResult:
    key_from_pcp(pcp, res, count_pcp=count_pcp, presence=SALIENCE_PRESENCE)
    if res.key is None:
        return res
    fps = feat.sr / 512.0
    n = feat.cqt.shape[1]
    win = int(win_s * fps)
    votes: list[tuple[int, str]] = []
    weights: list[float] = []
    for start in range(0, max(1, n - win // 2), win // 2):
        sl = slice(start, min(n, start + win))
        p, active = weighted_pcp(feat, sl)
        if active < 0.2 or pcp_entropy(p) >= FLAT_ENTROPY:
            continue
        c = key_scores(p)[0]
        if c.score < MIN_SCORE:
            continue
        votes.append((c.tonic, c.mode))
        weights.append(c.score)
    if not votes:
        return res
    w = np.asarray(weights)
    target = (res.key.tonic, res.key.mode)
    agree = float(w[[v == target for v in votes]].sum() / w.sum())
    rel = music.relative_key(*target)
    agree_rel = float(w[[v in (target, rel) for v in votes]].sum() / w.sum())
    res.metrics["section_agreement"] = round(agree, 3)
    res.evidence.append(f"{agree * 100:.0f}% of song sections agree on {res.key.name}")
    rank = res.reliability.rank
    if agree < 0.4 and agree_rel < 0.6:
        rank -= 2
        others = sorted({music.key_name(*v) for v in votes if v != target})
        res.evidence.append("Sections disagree – possible modulation (" + ", ".join(others[:3]) + ")")
    elif agree < 0.65:
        rank -= 1
    res.reliability = Reliability.from_rank(max(rank, 1))
    return res
