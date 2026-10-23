"""Audio-type classification (full song / drum loop / melodic loop / bass loop / vocal / one-shot).

Rule-based on interpretable features so every decision can be explained in the
UI. Filename keywords from sample packs ("kick", "vox", "bass loop", ...) add
evidence but never override strong acoustic evidence on their own.
"""
from __future__ import annotations

import os
import re
import warnings

import librosa
import numpy as np

from .features import Features
from .key import key_scores, weighted_pcp
from .tempo import onset_contrast
from .types import AudioType, Classification, Reliability

SONG_MIN_SECONDS = 50.0
ONESHOT_MAX_SECONDS = 6.0

_KEYWORDS: dict[AudioType, tuple[str, ...]] = {
    AudioType.ONE_SHOT: ("one shot", "oneshot", "one-shot", "kick", "snare", "clap", "hat", "hihat", "rim", "perc",
                         "shot", "stab", "hit", "808", "fx", "impact", "crash", "ride", "tom", "pluck", "chord"),
    AudioType.DRUM_LOOP: ("drum loop", "drumloop", "drums", "beat", "break", "groove", "top loop", "toploop",
                          "hat loop", "perc loop", "full drum"),
    AudioType.BASS_LOOP: ("bass", "sub", "reese", "808 loop"),
    AudioType.VOCAL: ("vox", "vocal", "acapella", "acappella", "voice", "chant", "adlib", "sing"),
    AudioType.MELODIC_LOOP: ("melody", "melodic", "keys", "piano", "synth", "pad", "lead", "guitar", "arp",
                             "chords", "strings", "bell", "music loop"),
    AudioType.FULL_SONG: ("mix", "master", "full track", "remix", "radio edit", "extended"),
}


def filename_type_hints(path: str | None) -> dict[AudioType, float]:
    if not path:
        return {}
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r"[_\-.]+", " ", name)
    hints: dict[AudioType, float] = {}
    for t, words in _KEYWORDS.items():
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", name):
                hints[t] = hints.get(t, 0.0) + 1.0
    if "loop" in name.split():
        hints[AudioType.ONE_SHOT] = hints.get(AudioType.ONE_SHOT, 0.0) - 1.5
    return hints


def _pitch_probe(feat: Features) -> dict[str, float]:
    """Short pYIN probe: voicing, register and how far pitch strays from the semitone grid."""
    y = feat.y_harmonic[: int(8.0 * feat.sr)]
    if len(y) < 4096 or np.max(np.abs(y)) < 1e-4:
        return {"voiced": 0.0, "f0_median": 0.0, "offgrid_cents": 0.0, "notes": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, prob = librosa.pyin(y, fmin=40.0, fmax=1200.0, sr=feat.sr, frame_length=2048, hop_length=512)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0][: len(f0)]
    loud = rms >= 0.05 * (rms.max() + 1e-12)
    v = voiced & loud & (prob > 0.5)
    if loud.sum() == 0 or v.sum() < 5:
        return {"voiced": float(v.sum() / max(loud.sum(), 1)), "f0_median": 0.0, "offgrid_cents": 0.0, "notes": 0}
    midi = librosa.hz_to_midi(f0[v]) - feat.tuning
    dev = (midi - np.round(midi)) * 100.0
    _, counts = np.unique(np.round(midi), return_counts=True)
    notes = int((counts >= 0.1 * counts.sum()).sum())  # distinct notes holding >= 10% of voiced time
    return {"voiced": float(v.sum() / loud.sum()), "f0_median": float(np.median(f0[v])),
            "offgrid_cents": float(np.sqrt(np.mean(dev ** 2))), "notes": notes}


def _envelope(feat: Features) -> dict[str, float]:
    rms = feat.rms
    if rms.size == 0 or rms.max() <= 0:
        return {"peak_pos": 0.0, "end_level": 0.0}
    peak = int(np.argmax(rms))
    tail = rms[int(len(rms) * 0.85):]
    return {"peak_pos": peak / max(len(rms), 1), "end_level": float(tail.mean() / rms.max()) if tail.size else 0.0}


def classify(feat: Features, path: str | None = None) -> Classification:
    d = feat.duration
    ev: list[str] = []
    m: dict[str, float] = {"duration": round(d, 3)}
    hints = filename_type_hints(path)
    scores = {t: 0.0 for t in AudioType}

    # ---- full songs: length is the dominant cue ------------------------------------------
    if d >= SONG_MIN_SECONDS:
        scores[AudioType.FULL_SONG] = 2.0 + min(2.0, (d - SONG_MIN_SECONDS) / 40.0)
        ev.append(f"Long file ({d:.0f} s)")

    onsets = feat.onset_times
    n_on = len(onsets)
    n_strong = feat.strong_onset_count
    reattack = feat.reattack_db
    single_event = n_strong <= 1 and reattack < 6.0
    env = _envelope(feat)
    perc = feat.percussive_ratio
    low = feat.low_energy_ratio
    contrast = onset_contrast(feat.onset_env)
    pcp, _ = weighted_pcp(feat)
    tonal_fit = key_scores(pcp)[0].score if pcp.any() else 0.0
    m.update({"onsets": n_on, "strong_onsets": n_strong, "percussive_ratio": round(perc, 3),
              "low_energy_ratio": round(low, 3), "peak_pos": round(env["peak_pos"], 3),
              "end_level": round(env["end_level"], 3), "onset_contrast": round(min(contrast, 1e4), 2),
              "tonal_fit": round(tonal_fit, 3), "reattack_db": round(reattack, 1)})

    # ---- one-shots ---------------------------------------------------------------------
    if d < 0.6:
        scores[AudioType.ONE_SHOT] += 3.0
        ev.append("Very short sound")
    elif d <= ONESHOT_MAX_SECONDS:
        first_onset = onsets[0] if n_on else 0.0
        if single_event and first_onset < 0.15:
            scores[AudioType.ONE_SHOT] += 2.0
            ev.append("Single sound event (no re-attack)")
        if env["peak_pos"] < 0.3 and env["end_level"] < 0.25:
            scores[AudioType.ONE_SHOT] += 1.0
            ev.append("Decaying envelope")
        if n_strong >= 3:
            scores[AudioType.ONE_SHOT] -= 1.5

    # ---- loop sub-types ------------------------------------------------------------------
    if d < SONG_MIN_SECONDS:
        probe = _pitch_probe(feat) if perc < 0.6 else {"voiced": 0.0, "f0_median": 0.0, "offgrid_cents": 0.0, "notes": 0}
        m.update({k: round(v, 3) for k, v in probe.items()})
        rhythmic = n_strong >= 2 and contrast >= 4.0
        tonal = 1.0 - perc
        mono = probe["voiced"] >= 0.5
        # tonal_fit: how well the pitch content fits any major/minor key (drums ~0.4, harmony ~0.8+)
        if rhythmic and tonal_fit < 0.6:
            scores[AudioType.DRUM_LOOP] += 1.5 + 2.0 * (0.6 - tonal_fit) + max(0.0, perc - 0.3)
            ev.append("Rhythmic attacks without a clear harmonic structure")
        if rhythmic and perc >= 0.45:
            scores[AudioType.DRUM_LOOP] += 1.0
            ev.append(f"Percussive energy {perc * 100:.0f}%")
        if tonal_fit >= 0.6:
            scores[AudioType.MELODIC_LOOP] += 1.0 + 3.0 * (tonal_fit - 0.6) + 0.5 * tonal
            if tonal_fit >= 0.75:
                ev.append("Clear harmonic (key) structure")
        if mono and 0 < probe["f0_median"] < 160 and low >= 0.45 and probe["notes"] >= 2:
            scores[AudioType.BASS_LOOP] += 1.5 + low
            ev.append(f"Monophonic low-register line (median {probe['f0_median']:.0f} Hz, {probe['notes']} notes)")
        elif low >= 0.7 and tonal >= 0.6 and probe["notes"] != 1:
            scores[AudioType.BASS_LOOP] += 1.5
            ev.append("Energy concentrated below 250 Hz")
        if mono and 150 <= probe["f0_median"] <= 1000 and probe["offgrid_cents"] >= 18:
            scores[AudioType.VOCAL] += 2.0 + min(1.0, (probe["offgrid_cents"] - 18) / 15)
            ev.append(f"Continuous, off-grid pitch (±{probe['offgrid_cents']:.0f}¢) typical of voice")

    for t, h in hints.items():
        scores[t] += 0.75 * h
    if hints:
        ev.append("Filename keywords: " + ", ".join(t.label for t, h in hints.items() if h > 0))

    loop_types = (AudioType.DRUM_LOOP, AudioType.MELODIC_LOOP, AudioType.BASS_LOOP, AudioType.VOCAL)
    decaying = env["peak_pos"] < 0.3 and env["end_level"] < 0.25
    if single_event and (d <= 2.5 or decaying):  # a loop needs repeated attacks
        for t in loop_types:
            scores[t] -= 1.5 if d <= ONESHOT_MAX_SECONDS else 0.5
    elif d < 1.0:
        for t in loop_types:
            scores[t] -= 1.0
    if max(scores.values()) <= 0.5:
        # nothing fits well (noise, FX, textures): fall back on the event structure
        fallback = AudioType.ONE_SHOT if (single_event and d <= ONESHOT_MAX_SECONDS) else AudioType.MELODIC_LOOP
        scores[fallback] = 0.5
        ev.append("No type fits well")
    if d >= SONG_MIN_SECONDS:
        scores[AudioType.ONE_SHOT] = -9.0

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, second = ranked[0], ranked[1]
    margin = best[1] - second[1]
    rel = Reliability.HIGH if margin >= 1.5 else Reliability.MEDIUM if margin >= 0.6 else Reliability.LOW
    detail = ""
    if best[0] == AudioType.ONE_SHOT:
        detail = "percussive" if perc >= 0.5 else "tonal"
    return Classification(audio_type=best[0], reliability=rel, detail=detail,
                          scores={t.value: round(s, 3) for t, s in scores.items()},
                          evidence=ev, metrics=m)
