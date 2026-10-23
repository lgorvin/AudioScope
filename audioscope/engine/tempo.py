"""BPM estimation.

Pipeline
--------
1. Onset envelope: multi-band spectral flux. Drum material uses the
   percussive component (transients); tonal material uses the full signal so
   note/chord changes count as onsets. Loops are analysed with wrap-around, as
   one period of an endlessly repeating signal.
2. Pulse gate: without distinct attacks (pads, drones, noise, legato vocals)
   no tempo is reported.
3. Periodicity salience: normalised autocorrelation (ACF) of the envelope at
   beat lags k*P (k = 1..4) for every candidate BPM. k = 3 penalises
   non-metrical periods such as 3/4 of the beat.
4. Metrical level: the metrical family {b, 2b} with the highest salience wins;
   the octave within the family is chosen by salience x a broad log-normal tempo
   prior (centre 120 BPM) x an event-density term (music usually has ~1-3 onsets
   per beat at the perceived level). Half/double-time relatives are always
   listed, and the result is flagged when the other octave is similarly plausible.
5. Loops: the exact file length is tested against 1..32 bars of 4/4. A length
   that agrees with the onset estimate (or with its octave relative) gives a
   sample-accurate BPM.
6. Songs: overlapping sections are analysed separately and aggregated; their
   agreement feeds the reliability rating.
7. Precision: the mean ACF over up to 64 beat multiples refines the BPM below
   frame resolution; a robust linear fit of tracked beat times gives a standard
   error used for integer snapping.
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import maximum_filter1d

from .features import ONSET_HOP, Features
from .types import AudioType, Reliability, TempoCandidate, TempoResult

BPM_MIN, BPM_MAX = 50.0, 220.0
GRID = np.round(np.arange(40.0, 260.0001, 0.1), 2)
PRIOR_CENTER = 120.0
PRIOR_WIDTH_OCT = 0.8
DENSITY_TARGET = 2.0  # onsets per beat at the perceived metrical level
DENSITY_WIDTH_OCT = 1.5  # deliberately soft
LOOP_BARS = (1, 2, 4, 8, 16, 32)
LOOP_BPM_RANGE = (55.0, 200.0)
COMB = (1, 2, 3, 4)

# Reliability thresholds on periodicity clarity (mean normalised ACF at beat multiples).
CLARITY_NA = 0.12
CLARITY_MEDIUM = 0.25
CLARITY_HIGH = 0.45
# flag half/double ambiguity when the other octave's (salience x prior) is this close
OCTAVE_AMBIGUITY_RATIO = 0.65
# below this onset contrast there are no distinct attacks to measure a pulse from
ONSET_CONTRAST_MIN = 4.0


def tempo_prior(bpm: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-0.5 * (np.log2(np.asarray(bpm) / PRIOR_CENTER) / PRIOR_WIDTH_OCT) ** 2)


def octave_weight(bpm: np.ndarray | float, onset_rate: float | None) -> np.ndarray | float:
    """Prior x event-density preference used to choose between half/double-time readings."""
    w = tempo_prior(bpm)
    if onset_rate:
        per_beat = onset_rate * 60.0 / np.asarray(bpm)
        w = w * np.exp(-0.5 * (np.log2(per_beat / DENSITY_TARGET) / DENSITY_WIDTH_OCT) ** 2)
    return w


def onset_rate(env: np.ndarray, fps: float) -> float | None:
    """Detected onsets per second (including soft ones such as hats)."""
    if env.size < 8 or env.max() <= 0:
        return None
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=fps * ONSET_HOP, hop_length=ONSET_HOP, delta=0.07,
                                        wait=3, units="frames")
    return len(frames) / (len(env) / fps) if frames.size else None


def normalized_acf(env: np.ndarray, circular: bool = False) -> np.ndarray:
    """Autocorrelation normalised to r[0] = 1.

    Linear mode is unbiased and returned up to 2/3 of the signal length. Circular
    mode treats the envelope as one period of a loop and returns 4 periods of lags.
    """
    x = env - np.mean(env)
    n = len(x)
    if n < 8 or not np.any(x):
        return np.zeros(max(n, 1))
    if circular:
        spec = np.fft.rfft(x)
        acf = np.fft.irfft(spec * np.conj(spec), n)
        acf /= acf[0] + 1e-12
        return np.tile(acf, 5)[: 4 * n + 1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    acf = acf / (n - np.arange(n))  # unbiased
    acf /= acf[0] + 1e-12
    return acf[: max(2, int(n * 2 / 3))]


def salience_curve(env: np.ndarray, fps: float, grid: np.ndarray = GRID, circular: bool = False) -> np.ndarray:
    """Mean ACF at lags k*P (k in COMB) for each BPM in grid. NaN where no lag fits."""
    acf = normalized_acf(env, circular)
    # tolerate slight timing jitter: local max over +-1 frame
    acf_s = np.maximum(acf, np.maximum(np.roll(acf, 1), np.roll(acf, -1)))
    acf_s[0] = acf[0]
    lags = np.arange(len(acf_s))
    period = 60.0 * fps / grid
    total = np.zeros_like(grid)
    count = np.zeros_like(grid)
    for k in COMB:
        lag = k * period
        ok = lag < len(acf_s) - 1
        total[ok] += np.interp(lag[ok], lags, acf_s)
        count[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        sal = np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)
    # need at least two beat periods (one for a loop that repeats) for a meaningful reading
    sal[(period if circular else 2 * period) >= len(env)] = np.nan
    return sal


def _at(curve: np.ndarray, bpm: float) -> float:
    if bpm < GRID[0] or bpm > GRID[-1]:
        return float("nan")
    return float(np.interp(bpm, GRID, curve))


def _refine_peak(curve: np.ndarray, bpm: float, rel: float = 0.04) -> float:
    m = (GRID >= bpm * (1 - rel)) & (GRID <= bpm * (1 + rel))
    if not np.any(m):
        return bpm
    return float(GRID[m][int(np.argmax(np.nan_to_num(curve[m], nan=-1.0)))])


def _pick(curve: np.ndarray, rate: float | None = None) -> tuple[float, float]:
    """Best BPM in [BPM_MIN, BPM_MAX]. Returns (bpm, salience).

    Two steps, so the tempo prior only arbitrates between octave relatives:
    1. choose the metrical family {b, 2b} (b in 60..120) with the highest mean
       salience over its members inside 60..200 BPM;
    2. choose the octave within that family by salience x octave_weight.
    """
    c = maximum_filter1d(np.nan_to_num(curve, nan=-1.0), size=7)
    best_score, best_b = -np.inf, 120.0
    for b in GRID[(GRID >= 60.0) & (GRID < 120.0)]:
        vals = [np.interp(m, GRID, c) for m in (b, 2 * b) if 60.0 <= m <= 200.0]
        score = float(np.mean(vals))
        if score > best_score:
            best_score, best_b = score, float(b)
    options = [best_b * f for f in (0.5, 1.0, 2.0, 4.0) if BPM_MIN <= best_b * f <= BPM_MAX]
    choice = max(options, key=lambda b: np.interp(b, GRID, c) * octave_weight(b, rate))
    choice = _refine_peak(curve, choice, 0.01)
    return float(choice), float(np.nan_to_num(_at(curve, choice), nan=0.0))


def _peaks(curve: np.ndarray, n: int = 6) -> list[float]:
    c = np.nan_to_num(curve, nan=-1.0)
    idx = [i for i in range(1, len(c) - 1) if c[i] >= c[i - 1] and c[i] > c[i + 1] and c[i] > 0
           and BPM_MIN <= GRID[i] <= BPM_MAX]
    idx.sort(key=lambda i: -c[i] * tempo_prior(GRID[i]))
    out: list[float] = []
    for i in idx:
        if all(abs(GRID[i] / b - 1) > 0.03 for b in out):
            out.append(float(GRID[i]))
        if len(out) >= n:
            break
    return out


def fine_tune(env: np.ndarray, fps: float, bpm: float, circular: bool = False, rel: float = 0.015) -> float:
    """Sub-frame tempo refinement: maximise the mean ACF over many beat multiples.

    A single-lag ACF peak is quantised to whole frames (~0.7 % at 124 BPM).
    Averaging r(k*P) for k = 1..64 shrinks that error by the number of beats.
    """
    acf = normalized_acf(env, circular)
    lags = np.arange(len(acf))
    cands = np.arange(bpm * (1 - rel), bpm * (1 + rel), 0.01)
    period = 60.0 * fps / cands
    k_max = int(min(64, (len(acf) - 2) // period.max()))
    if k_max < 2:
        return bpm
    lag = np.outer(period, np.arange(1, k_max + 1))
    score = np.interp(lag, lags, acf).mean(axis=1)
    return float(cands[int(np.argmax(score))])


def track_beats(env: np.ndarray, fps: float, bpm: float, sr: int) -> tuple[np.ndarray, float | None, float | None]:
    """Beat times at ~bpm plus a robust linear fit of beat_time = t0 + i * period.

    Returns (beat_times, fitted_bpm, standard_error_bpm); the fit values are None
    if there are too few consistent beats.
    """
    if env.size < 16 or env.max() <= 0:
        return np.zeros(0), None, None
    try:
        _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=ONSET_HOP, bpm=bpm,
                                           tightness=100, trim=False, units="frames")
    except Exception:  # noqa: BLE001
        return np.zeros(0), None, None
    beats = np.asarray(beats, dtype=int)
    if beats.size < 6:
        return beats / fps, None, None
    refined = []  # sub-frame onset peak near each tracked beat
    for b in beats:
        lo, hi = max(b - 3, 1), min(b + 4, len(env) - 1)
        if hi <= lo:
            refined.append(float(b))
            continue
        j = lo + int(np.argmax(env[lo:hi]))
        a, c, d = env[j - 1], env[j], env[min(j + 1, len(env) - 1)]
        den = a - 2 * c + d
        off = 0.5 * (a - d) / den if abs(den) > 1e-12 else 0.0
        refined.append(j + float(np.clip(off, -0.5, 0.5)))
    t = np.asarray(refined) / fps
    period = 60.0 / bpm
    idx = np.round((t - t[0]) / period)
    keep = np.ones(len(t), dtype=bool)
    slope, icpt = period, t[0]
    for _ in range(4):
        if keep.sum() < 6:
            break
        A = np.vstack([idx[keep], np.ones(keep.sum())]).T
        (slope, icpt), *_ = np.linalg.lstsq(A, t[keep], rcond=None)
        new_keep = np.abs(t - (slope * idx + icpt)) < 0.1 * period
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if keep.sum() < 6 or slope <= 0:
        return t, None, None
    resid = t[keep] - (slope * idx[keep] + icpt)
    s2 = float(np.sum(resid ** 2) / max(keep.sum() - 2, 1))
    sxx = float(np.sum((idx[keep] - idx[keep].mean()) ** 2)) + 1e-12
    fit_bpm = 60.0 / slope
    se_bpm = fit_bpm * np.sqrt(s2 / sxx) / slope
    if abs(fit_bpm / bpm - 1) > 0.02:  # tracker settled on another metrical level
        return t, None, None
    return t, float(fit_bpm), float(se_bpm)


def loop_length_candidates(duration: float) -> list[tuple[int, float]]:
    out = []
    for bars in LOOP_BARS:
        bpm = 240.0 * bars / duration
        if LOOP_BPM_RANGE[0] <= bpm <= LOOP_BPM_RANGE[1]:
            out.append((bars, bpm))
    return out


def _relation(bpm: float, primary: float) -> str:
    r = bpm / primary
    for ratio, name in ((0.5, "half-time"), (2.0, "double-time"), (1.5, "3:2"), (2 / 3, "2:3"),
                        (0.25, "quarter-time"), (4.0, "4x")):
        if abs(r / ratio - 1) < 0.03:
            return name
    return "alternative"


def onset_contrast(env: np.ndarray) -> float:
    """95th percentile / median of the onset envelope. Distinct attacks give large values;
    pads, drones, noise and legato material stay around 1-3."""
    if env.size == 0 or env.max() <= 0:
        return 0.0
    return float(np.percentile(env, 95) / (np.median(env) + 1e-9 * env.max()))


def _length_candidates(duration: float, audio_type: AudioType) -> list[TempoCandidate]:
    """Integer BPMs implied by a bar-exact file length. Weak evidence: offered as alternatives only."""
    if not audio_type.is_loop:
        return []
    return [TempoCandidate(bpm=float(round(b)), strength=0.0, relation="loop-length")
            for _, b in loop_length_candidates(duration) if abs(b - round(b)) < 0.02]


def estimate_tempo(feat: Features, audio_type: AudioType, is_lossy: bool = False) -> TempoResult:
    res = TempoResult()
    if audio_type == AudioType.ONE_SHOT:
        res.evidence.append("One-shots have no tempo")
        return res
    min_len = 0.9 if audio_type.is_loop else 1.5
    if feat.duration < min_len or feat.is_silent:
        res.evidence.append("Too short or silent for tempo analysis")
        return res

    fps = feat.onset_fps
    circular = audio_type.is_loop and feat.duration <= 40.0
    percussive = audio_type == AudioType.DRUM_LOOP
    if percussive:
        res.evidence.append("Transient (percussive) onsets used")
    if circular:
        # a loop repeats: analyse it as one period of an endless signal
        env = feat.loop_onset_env(percussive)
    elif audio_type == AudioType.FULL_SONG:
        env = 0.5 * feat.onset_env + 0.5 * feat.onset_env_percussive
    else:
        env = feat.onset_env_percussive if percussive else feat.onset_env

    contrast = onset_contrast(env)
    res.metrics["onset_contrast"] = round(min(contrast, 1e4), 2)
    if contrast < ONSET_CONTRAST_MIN:
        res.evidence.append("No distinct note or drum attacks – no measurable pulse")
        res.candidates = _length_candidates(feat.duration, audio_type)
        if res.candidates:
            res.evidence.append("Loop-length readings are listed as alternatives only")
        return res

    section_agreement, section_bpms = None, []
    if audio_type == AudioType.FULL_SONG and feat.duration > 50:
        curve, section_agreement, section_bpms = _song_sections(env, fps)
    else:
        curve = salience_curve(env, fps, circular=circular)
    if np.all(np.isnan(curve)):
        res.evidence.append("Audio too short for periodicity analysis")
        return res

    rate = onset_rate(env, fps)
    best, clarity = _pick(curve, rate)
    res.metrics["periodicity"] = round(clarity, 4)
    if rate:
        res.metrics["onsets_per_beat"] = round(rate * 60.0 / best, 2)

    # ---- half/double-time ambiguity --------------------------------------------------
    alts = []
    for factor in (0.5, 2.0):
        b = best * factor
        if 40 <= b <= 260:
            b = _refine_peak(curve, b, 0.03)
            alts.append((b, _at(curve, b)))
    for b, sal in alts:
        if np.isnan(sal) or not (BPM_MIN <= b <= BPM_MAX):
            continue
        if sal * octave_weight(b, rate) > OCTAVE_AMBIGUITY_RATIO * clarity * octave_weight(best, rate):
            res.octave_ambiguous = True

    # ---- loops: exact bar length -----------------------------------------------------------
    bpm, bars, loop_match = best, None, False
    if audio_type.is_loop:
        tol = 0.03 if is_lossy else 0.02
        cands = loop_length_candidates(feat.duration)
        matches, factor = [], 1.0
        for factor in (1.0, 2.0, 0.5):  # prefer the onset reading, then its octave relatives
            matches = [(n, b) for n, b in cands if abs(b / (best * factor) - 1) < tol]
            if matches:
                break
        if matches:
            bars, bpm = matches[0]
            loop_match = True
            if factor != 1.0:
                res.evidence.append(f"Octave chosen so the file is a whole number of bars ({best:.1f} would not be)")
                s_new = _at(curve, bpm)
                clarity = clarity if np.isnan(s_new) else s_new
            res.evidence.append(f"File length = {bars} bar{'s' if bars > 1 else ''} of 4/4 at {bpm:.2f} BPM")
        elif cands:
            res.evidence.append("File length is not a whole number of bars at this tempo "
                                "(tail, irregular cut or not 4/4)")

    # ---- precise tempo -------------------------------------------------------------------
    if not loop_match:
        bpm = fine_tune(env, fps, bpm, circular)
    beat_times, fit_bpm, se = track_beats(env, fps, bpm, feat.sr)
    if fit_bpm is not None:
        res.metrics["beat_fit_bpm"] = round(fit_bpm, 3)
        res.metrics["beat_fit_std_error"] = round(se, 4)
        if loop_match and abs(fit_bpm / bpm - 1) > 0.01:
            res.evidence.append(f"Beat spacing ({fit_bpm:.1f}) differs slightly from the loop-length tempo")
    if loop_match:
        if abs(bpm - round(bpm)) < 0.02:
            bpm = float(round(bpm))
    else:
        # integer tempos are the norm for DAW-made music; snap only within measurement precision
        tol = max(3.0 * se, 0.08) if se is not None else 0.08
        if abs(bpm - round(bpm)) <= min(tol, 0.3):
            bpm = float(round(bpm))
            res.evidence.append("Consistent with an integer tempo")

    # ---- reliability ---------------------------------------------------------------------
    rank = 3 if clarity >= CLARITY_HIGH else 2 if clarity >= CLARITY_MEDIUM else 1 if clarity >= CLARITY_NA else 0
    if loop_match and rank > 0:
        rank = min(3, rank + 1)
    if section_agreement is not None:
        res.metrics["section_agreement"] = round(section_agreement, 3)
        res.evidence.append(f"{section_agreement * 100:.0f}% of song sections agree (half/double counted)")
        if section_agreement < 0.4:
            rank -= 2
            res.evidence.append("Sections disagree – tempo may change during the track")
        elif section_agreement < 0.7:
            rank -= 1
        if section_bpms:
            res.metrics["section_bpm_min"] = round(min(section_bpms), 1)
            res.metrics["section_bpm_max"] = round(max(section_bpms), 1)
    if res.octave_ambiguous:
        rank = min(rank, 2)
        res.evidence.append("Half/double-time reading is similarly plausible")

    if rank <= 0:
        res.evidence.append("No clear rhythmic pulse")
        res.candidates = _length_candidates(feat.duration, audio_type)
        return res

    res.bpm = round(float(bpm), 2)
    res.bars = bars
    res.reliability = Reliability.from_rank(rank)
    res.beat_times = [round(float(t), 4) for t in beat_times]

    # ---- alternatives ------------------------------------------------------------------
    seen = [res.bpm]
    res.candidates.append(TempoCandidate(bpm=res.bpm, strength=1.0, relation="primary"))
    for b in [b for b, _ in alts] + _peaks(curve):
        if not (40 <= b <= 260) or any(abs(b / x - 1) < 0.03 for x in seen):
            continue
        sal = _at(curve, b)
        if np.isnan(sal) or sal <= 0:
            continue
        rel = _relation(b, res.bpm)
        exact = {"half-time": 0.5, "double-time": 2.0, "quarter-time": 0.25, "4x": 4.0}.get(rel)
        b_out = res.bpm * exact if exact else b
        res.candidates.append(TempoCandidate(bpm=round(b_out, 2), strength=round(max(0.0, sal / max(clarity, 1e-6)), 3),
                                             relation=rel))
        seen.append(b_out)
        if len(res.candidates) >= 5:
            break
    return res


def _song_sections(env: np.ndarray, fps: float, win_s: float = 24.0, hop_s: float = 12.0):
    """Aggregate salience over overlapping sections. Returns (curve, agreement, section_bpms)."""
    win, hop = int(win_s * fps), int(hop_s * fps)
    curves, weights, bests = [], [], []
    for start in range(0, max(1, len(env) - win + 1), hop):
        seg = env[start:start + win]
        if seg.size < win // 2 or seg.max() <= 0 or onset_contrast(seg) < ONSET_CONTRAST_MIN:
            continue
        c = salience_curve(seg, fps)
        b, sal = _pick(c, onset_rate(seg, fps))
        if sal < CLARITY_NA:  # beatless intro/breakdown: no vote
            continue
        curves.append(np.nan_to_num(c, nan=0.0))
        weights.append(sal)
        bests.append(b)
    if not curves:
        return salience_curve(env, fps), 0.0, []
    w = np.asarray(weights)
    curve = np.average(np.vstack(curves), axis=0, weights=w)
    # whole-track salience keeps long-lag detail; blend with the section consensus
    full = np.nan_to_num(salience_curve(env, fps), nan=0.0)
    curve = 0.6 * curve + 0.4 * full
    g, _ = _pick(curve, onset_rate(env, fps))
    agree = 0.0
    for b, wt in zip(bests, w):
        r = b / g
        if min(abs(r - 1), abs(r - 2) / 2, abs(r - 0.5) / 0.5) < 0.03:
            agree += wt
    return curve, float(agree / w.sum()), bests
