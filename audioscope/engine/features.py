"""Shared, lazily computed signal features.

Detectors pull what they need from a :class:`Features` instance so expensive
steps (STFT, harmonic/percussive separation, constant-Q transform) run at most
once per analysis.
"""
from __future__ import annotations

import warnings
from functools import cached_property

import librosa
import numpy as np

ONSET_HOP = 256  # ~86 frames/s at 22.05 kHz; resolution for tempo analysis
CQT_HOP = 512
CQT_BINS_PER_OCTAVE = 36
CQT_FMIN = librosa.note_to_hz("C1")
CQT_OCTAVES = 7
SALIENCE_DECAY = 0.8


class Features:
    def __init__(self, y: np.ndarray, sr: int):
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.sr = sr
        self.duration = len(y) / float(sr)

    # ---- level / energy -------------------------------------------------
    @cached_property
    def rms(self) -> np.ndarray:
        return librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]

    @cached_property
    def peak(self) -> float:
        return float(np.max(np.abs(self.y))) if self.y.size else 0.0

    @property
    def is_silent(self) -> bool:
        return self.peak < 1e-4

    # ---- harmonic / percussive separation -------------------------------
    @cached_property
    def stft(self) -> np.ndarray:
        return librosa.stft(self.y, n_fft=2048, hop_length=512)

    @cached_property
    def hpss(self) -> tuple[np.ndarray, np.ndarray]:
        h, p = librosa.decompose.hpss(self.stft, kernel_size=31, margin=1.0)
        return h, p

    @cached_property
    def y_harmonic(self) -> np.ndarray:
        return librosa.istft(self.hpss[0], hop_length=512, length=len(self.y))

    @cached_property
    def y_percussive(self) -> np.ndarray:
        return librosa.istft(self.hpss[1], hop_length=512, length=len(self.y))

    @cached_property
    def percussive_ratio(self) -> float:
        """Share of spectral energy assigned to the percussive component (0..1)."""
        h, p = self.hpss
        eh = float(np.sum(np.abs(h) ** 2))
        ep = float(np.sum(np.abs(p) ** 2))
        return ep / (eh + ep + 1e-12)

    @cached_property
    def low_energy_ratio(self) -> float:
        """Share of harmonic energy below 250 Hz."""
        h = np.abs(self.hpss[0]) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        total = float(h.sum()) + 1e-12
        return float(h[freqs < 250].sum()) / total

    @cached_property
    def spectral_flatness(self) -> float:
        return float(np.median(librosa.feature.spectral_flatness(S=np.abs(self.stft))))

    # ---- onset envelopes ---------------------------------------------------
    def _multiband_onset(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=1024, hop_length=ONSET_HOP, n_mels=96, fmax=self.sr / 2)
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
        mel_f = librosa.mel_frequencies(n_mels=96, fmax=self.sr / 2)
        lo = int(np.searchsorted(mel_f, 200.0))
        hi = int(np.searchsorted(mel_f, 2500.0))
        bands = [slice(0, max(lo, 1)), slice(lo, hi), slice(hi, 96)]
        env = librosa.onset.onset_strength_multi(S=log_mel, sr=self.sr, hop_length=ONSET_HOP, channels=bands, lag=1, max_size=3)
        # balance the bands (hats vs kicks), but never amplify a nearly empty band (codec noise,
        # silence) by more than 10x relative to the strongest one
        scales = np.array([np.percentile(b, 99) if b.size else 0.0 for b in env])
        floor = 0.1 * scales.max() if scales.size and scales.max() > 0 else 1.0
        out = np.zeros(env.shape[1])
        for band, scale in zip(env, scales):
            out += band / max(scale, floor, 1e-9)
        return out / len(env)

    @cached_property
    def onset_env(self) -> np.ndarray:
        """Multi-band spectral-flux onset envelope of the full signal."""
        return self._multiband_onset(self.y)

    @cached_property
    def onset_env_percussive(self) -> np.ndarray:
        """Onset envelope of the percussive component (transients)."""
        return self._multiband_onset(self.y_percussive.astype(np.float32))

    def loop_onset_env(self, percussive: bool = False) -> np.ndarray:
        """Onset envelope of exactly one loop period, computed with wrap-around so
        an onset at the loop start is measured against the loop's own ending."""
        y = self.y_percussive.astype(np.float32) if percussive else self.y
        env = self._multiband_onset(np.concatenate([y, y]))
        n = int(round(len(y) / ONSET_HOP))
        return env[n:2 * n] if env.size >= 2 * n else env[:n]

    @property
    def onset_fps(self) -> float:
        return self.sr / ONSET_HOP

    @cached_property
    def onset_times(self) -> np.ndarray:
        """Times of clearly detectable onsets (s)."""
        env = self.onset_env
        if env.size == 0 or env.max() <= 0:
            return np.zeros(0)
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=self.sr, hop_length=ONSET_HOP, backtrack=False,
                                            units="frames", delta=0.15, wait=8)
        strong = frames[env[frames] >= 0.25 * env.max()] if frames.size else frames
        return librosa.frames_to_time(strong, sr=self.sr, hop_length=ONSET_HOP)

    @cached_property
    def strong_onset_count(self) -> int:
        """Onsets reaching at least half the strongest attack (re-attacks, not decay ripples).

        Measured on the wrap-around envelope: a loop's first hit is then judged against the
        loop's own ending (an ordinary onset), while a one-shot's attack still stands alone.
        """
        env = self.loop_onset_env()
        if env.size == 0 or env.max() <= 0:
            return 0
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=self.sr, hop_length=ONSET_HOP, delta=0.15, wait=8,
                                            units="frames")
        return int((env[frames] >= 0.5 * env.max()).sum()) if frames.size else 0

    @cached_property
    def reattack_db(self) -> float:
        """Largest loudness rise (dB) after the loudest moment, relative to the quietest point
        reached since. ~0 dB for a single decaying sound; chord changes and hits give 10+ dB."""
        rms = librosa.feature.rms(y=self.y, frame_length=1024, hop_length=256)[0]
        if rms.size == 0 or rms.max() <= 0:
            return 0.0
        rms = np.convolve(rms, np.ones(3) / 3, mode="same")
        peak = int(np.argmax(rms))
        running_min = np.minimum.accumulate(rms[peak:])
        audible = rms[peak:] > 0.1 * rms.max()
        ratio = np.where(audible & (running_min > 0), rms[peak:] / np.maximum(running_min, 1e-12), 1.0)
        return float(20 * np.log10(ratio.max()))

    # ---- pitch content ----------------------------------------------------------
    @cached_property
    def tuning(self) -> float:
        """Tuning offset of the material relative to A440, in semitones (-0.5..0.5)."""
        y = self.y_harmonic
        if np.max(np.abs(y)) < 1e-4:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = librosa.estimate_tuning(y=y, sr=self.sr, bins_per_octave=12, resolution=0.01)
        return float(np.nan_to_num(t))

    @cached_property
    def cqt(self) -> np.ndarray:
        """Magnitude CQT of the harmonic component, C1..B7, 3 bins/semitone, shifted by the tuning offset."""
        y = self.y_harmonic
        min_len = 4 * CQT_HOP
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # librosa expects tuning in fractions of a CQT bin
            return np.abs(librosa.cqt(y, sr=self.sr, hop_length=CQT_HOP, fmin=CQT_FMIN,
                                      n_bins=CQT_OCTAVES * CQT_BINS_PER_OCTAVE, bins_per_octave=CQT_BINS_PER_OCTAVE,
                                      tuning=self.tuning * CQT_BINS_PER_OCTAVE / 12.0))

    @cached_property
    def cqt_semitones(self) -> np.ndarray:
        """CQT collapsed to one bin per semitone, shape (84, frames). Row 0 = C1 (MIDI 24).

        Bin 3k is centred on semitone k, so each semitone pools bins 3k-1, 3k, 3k+1.
        """
        c = np.vstack([np.zeros((1, self.cqt.shape[1])), self.cqt])
        n_semis = CQT_OCTAVES * 12
        c3 = c[: n_semis * 3].reshape(n_semis, 3, -1)
        return 0.25 * c3[:, 0] + 0.5 * c3[:, 1] + 0.25 * c3[:, 2]

    @cached_property
    def pitch_salience(self) -> np.ndarray:
        """Harmonic-summation pitch salience per semitone (84, frames).

        A note's salience combines its own energy with its first 8 partials
        (decaying weights), so a fundamental outweighs its own overtones (the 3rd
        and 5th partials would otherwise add a fifth and a major third).
        """
        s = self.cqt_semitones
        support = np.zeros_like(s)
        for h in range(1, 9):
            off = int(round(12 * np.log2(h)))
            if off >= s.shape[0]:
                break
            support[: s.shape[0] - off] += SALIENCE_DECAY ** (h - 1) * s[off:]
        # geometric mean with the fundamental's own energy: an empty bin below a note
        # (a "subharmonic") gets no salience even though the note's partials line up with it
        return np.sqrt(s * support)

    def chroma(self, min_midi: int = 24, max_midi: int = 107) -> np.ndarray:
        """Square-root-compressed chroma (12, frames) from the semitone CQT within a MIDI range.

        (Harmonic-salience and log-compressed variants were evaluated and were no
        better for key finding; see eval/README.md.)
        """
        semis = self.cqt_semitones
        lo, hi = max(0, min_midi - 24), min(semis.shape[0], max_midi - 24 + 1)
        band = np.sqrt(semis[lo:hi] / (semis.max() + 1e-12))
        chroma = np.zeros((12, band.shape[1]))
        for i in range(band.shape[0]):
            chroma[(lo + i) % 12] += band[i]
        return chroma

    @cached_property
    def harmonic_frame_energy(self) -> np.ndarray:
        return self.cqt_semitones.sum(axis=0)
