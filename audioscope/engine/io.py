"""Audio decoding.

WAV, FLAC and MP3 are decoded with libsndfile (via ``soundfile``; MP3 support
requires libsndfile >= 1.1, which the soundfile wheels ship). If libsndfile
fails, FFmpeg is used when it is available on PATH or next to the executable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import soundfile as sf

ANALYSIS_SR = 22050
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg")


class AudioLoadError(Exception):
    pass


@dataclass
class AudioData:
    y: np.ndarray  # mono float32 at ``sr``
    sr: int
    native_sr: int
    channels: int
    duration: float  # seconds, from the native-rate sample count
    path: str
    codec: str


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def find_ffmpeg() -> str | None:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(sys.argv[0] or "."))
    for cand in (os.path.join(base, exe), os.path.join(base, "ffmpeg", exe)):
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffmpeg")


def read_native(path: str) -> tuple[np.ndarray, int, str]:
    """Read a file at its native rate. Returns (samples[frames, channels], sr, codec)."""
    if not os.path.isfile(path):
        raise AudioLoadError(f"File not found: {path}")
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        codec = sf.info(path).format
        return data, int(sr), codec
    except Exception as sf_err:  # noqa: BLE001 - fall through to ffmpeg
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise AudioLoadError(f"Could not decode {os.path.basename(path)}: {sf_err}") from sf_err
        return _read_ffmpeg(ffmpeg, path)


def _read_ffmpeg(ffmpeg: str, path: str) -> tuple[np.ndarray, int, str]:
    sr = 44100
    cmd = [ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(cmd, capture_output=True, creationflags=flags)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioLoadError(f"FFmpeg could not decode {os.path.basename(path)}: {proc.stderr.decode(errors='ignore')[:300]}")
    data = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2)
    return data, sr, "ffmpeg"


def load_audio(path: str, sr: int = ANALYSIS_SR) -> AudioData:
    """Decode ``path`` to mono float32 at ``sr`` for analysis."""
    data, native_sr, codec = read_native(path)
    if data.size == 0:
        raise AudioLoadError(f"{os.path.basename(path)} contains no audio")
    channels = data.shape[1]
    duration = data.shape[0] / float(native_sr)
    mono = data.mean(axis=1).astype(np.float32)
    if native_sr != sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=native_sr, target_sr=sr, res_type="soxr_hq").astype(np.float32)
    mono = np.nan_to_num(mono)
    return AudioData(y=mono, sr=sr, native_sr=native_sr, channels=channels, duration=duration, path=path, codec=codec)
