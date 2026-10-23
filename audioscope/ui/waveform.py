"""Interactive waveform: min/max overview, beat/bar grid, playhead, click/drag to seek."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QRunnable, QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..engine.io import read_native
from . import theme

BLOCK = 128  # samples per peak block


class Peaks:
    def __init__(self, mins: np.ndarray, maxs: np.ndarray, sr: int, duration: float):
        self.mins, self.maxs, self.sr, self.duration = mins, maxs, sr, duration


def compute_peaks(path: str) -> Peaks:
    data, sr, _ = read_native(path)
    mono = data.mean(axis=1)
    n = len(mono) // BLOCK * BLOCK
    if n == 0:
        return Peaks(np.zeros(1), np.zeros(1), sr, len(mono) / sr)
    blocks = mono[:n].reshape(-1, BLOCK)
    mins, maxs = blocks.min(axis=1), blocks.max(axis=1)
    peak = max(float(np.max(np.abs(mono))), 1e-9)
    return Peaks(mins / peak, maxs / peak, sr, len(mono) / sr)


class _PeakSignals(QObject):
    done = Signal(str, object)
    failed = Signal(str, str)


class PeakJob(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = _PeakSignals()

    def run(self):
        try:
            self.signals.done.emit(self.path, compute_peaks(self.path))
        except Exception as e:  # noqa: BLE001
            self.signals.failed.emit(self.path, str(e))


class WaveformView(QWidget):
    seekRequested = Signal(float)

    RULER_H = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.peaks: Peaks | None = None
        self.position = 0.0
        self.hover_x: float | None = None
        self.beats: list[float] = []
        self.bar_every = 4
        self.message = "Drop audio files to begin"
        self._cache_w = -1
        self._cols: tuple[np.ndarray, np.ndarray] | None = None

    # ---- data -----------------------------------------------------------------
    def set_peaks(self, peaks: Peaks | None, message: str = ""):
        self.peaks = peaks
        self.position = 0.0
        self._cache_w = -1
        self.message = message
        self.update()

    def set_grid(self, beats: list[float], bar_every: int = 4):
        self.beats = beats
        self.bar_every = bar_every
        self.update()

    def set_position(self, sec: float):
        self.position = sec
        self.update()

    def grid_from_tempo(self, bpm: float | None, duration: float, beat_times: list[float], exact_loop: bool):
        if not bpm:
            self.set_grid([])
            return
        if exact_loop or not beat_times:
            period = 60.0 / bpm
            start = 0.0 if exact_loop or not beat_times else beat_times[0] % period
            beats = list(np.arange(start, duration, period))
        else:
            beats = beat_times
        self.set_grid(beats)

    # ---- geometry --------------------------------------------------------------
    def _wave_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), self.height() - self.RULER_H)

    def _x_to_sec(self, x: float) -> float:
        if not self.peaks:
            return 0.0
        return max(0.0, min(self.peaks.duration, x / max(1, self.width()) * self.peaks.duration))

    def _sec_to_x(self, sec: float) -> float:
        if not self.peaks or self.peaks.duration <= 0:
            return 0.0
        return sec / self.peaks.duration * self.width()

    def _columns(self, w: int):
        if self._cache_w == w and self._cols is not None:
            return self._cols
        mins, maxs = self.peaks.mins, self.peaks.maxs
        edges = np.linspace(0, len(mins), w + 1).astype(int)
        edges[-1] = len(mins)
        lo = np.empty(w)
        hi = np.empty(w)
        for i in range(w):
            a, b = edges[i], max(edges[i + 1], edges[i] + 1)
            lo[i] = mins[a:b].min() if b <= len(mins) else 0
            hi[i] = maxs[a:b].max() if b <= len(maxs) else 0
        self._cols = (lo, hi)
        self._cache_w = w
        return self._cols

    # ---- painting ----------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        r = self._wave_rect()
        p.fillRect(self.rect(), QColor(theme.PANEL))
        mid = r.center().y()
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(QPointF(0, mid), QPointF(r.width(), mid))

        if not self.peaks:
            p.setPen(QColor(theme.FAINT))
            p.drawText(r, Qt.AlignCenter, self.message)
            p.end()
            return

        w = max(1, int(r.width()))
        lo, hi = self._columns(w)
        half = r.height() * 0.46
        play_x = self._sec_to_x(self.position)

        # beat / bar grid
        if self.beats:
            for i, t in enumerate(self.beats):
                x = self._sec_to_x(t)
                bar = i % self.bar_every == 0
                p.setPen(QPen(QColor(255, 255, 255, 34 if bar else 13), 1))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))

        unplayed = QLinearGradient(0, r.top(), 0, r.bottom())
        unplayed.setColorAt(0.0, QColor(theme.WAVE))
        unplayed.setColorAt(0.5, QColor("#2A8F88"))
        unplayed.setColorAt(1.0, QColor(theme.WAVE))
        played = QLinearGradient(0, r.top(), 0, r.bottom())
        played.setColorAt(0.0, QColor(theme.WAVE_PLAYED))
        played.setColorAt(0.5, QColor(theme.ACCENT))
        played.setColorAt(1.0, QColor(theme.WAVE_PLAYED))
        pen_u = QPen(unplayed, 1)
        pen_p = QPen(played, 1)
        for x in range(w):
            p.setPen(pen_p if x <= play_x else pen_u)
            y1 = mid - hi[x] * half
            y2 = mid - lo[x] * half
            if y2 - y1 < 1:
                y2 = y1 + 1
            p.drawLine(QPointF(x + 0.5, y1), QPointF(x + 0.5, y2))

        # hover line + time
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.hover_x is not None:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            p.drawLine(QPointF(self.hover_x, r.top()), QPointF(self.hover_x, r.bottom()))
            p.setPen(QColor(theme.TEXT))
            p.setFont(QFont("Segoe UI", 8))
            label = fmt_time(self._x_to_sec(self.hover_x))
            tx = min(self.hover_x + 6, r.width() - 50)
            p.drawText(QPointF(tx, r.top() + 14), label)

        # playhead
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.drawLine(QPointF(play_x, r.top()), QPointF(play_x, r.bottom()))

        self._draw_ruler(p, r)
        p.end()

    def _draw_ruler(self, p: QPainter, r: QRectF):
        dur = self.peaks.duration
        y0 = r.bottom()
        p.fillRect(QRectF(0, y0, self.width(), self.RULER_H), QColor(theme.RAISED))
        p.setFont(QFont("Segoe UI", 7.5))
        target = max(1, self.width() // 90)
        step = next((s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120) if dur / s <= target), 300)
        t = 0.0
        while t <= dur + 1e-9:
            x = self._sec_to_x(t)
            p.setPen(QColor(theme.BORDER))
            p.drawLine(QPointF(x, y0), QPointF(x, y0 + 4))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QPointF(x + 3, y0 + 13), fmt_time(t, step < 1))
            t += step

    # ---- interaction ---------------------------------------------------------------------
    def mousePressEvent(self, ev):
        if self.peaks and ev.button() == Qt.LeftButton:
            self.seekRequested.emit(self._x_to_sec(ev.position().x()))

    def mouseMoveEvent(self, ev):
        self.hover_x = ev.position().x()
        if self.peaks and ev.buttons() & Qt.LeftButton:
            self.seekRequested.emit(self._x_to_sec(self.hover_x))
        self.update()

    def leaveEvent(self, _ev):
        self.hover_x = None
        self.update()

    def resizeEvent(self, ev):
        self._cache_w = -1
        super().resizeEvent(ev)


def fmt_time(sec: float, fine: bool = False) -> str:
    m, s = divmod(max(0.0, sec), 60)
    return f"{int(m)}:{s:05.2f}" if fine else f"{int(m)}:{int(s):02d}"
