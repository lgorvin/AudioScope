"""Vector icons drawn with QPainter (crisp at any DPI, no image assets needed)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

from . import theme


def _canvas(size: int) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    return pm, p


def _icon(draw, color: str = theme.TEXT, size: int = 64) -> QIcon:
    pm, p = _canvas(size)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    draw(p, size, QColor(color))
    p.end()
    return QIcon(pm)


def play(color=theme.TEXT) -> QIcon:
    def d(p, s, c):
        path = QPainterPath()
        path.moveTo(s * 0.34, s * 0.24)
        path.lineTo(s * 0.78, s * 0.5)
        path.lineTo(s * 0.34, s * 0.76)
        path.closeSubpath()
        p.drawPath(path)
    return _icon(d, color)


def pause(color=theme.TEXT) -> QIcon:
    def d(p, s, c):
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.24, s * 0.15, s * 0.52), 3, 3)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.24, s * 0.15, s * 0.52), 3, 3)
    return _icon(d, color)


def stop(color=theme.TEXT) -> QIcon:
    def d(p, s, c):
        p.drawRoundedRect(QRectF(s * 0.29, s * 0.29, s * 0.42, s * 0.42), 4, 4)
    return _icon(d, color)


def loop(color=theme.TEXT) -> QIcon:
    def d(p, s, c):
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(c, s * 0.075, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        r = QRectF(s * 0.2, s * 0.3, s * 0.6, s * 0.4)
        p.drawRoundedRect(r, s * 0.2, s * 0.2)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        tri = QPainterPath()
        tri.moveTo(s * 0.56, s * 0.18)
        tri.lineTo(s * 0.70, s * 0.30)
        tri.lineTo(s * 0.56, s * 0.42)
        tri.closeSubpath()
        p.drawPath(tri)
    return _icon(d, color)


def volume(color=theme.MUTED) -> QIcon:
    def d(p, s, c):
        path = QPainterPath()
        path.moveTo(s * 0.18, s * 0.4)
        path.lineTo(s * 0.32, s * 0.4)
        path.lineTo(s * 0.5, s * 0.24)
        path.lineTo(s * 0.5, s * 0.76)
        path.lineTo(s * 0.32, s * 0.6)
        path.lineTo(s * 0.18, s * 0.6)
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(c, s * 0.06, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(s * 0.42, s * 0.32, s * 0.24, s * 0.36), -50 * 16, 100 * 16)
        p.drawArc(QRectF(s * 0.40, s * 0.22, s * 0.38, s * 0.56), -50 * 16, 100 * 16)
    return _icon(d, color)


def app_pixmap(size: int = 256) -> QPixmap:
    """Application icon: an orange-to-teal waveform inside a dark rounded square."""
    pm, p = _canvas(size)
    s = float(size)
    p.setBrush(QColor("#15181D"))
    p.setPen(QPen(QColor("#2A3038"), s * 0.02))
    p.drawRoundedRect(QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92), s * 0.22, s * 0.22)
    grad = QLinearGradient(QPointF(s * 0.15, 0), QPointF(s * 0.85, 0))
    grad.setColorAt(0.0, QColor(theme.ACCENT))
    grad.setColorAt(1.0, QColor(theme.WAVE))
    heights = [0.16, 0.30, 0.52, 0.38, 0.66, 0.44, 0.28, 0.50, 0.24, 0.14]
    bar_w = s * 0.052
    gap = (s * 0.70 - bar_w * len(heights)) / (len(heights) - 1)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(grad))
    for i, h in enumerate(heights):
        x = s * 0.15 + i * (bar_w + gap)
        hh = s * h
        p.drawRoundedRect(QRectF(x, s * 0.5 - hh / 2, bar_w, hh), bar_w / 2, bar_w / 2)
    p.end()
    return pm


def app_icon() -> QIcon:
    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(app_pixmap(sz))
    return icon
