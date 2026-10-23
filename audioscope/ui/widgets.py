"""Reusable widgets: drop zone, result cards, reliability badge, alternatives list."""
from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..engine.io import is_supported
from ..engine.types import Reliability
from . import theme


def collect_audio_paths(paths: list[str]) -> list[str]:
    """Expand folders recursively; keep supported audio files only."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in sorted(files) if is_supported(f))
        elif is_supported(p):
            out.append(p)
    return out


class DropArea(QWidget):
    """Mixin-like base: accepts file/folder drops anywhere on the widget."""

    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._hover = False

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self._hover = True
            self.update()

    def dragLeaveEvent(self, _ev):
        self._hover = False
        self.update()

    def dropEvent(self, ev):
        self._hover = False
        self.update()
        paths = collect_audio_paths([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])
        if paths:
            self.filesDropped.emit(paths)


class DropHint(DropArea):
    """Large dashed drop target shown while the batch is empty."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(160)

    def mousePressEvent(self, _ev):
        self.clicked.emit()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        color = QColor(theme.ACCENT if self._hover else theme.FAINT)
        pen = QPen(color, 1.6, Qt.DashLine)
        pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.setBrush(QColor(255, 138, 61, 18) if self._hover else Qt.NoBrush)
        p.drawRoundedRect(r, 14, 14)
        # arrow glyph
        cx, cy = r.center().x(), r.center().y() - 26
        p.setPen(QPen(color, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for x1, y1, x2, y2 in ((cx, cy - 16, cx, cy + 12), (cx - 10, cy + 2, cx, cy + 12),
                               (cx + 10, cy + 2, cx, cy + 12), (cx - 18, cy + 20, cx + 18, cy + 20)):
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        p.setPen(QColor(theme.TEXT))
        f = p.font()
        f.setPointSizeF(11.5)
        f.setWeight(f.Weight.DemiBold)
        p.setFont(f)
        p.drawText(QRectF(r.left(), cy + 30, r.width(), 24), Qt.AlignCenter, "Drop audio files or folders")
        f.setPointSizeF(9)
        f.setWeight(f.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(r.left(), cy + 54, r.width(), 20), Qt.AlignCenter,
                   "WAV · MP3 · FLAC — full songs, loops and one-shots · click to browse")
        p.end()


class ReliabilityBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_reliability(None)

    def set_reliability(self, rel: Reliability | None, prefix: str = ""):
        if rel is None:
            self.setText("")
            self.setStyleSheet("")
            return
        c = QColor(theme.RELIABILITY_COLORS[rel])
        rgb = f"{c.red()}, {c.green()}, {c.blue()}"
        label = "N/A" if rel == Reliability.NONE else f"{rel.label} reliability"
        self.setText(f"●  {prefix}{label}")
        self.setStyleSheet(f"color: {c.name()}; background: rgba({rgb}, 26); border: 1px solid rgba({rgb}, 85);"
                           f" border-radius: 9px; padding: 2px 8px; font-size: 8.5pt; font-weight: 600;")


class ResultCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        top = QHBoxLayout()
        self.title = QLabel(title.upper())
        self.title.setObjectName("cardTitle")
        self.edited = QLabel("EDITED")
        self.edited.setObjectName("editedChip")
        self.edited.hide()
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.edited)
        lay.addLayout(top)
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.value.setMinimumWidth(10)
        self.value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.value)
        self.sub = QLabel("")
        self.sub.setObjectName("cardSub")
        self.sub.setWordWrap(True)
        self.sub.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.sub)
        self.badge = ReliabilityBadge()
        row = QHBoxLayout()
        row.addWidget(self.badge)
        row.addStretch()
        lay.addSpacing(4)
        lay.addLayout(row)

    def set(self, value: str, sub: str = "", rel: Reliability | None = None, edited: bool = False, dim: bool = False):
        self.value.setText(value)
        style = f"color: {theme.FAINT};" if dim else ""
        if len(value) > 9 and "<" not in value:  # long words (e.g. 'Melodic loop') get a smaller size
            style += " font-size: 16pt;"
        self.value.setStyleSheet(style)
        self.sub.setText(sub)
        self.sub.setVisible(bool(sub))
        self.badge.set_reliability(None if edited else rel)
        self.edited.setVisible(edited)


class StrengthBar(QWidget):
    """Thin bar showing relative evidence (1.0 = as strong as the chosen value). Not a probability."""

    def __init__(self, value: float | None, parent=None):
        super().__init__(parent)
        self.value = value
        self.setFixedSize(70, 8)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.BORDER))
        p.drawRoundedRect(r, 4, 4)
        if self.value is not None:
            w = max(3.0, r.width() * max(0.0, min(1.0, self.value)))
            p.setBrush(QColor(theme.WAVE))
            p.drawRoundedRect(QRectF(0, 0, w, r.height()), 4, 4)
        p.end()


class AlternativesList(QWidget):
    """Rows of (value, relation/detail, relative strength bar)."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(5)
        head = QLabel(title.upper())
        head.setObjectName("sectionTitle")
        self.lay.addWidget(head)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(4)
        self.lay.addLayout(self.rows)

    def set_rows(self, rows: list[tuple[str, str, float | None]], empty: str = "None"):
        while self.rows.count():
            item = self.rows.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        if not rows:
            lbl = QLabel(empty)
            lbl.setObjectName("muted")
            self.rows.addWidget(lbl)
            return
        for value, detail, strength in rows:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            v = QLabel(value)
            v.setMinimumWidth(92)
            v.setStyleSheet("font-weight: 600;")
            d = QLabel(detail)
            d.setObjectName("muted")
            h.addWidget(v)
            h.addWidget(d, 1)
            if strength is not None:
                bar = StrengthBar(strength)
                bar.setToolTip("Relative evidence compared with the chosen value (not a probability)")
                h.addWidget(bar)
            self.rows.addWidget(row)


class EvidenceList(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def set_sections(self, sections: list[tuple[str, list[str]]]):
        html = []
        for title, items in sections:
            if not items:
                continue
            html.append(f"<div style='color:{theme.MUTED}; font-size:8.5pt; font-weight:600; letter-spacing:1px;"
                        f" margin-top:6px'>{title.upper()}</div>")
            html.append("<ul style='margin-left:-22px; margin-top:2px'>" + "".join(
                f"<li style='margin-bottom:2px'>{_esc(i)}</li>" for i in items) + "</ul>")
        self.setText("".join(html))


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class FileDragChip(QLabel):
    """A chip the user drags out of the app (e.g. into FL Studio's playlist or piano roll).

    ``provider`` is called when the drag starts and returns the path of the file to drag
    (or None); the file is handed over as a regular file drop.
    """

    def __init__(self, text: str, provider, parent=None):
        super().__init__(text, parent)
        self.provider = provider
        self._press = None
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("Drag into FL Studio (playlist, channel rack or piano roll) or any folder")
        self.setStyleSheet(f"QLabel {{ background: {theme.RAISED_HI}; border: 1px dashed {theme.FAINT}; border-radius: 7px;"
                           f" padding: 5px 10px; }} QLabel:hover {{ border-color: {theme.ACCENT}; }}"
                           f" QLabel:disabled {{ color: {theme.FAINT}; }}")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._press = ev.position().toPoint()

    def mouseMoveEvent(self, ev):
        from PySide6.QtCore import QMimeData, QUrl
        from PySide6.QtGui import QDrag
        from PySide6.QtWidgets import QApplication

        if self._press is None or not (ev.buttons() & Qt.LeftButton):
            return
        if (ev.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        path = self.provider()
        if not path:
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path)])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.exec(Qt.CopyAction)
