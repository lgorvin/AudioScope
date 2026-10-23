"""Batch data model: one row per file, with detected results and user corrections."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QStandardPaths, Qt
from PySide6.QtGui import QColor

from ..engine import music
from ..engine.types import AnalysisResult, format_bpm
from . import theme

STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR = "Queued", "Analysing…", "Done", "Error"


@dataclass
class Correction:
    bpm: float | None = None
    key: tuple[int, str] | None = None  # (tonic, mode); ("na") handled as key_na
    key_na: bool = False
    audio_type: str | None = None  # manual type override (triggers re-analysis)

    def is_empty(self) -> bool:
        return self.bpm is None and self.key is None and not self.key_na and self.audio_type is None

    def to_json(self) -> dict:
        return {"bpm": self.bpm, "key": list(self.key) if self.key else None, "key_na": self.key_na,
                "audio_type": self.audio_type}

    @staticmethod
    def from_json(d: dict) -> "Correction":
        return Correction(bpm=d.get("bpm"), key=tuple(d["key"]) if d.get("key") else None,
                          key_na=bool(d.get("key_na")), audio_type=d.get("audio_type"))


@dataclass
class Entry:
    path: str
    status: str = STATUS_QUEUED
    result: AnalysisResult | None = None
    error: str = ""
    correction: Correction = field(default_factory=Correction)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    # effective (corrected-or-detected) values
    def bpm(self) -> float | None:
        if self.correction.bpm is not None:
            return self.correction.bpm
        return self.result.tempo.bpm if self.result else None

    def key(self) -> tuple[int, str] | None:
        if self.correction.key_na:
            return None
        if self.correction.key is not None:
            return self.correction.key
        if self.result and self.result.key.key:
            return self.result.key.key.tonic, self.result.key.key.mode
        return None

    def key_text(self) -> str:
        k = self.key()
        return music.key_name(*k) if k else ("N/A" if self.result else "")

    def camelot_text(self) -> str:
        k = self.key()
        return music.camelot(*k) if k else ("N/A" if self.result else "")

    def bpm_text(self) -> str:
        b = self.bpm()
        return format_bpm(b) if b else ("N/A" if self.result else "")

    def bpm_edited(self) -> bool:
        return self.correction.bpm is not None

    def key_edited(self) -> bool:
        return self.correction.key is not None or self.correction.key_na


COLUMNS = ["File", "Type", "BPM", "Key", "Camelot", "Root", "Status"]


class BatchModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[Entry] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section].upper()
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        e = self.entries[index.row()]
        col = COLUMNS[index.column()]
        r = e.result
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == "File":
                return e.name
            if col == "Type":
                return r.classification.audio_type.label if r else ""
            if col == "BPM":
                return e.bpm_text() + (" ✎" if e.bpm_edited() else "")
            if col == "Key":
                return e.key_text() + (" ✎" if e.key_edited() else "")
            if col == "Camelot":
                return e.camelot_text()
            if col == "Root":
                return (r.pitch.chord or r.pitch.root_name or "N/A") if r else ""
            if col == "Status":
                return e.status if e.status != STATUS_ERROR else "Error"
        if role == Qt.UserRole:  # sort key
            if col == "BPM":
                return e.bpm() or 0.0
            if col == "Camelot":
                k = e.key()
                if not k:
                    return 99.0
                code = music.camelot(*k)
                return int(code[:-1]) + (0.5 if code.endswith("B") else 0.0)
            return str(self.data(index, Qt.DisplayRole) or "").lower()
        if role == Qt.DecorationRole and r:
            rel = None
            if col == "BPM" and not e.bpm_edited():
                rel = r.tempo.reliability
            elif col in ("Key",) and not e.key_edited():
                rel = r.key.reliability
            elif col == "Root":
                rel = r.pitch.reliability
            if rel is not None:
                return theme.rel_color(rel)
        if role == Qt.ForegroundRole:
            if col == "Status":
                return QColor({STATUS_DONE: theme.MUTED, STATUS_ERROR: "#FF6B6B"}.get(e.status, theme.ACCENT))
            if col in ("BPM", "Key", "Camelot", "Root") and r:
                txt = self.data(index, Qt.DisplayRole)
                if txt and txt.startswith("N/A"):
                    return QColor(theme.FAINT)
            if col == "Type" and r and r.classification.manual:
                return QColor(theme.ACCENT)
        if role == Qt.ToolTipRole:
            if e.status == STATUS_ERROR:
                return e.error
            if col == "File":
                return e.path
            if r and col == "BPM":
                return f"Reliability: {r.tempo.reliability.label}\n" + "\n".join(r.tempo.evidence)
            if r and col == "Key":
                return f"Reliability: {r.key.reliability.label}\n" + "\n".join(r.key.evidence)
        if role == Qt.TextAlignmentRole and col in ("BPM", "Camelot"):
            return int(Qt.AlignCenter)
        return None

    # ---- mutation ---------------------------------------------------------------------
    def add(self, paths: list[str]) -> list[Entry]:
        known = {e.path for e in self.entries}
        new = [Entry(p) for p in paths if p not in known]
        if new:
            self.beginInsertRows(QModelIndex(), len(self.entries), len(self.entries) + len(new) - 1)
            self.entries.extend(new)
            self.endInsertRows()
        return new

    def remove_rows(self, rows: list[int]):
        for row in sorted(set(rows), reverse=True):
            self.beginRemoveRows(QModelIndex(), row, row)
            del self.entries[row]
            self.endRemoveRows()

    def clear(self):
        self.beginResetModel()
        self.entries.clear()
        self.endResetModel()

    def row_of(self, path: str) -> int:
        for i, e in enumerate(self.entries):
            if e.path == path:
                return i
        return -1

    def entry(self, path: str) -> Entry | None:
        i = self.row_of(path)
        return self.entries[i] if i >= 0 else None

    def touch(self, path: str):
        i = self.row_of(path)
        if i >= 0:
            self.dataChanged.emit(self.index(i, 0), self.index(i, len(COLUMNS) - 1))


class CorrectionStore:
    """Persists manual corrections per file (path + size + modification time) between sessions."""

    def __init__(self):
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or os.path.expanduser("~/.audioscope")
        os.makedirs(base, exist_ok=True)
        self.file = os.path.join(base, "corrections.json")
        try:
            with open(self.file, encoding="utf-8") as f:
                self.data: dict = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def _key(path: str) -> str | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{int(st.st_mtime)}"

    def get(self, path: str) -> Correction:
        k = self._key(path)
        return Correction.from_json(self.data[k]) if k and k in self.data else Correction()

    def put(self, path: str, c: Correction):
        k = self._key(path)
        if not k:
            return
        if c.is_empty():
            self.data.pop(k, None)
        else:
            self.data[k] = c.to_json()
        try:
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=1)
        except OSError:
            pass
