"""AudioScope main window: batch table + detail panel (waveform, results, corrections)."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections import OrderedDict

from PySide6.QtCore import QSortFilterProxyModel, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QSplitter, QStackedWidget, QTableView,
                               QVBoxLayout, QWidget)

from ..engine import ENGINE_VERSION, music
from ..engine.io import SUPPORTED_EXTENSIONS
from ..engine.midi import notes_to_midi
from ..engine.transcribe import BassTranscription, quantize
from ..engine.types import AnalysisResult, AudioType, Reliability, format_bpm
from . import icons, theme
from .jobs import AnalysisManager
from .model import (COLUMNS, STATUS_DONE, STATUS_ERROR, STATUS_QUEUED, STATUS_RUNNING, BatchModel, Correction,
                    CorrectionStore, Entry)
from .player import Player
from .waveform import PeakJob, WaveformView, fmt_time
from .widgets import (AlternativesList, DropArea, DropHint, EvidenceList, FileDragChip, ReliabilityBadge, ResultCard,
                      collect_audio_paths)

KEY_CHOICES = [(pc, mode) for mode in (music.MINOR, music.MAJOR) for pc in range(12)]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioScope")
        self.setWindowIcon(icons.app_icon())
        self.resize(1440, 900)
        self.setMinimumSize(1080, 680)

        self.model = BatchModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.UserRole)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.store = CorrectionStore()
        self.manager = AnalysisManager(parent=self)
        self.manager.started.connect(self._on_started)
        self.manager.finished.connect(self._on_finished)
        self.manager.failed.connect(self._on_failed)
        self.manager.bassFinished.connect(self._on_bass_finished)
        self.manager.bassFailed.connect(self._on_bass_failed)
        self._bass_export_dir: str | None = None
        self._bass_export_pending: set[str] = set()
        self._bass_export_written = 0
        self.player = Player(self)
        self.player.positionChanged.connect(self._on_position)
        self.player.playingChanged.connect(self._on_playing)
        self.player.errorOccurred.connect(lambda m: self.statusBar().showMessage(f"Playback: {m}", 5000))
        self.peaks_cache: OrderedDict[str, object] = OrderedDict()
        self.current: str | None = None
        self._taps: list[float] = []
        self._updating = False

        self._build()
        self._update_status()

    # =================================================================== layout
    def _build(self):
        central = DropArea()
        central.setObjectName("central")
        central.filesDropped.connect(self.add_paths)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 8)
        root.setSpacing(10)
        root.addLayout(self._build_header())

        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(10)
        split.addWidget(self._build_batch_panel())
        split.addWidget(self._build_detail_panel())
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 6)
        split.setSizes([620, 800])
        root.addWidget(split, 1)

        QShortcut(QKeySequence(Qt.Key_Space), self, self.player.toggle)
        QShortcut(QKeySequence.Open, self, self.browse_files)
        QShortcut(QKeySequence.Delete, self, self.remove_selected)
        QShortcut(QKeySequence("Ctrl+E"), self, lambda: self.export("csv"))

    def _build_header(self):
        h = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(icons.app_pixmap(34))
        h.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        t = QLabel("AudioScope")
        t.setObjectName("appTitle")
        s = QLabel("Key, BPM & pitch analysis for producers")
        s.setObjectName("appSubtitle")
        titles.addWidget(t)
        titles.addWidget(s)
        h.addLayout(titles)
        h.addStretch()
        add = QPushButton("Add files…")
        add.setObjectName("primary")
        add.clicked.connect(self.browse_files)
        folder = QPushButton("Add folder…")
        folder.clicked.connect(self.browse_folder)
        self.reanalyse_btn = QPushButton("Re-analyse")
        self.reanalyse_btn.setToolTip("Analyse the selected files again (all files if none selected)")
        self.reanalyse_btn.clicked.connect(self.reanalyse)
        export = QPushButton("Export ▾")
        menu = QMenu(export)
        menu.addAction("Export CSV…", lambda: self.export("csv"))
        menu.addAction("Export JSON (full detail)…", lambda: self.export("json"))
        menu.addSeparator()
        menu.addAction("Bass lines to MIDI (all files)…", self.export_bass_batch)
        export.setMenu(menu)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_all)
        for b in (add, folder, self.reanalyse_btn, export, clear):
            h.addWidget(b)
        return h

    def _build_batch_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        top = QHBoxLayout()
        lbl = QLabel("BATCH")
        lbl.setObjectName("sectionTitle")
        top.addWidget(lbl)
        self.count_label = QLabel("")
        self.count_label.setObjectName("muted")
        top.addWidget(self.count_label)
        top.addStretch()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter (name, key, type…)")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(220)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        top.addWidget(self.search)
        v.addLayout(top)

        self.stack = QStackedWidget()
        self.drop_hint = DropHint()
        self.drop_hint.clicked.connect(self.browse_files)
        self.drop_hint.filesDropped.connect(self.add_paths)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(30)
        hh = self.table.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for col, w in zip(range(1, len(COLUMNS)), (104, 70, 112, 70, 64, 84)):
            self.table.setColumnWidth(col, w)
        self.table.selectionModel().selectionChanged.connect(self._on_selection)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        self.stack.addWidget(self.drop_hint)
        self.stack.addWidget(self.table)
        v.addWidget(self.stack, 1)

        legend = QHBoxLayout()
        for rel in (Reliability.HIGH, Reliability.MEDIUM, Reliability.LOW, Reliability.NONE):
            l = QLabel(f"<span style='color:{theme.RELIABILITY_COLORS[rel]}'>■</span> {rel.label}")
            l.setObjectName("muted")
            legend.addWidget(l)
        legend.addStretch()
        hint = QLabel("✎ = manually corrected")
        hint.setObjectName("muted")
        legend.addWidget(hint)
        v.addLayout(legend)
        return panel

    def _build_detail_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        scroll.setWidget(body)
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 4, 0)
        v.setSpacing(10)

        # --- title row
        title_row = QHBoxLayout()
        tcol = QVBoxLayout()
        tcol.setSpacing(1)
        self.file_title = QLabel("No file selected")
        self.file_title.setObjectName("fileTitle")
        self.file_meta = QLabel("Add audio to analyse key, tempo and pitch.")
        self.file_meta.setObjectName("muted")
        tcol.addWidget(self.file_title)
        tcol.addWidget(self.file_meta)
        title_row.addLayout(tcol, 1)
        self.copy_btn = QPushButton("Copy result")
        self.copy_btn.setToolTip("Copy BPM / key / Camelot to the clipboard")
        self.copy_btn.clicked.connect(self.copy_result)
        title_row.addWidget(self.copy_btn, 0, Qt.AlignTop)
        v.addLayout(title_row)

        # --- waveform + transport
        wave_panel = QFrame()
        wave_panel.setObjectName("panel")
        wv = QVBoxLayout(wave_panel)
        wv.setContentsMargins(10, 10, 10, 10)
        self.wave = WaveformView()
        self.wave.seekRequested.connect(self.player.seek)
        wv.addWidget(self.wave, 1)
        tr = QHBoxLayout()
        self.play_btn = self._transport(icons.play(), "Play / pause (Space)", self.player.toggle)
        self.stop_btn = self._transport(icons.stop(), "Stop", self.player.stop)
        self.loop_btn = self._transport(icons.loop(), "Loop playback", None)
        self.loop_btn.setCheckable(True)
        self.loop_btn.toggled.connect(self.player.set_loop)
        for b in (self.play_btn, self.stop_btn, self.loop_btn):
            tr.addWidget(b)
        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setStyleSheet("font-family: Consolas, 'Cascadia Mono', monospace; font-size: 10.5pt;")
        tr.addSpacing(8)
        tr.addWidget(self.time_label)
        tr.addStretch()
        vol_icon = QLabel()
        vol_icon.setPixmap(icons.volume().pixmap(18, 18))
        tr.addWidget(vol_icon)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(110)
        self.volume.valueChanged.connect(lambda x: self.player.set_volume(x / 100))
        tr.addWidget(self.volume)
        wv.addLayout(tr)
        wave_panel.setMinimumHeight(230)
        v.addWidget(wave_panel)

        # --- result cards
        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.card_bpm = ResultCard("Tempo (BPM)")
        self.card_key = ResultCard("Key · Camelot")
        self.card_root = ResultCard("Root note")
        self.card_type = ResultCard("Audio type")
        for c in (self.card_bpm, self.card_key, self.card_root, self.card_type):
            cards.addWidget(c)
        v.addLayout(cards)

        # --- alternatives + corrections / evidence
        lower = QHBoxLayout()
        lower.setSpacing(10)
        alt_panel = QFrame()
        alt_panel.setObjectName("panel")
        av = QVBoxLayout(alt_panel)
        av.setContentsMargins(14, 12, 14, 12)
        av.setSpacing(12)
        self.alt_bpm = AlternativesList("Alternative tempos")
        self.alt_key = AlternativesList("Alternative keys")
        self.alt_pitch = AlternativesList("Dominant pitches")
        av.addWidget(self.alt_bpm)
        av.addWidget(self.alt_key)
        av.addWidget(self.alt_pitch)
        av.addStretch()
        lower.addWidget(alt_panel, 1)

        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(self._build_corrections())
        right.addWidget(self._build_bass_panel())
        ev_panel = QFrame()
        ev_panel.setObjectName("panel")
        evl = QVBoxLayout(ev_panel)
        evl.setContentsMargins(14, 12, 14, 12)
        head = QLabel("WHY THESE RESULTS")
        head.setObjectName("sectionTitle")
        evl.addWidget(head)
        self.evidence = EvidenceList()
        evl.addWidget(self.evidence, 1)
        right.addWidget(ev_panel, 1)
        lower.addLayout(right, 1)
        v.addLayout(lower, 1)
        self._set_detail_enabled(False)
        return scroll

    def _transport(self, icon, tip, slot):
        b = QPushButton()
        b.setObjectName("transport")
        b.setIcon(icon)
        b.setToolTip(tip)
        if slot:
            b.clicked.connect(slot)
        return b

    def _build_corrections(self):
        panel = QFrame()
        panel.setObjectName("panel")
        g = QGridLayout(panel)
        g.setContentsMargins(14, 12, 14, 12)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(8)
        head = QLabel("MANUAL CORRECTION")
        head.setObjectName("sectionTitle")
        g.addWidget(head, 0, 0, 1, 4)

        g.addWidget(QLabel("BPM"), 1, 0)
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(0.0, 400.0)
        self.bpm_spin.setDecimals(2)
        self.bpm_spin.setSpecialValueText("N/A")
        self.bpm_spin.setKeyboardTracking(False)
        self.bpm_spin.editingFinished.connect(self._commit_bpm)
        g.addWidget(self.bpm_spin, 1, 1)
        bh = QHBoxLayout()
        for text, factor in (("½×", 0.5), ("2×", 2.0)):
            b = QPushButton(text)
            b.setObjectName("small")
            b.setToolTip("Half-time" if factor < 1 else "Double-time")
            b.clicked.connect(lambda _=False, f=factor: self._scale_bpm(f))
            bh.addWidget(b)
        self.tap_btn = QPushButton("Tap")
        self.tap_btn.setObjectName("small")
        self.tap_btn.setToolTip("Tap along with the beat (4+ taps)")
        self.tap_btn.clicked.connect(self._tap)
        bh.addWidget(self.tap_btn)
        g.addLayout(bh, 1, 2, 1, 2)

        g.addWidget(QLabel("Key"), 2, 0)
        self.key_combo = QComboBox()
        self.key_combo.addItem("Detected", None)
        for pc, mode in KEY_CHOICES:
            self.key_combo.addItem(f"{music.key_name(pc, mode)}  ·  {music.camelot(pc, mode)}", (pc, mode))
        self.key_combo.addItem("N/A (no key)", "na")
        self.key_combo.setMaxVisibleItems(16)
        self.key_combo.activated.connect(self._commit_key)
        g.addWidget(self.key_combo, 2, 1, 1, 3)

        g.addWidget(QLabel("Type"), 3, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItem("Auto-detect", None)
        for t in AudioType:
            self.type_combo.addItem(t.label, t.value)
        self.type_combo.setToolTip("Changing the type re-analyses the file with that type's strategy")
        self.type_combo.activated.connect(self._commit_type)
        g.addWidget(self.type_combo, 3, 1, 1, 2)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("small")
        self.reset_btn.setToolTip("Discard all manual corrections for this file")
        self.reset_btn.clicked.connect(self._reset_corrections)
        g.addWidget(self.reset_btn, 3, 3)
        g.setColumnStretch(1, 1)
        self.corr_panel = panel
        return panel

    def _build_bass_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("BASS → MIDI")
        title.setObjectName("sectionTitle")
        head.addWidget(title)
        head.addStretch()
        self.bass_badge = ReliabilityBadge()
        head.addWidget(self.bass_badge)
        v.addLayout(head)
        self.bass_summary = QLabel()
        self.bass_summary.setObjectName("muted")
        self.bass_summary.setWordWrap(True)
        self.bass_summary.setTextFormat(Qt.RichText)
        v.addWidget(self.bass_summary)
        row = QHBoxLayout()
        self.bass_btn = QPushButton("Detect bass notes")
        self.bass_btn.setToolTip("Transcribe the bass line (or 808 / tuned kick) into notes")
        self.bass_btn.clicked.connect(self._detect_bass)
        row.addWidget(self.bass_btn)
        self.quant_chk = QCheckBox("Quantize to 1/16")
        self.quant_chk.setToolTip("Snap note starts and ends to 16th notes at the file's BPM when exporting")
        row.addWidget(self.quant_chk)
        row.addStretch()
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.midi_btn = QPushButton("Export MIDI…")
        self.midi_btn.clicked.connect(self._export_midi)
        row2.addWidget(self.midi_btn)
        self.drag_chip = FileDragChip("⠿  Drag MIDI into FL Studio", self._midi_temp_file)
        row2.addWidget(self.drag_chip)
        row2.addStretch()
        v.addLayout(row2)
        self.bass_panel = panel
        return panel

    # =================================================================== files
    def browse_files(self):
        exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio files", "", f"Audio ({exts});;All files (*)")
        if files:
            self.add_paths(collect_audio_paths(files))

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder (scanned recursively)")
        if d:
            paths = collect_audio_paths([d])
            if not paths:
                QMessageBox.information(self, "AudioScope", "No WAV, MP3 or FLAC files found in that folder.")
            self.add_paths(paths)

    def add_paths(self, paths: list[str]):
        paths = [os.path.normpath(p) for p in paths]
        new = self.model.add(paths)
        for e in new:
            e.correction = self.store.get(e.path)
            self.manager.submit(e.path, e.correction.audio_type)
        if new:
            self.stack.setCurrentWidget(self.table)
            if self.current is None:
                self._select_path(new[0].path)
        self._update_status()

    def remove_selected(self):
        rows = [self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()]
        if not rows:
            return
        if self.current and self.model.row_of(self.current) in rows:
            self.player.unload()
            self.current = None
            self._show_entry(None)
        self.model.remove_rows(rows)
        if not self.model.entries:
            self.stack.setCurrentWidget(self.drop_hint)
        self._update_status()

    def clear_all(self):
        self.manager.cancel_pending()
        self.player.unload()
        self.model.clear()
        self.current = None
        self._show_entry(None)
        self.stack.setCurrentWidget(self.drop_hint)
        self._update_status()

    def reanalyse(self):
        rows = [self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()]
        entries = [self.model.entries[r] for r in rows] if rows else list(self.model.entries)
        for e in entries:
            e.status = STATUS_QUEUED
            self.model.touch(e.path)
            self.manager.submit(e.path, e.correction.audio_type)
        self._update_status()

    def _table_menu(self, pos):
        m = QMenu(self)
        m.addAction("Re-analyse", self.reanalyse)
        m.addAction("Show in Explorer", self._reveal)
        m.addSeparator()
        m.addAction("Remove", self.remove_selected)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def _reveal(self):
        if self.current and os.name == "nt":
            import subprocess

            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.current)])

    # =================================================================== analysis callbacks
    def _on_started(self, path):
        e = self.model.entry(path)
        if e:
            e.status = STATUS_RUNNING
            self.model.touch(path)
        self._update_status()

    def _on_finished(self, path, d):
        e = self.model.entry(path)
        if not e:
            return
        e.result = AnalysisResult.from_dict(d)
        e.status = STATUS_DONE
        e.error = ""
        self.model.touch(path)
        if path == self.current:
            self._show_entry(e)
        self._update_status()

    def _on_failed(self, path, err):
        e = self.model.entry(path)
        if not e:
            return
        e.status = STATUS_ERROR
        e.error = err
        self.model.touch(path)
        if path == self.current:
            self._show_entry(e)
        self._update_status()

    def _update_status(self):
        n = len(self.model.entries)
        done = sum(e.status == STATUS_DONE for e in self.model.entries)
        err = sum(e.status == STATUS_ERROR for e in self.model.entries)
        pending = self.manager.pending()
        self.count_label.setText(f"{n} file{'s' if n != 1 else ''}" if n else "")
        parts = [f"{done}/{n} analysed"] if n else ["Ready"]
        if pending:
            parts.append(f"{pending} in progress ({self.manager.workers} workers)")
        if err:
            parts.append(f"{err} failed")
        parts.append(f"engine {ENGINE_VERSION}")
        self.statusBar().showMessage("   ·   ".join(parts))

    # =================================================================== selection / detail
    def _select_path(self, path):
        row = self.model.row_of(path)
        if row < 0:
            return
        idx = self.proxy.mapFromSource(self.model.index(row, 0))
        self.table.selectRow(idx.row())

    def _on_selection(self, *_):
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return
        e = self.model.entries[self.proxy.mapToSource(idxs[0]).row()]
        if e.path != self.current:
            self.current = e.path
            self._load_waveform(e.path)
            self.player.load(e.path)
        self._show_entry(e)

    def _load_waveform(self, path):
        if path in self.peaks_cache:
            self.peaks_cache.move_to_end(path)
            self.wave.set_peaks(self.peaks_cache[path])
            self._on_position(0.0)
            return
        self.wave.set_peaks(None, "Loading waveform…")
        job = PeakJob(path)
        job.signals.done.connect(self._on_peaks)
        job.signals.failed.connect(lambda p, m: p == self.current and self.wave.set_peaks(None, f"Cannot read audio: {m}"))
        QThreadPool.globalInstance().start(job)

    def _on_peaks(self, path, peaks):
        self.peaks_cache[path] = peaks
        while len(self.peaks_cache) > 24:
            self.peaks_cache.popitem(last=False)
        if path == self.current:
            self.wave.set_peaks(peaks)
            self._on_position(0.0)
            e = self.model.entry(path)
            if e:
                self._update_grid(e)

    def _set_detail_enabled(self, on: bool):
        for w in (self.corr_panel, self.copy_btn, self.play_btn, self.stop_btn, self.loop_btn):
            w.setEnabled(on)

    def _show_entry(self, e: Entry | None):
        self._updating = True
        try:
            self._fill_detail(e)
        finally:
            self._updating = False

    def _fill_detail(self, e: Entry | None):
        empty_cards = ((self.card_bpm, "—"), (self.card_key, "—"), (self.card_root, "—"), (self.card_type, "—"))
        if e is None:
            self.file_title.setText("No file selected")
            self.file_meta.setText("Add audio to analyse key, tempo and pitch.")
            for c, v in empty_cards:
                c.set(v, dim=True)
            self.wave.set_peaks(None, "Drop audio files to begin")
            self.alt_bpm.set_rows([])
            self.alt_key.set_rows([])
            self.alt_pitch.set_rows([])
            self.evidence.set_sections([])
            self._set_detail_enabled(False)
            self._fill_bass(None)
            return
        self.file_title.setText(e.name)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.loop_btn.setEnabled(True)
        r = e.result
        if r is None:
            msg = {STATUS_ERROR: f"Analysis failed: {e.error}", STATUS_RUNNING: "Analysing…"}.get(e.status, "Queued…")
            self.file_meta.setText(msg)
            for c, _ in empty_cards:
                c.set("…" if e.status != STATUS_ERROR else "—", dim=True)
            self.alt_bpm.set_rows([])
            self.alt_key.set_rows([])
            self.alt_pitch.set_rows([])
            self.evidence.set_sections([("Error", [e.error])] if e.error else [])
            self.corr_panel.setEnabled(False)
            self.copy_btn.setEnabled(False)
            self._fill_bass(None)
            return
        self._set_detail_enabled(True)
        ch = "mono" if r.channels == 1 else "stereo" if r.channels == 2 else f"{r.channels} ch"
        ext = os.path.splitext(e.path)[1].upper().lstrip(".")
        self.file_meta.setText(f"{fmt_time(r.duration, True)}  ·  {r.sample_rate / 1000:g} kHz {ch}  ·  {ext}"
                               f"  ·  analysed in {r.elapsed:.1f} s")
        self._fill_cards(e, r)
        self._fill_alternatives(r)
        self._fill_evidence(r)
        self._fill_corrections(e, r)
        self._fill_bass(e)
        self._update_grid(e)

    def _fill_cards(self, e: Entry, r: AnalysisResult):
        t = r.tempo
        if e.bpm_edited():
            self.card_bpm.set(format_bpm(e.correction.bpm), f"detected: {t.display}", edited=True)
        elif t.bpm is None:
            self.card_bpm.set("N/A", t.evidence[-1] if t.evidence else "", Reliability.NONE, dim=True)
        else:
            sub = []
            if t.bars:
                sub.append(f"{t.bars} bar{'s' if t.bars > 1 else ''} of 4/4")
            if t.octave_ambiguous:
                alt = next((c for c in t.candidates if c.relation in ("half-time", "double-time")), None)
                if alt:
                    sub.append(f"or {format_bpm(alt.bpm)} ({alt.relation})")
            if "section_agreement" in t.metrics:
                sub.append(f"{t.metrics['section_agreement'] * 100:.0f}% of sections agree")
            self.card_bpm.set(t.display, " · ".join(sub), t.reliability)

        k = r.key
        if e.key_edited():
            kk = e.key()
            txt = self._key_rich(*kk) if kk else "N/A"
            self.card_key.set(txt, f"detected: {k.display}", edited=True, dim=kk is None)
        elif k.key is None:
            self.card_key.set("N/A", k.evidence[-1] if k.evidence else "", Reliability.NONE, dim=True)
        else:
            sub = [k.key.short_name]
            if k.tuning_cents is not None and abs(k.tuning_cents) >= 10:
                sub.append(f"tuned {k.tuning_cents:+.0f}¢")
            rel_alt = music.relative_key(k.key.tonic, k.key.mode)
            if k.candidates[1:2] and (k.candidates[1].tonic, k.candidates[1].mode) == rel_alt \
                    and k.reliability != Reliability.HIGH:
                sub.append(f"or relative {music.key_name(*rel_alt)}")
            self.card_key.set(self._key_rich(k.key.tonic, k.key.mode), " · ".join(sub), k.reliability)

        p = r.pitch
        if p.root_midi is None:
            self.card_root.set("N/A", p.evidence[-1] if p.evidence else "", Reliability.NONE, dim=True)
        else:
            value = p.chord if p.chord else p.root_name
            sub = [p.root_kind]
            if p.chord:
                sub.append(f"root {p.root_name}")
            if p.root_hz:
                sub.append(f"{p.root_hz:.1f} Hz")
            if p.root_cents is not None and abs(p.root_cents) >= 5:
                sub.append(f"{p.root_cents:+.0f}¢")
            self.card_root.set(value, " · ".join(sub), p.reliability)

        c = r.classification
        detail = f"{c.detail} · " if c.detail else ""
        if c.manual:
            self.card_type.set(c.audio_type.label, f"{detail}set manually", edited=True)
        else:
            self.card_type.set(c.audio_type.label, f"{detail}auto-detected", c.reliability)

    @staticmethod
    def _key_rich(tonic: int, mode: str) -> str:
        return (f"{music.key_name(tonic, mode)} <span style='color:{theme.ACCENT}; font-size:17pt'>&nbsp;"
                f"{music.camelot(tonic, mode)}</span>")

    def _fill_alternatives(self, r: AnalysisResult):
        t = r.tempo
        rows = []
        for c in t.candidates:
            if c.relation == "primary":
                continue
            label = {"loop-length": "from loop length only"}.get(c.relation, c.relation)
            rows.append((f"{format_bpm(c.bpm)} BPM", label, c.strength if c.relation != "loop-length" else None))
        self.alt_bpm.set_rows(rows, "No alternatives")
        krows = []
        best = r.key.key
        for c in (r.key.candidates if best else []):  # nothing to offer when the key is N/A
            if best and (c.tonic, c.mode) == (best.tonic, best.mode):
                continue
            rel = ""
            if best:
                if (c.tonic, c.mode) == music.relative_key(best.tonic, best.mode):
                    rel = "relative · "
                elif c.mode == best.mode and (c.tonic - best.tonic) % 12 in (5, 7):
                    rel = "neighbour (fifth) · "
                elif c.tonic == best.tonic:
                    rel = "parallel · "
            krows.append((f"{c.short_name}  {c.camelot}", f"{rel}r = {c.score:.2f}", None))
        self.alt_key.set_rows(krows[:4], "No alternatives" if best else "None – no key detected")
        self.alt_pitch.set_rows([(name, "", strength) for name, strength in r.pitch.dominant_pitches], "None")

    def _fill_evidence(self, r: AnalysisResult):
        sections = [
            ("Type", r.classification.evidence),
            ("Tempo", r.tempo.evidence),
            ("Key", r.key.evidence),
            ("Root note", r.pitch.evidence),
        ]
        hints = r.filename_hints
        if hints:
            items = []
            if "bpm" in hints:
                agree = r.tempo.bpm and abs(hints["bpm"] - r.tempo.bpm) < 0.6
                items.append(f"Filename says {format_bpm(hints['bpm'])} BPM" + (" (agrees)" if agree else " (differs from detection)"))
            if "key" in hints:
                agree = r.key.key and hints["key"] == r.key.key.name
                items.append(f"Filename says {hints['key']} ({hints['camelot']})" + (" (agrees)" if agree else " (differs from detection)"))
            sections.append(("Filename tags (not used for detection)", items))
        if r.warnings:
            sections.append(("Warnings", r.warnings))
        self.evidence.set_sections(sections)

    def _fill_corrections(self, e: Entry, r: AnalysisResult):
        self.bpm_spin.setValue(e.bpm() or 0.0)
        if e.correction.key_na:
            self.key_combo.setCurrentIndex(self.key_combo.count() - 1)
        elif e.correction.key is not None:
            self.key_combo.setCurrentIndex(1 + KEY_CHOICES.index(tuple(e.correction.key)))
        else:
            self.key_combo.setCurrentIndex(0)
        self.key_combo.setItemText(0, f"Detected ({r.key.key.short_name if r.key.key else 'N/A'})")
        idx = 0 if not e.correction.audio_type else 1 + [t.value for t in AudioType].index(e.correction.audio_type)
        self.type_combo.setCurrentIndex(idx)

    def _update_grid(self, e: Entry):
        r = e.result
        if not r or not self.wave.peaks:
            self.wave.set_grid([])
            return
        bpm = e.bpm()
        exact = bool(r.tempo.bars) and not e.bpm_edited()
        beats = r.tempo.beat_times if not e.bpm_edited() else []
        self.wave.grid_from_tempo(bpm, self.wave.peaks.duration, beats, exact or (e.bpm_edited() and r.classification.audio_type.is_loop))

    # =================================================================== corrections
    def _entry(self) -> Entry | None:
        return self.model.entry(self.current) if self.current else None

    def _save_correction(self, e: Entry):
        self.store.put(e.path, e.correction)
        self.model.touch(e.path)
        self._show_entry(e)

    def _commit_bpm(self):
        e = self._entry()
        if self._updating or not e or not e.result:
            return
        v = round(self.bpm_spin.value(), 2)
        detected = e.result.tempo.bpm
        e.correction.bpm = None if (v == 0 and detected is None) or (detected and abs(v - detected) < 0.005) else (v or None)
        self._save_correction(e)

    def _scale_bpm(self, f: float):
        e = self._entry()
        if not e or not e.bpm():
            return
        self.bpm_spin.setValue(round(e.bpm() * f, 2))
        self._commit_bpm()

    def _tap(self):
        now = time.perf_counter()
        if self._taps and now - self._taps[-1] > 2.0:
            self._taps = []
        self._taps.append(now)
        self._taps = self._taps[-9:]
        if len(self._taps) >= 4:
            intervals = sorted(b - a for a, b in zip(self._taps, self._taps[1:]))
            med = intervals[len(intervals) // 2]
            self.bpm_spin.setValue(round(60.0 / med, 1))
            self._commit_bpm()
        self.tap_btn.setText(f"Tap ({len(self._taps)})")
        QTimer.singleShot(2100, lambda: (time.perf_counter() - self._taps[-1] > 2.0 if self._taps else True)
                          and self.tap_btn.setText("Tap"))

    def _commit_key(self, _idx):
        e = self._entry()
        if self._updating or not e or not e.result:
            return
        data = self.key_combo.currentData()
        e.correction.key_na = data == "na"
        e.correction.key = tuple(data) if isinstance(data, tuple) else None
        det = e.result.key.key
        if e.correction.key and det and e.correction.key == (det.tonic, det.mode):
            e.correction.key = None
        if e.correction.key_na and det is None:
            e.correction.key_na = False
        self._save_correction(e)

    def _commit_type(self, _idx):
        e = self._entry()
        if self._updating or not e:
            return
        e.correction.audio_type = self.type_combo.currentData()
        self.store.put(e.path, e.correction)
        e.bass = None  # extraction mode depends on the type
        e.status = STATUS_QUEUED
        self.model.touch(e.path)
        self.manager.submit(e.path, e.correction.audio_type)
        self._show_entry(e)
        self._update_status()

    def _reset_corrections(self):
        e = self._entry()
        if not e:
            return
        retype = e.correction.audio_type is not None
        e.correction = Correction()
        self.store.put(e.path, e.correction)
        if retype:
            e.bass = None
            e.status = STATUS_QUEUED
            self.manager.submit(e.path, None)
        self.model.touch(e.path)
        self._show_entry(e)

    # =================================================================== bass -> MIDI
    def _detect_bass(self):
        e = self._entry()
        if not e or not e.result:
            return
        e.bass_error = ""
        self.manager.submit_bass(e.path, e.result.classification.audio_type.value)
        self._fill_bass(e)
        self._update_status()

    def _on_bass_finished(self, path, d):
        e = self.model.entry(path)
        if not e:
            return
        e.bass = BassTranscription.from_dict(d)
        e.bass_error = ""
        if path == self.current:
            self._fill_bass(e)
        self._update_status()
        self._continue_bass_export(e)  # after the status update, so its message stays visible

    def _on_bass_failed(self, path, err):
        e = self.model.entry(path)
        if not e:
            return
        e.bass, e.bass_error = None, err
        if path == self.current:
            self._fill_bass(e)
        self._update_status()
        self._continue_bass_export(e)  # after the status update, so its message stays visible

    def _fill_bass(self, e: Entry | None):
        running = bool(e) and self.manager.bass_running(e.path)
        has_notes = bool(e and e.bass and e.bass.notes)
        self.bass_btn.setEnabled(bool(e and e.result) and not running)
        self.bass_btn.setText("Detecting…" if running else "Detect again" if e and e.bass else "Detect bass notes")
        self.midi_btn.setEnabled(has_notes)
        self.drag_chip.setEnabled(has_notes)
        bpm = e.bpm() if e and e.result else None
        self.quant_chk.setEnabled(bool(bpm))
        self.quant_chk.setText(f"Quantize to 1/16 at {format_bpm(bpm)} BPM" if bpm else "Quantize (needs a BPM)")
        self.bass_badge.set_reliability(e.bass.reliability if e and e.bass else None)
        if e is None or e.result is None:
            self.bass_summary.setText("Find the notes of the bass line and bounce them to MIDI.")
        elif e.bass_error:
            self.bass_summary.setText(f"Transcription failed: {e.bass_error}")
        elif running:
            self.bass_summary.setText("Tracking the bass line…")
        elif e.bass is None:
            self.bass_summary.setText("Find the notes of the bass line and bounce them to MIDI.")
        elif not has_notes:
            self.bass_summary.setText(e.bass.evidence[-1] if e.bass.evidence else "No bass notes found.")
        else:
            b = e.bass
            name = self._note_namer(e)
            names = " ".join(name(n.midi) for n in b.notes[:24]) + (" …" if len(b.notes) > 24 else "")
            lo, hi = min(n.midi for n in b.notes), max(n.midi for n in b.notes)
            span = name(lo) if lo == hi else f"{name(lo)}–{name(hi)}"
            tuning = f" · tuned {b.tuning_cents:+.0f}¢" if abs(b.tuning_cents) >= 10 else ""
            caveat = "" if b.reliability == Reliability.HIGH else \
                f"<br><span style='color:{theme.FAINT}'>{b.evidence[0]}</span>"
            self.bass_summary.setText(
                f"<span style='color:{theme.TEXT}'><b>{len(b.notes)} note{'s' if len(b.notes) != 1 else ''}</b>"
                f" · {span}{tuning}</span><br>{names}{caveat}")
        self.wave.set_notes(e.bass.notes if has_notes else [], self._note_namer(e) if has_notes else None)

    @staticmethod
    def _note_namer(e: Entry):
        """Note-name function spelled for the entry's key (Eb in G minor, D# in E major)."""
        k = e.key()
        flats = music.key_uses_flats(*k) if k else False
        return lambda m: music.midi_to_name(m, flats)

    def _bass_midi(self, e: Entry) -> tuple[bytes, str]:
        """MIDI bytes and a file name for the entry's bass line (quantized if requested)."""
        bpm = e.bpm()
        notes = quantize(e.bass.notes, bpm) if self.quant_chk.isChecked() and bpm else e.bass.notes
        stem = os.path.splitext(e.name)[0]
        return notes_to_midi(notes, bpm, track_name=f"{stem} bass"), f"{stem}_bass.mid"

    def _export_midi(self):
        e = self._entry()
        if not e or not e.bass or not e.bass.notes:
            return
        data, name = self._bass_midi(e)
        path, _ = QFileDialog.getSaveFileName(self, "Export bass MIDI", os.path.join(os.path.dirname(e.path), name),
                                              "MIDI (*.mid)")
        if not path:
            return
        try:
            with open(path, "wb") as f:
                f.write(data)
            self.statusBar().showMessage(f"Saved {path}", 5000)
        except OSError as err:
            QMessageBox.warning(self, "Export failed", str(err))

    def _midi_temp_file(self) -> str | None:
        e = self._entry()
        if not e or not e.bass or not e.bass.notes:
            return None
        data, name = self._bass_midi(e)
        folder = os.path.join(tempfile.gettempdir(), "AudioScope", "midi")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def export_bass_batch(self):
        entries = [e for e in self.model.entries if e.result]
        if not entries:
            QMessageBox.information(self, "AudioScope", "Nothing analysed yet.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for bass MIDI files")
        if not folder:
            return
        self._bass_export_dir, self._bass_export_pending, self._bass_export_written = folder, set(), 0
        for e in entries:
            if e.bass is not None:
                self._write_bass_file(e)
            else:
                self._bass_export_pending.add(e.path)
                self.manager.submit_bass(e.path, e.result.classification.audio_type.value)
        self._report_bass_export()

    def _continue_bass_export(self, e: Entry):
        if e.path in self._bass_export_pending:
            self._bass_export_pending.discard(e.path)
            self._write_bass_file(e)
            self._report_bass_export()

    def _write_bass_file(self, e: Entry):
        if not self._bass_export_dir or not e.bass or not e.bass.notes:
            return
        data, name = self._bass_midi(e)
        try:
            with open(os.path.join(self._bass_export_dir, name), "wb") as f:
                f.write(data)
            self._bass_export_written += 1
        except OSError as err:
            self.statusBar().showMessage(f"Could not write {name}: {err}", 5000)

    def _report_bass_export(self):
        left = len(self._bass_export_pending)
        msg = f"Bass MIDI: {self._bass_export_written} file(s) written to {self._bass_export_dir}"
        self.statusBar().showMessage(msg + (f" · {left} still transcribing" if left else ""), 8000)

    # =================================================================== playback
    def _on_position(self, sec):
        self.wave.set_position(sec)
        dur = self.wave.peaks.duration if self.wave.peaks else 0.0
        self.time_label.setText(f"{fmt_time(sec, True)} / {fmt_time(dur, True)}")

    def _on_playing(self, playing):
        self.play_btn.setIcon(icons.pause() if playing else icons.play())

    # =================================================================== clipboard / export
    def copy_result(self):
        e = self._entry()
        if not e or not e.result:
            return
        parts = [e.name]
        if e.bpm():
            parts.append(f"{format_bpm(e.bpm())} BPM")
        k = e.key()
        if k:
            parts.append(f"{music.key_name(*k)} ({music.camelot(*k)})")
        p = e.result.pitch
        if p.root_midi is not None:
            parts.append(f"root {p.chord or p.root_name}")
        QGuiApplication.clipboard().setText(" · ".join(parts))
        self.statusBar().showMessage("Copied: " + " · ".join(parts[1:]), 3000)

    def export(self, kind: str):
        entries = [e for e in self.model.entries if e.result]
        if not entries:
            QMessageBox.information(self, "AudioScope", "Nothing analysed yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", f"audioscope_results.{kind}",
                                              "CSV (*.csv)" if kind == "csv" else "JSON (*.json)")
        if not path:
            return
        try:
            if kind == "csv":
                write_csv(path, entries)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([{"result": e.result.to_dict(), "correction": e.correction.to_json()} for e in entries],
                              f, indent=1)
            self.statusBar().showMessage(f"Exported {len(entries)} results to {path}", 5000)
        except OSError as err:
            QMessageBox.warning(self, "Export failed", str(err))

    def closeEvent(self, ev):
        self.player.unload()
        self.manager.shutdown()
        super().closeEvent(ev)


CSV_FIELDS = ["file", "path", "duration_s", "type", "type_reliability", "type_set_manually",
              "bpm", "bpm_reliability", "bpm_detected", "bpm_alternatives", "loop_bars",
              "key", "camelot", "key_reliability", "key_detected", "key_alternatives", "tuning_cents",
              "root_note", "root_cents", "chord", "root_reliability", "manually_corrected",
              "filename_bpm", "filename_key"]


def write_csv(path: str, entries: list[Entry]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for e in entries:
            r = e.result
            k = e.key()
            w.writerow({
                "file": e.name, "path": e.path, "duration_s": r.duration,
                "type": r.classification.audio_type.label, "type_reliability": r.classification.reliability.label,
                "type_set_manually": r.classification.manual,
                "bpm": format_bpm(e.bpm()) if e.bpm() else "N/A",
                "bpm_reliability": "edited" if e.bpm_edited() else r.tempo.reliability.label,
                "bpm_detected": r.tempo.display,
                "bpm_alternatives": "; ".join(f"{format_bpm(c.bpm)} ({c.relation})" for c in r.tempo.candidates
                                              if c.relation != "primary"),
                "loop_bars": r.tempo.bars or "",
                "key": music.key_name(*k) if k else "N/A", "camelot": music.camelot(*k) if k else "N/A",
                "key_reliability": "edited" if e.key_edited() else r.key.reliability.label,
                "key_detected": r.key.display,
                "key_alternatives": "; ".join(f"{c.name} ({c.camelot})" for c in r.key.candidates[1:4]) if r.key.key else "",
                "tuning_cents": r.key.tuning_cents if r.key.tuning_cents is not None else "",
                "root_note": r.pitch.root_name or "N/A", "root_cents": r.pitch.root_cents if r.pitch.root_cents is not None else "",
                "chord": r.pitch.chord or "", "root_reliability": r.pitch.reliability.label,
                "manually_corrected": not e.correction.is_empty(),
                "filename_bpm": r.filename_hints.get("bpm", ""), "filename_key": r.filename_hints.get("key", ""),
            })
