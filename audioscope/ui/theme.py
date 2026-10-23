"""Dark theme: palette, reliability colours and the application stylesheet."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from ..engine.types import Reliability

BG = "#0F1114"
PANEL = "#16191E"
RAISED = "#1D2127"
RAISED_HI = "#252A31"
BORDER = "#2A3038"
TEXT = "#E8EAED"
MUTED = "#8C939E"
FAINT = "#5C636E"
ACCENT = "#FF8A3D"  # warm orange (transport, selection, playhead)
ACCENT_DIM = "#7A4524"
WAVE = "#3CC7BC"  # teal waveform
WAVE_PLAYED = "#FFB27A"

RELIABILITY_COLORS = {
    Reliability.HIGH: "#3DDC84",
    Reliability.MEDIUM: "#F2C14E",
    Reliability.LOW: "#FF7A59",
    Reliability.NONE: "#5C636E",
}


def rel_color(rel: Reliability) -> QColor:
    return QColor(RELIABILITY_COLORS[rel])


STYLESHEET = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#central {{ background: {BG}; }}
QToolTip {{ background: {RAISED_HI}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#appTitle {{ font-size: 15pt; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#appSubtitle {{ color: {MUTED}; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1.2px; }}
QLabel#cardValue {{ font-size: 21pt; font-weight: 600; }}
QLabel#cardSub {{ color: {MUTED}; font-size: 9.5pt; }}
QLabel#sectionTitle {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 1.2px; }}
QLabel#fileTitle {{ font-size: 12.5pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#editedChip {{ background: {ACCENT_DIM}; color: #FFD2B3; border-radius: 6px; padding: 1px 6px; font-size: 8pt; }}
QPushButton {{ background: {RAISED_HI}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {FAINT}; border-color: {RAISED_HI}; }}
QPushButton#primary {{ background: {ACCENT}; color: #1A0E06; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: #FF9E5C; }}
QPushButton#transport {{ border-radius: 18px; padding: 0; min-width: 36px; max-width: 36px; min-height: 36px; max-height: 36px; }}
QPushButton#transport:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QPushButton#small {{ padding: 3px 8px; font-size: 9pt; }}
QComboBox, QDoubleSpinBox, QLineEdit {{ background: {RAISED_HI}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px; selection-background-color: {ACCENT}; }}
QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {FAINT}; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM}; }}
QTableView {{ background: {PANEL}; alternate-background-color: #181C21; border: none; gridline-color: transparent;
             selection-background-color: #2B2219; selection-color: {TEXT}; }}
QTableView::item {{ padding: 4px; border-bottom: 1px solid #1E2228; }}
QTableView::item:selected {{ background: #33261B; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
                        padding: 6px 4px; font-size: 8.5pt; font-weight: 600; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }}
QSplitter::handle {{ background: {BG}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QScrollArea {{ background: transparent; border: none; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, PANEL), (QPalette.AlternateBase, RAISED),
        (QPalette.Text, TEXT), (QPalette.Button, RAISED_HI), (QPalette.ButtonText, TEXT),
        (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#1A0E06"), (QPalette.ToolTipBase, RAISED_HI),
        (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, FAINT),
    ):
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(FAINT))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(FAINT))
    app.setPalette(pal)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)
