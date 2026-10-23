"""Audio playback wrapper around QMediaPlayer."""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class Player(QObject):
    positionChanged = Signal(float)  # seconds
    playingChanged = Signal(bool)
    errorOccurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._out = QAudioOutput(self)
        self._out.setVolume(0.8)
        self._mp = QMediaPlayer(self)
        self._mp.setAudioOutput(self._out)
        self._mp.playbackStateChanged.connect(self._on_state)
        self._mp.errorOccurred.connect(lambda _e, msg: self.errorOccurred.emit(msg))
        self._timer = QTimer(self)
        self._timer.setInterval(30)  # smooth playhead
        self._timer.timeout.connect(lambda: self.positionChanged.emit(self._mp.position() / 1000.0))
        self._loop = False
        self.path: str | None = None

    def load(self, path: str):
        if path == self.path:
            return
        self._mp.stop()
        self.path = path
        self._mp.setSource(QUrl.fromLocalFile(path))
        self.positionChanged.emit(0.0)

    def unload(self):
        self._mp.stop()
        self._mp.setSource(QUrl())
        self.path = None

    def is_playing(self) -> bool:
        return self._mp.playbackState() == QMediaPlayer.PlayingState

    def toggle(self):
        if not self.path:
            return
        if self.is_playing():
            self._mp.pause()
        else:
            self._mp.play()

    def stop(self):
        self._mp.stop()
        self.positionChanged.emit(0.0)

    def seek(self, sec: float):
        self._mp.setPosition(int(sec * 1000))
        self.positionChanged.emit(sec)

    def set_loop(self, on: bool):
        self._loop = on
        self._mp.setLoops(QMediaPlayer.Infinite if on else 1)

    def set_volume(self, v: float):
        self._out.setVolume(max(0.0, min(1.0, v)))

    def _on_state(self, state):
        playing = state == QMediaPlayer.PlayingState
        if playing:
            self._timer.start()
        else:
            self._timer.stop()
            self.positionChanged.emit(self._mp.position() / 1000.0)
        self.playingChanged.emit(playing)
