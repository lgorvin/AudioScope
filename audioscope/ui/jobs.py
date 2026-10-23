"""Background analysis in worker processes (keeps the UI responsive and uses all cores)."""
from __future__ import annotations

import multiprocessing
import os
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor

from PySide6.QtCore import QObject, Signal


def _analyze_in_worker(path: str, audio_type: str | None) -> dict:
    """Runs in a worker process. Returns a JSON-able dict (results cross the process boundary)."""
    try:
        from ..engine import AudioType, analyze_file

        result = analyze_file(path, AudioType(audio_type) if audio_type else None)
        return {"ok": True, "result": result.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _transcribe_in_worker(path: str, audio_type: str | None) -> dict:
    """Bass transcription in a worker process."""
    try:
        from ..engine import AudioType, transcribe_file

        return {"ok": True, "result": transcribe_file(path, AudioType(audio_type) if audio_type else None).to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


class AnalysisManager(QObject):
    started = Signal(str)
    finished = Signal(str, dict)  # path, result dict
    failed = Signal(str, str)
    bassFinished = Signal(str, dict)  # path, BassTranscription dict
    bassFailed = Signal(str, str)
    _done = Signal(str, dict)  # marshals results from executor threads onto the GUI thread
    _bass_done = Signal(str, dict)

    def __init__(self, workers: int | None = None, parent=None):
        super().__init__(parent)
        self.workers = workers or default_workers()
        self._pool: ProcessPoolExecutor | None = None
        self._queue: deque[tuple[str, str | None]] = deque()
        self._running: dict[str, Future] = {}
        self._done.connect(self._on_done)
        self._bass_done.connect(self._on_bass_done)
        self._bass_running: set[str] = set()

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))

    def submit(self, path: str, audio_type: str | None = None):
        if path in self._running or any(p == path for p, _ in self._queue):
            return
        self._queue.append((path, audio_type))
        self._pump()

    def pending(self) -> int:
        return len(self._queue) + len(self._running) + len(self._bass_running)

    def submit_bass(self, path: str, audio_type: str | None):
        """Bass transcription is on demand, so it skips the analysis queue."""
        if path in self._bass_running:
            return
        self._ensure_pool()
        self._bass_running.add(path)
        fut = self._pool.submit(_transcribe_in_worker, path, audio_type)
        fut.add_done_callback(lambda f, p=path: self._bass_done.emit(p, self._safe_result(f)))

    def bass_running(self, path: str) -> bool:
        return path in self._bass_running

    def _on_bass_done(self, path: str, payload: dict):
        self._bass_running.discard(path)
        if payload.get("ok"):
            self.bassFinished.emit(path, payload["result"])
        else:
            self.bassFailed.emit(path, payload.get("error", "unknown error"))

    def cancel_pending(self):
        self._queue.clear()

    def _pump(self):
        self._ensure_pool()
        while self._queue and len(self._running) < self.workers:
            path, atype = self._queue.popleft()
            fut = self._pool.submit(_analyze_in_worker, path, atype)
            self._running[path] = fut
            self.started.emit(path)
            fut.add_done_callback(lambda f, p=path: self._done.emit(p, self._safe_result(f)))

    @staticmethod
    def _safe_result(f: Future) -> dict:
        try:
            return f.result()
        except Exception as e:  # noqa: BLE001 - e.g. a worker process crashed
            return {"ok": False, "error": f"Worker failed: {e}"}

    def _on_done(self, path: str, payload: dict):
        self._running.pop(path, None)
        if payload.get("ok"):
            self.finished.emit(path, payload["result"])
        else:
            self.failed.emit(path, payload.get("error", "unknown error"))
        self._pump()

    def shutdown(self):
        self._queue.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
