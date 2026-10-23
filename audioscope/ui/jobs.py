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


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


class AnalysisManager(QObject):
    started = Signal(str)
    finished = Signal(str, dict)  # path, result dict
    failed = Signal(str, str)
    _done = Signal(str, dict)  # marshals results from executor threads onto the GUI thread

    def __init__(self, workers: int | None = None, parent=None):
        super().__init__(parent)
        self.workers = workers or default_workers()
        self._pool: ProcessPoolExecutor | None = None
        self._queue: deque[tuple[str, str | None]] = deque()
        self._running: dict[str, Future] = {}
        self._done.connect(self._on_done)

    def _ensure_pool(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))

    def submit(self, path: str, audio_type: str | None = None):
        if path in self._running or any(p == path for p, _ in self._queue):
            return
        self._queue.append((path, audio_type))
        self._pump()

    def pending(self) -> int:
        return len(self._queue) + len(self._running)

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
