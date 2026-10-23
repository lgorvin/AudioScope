"""GUI entry point."""
from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile


def _prepare_environment():
    # numba (used by librosa) caches compiled code; a frozen app needs a writable location
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "audioscope_numba"))


def _selftest(audio_path: str, out_path: str) -> int:
    """Headless check of a packaged build: analyse a file in-process and in a worker process."""
    import json
    from concurrent.futures import ProcessPoolExecutor

    from .ui.jobs import _analyze_in_worker

    report = {"in_process": _analyze_in_worker(audio_path, None)}
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        report["worker"] = ex.submit(_analyze_in_worker, audio_path, None).result(timeout=300)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return 0 if report["in_process"].get("ok") and report["worker"].get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    _prepare_environment()
    argv = sys.argv if argv is None else argv
    if len(argv) == 4 and argv[1] == "--selftest":
        return _selftest(argv[2], argv[3])

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from .engine.io import is_supported
    from .ui.main_window import MainWindow
    from .ui.theme import apply_theme
    from .ui.widgets import collect_audio_paths

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName("AudioScope")
    app.setOrganizationName("AudioScope")
    apply_theme(app)
    win = MainWindow()
    win.show()
    files = collect_audio_paths([a for a in argv[1:] if os.path.isdir(a) or is_supported(a)])
    if files:
        win.add_paths(files)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
