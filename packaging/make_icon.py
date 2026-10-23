"""Render packaging/audioscope.ico from the app's vector icon (run before PyInstaller)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication, QImageWriter  # noqa: E402

from audioscope.ui.icons import app_pixmap  # noqa: E402

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audioscope.ico")
    writer = QImageWriter(out, b"ico")
    if not writer.write(app_pixmap(256).toImage()):
        raise SystemExit(f"could not write icon: {writer.errorString()}")
    print("wrote", out)
