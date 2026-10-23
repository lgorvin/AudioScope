# PyInstaller spec: one folder (dist/AudioScope) with the GUI and a console CLI sharing libraries.
#   python -m PyInstaller packaging/audioscope.spec --noconfirm
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(SPECPATH, "audioscope.ico")

datas = collect_data_files("librosa")  # lazy-loader .pyi stubs + data registry
# librosa.segment would pull in scikit-learn; AudioScope does not use it
hiddenimports = [m for m in collect_submodules("librosa") if not m.startswith("librosa.segment")]
hiddenimports += ["soxr", "scipy.signal", "scipy.ndimage"]
excludes = ["tkinter", "matplotlib", "IPython", "pandas", "pytest", "sklearn", "PySide6.QtWebEngineCore",
            "PySide6.QtWebEngineWidgets", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtPdf"]
# not needed by a Qt Widgets app: software OpenGL fallback, translations, PDF image plugin
DROP = ("opengl32sw.dll", os.path.join("PySide6", "translations"), "qpdf.dll", "Qt6Pdf.dll")


def _keep(entry):
    return not any(d.lower() in entry[0].lower() for d in DROP)


def analysis(script):
    a = Analysis([os.path.join(SPECPATH, script)], pathex=[ROOT], datas=datas, hiddenimports=hiddenimports,
                 excludes=excludes, noarchive=False)
    a.binaries = [b for b in a.binaries if _keep(b)]
    a.datas = [d for d in a.datas if _keep(d)]
    return a


gui_a = analysis("gui_entry.py")
cli_a = analysis("cli_entry.py")

gui_exe = EXE(PYZ(gui_a.pure), gui_a.scripts, [], exclude_binaries=True, name="AudioScope", console=False,
              icon=ICON, upx=False)
cli_exe = EXE(PYZ(cli_a.pure), cli_a.scripts, [], exclude_binaries=True, name="audioscope-cli", console=True,
              icon=ICON, upx=False)

coll = COLLECT(gui_exe, gui_a.binaries, gui_a.datas, cli_exe, cli_a.binaries, cli_a.datas,
               name="AudioScope", upx=False)
