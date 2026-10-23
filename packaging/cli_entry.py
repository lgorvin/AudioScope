import multiprocessing
import sys

from audioscope.app import _prepare_environment
from audioscope.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    _prepare_environment()
    sys.exit(main())
