import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=UserWarning)

from audioscope.engine.features import Features  # noqa: E402
from audioscope.engine.io import ANALYSIS_SR  # noqa: E402
from tests import synth  # noqa: E402


def to_features(y: np.ndarray, sr: int = synth.SR) -> Features:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if sr != ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR)
    return Features(y, ANALYSIS_SR)


@pytest.fixture
def feat():
    return to_features
