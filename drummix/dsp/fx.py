"""Bridge to pedalboard: our (samples, 2) float64 <-> its (2, samples) float32."""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from drummix.audio import SR


def process(x: np.ndarray, plugins: list) -> np.ndarray:
    y = pb.Pedalboard(plugins)(np.ascontiguousarray(x.T, dtype=np.float32), SR)
    return np.ascontiguousarray(y.T, dtype=np.float64)
