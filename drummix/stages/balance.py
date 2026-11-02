"""Set the drums a fixed number of LU under the band, measured where both play."""
from __future__ import annotations

import numpy as np

from drummix.audio import gain, lufs


def balance(drums: dict[str, np.ndarray], band: np.ndarray, overlap: slice, below_lu: float):
    bus = sum(drums.values())
    g = (lufs(band[overlap]) - below_lu) - lufs(bus[overlap])
    if not np.isfinite(g):
        raise ValueError("bateria ou banda em silêncio na região comum — não dá para equilibrar")
    return {k: v * gain(g) for k, v in drums.items()}, float(g)
