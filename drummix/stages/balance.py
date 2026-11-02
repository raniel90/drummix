"""Set the drums a fixed number of LU under the band, measured where both play."""
from __future__ import annotations

import numpy as np

from drummix.audio import gain, lufs


def scale_in_place(stems: dict[str, np.ndarray], g: float) -> None:
    """Multiply every stem by g without a second dict of copies. An array that shares memory with
    an earlier entry is copied (before anything is scaled), so nothing is ever scaled twice."""
    seen: list[np.ndarray] = []
    for k, v in stems.items():
        if any(np.may_share_memory(v, w) for w in seen):
            v = stems[k] = v.copy()
        seen.append(v)
    for v in stems.values():
        v *= g


def balance(drums: dict[str, np.ndarray], band: np.ndarray, overlap: slice, below_lu: float):
    """Scales `drums` in place (peak memory: no second full set of stems) and returns it."""
    bus = sum(drums.values())
    g = (lufs(band[overlap]) - below_lu) - lufs(bus[overlap])
    del bus
    if not np.isfinite(g):
        raise ValueError("bateria ou banda em silêncio na região comum — não dá para equilibrar")
    scale_in_place(drums, gain(g))
    return drums, float(g)
