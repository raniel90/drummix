"""Backing track: static carve EQ and placement on the drum/video timeline."""
from __future__ import annotations

import numpy as np

from drummix.audio import SR
from drummix.dsp.filters import eq


def prepare_band(band: np.ndarray, delay_s: float, length: int, eq_bands) -> tuple[np.ndarray, slice]:
    y = eq(band, eq_bands) if eq_bands else band
    shift = int(round(delay_s * SR))
    out = np.zeros((length, 2))
    if shift >= 0:
        seg = y[:max(0, length - shift)]
        out[shift:shift + len(seg)] = seg
        return out, slice(shift, shift + len(seg))
    seg = y[-shift:-shift + length]
    out[:len(seg)] = seg
    return out, slice(0, len(seg))
