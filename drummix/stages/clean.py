"""Per-piece cleanup of separated stems: filters, gate, EQ, compression, width, gain."""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from drummix.audio import gain
from drummix.dsp import fx
from drummix.dsp.filters import apply_sos, eq, highpass_sos, lowpass_sos


def _width(x: np.ndarray, amount: float) -> np.ndarray:
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2 * amount
    return np.stack([mid + side, mid - side], axis=1)


def clean_piece(x: np.ndarray, p: dict) -> np.ndarray:
    y = x
    if p.get("mono"):
        y = _width(y, 0.0)
    if "highpass_hz" in p:
        y = apply_sos(highpass_sos(p["highpass_hz"]), y)
    if "lowpass_hz" in p:
        y = apply_sos(lowpass_sos(p["lowpass_hz"]), y)
    if "gate" in p:
        y = fx.process(y, [pb.NoiseGate(**p["gate"])])
    if p.get("eq"):
        y = eq(y, p["eq"])
    if "compressor" in p:
        y = fx.process(y, [pb.Compressor(**p["compressor"])])
    if "width" in p:
        y = _width(y, p["width"])
    if "gain_db" in p:
        y = y * gain(p["gain_db"])
    return y


def clean(stems: dict[str, np.ndarray], params: dict) -> dict[str, np.ndarray]:
    out = {}
    for piece, x in stems.items():
        p = params.get(piece) or {}
        if p.get("drop"):
            continue
        out[piece] = clean_piece(x, p) if p else x
    return out
