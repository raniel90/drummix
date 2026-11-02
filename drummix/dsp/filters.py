"""Biquads (RBJ peaking, Butterworth HP/LP) and octave-band level analysis."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from drummix.audio import SR

OCTAVES = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def peaking_sos(fc: float, gain_db: float, q: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * fc / SR
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])[None, :]


def highpass_sos(fc: float, order: int = 2) -> np.ndarray:
    return butter(order, fc, "high", fs=SR, output="sos")


def lowpass_sos(fc: float, order: int = 2) -> np.ndarray:
    return butter(order, fc, "low", fs=SR, output="sos")


def apply_sos(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sosfilt(sos, x, axis=0)


def eq(x: np.ndarray, bands) -> np.ndarray:
    """bands: [[freq_hz, gain_db, q], ...]; bands too close to Nyquist are skipped."""
    sections = [peaking_sos(f, g, q) for f, g, q in bands if abs(g) > 0.05 and f < 0.8 * SR / 2]
    return apply_sos(np.vstack(sections), x) if sections else x


def octave_levels(x: np.ndarray) -> np.ndarray:
    """Level of each octave band in dB relative to the full-band level (mono sum)."""
    m = np.asarray(x).mean(axis=1)
    total = 10 * np.log10(np.mean(m ** 2) + 1e-20)
    out = []
    for fc in OCTAVES:
        lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), 0.95 * SR / 2)
        band = sosfiltfilt(butter(4, [lo, hi], "band", fs=SR, output="sos"), m)
        out.append(10 * np.log10(np.mean(band ** 2) + 1e-20) - total)
    return np.array(out)
