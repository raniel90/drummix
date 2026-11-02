"""Kick reinforcement: detect hits on the kick stem and lay a tuned synthetic kick under them."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from drummix.audio import SR, gain, rms_db
from drummix.dsp.midi import Hit

_HOP = 64


def _low(kick: np.ndarray) -> np.ndarray:
    return sosfiltfilt(butter(4, 150, "low", fs=SR, output="sos"), kick.mean(axis=1))


def detect_kicks(kick: np.ndarray, min_rise_db: float = 12.0, min_gap_ms: float = 100.0,
                 floor_below_peak_db: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (start sample of each hit, low-band peak amplitude of each hit)."""
    low = _low(kick)
    win = int(0.003 * SR)
    env = np.sqrt(np.convolve(low ** 2, np.ones(win) / win, "same"))
    e = 20 * np.log10(env[::_HOP] + 1e-9)
    lag = max(1, int(0.008 * SR / _HOP))
    flux = np.maximum(e - np.roll(e, lag), 0)
    flux[:lag] = 0
    floor = np.percentile(e, 99.5) - floor_below_peak_db
    pk, _ = find_peaks(flux, height=min_rise_db, distance=max(1, int(min_gap_ms / 1000 * SR / _HOP)))
    pk = pk[e[np.minimum(pk + lag, len(e) - 1)] > floor]
    starts = np.maximum(pk - lag, 0) * _HOP
    peaks = np.array([np.max(np.abs(low[s:s + int(0.04 * SR)])) for s in starts]) if len(starts) else np.zeros(0)
    return starts, peaks


def synth_kick(tonic_hz: float, sweep_from_hz: float = 160.0, decay_s: float = 0.11,
               click: float = 0.25, seed: int = 7) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = tonic_hz + (sweep_from_hz - tonic_hz) * np.exp(-t / 0.018)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay_s)
    rng = np.random.default_rng(seed)
    noise = sosfiltfilt(butter(2, [2500, 7000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    k = body + click * noise * np.exp(-t / 0.0025) / np.max(np.abs(noise))
    fade = int(0.0005 * SR)
    k[:fade] *= np.linspace(0, 1, fade)
    return k / np.max(np.abs(k))


def kick_sample(kick: np.ndarray, tonic_hz: float, reference_db: float, level_db: float = -9.0,
                align_window_ms: tuple[float, float] = (-3.0, 12.0), **synth) -> tuple[np.ndarray, list[Hit]]:
    """Sample track (same length as `kick`) at `level_db` relative to `reference_db` RMS, plus hits."""
    starts, peaks = detect_kicks(kick)
    low = _low(kick)
    sample = synth_kick(tonic_hz, **synth)
    n, L = len(sample), int(0.03 * SR)
    out = np.zeros(len(kick) + n)
    hits = []
    pref = np.percentile(peaks, 95) if len(peaks) else 1.0
    # the flux detector fires ~5-8 ms before the true attack, so the search window is asymmetric
    lo, hi = (int(ms / 1000 * SR) for ms in align_window_ms)
    for s, p in zip(starts, peaks):
        best_c, best_a = -np.inf, int(s)
        for a in range(int(s) + lo, int(s) + hi + 1, 4):
            ref = low[max(a, 0):a + L]
            if a < 0 or len(ref) < L:
                continue
            c = float(np.dot(ref, sample[:L]))
            if c > best_c:
                best_c, best_a = c, a
        g = min(1.0, p / pref)
        out[best_a:best_a + n] += sample * g
        hits.append(Hit(best_a / SR, int(np.clip(40 + 87 * g, 1, 127))))
    mono = out[:len(kick)]
    track = np.stack([mono, mono], axis=1)
    if not hits:
        return track, hits
    return track * gain(reference_db + level_db - rms_db(track)), hits
