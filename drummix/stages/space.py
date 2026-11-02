"""Short synthetic room: decaying decorrelated noise IR fed by a weighted send of the stems."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, oaconvolve, sosfiltfilt

from drummix.audio import SR, gain, rms_db


def room_ir(rt60_s: float = 0.45, predelay_ms: float = 10.0, band_hz=(300.0, 6000.0), seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(1.33 * rt60_s * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60_s)[:, None]
    ir = sosfiltfilt(butter(2, list(band_hz), "band", fs=SR, output="sos"), ir, axis=0)
    return np.vstack([np.zeros((int(predelay_ms / 1000 * SR), 2)), ir])


def room(stems: dict[str, np.ndarray], sends: dict[str, float], level_db: float = -15.0, **ir_params) -> np.ndarray:
    send = np.zeros_like(next(iter(stems.values())), dtype=float)
    used = False
    for k, w in sends.items():
        if k in stems:
            send += w * stems[k]
            used = True
    if not used:
        return send  # nothing feeds the room: silence, not a crash
    ir = room_ir(**ir_params)
    n = len(send)
    # overlap-add: same result as fftconvolve without one FFT the size of the whole song
    wet = np.stack([oaconvolve(send[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
    return wet * gain(rms_db(send) + level_db - rms_db(wet))
