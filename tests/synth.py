"""Synthetic song for tests: Moises-named drum stems, drumless band, original, known delay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from drummix import audio
from drummix.audio import SR

_NAMES = {"kick": "Kick", "snare": "Snare", "toms": "Toms", "hihat": "Hi-Hat",
          "cymbals": "Cymbals", "other": "Other", "other_kit": "other_kit"}


def _decay(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def _kick(rng):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 55 + 105 * np.exp(-t / 0.02)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.12)


def _noise(rng, n, lo, hi, tau):
    sos = butter(2, [lo, hi], "band", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * _decay(n, tau)


def _place(track, sound, t_s, amp):
    i = int(round(t_s * SR))
    if 0 <= i < len(track):
        seg = sound[: len(track) - i]
        track[i:i + len(seg)] += amp * seg


def make_song(folder, seconds=20.0, delay_s=1.5, bpm=120, seed=0) -> dict:
    """Drums are recorded `delay_s` later than the band timeline (like the EAD10 take)."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    n = int(seconds * SR)
    beat = 60.0 / bpm
    music_len = seconds - delay_s - 0.5
    pieces = {k: np.zeros(n) for k in _NAMES}
    ref = {k: np.zeros(n) for k in _NAMES}  # same drums on the band timeline
    kick_times, snare_times = [], []
    k = 0
    while k * beat < music_len:
        t_band = k * beat
        jitter = rng.uniform(-0.003, 0.003)
        for track, t in ((pieces, t_band + delay_s + jitter), (ref, t_band)):
            if k % 2 == 0:
                _place(track["kick"], _kick(rng), t, 0.8)
            else:
                _place(track["snare"], _noise(rng, int(0.2 * SR), 200, 8000, 0.08), t, 0.6)
            _place(track["hihat"], _noise(rng, int(0.05 * SR), 7000, 15000, 0.02), t, 0.15)
            _place(track["hihat"], _noise(rng, int(0.05 * SR), 7000, 15000, 0.02), t + beat / 2, 0.1)
            if k % 8 == 0:
                _place(track["cymbals"], _noise(rng, int(1.2 * SR), 3000, 16000, 0.6), t, 0.2)
            if k % 2 == 1:  # ghost note no contratempo
                _place(track["snare"], _noise(rng, int(0.1 * SR), 200, 8000, 0.04), t + beat / 2, 0.08)
        (kick_times if k % 2 == 0 else snare_times).append(t_band + delay_s + jitter)
        k += 1
    t = np.arange(n) / SR
    band = np.zeros(n)
    for f in (110.0, 130.81, 164.81):
        band += 0.08 * np.sin(2 * np.pi * f * t)
    band += 0.15 * np.sin(2 * np.pi * 55.0 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / beat)))
    band[int(music_len * SR):] = 0.0
    pieces["toms"] += 1e-4 * rng.standard_normal(n)
    pieces["other"] += 1e-6 * rng.standard_normal(n)
    pieces["other_kit"] += 1e-4 * rng.standard_normal(n)

    stems_dir = folder / "stems"
    for piece, x in pieces.items():
        audio.save(stems_dir / f"Synth - Audio-{_NAMES[piece]}-A minor-{bpm}bpm-440hz.wav", np.stack([x, x], 1))
    ref_drums = sum(ref.values())
    audio.save(folder / "band.wav", np.stack([band, band], 1))
    audio.save(folder / "original.wav", np.stack([band + ref_drums, band + ref_drums], 1))
    return {"stems_dir": stems_dir, "band": folder / "band.wav", "original": folder / "original.wav",
            "delay_s": delay_s, "kick_times": kick_times, "snare_times": snare_times}
