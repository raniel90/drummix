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
    burst = sosfilt(sos, rng.standard_normal(n)) * _decay(n, tau)
    return burst / np.max(np.abs(burst))  # unit peak, so `amp` is the hit's real peak level


def _place(track, sound, t_s, amp):
    i = int(round(t_s * SR))
    if 0 <= i < len(track):
        seg = sound[: len(track) - i]
        track[i:i + len(seg)] += amp * seg


def _tom(rng, freq):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    f = freq * (1 + 0.3 * np.exp(-t / 0.03))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.1)


def _vary(rng, amp):
    return amp * rng.uniform(0.7, 1.3)  # dinâmica humana: ±30% por golpe


def make_song(folder, seconds=20.0, delay_s=1.5, bpm=120, seed=0) -> dict:
    """Drums are recorded `delay_s` later than the band timeline (like the EAD10 take).

    The groove is deliberately not periodic (random dynamics, dropped hi-hats and ghost notes,
    tom fills, crashes at random bars) so onset correlation has a single true lag.
    Take and reference hold the same hits; only the delay and ±3 ms jitter differ.
    """
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    n = int(seconds * SR)
    beat = 60.0 / bpm
    music_len = seconds - delay_s - 0.5
    n_beats = int(np.ceil(music_len / beat))
    # uma virada de tons a cada ~4 compassos (16 tempos), em um tempo aleatório do bloco
    fill_beats = {b0 + int(rng.integers(0, 16)) for b0 in range(0, n_beats, 16)}
    pieces = {k: np.zeros(n) for k in _NAMES}
    ref = {k: np.zeros(n) for k in _NAMES}  # same drums on the band timeline
    kick_times, snare_times = [], []
    for k in range(n_beats):
        t_band = k * beat
        jitter = rng.uniform(-0.003, 0.003)
        hits = []  # (piece, sound, offset from beat in s, amp), shared by take and reference
        if k % 2 == 0:
            hits.append(("kick", _kick(rng), 0.0, _vary(rng, 0.6)))
        else:
            hits.append(("snare", _noise(rng, int(0.2 * SR), 200, 8000, 0.08), 0.0, _vary(rng, 0.6)))
            if rng.random() > 0.3:  # ghost note no contratempo
                hits.append(("snare", _noise(rng, int(0.1 * SR), 200, 8000, 0.04), beat / 2, _vary(rng, 0.08)))
        for off, amp in ((0.0, 0.15), (beat / 2, 0.1)):
            if rng.random() > 0.2:
                hits.append(("hihat", _noise(rng, int(0.05 * SR), 7000, 15000, 0.02), off, _vary(rng, amp)))
        if k % 4 == 0 and rng.random() < 0.3:  # prato no início de compassos aleatórios
            hits.append(("cymbals", _noise(rng, int(1.2 * SR), 3000, 16000, 0.6), 0.0, _vary(rng, 0.2)))
        if k in fill_beats:
            n_hits = int(rng.integers(2, 5))
            for i, f in enumerate(np.linspace(120.0, 85.0, n_hits)):
                hits.append(("toms", _tom(rng, f), i * beat / 4, _vary(rng, 0.4)))
        for track, t in ((pieces, t_band + delay_s + jitter), (ref, t_band)):
            for piece, sound, off, amp in hits:
                if t_band + off < music_len:
                    _place(track[piece], sound, t + off, amp)
        (kick_times if k % 2 == 0 else snare_times).append(t_band + delay_s + jitter)
    t = np.arange(n) / SR
    band = np.zeros(n)
    for f in (110.0, 130.81, 164.81):
        band += 0.08 * np.sin(2 * np.pi * f * t)
    band += 0.15 * np.sin(2 * np.pi * 55.0 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / beat)))
    band[int(music_len * SR):] = 0.0
    pieces["toms"] += 1e-4 * rng.standard_normal(n)
    pieces["other"] += 1e-6 * rng.standard_normal(n)
    pieces["other_kit"] += 1e-4 * rng.standard_normal(n)

    # one common scale for every file keeps original - band == reference drums, without clipping
    ref_drums = sum(ref.values())
    scale = min(1.0, 0.9 / max(np.max(np.abs(band + ref_drums)), np.max(np.abs(sum(pieces.values())))))
    pieces = {k: v * scale for k, v in pieces.items()}
    band, ref_drums = band * scale, ref_drums * scale

    stems_dir = folder / "stems"
    for piece, x in pieces.items():
        audio.save(stems_dir / f"Synth - Audio-{_NAMES[piece]}-A minor-{bpm}bpm-440hz.wav", np.stack([x, x], 1))
    audio.save(folder / "band.wav", np.stack([band, band], 1))
    audio.save(folder / "original.wav", np.stack([band + ref_drums, band + ref_drums], 1))
    return {"stems_dir": stems_dir, "band": folder / "band.wav", "original": folder / "original.wav",
            "delay_s": delay_s, "kick_times": kick_times, "snare_times": snare_times}
