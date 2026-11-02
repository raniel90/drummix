"""Delay of the recorded drums relative to the band timeline, via onset cross-correlation.

Positive delay_s: the drums happen delay_s later than the reference, so the band must be
shifted later by delay_s to line up (the drum/video timeline is the master).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from drummix.audio import SR

FRAME_SR = 200  # onset frames per second (5 ms)


class AlignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Alignment:
    delay_s: float
    spread_s: float
    mean_corr: float
    windows: int
    used: int


def onset_envelope(x: np.ndarray) -> np.ndarray:
    mono = np.asarray(x).mean(axis=1)
    a = np.abs(resample_poly(mono, 40, 441))  # 44100 -> 4000 Hz
    hop = 4000 // FRAME_SR
    a = np.convolve(a, np.ones(hop) / hop, "same")[::hop]
    return np.maximum(np.diff(a, prepend=a[0]), 0)


def estimate_delay(drums, reference, search_s=(-5.0, 30.0), window_s=15, step_s=20,
                   min_corr=0.05, agree_s=0.05) -> Alignment:
    d, r = onset_envelope(drums), onset_envelope(reference)
    lags = np.arange(int(search_s[0] * FRAME_SR), int(search_s[1] * FRAME_SR) + 1)
    win = int(window_s * FRAME_SR)
    found = []
    windows = 0
    for i0 in range(int(step_s * FRAME_SR), len(d) - win, int(step_s * FRAME_SR)):
        seg = d[i0:i0 + win]
        if seg.std() == 0:
            continue
        windows += 1
        seg = (seg - seg.mean()) / seg.std()
        best_c, best_l = -1.0, 0
        for lag in lags:
            j0 = i0 - lag
            if j0 < 0 or j0 + win > len(r):
                continue
            w = r[j0:j0 + win]
            s = w.std()
            if s == 0:
                continue
            c = float(np.dot(seg, (w - w.mean()) / s) / win)
            if c > best_c:
                best_c, best_l = c, lag
        if best_c >= min_corr:
            found.append((best_c, best_l))
    if not found:
        raise AlignError("nenhuma janela correlacionou bateria e referência")
    corr = np.array([c for c, _ in found])
    lag = np.array([l for _, l in found], dtype=float)
    keep = _pick_cluster(lag, corr, agree_s * FRAME_SR)
    return Alignment(delay_s=float(np.median(lag[keep]) / FRAME_SR),
                     spread_s=float(np.std(lag[keep]) / FRAME_SR),
                     mean_corr=float(corr[keep].mean()),
                     windows=windows, used=int(keep.size))


def _pick_cluster(lags: np.ndarray, corrs: np.ndarray, agree_frames: float) -> np.ndarray:
    """Indices of the largest group of window lags that chain within agree_frames of each other.

    Ties go to the group with the higher mean correlation. Never empty for non-empty input.
    """
    order = np.argsort(lags, kind="stable")
    splits = np.flatnonzero(np.diff(lags[order]) > agree_frames) + 1
    clusters = np.split(order, splits)
    return max(clusters, key=lambda idx: (idx.size, float(corrs[idx].mean())))
