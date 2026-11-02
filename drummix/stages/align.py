"""Delay of the recorded drums relative to the band timeline, via onset cross-correlation.

Positive delay_s: the drums happen delay_s later than the reference, so the band must be
shifted later by delay_s to line up (the drum/video timeline is the master).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, resample_poly

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
    """Global delay = peak of the per-window correlation curves averaged over all windows.

    Music repeats every bar, so a single window often prefers a lag one or two bars off by a
    thin margin; averaging the curves lets every window vote with its whole curve, not only
    its argmax. Reported alongside:
      used      — windows whose own best lag (>= min_corr) is within agree_s of the global one;
      spread_s  — std of those agreeing lags; if none agree, the RMS distance of every
                  window's best lag to the global one (large, so the pipeline warns);
      mean_corr — mean of the agreeing windows' own best correlation (same meaning as before
                  the summed estimate, so the pipeline's 0.1 threshold still applies);
                  if none agree, the averaged curve's peak value. The averaged peak is always
                  lower (disagreeing windows dilute it): 0.11 vs 0.24 on The Hornet.
    """
    d, r = onset_envelope(drums), onset_envelope(reference)
    lags = np.arange(int(search_s[0] * FRAME_SR), int(search_s[1] * FRAME_SR) + 1)
    win = int(window_s * FRAME_SR)
    starts = list(range(int(step_s * FRAME_SR), len(d) - win, int(step_s * FRAME_SR)))
    if not starts:  # song shorter than step_s + window_s: one central window instead of none
        win = min(win, len(d) * 3 // 4)
        starts = [(len(d) - win) // 2]
    curves = _window_curves(d, r, starts, win, lags)
    windows = len(curves)
    if windows == 0:
        raise AlignError("nenhuma janela correlacionou bateria e referência")
    curves = np.stack(curves)
    valid = np.isfinite(curves)
    count = valid.sum(axis=0)
    mean = np.where(valid, curves, 0.0).sum(axis=0) / np.maximum(count, 1)
    # a lag seen by only a few windows (search edges) must not win on a lucky window
    mean[count < max(1, (count.max() + 1) // 2)] = -np.inf
    g = int(np.argmax(mean))
    if not mean[g] >= min_corr:
        raise AlignError("nenhuma janela correlacionou bateria e referência")
    filled = np.where(valid, curves, -np.inf)
    best = filled.max(axis=1)
    best_lag = lags[np.argmax(filled, axis=1)].astype(float)
    ok = best >= min_corr
    agree = ok & (np.abs(best_lag - lags[g]) <= agree_s * FRAME_SR)
    # some window has curve[g] >= mean[g] >= min_corr, so `ok` is never empty here
    spread = (np.std(best_lag[agree]) if agree.any()
              else np.sqrt(np.mean((best_lag[ok] - lags[g]) ** 2)))
    corr = best[agree].mean() if agree.any() else mean[g]
    return Alignment(delay_s=float(lags[g] / FRAME_SR), spread_s=float(spread) / FRAME_SR,
                     mean_corr=float(corr), windows=windows, used=int(agree.sum()))


def _window_curves(d: np.ndarray, r: np.ndarray, starts, win: int, lags: np.ndarray) -> list[np.ndarray]:
    """Normalised correlation of each non-silent drum window against r at every lag (NaN = no data)."""
    cs = np.concatenate([[0.0], np.cumsum(r)])
    cs2 = np.concatenate([[0.0], np.cumsum(r * r)])
    n_pos = len(r) - win + 1  # reference windows r[j0:j0+win], j0 in [0, n_pos)
    if n_pos <= 0:
        return []
    m = (cs[win:] - cs[:-win]) / win
    var = (cs2[win:] - cs2[:-win]) / win - m * m
    std = np.sqrt(np.where(var > 1e-12 * max(float(r.var()), 0.0), var, np.nan))
    out = []
    for i0 in starts:
        seg = d[i0:i0 + win]
        if seg.std() == 0:
            continue
        seg = (seg - seg.mean()) / seg.std()
        dots = correlate(r, seg, mode="valid")  # dots[j0] = seg · r[j0:j0+win]
        j0 = i0 - lags
        inside = (j0 >= 0) & (j0 < n_pos)
        c = np.full(len(lags), np.nan)
        jj = j0[inside]
        c[inside] = dots[jj] / (win * std[jj])  # seg has zero mean, so the window mean drops out
        out.append(c)
    return out
