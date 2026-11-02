"""Move the drum bus's octave-band balance part of the way toward the reference drums.

The same EQ goes on every stem, so the stems still sum to the matched bus.
"""
from __future__ import annotations

import numpy as np

from drummix.dsp.filters import OCTAVES, eq, octave_levels


def tone_match(stems: dict[str, np.ndarray], reference: np.ndarray, amount: float,
               max_step_db: float = 6.0, max_sub_boost_db: float = 3.0, max_top_cut_db: float = 3.0,
               iterations: int = 3, q: float = 1.41):
    ref = octave_levels(reference)
    before = octave_levels(sum(stems.values()))
    target = before + amount * (ref - before)
    total = np.zeros(len(OCTAVES))
    out = dict(stems)
    for _ in range(iterations):
        step = np.clip(target - octave_levels(sum(out.values())), -max_step_db, max_step_db)
        step[0] = min(step[0], max_sub_boost_db - total[0])  # don't chase sub rumble
        step[-1] = max(step[-1], -max_top_cut_db - total[-1])  # keep cymbal air
        total += step
        bands = [[f, g, q] for f, g in zip(OCTAVES, step)]
        out = {k: eq(v, bands) for k, v in out.items()}
    after = octave_levels(sum(out.values()))
    report = {"bands_hz": list(OCTAVES), "reference": ref.round(2).tolist(), "before": before.round(2).tolist(),
              "after": after.round(2).tolist(), "eq_db": total.round(2).tolist(), "amount": amount}
    return out, report
