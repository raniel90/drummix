"""Move the drum bus's octave-band balance part of the way toward the reference drums.

The EQ is solved on the summed bus (EQ is linear, so iterating on the sum is enough) and the
final accumulated EQ is applied once, identically, to every stem: the stems still sum to the
matched bus.
"""
from __future__ import annotations

import numpy as np

from drummix.dsp.filters import OCTAVES, eq, octave_levels


def tone_match(stems: dict[str, np.ndarray], reference: np.ndarray, amount: float,
               max_step_db: float = 6.0, max_sub_boost_db: float = 3.0, max_top_cut_db: float = 3.0,
               iterations: int = 3, q: float = 1.41):
    """max_step_db caps the *accumulated* correction per band (spec §5: ±6 dB)."""
    ref = octave_levels(reference)
    bus = np.zeros_like(next(iter(stems.values())), dtype=float)
    for v in stems.values():
        bus += v
    before = octave_levels(bus)
    target = before + amount * (ref - before)
    total = np.zeros(len(OCTAVES))
    cur = bus
    for _ in range(iterations):
        step = target - octave_levels(cur)
        # levels are relative to the full band, so a uniform offset is invisible to the measure;
        # dropping it keeps a capped band from dragging every other band along (drift)
        step -= np.median(step)
        step = np.clip(total + step, -max_step_db, max_step_db) - total
        step[0] = min(step[0], max_sub_boost_db - total[0])  # don't chase sub rumble
        step[-1] = max(step[-1], -max_top_cut_db - total[-1])  # keep cymbal air
        total += step
        cur = eq(cur, [[f, g, q] for f, g in zip(OCTAVES, step)])
    total = total.round(2)  # what is applied is exactly what is reported
    bands = [[f, g, q] for f, g in zip(OCTAVES, total)]
    del cur
    after = octave_levels(eq(bus, bands))  # == the sum of the output stems (linear EQ)
    del bus
    out = {k: eq(v, bands) for k, v in stems.items()}
    report = {"bands_hz": list(OCTAVES), "reference": ref.round(2).tolist(), "before": before.round(2).tolist(),
              "after": after.round(2).tolist(), "eq_db": total.tolist(), "amount": amount}
    return out, report
