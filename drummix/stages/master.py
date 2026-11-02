"""Glue compression, then gain into a true-peak brickwall limiter until the loudness target is met."""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from drummix.audio import gain, lufs, true_peak_db
from drummix.dsp import fx


class MasterError(RuntimeError):
    pass


def master(premaster: np.ndarray, target_lufs: float = -14.0, ceiling_dbtp: float = -1.0,
           tolerance_lu: float = 0.3, glue: dict | None = None, max_iter: int = 8):
    max_iter = max(1, max_iter)
    x = fx.process(premaster, [pb.Compressor(**glue)]) if glue else premaster
    start = lufs(x)
    if not np.isfinite(start):
        raise MasterError("premaster sem conteúdo mensurável (silêncio)")
    g = target_lufs - start
    level = start
    for _ in range(max_iter):
        y = fx.process(x * gain(g), [pb.BrickwallLimiter(ceiling_db=ceiling_dbtp, true_peak=True)])
        level = lufs(y)
        if abs(level - target_lufs) <= tolerance_lu:
            break
        g += target_lufs - level
    else:
        raise MasterError(f"não atingiu {target_lufs} LUFS em {max_iter} iterações (último {level:.2f})")
    tp = true_peak_db(y)
    if tp > ceiling_dbtp + 0.05:
        raise MasterError(f"true peak {tp:.2f} dBTP acima do teto {ceiling_dbtp}")
    return y, {"gain_db": round(g, 2), "lufs": round(level, 2), "true_peak_db": round(tp, 2)}
