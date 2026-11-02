import numpy as np
import pytest

from drummix import audio
from drummix.audio import SR
from drummix.stages.master import MasterError, master


def _music(seed=0, seconds=10.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((int(seconds * SR), 2)) * 0.02
    x[::SR // 4] += 0.5  # transientes
    return x


def test_hits_loudness_and_true_peak():
    y, info = master(_music(), target_lufs=-14.0, ceiling_dbtp=-1.0, tolerance_lu=0.3,
                     glue={"threshold_db": -24, "ratio": 2.5, "attack_ms": 30, "release_ms": 150})
    assert audio.lufs(y) == pytest.approx(-14.0, abs=0.3)
    assert audio.true_peak_db(y) <= -1.0 + 0.05
    assert info["true_peak_db"] <= -0.95


def test_keeps_time_alignment():
    x = np.zeros((SR * 4, 2))
    x[SR] = 0.5
    x += np.random.default_rng(1).standard_normal(x.shape) * 1e-3
    y, _ = master(x, target_lufs=-30.0, ceiling_dbtp=-1.0, tolerance_lu=0.5, glue=None)
    assert int(np.argmax(np.abs(y[:, 0]))) == SR


def test_raises_if_target_unreachable():
    with pytest.raises(MasterError):
        master(np.zeros((SR * 2, 2)), target_lufs=-14.0, ceiling_dbtp=-1.0, tolerance_lu=0.3, glue=None)
