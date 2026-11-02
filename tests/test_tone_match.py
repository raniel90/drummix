import numpy as np
import pytest

from drummix.audio import SR
from drummix.dsp.filters import eq, octave_levels
from drummix.stages.tone_match import tone_match


def _pink(seed, seconds=4.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n)
    x /= np.max(np.abs(x))
    return np.stack([x, x], 1) * 0.3


def test_moves_spectrum_by_requested_amount():
    ref = _pink(0)
    mine = eq(_pink(1), [[1000, 6.0, 1.41]])  # 6 dB a mais em 1 kHz
    out, rep = tone_match({"a": mine * 0.5, "b": mine * 0.5}, ref, amount=0.5)
    i = 5  # banda de 1 kHz
    before = np.array(rep["before"])[i] - np.array(rep["reference"])[i]
    after = np.array(rep["after"])[i] - np.array(rep["reference"])[i]
    assert after == pytest.approx(before * 0.5, abs=0.8)


def test_same_eq_applied_to_every_stem():
    ref = _pink(0)
    a, b = _pink(1), _pink(2)
    out, rep = tone_match({"a": a, "b": b}, ref, amount=0.65)
    assert np.allclose(octave_levels(out["a"]) - octave_levels(a), octave_levels(out["b"]) - octave_levels(b), atol=0.6)


def test_caps_sub_boost_and_top_cut():
    ref = eq(_pink(0), [[31.5, 12.0, 1.41], [16000, -12.0, 1.41]])
    out, rep = tone_match({"a": _pink(1)}, ref, amount=1.0, max_sub_boost_db=3.0, max_top_cut_db=3.0)
    assert rep["eq_db"][0] <= 3.0 + 1e-9
    assert rep["eq_db"][-1] >= -3.0 - 1e-9
