import numpy as np
import pytest

from drummix.audio import SR, rms_db
from drummix.stages.clean import clean


def _stereo_noise(seed=0, seconds=1.0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(seconds * SR), 2)) * 0.1


def test_mono_makes_channels_equal():
    out = clean({"kick": _stereo_noise()}, {"kick": {"mono": True}})
    assert np.allclose(out["kick"][:, 0], out["kick"][:, 1])


def test_width_narrows_side():
    x = _stereo_noise()
    y = clean({"hihat": x}, {"hihat": {"width": 0.5}})["hihat"]
    side = lambda s: rms_db((s[:, 0] - s[:, 1])[:, None] / 2)
    assert side(y) - side(x) == pytest.approx(-6.02, abs=0.05)


def test_gain_and_drop_and_untouched_piece():
    x = _stereo_noise()
    out = clean({"snare": x, "other": x, "toms": x}, {"snare": {"gain_db": 2.0}, "other": {"drop": True}})
    assert "other" not in out
    assert rms_db(out["snare"]) - rms_db(x) == pytest.approx(2.0, abs=0.01)
    assert np.array_equal(out["toms"], x)


def test_gate_silences_low_level_residue():
    x = _stereo_noise() * 0.001  # ~ -70 dBFS
    y = clean({"toms": x}, {"toms": {"gate": {"threshold_db": -36, "ratio": 6, "attack_ms": 1, "release_ms": 200}}})["toms"]
    assert rms_db(y) < rms_db(x) - 10
