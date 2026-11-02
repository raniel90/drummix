import pedalboard
import pyebur128
import pyloudnorm


def test_dependencies_import():
    assert hasattr(pedalboard, "BrickwallLimiter")
    assert hasattr(pyebur128, "R128State")
    assert hasattr(pyloudnorm, "Meter")
