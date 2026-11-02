import numpy as np

from drummix.stages.balance import scale_in_place


def test_scale_in_place_scales_every_entry_once_even_if_arrays_are_shared():
    a = np.ones((10, 2))
    b = np.full((10, 2), 2.0)
    stems = {"x": a, "y": a, "z": b, "w": b[2:5]}
    scale_in_place(stems, 0.5)
    assert np.all(stems["x"] == 0.5) and np.all(stems["y"] == 0.5)
    assert np.all(stems["z"] == 1.0) and np.all(stems["w"] == 1.0)


def test_scale_in_place_does_not_copy_unshared_arrays():
    a = np.ones((10, 2))
    stems = {"x": a}
    scale_in_place(stems, 0.5)
    assert stems["x"] is a and np.all(a == 0.5)
