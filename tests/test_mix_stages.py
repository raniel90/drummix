import numpy as np
import pytest

from drummix import audio
from drummix.audio import SR, rms_db
from drummix.stages.balance import balance
from drummix.stages.band import prepare_band
from drummix.stages.space import room


def _clicks(seconds=3.0):
    x = np.zeros((int(seconds * SR), 2))
    x[::SR // 2] = 0.8
    return x


def test_room_level_relative_to_send_and_deterministic():
    stems = {"snare": _clicks(), "kick": _clicks()}
    wet = room(stems, sends={"snare": 1.0, "kick": 0.3}, level_db=-15.0)
    send = stems["snare"] + 0.3 * stems["kick"]
    assert wet.shape == send.shape
    assert rms_db(wet) - rms_db(send) == pytest.approx(-15.0, abs=0.05)
    assert np.array_equal(wet, room(stems, sends={"snare": 1.0, "kick": 0.3}, level_db=-15.0))


def test_prepare_band_shifts_later_and_reports_overlap():
    band = np.ones((SR * 2, 2))
    out, overlap = prepare_band(band, delay_s=0.5, length=SR * 3, eq_bands=[])
    assert out.shape == (SR * 3, 2)
    assert out[: SR // 2].sum() == 0
    assert overlap == slice(SR // 2, SR // 2 + SR * 2)


def test_prepare_band_negative_delay_trims_start():
    band = np.arange(SR * 2, dtype=float)[:, None].repeat(2, 1)
    out, overlap = prepare_band(band, delay_s=-0.5, length=SR, eq_bands=[])
    assert out[0, 0] == SR // 2
    assert overlap == slice(0, SR)


def test_balance_puts_drums_below_band():
    rng = np.random.default_rng(0)
    band = rng.standard_normal((SR * 5, 2)) * 0.1
    drums = {"kick": rng.standard_normal((SR * 5, 2)) * 0.01, "snare": rng.standard_normal((SR * 5, 2)) * 0.02}
    out, g = balance(drums, band, slice(0, SR * 5), below_lu=2.0)
    assert audio.lufs(sum(out.values())) == pytest.approx(audio.lufs(band) - 2.0, abs=0.05)


def test_balance_raises_on_silent_drums():
    rng = np.random.default_rng(0)
    band = rng.standard_normal((SR * 5, 2)) * 0.1
    drums = {"kick": np.zeros((SR * 5, 2))}
    with pytest.raises(ValueError):
        balance(drums, band, slice(0, SR * 5), below_lu=2.0)


def test_room_uses_overlap_add_with_same_result_as_fft_convolution(monkeypatch):
    from scipy.signal import fftconvolve

    from drummix.stages import space
    rng = np.random.default_rng(4)
    stems = {"snare": rng.standard_normal((SR // 2, 2)) * 0.1}
    calls = []
    real = space.oaconvolve
    monkeypatch.setattr(space, "oaconvolve", lambda *a, **k: calls.append(1) or real(*a, **k))
    wet = room(stems, sends={"snare": 1.0}, level_db=-15.0)
    assert calls
    ir = space.room_ir()
    send = stems["snare"]
    ref = np.stack([fftconvolve(send[:, c], ir[:, c])[:len(send)] for c in range(2)], axis=1)
    ref *= audio.gain(rms_db(send) - 15.0 - rms_db(ref))
    assert np.max(np.abs(wet - ref)) < 1e-9


def test_room_without_matching_sends_is_silent():
    stems = {"hihat": _clicks()}
    wet = room(stems, sends={"snare": 1.0}, level_db=-15.0)
    assert wet.shape == stems["hihat"].shape and not wet.any()


def test_balance_scales_in_place_without_a_second_set_of_stems():
    rng = np.random.default_rng(0)
    band = rng.standard_normal((SR * 5, 2)) * 0.1
    kick = rng.standard_normal((SR * 5, 2)) * 0.01
    drums = {"kick": kick}
    out, g = balance(drums, band, slice(0, SR * 5), below_lu=2.0)
    assert out is drums and out["kick"] is kick
