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


def test_cumulative_eq_capped_at_max_step():
    # +18 dB em 1 kHz: cada iteração é limitada, mas o acumulado também precisa ser (spec §5: ±6 dB)
    ref = eq(_pink(0), [[1000, 18.0, 1.41]])
    out, rep = tone_match({"a": _pink(1)}, ref, amount=1.0, max_step_db=6.0, iterations=5)
    assert max(abs(g) for g in rep["eq_db"]) <= 6.0 + 1e-9
    assert rep["eq_db"][5] == pytest.approx(6.0, abs=1e-9)


def test_pinned_top_band_does_not_drag_other_bands():
    # rosa + branco: a banda de 16 kHz pesa no nível total. A referência é o próprio bus com tudo
    # acima de 11.3 kHz (só a banda de 16 kHz) 15 dB mais baixo; o corte fica preso em -3 dB e as
    # demais bandas, idênticas à referência, não podem derivar (antes: até +2.8 dB).
    w = np.random.default_rng(5).standard_normal(int(4 * SR))
    x = _pink(1)[:, 0] + 0.3 * w / np.max(np.abs(w))
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1 / SR) > 16000 / np.sqrt(2)] *= 10 ** (-15 / 20)
    y = np.fft.irfft(spec, len(x))
    bus, ref = np.stack([x, x], 1), np.stack([y, y], 1)
    out, rep = tone_match({"a": bus * 0.5, "b": bus * 0.5}, ref, amount=1.0, max_top_cut_db=3.0, iterations=5)
    assert rep["eq_db"][-1] == pytest.approx(-3.0, abs=1e-9)
    assert max(abs(g) for g in rep["eq_db"][:-1]) < 0.5


def test_output_stems_get_one_eq_with_accumulated_gains():
    ref = eq(_pink(0), [[250, 4.0, 1.41], [4000, -4.0, 1.41]])
    a, b = _pink(1), _pink(2)
    out, rep = tone_match({"a": a, "b": b}, ref, amount=1.0)
    bands = [[f, g, 1.41] for f, g in zip(rep["bands_hz"], rep["eq_db"])]
    assert np.allclose(out["a"], eq(a, bands), atol=1e-6)
    assert np.allclose(out["b"], eq(b, bands), atol=1e-6)
