import numpy as np
import pedalboard as pb
import pretty_midi
import pytest

from drummix.audio import SR, rms_db
from drummix.dsp import filters, fx, midi


def _sine(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], 1)


def test_peaking_boosts_at_center_only():
    x = _sine(1000)
    y = filters.eq(x, [[1000, 6.0, 1.0]])
    assert rms_db(y[SR // 2:]) - rms_db(x[SR // 2:]) == pytest.approx(6.0, abs=0.3)
    far = _sine(100)
    assert rms_db(filters.eq(far, [[1000, 6.0, 1.0]])[SR // 2:]) - rms_db(far[SR // 2:]) == pytest.approx(0, abs=0.3)


def test_highpass_attenuates_below_cutoff():
    y = filters.apply_sos(filters.highpass_sos(200, order=2), _sine(50))
    assert rms_db(y[SR // 2:]) - rms_db(_sine(50)[SR // 2:]) < -20


def test_octave_levels_peak_where_energy_is():
    levels = filters.octave_levels(_sine(1000))
    assert filters.OCTAVES[int(np.argmax(levels))] == 1000


def test_fx_process_keeps_shape_and_alignment():
    x = np.zeros((SR, 2))
    x[10000] = 0.5
    y = fx.process(x, [pb.Gain(gain_db=0.0)])
    assert y.shape == x.shape and y.dtype == np.float64
    assert int(np.argmax(np.abs(y[:, 0]))) == 10000


def test_write_drum_midi(tmp_path):
    hits = {36: [midi.Hit(0.5, 100), midi.Hit(1.0, 60)], 38: [midi.Hit(0.75, 90)]}
    midi.write_drum_midi(tmp_path / "d.mid", hits, bpm=131)
    pm = pretty_midi.PrettyMIDI(str(tmp_path / "d.mid"))
    notes = sorted((n.start, n.pitch, n.velocity) for n in pm.instruments[0].notes)
    assert pm.instruments[0].is_drum
    assert [(p, v) for _, p, v in notes] == [(36, 100), (38, 90), (36, 60)]
    assert [s for s, _, _ in notes] == pytest.approx([0.5, 0.75, 1.0], abs=0.003)  # MIDI tick quantization
    assert pm.get_tempo_changes()[1][0] == pytest.approx(131, abs=0.01)
