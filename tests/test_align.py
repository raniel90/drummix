import numpy as np
import pytest

from drummix import audio, song
from drummix.stages.align import AlignError, estimate_delay
from tests.synth import make_song


def _drums_and_reference(tmp_path, delay_s, seconds=60.0):
    info = make_song(tmp_path, seconds=seconds, delay_s=delay_s)
    stems, _, _ = song.find_stems(info["stems_dir"])
    drums = sum(audio.load(p) for p in stems.values())
    ref = audio.load(info["original"]) - audio.load(info["band"])
    return drums, ref


def test_recovers_known_delay(tmp_path):
    drums, ref = _drums_and_reference(tmp_path, delay_s=2.37)
    a = estimate_delay(drums, ref, search_s=(1.9, 2.9), window_s=10, step_s=8)
    assert a.delay_s == pytest.approx(2.37, abs=0.01)
    assert a.spread_s < 0.01
    assert a.used >= 3


def test_raises_when_nothing_correlates():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((audio.SR * 40, 2)) * 1e-9
    with pytest.raises(AlignError):
        estimate_delay(noise, noise * 0, search_s=(0, 2), window_s=10, step_s=8)
