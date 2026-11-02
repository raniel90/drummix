import numpy as np
import pytest

from drummix import audio, song
from drummix.stages.align import AlignError, _pick_cluster, estimate_delay
from tests.synth import make_song


def _drums_and_reference(tmp_path, delay_s, seconds=60.0):
    info = make_song(tmp_path, seconds=seconds, delay_s=delay_s)
    stems, _, _ = song.find_stems(info["stems_dir"])
    drums = sum(audio.load(p) for p in stems.values())
    ref = audio.load(info["original"]) - audio.load(info["band"])
    return drums, ref


def test_recovers_known_delay(tmp_path):
    drums, ref = _drums_and_reference(tmp_path, delay_s=2.37)
    a = estimate_delay(drums, ref, search_s=(-1.0, 5.0), window_s=10, step_s=8)
    assert a.delay_s == pytest.approx(2.37, abs=0.01)
    assert a.spread_s < 0.01
    assert a.used >= 3


@pytest.mark.parametrize("delay_s", [0.4, 11.73])
def test_recovers_delay_over_wide_search(tmp_path, delay_s):
    drums, ref = _drums_and_reference(tmp_path, delay_s=delay_s, seconds=60.0)
    a = estimate_delay(drums, ref, search_s=(-5.0, 30.0))
    assert a.delay_s == pytest.approx(delay_s, abs=0.01)
    assert a.spread_s < 0.01
    assert a.used >= 2


def test_raises_when_nothing_correlates():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((audio.SR * 40, 2)) * 1e-9
    with pytest.raises(AlignError):
        estimate_delay(noise, noise * 0, search_s=(0, 2), window_s=10, step_s=8)


def _clicks(times, n):
    x = np.zeros(n)
    burst = np.random.default_rng(7).standard_normal(int(0.02 * audio.SR)) * np.exp(-np.arange(int(0.02 * audio.SR)) / 200)
    for t in times:
        i = int(round(t * audio.SR))
        if 0 <= i and i + len(burst) <= n:
            x[i:i + len(burst)] += burst
    return np.stack([x, x], 1)


def test_disagreeing_windows_use_majority_cluster():
    # 6 janelas: 3 com atraso 1.0 s, 2 com 2.0 s, 1 com 3.0 s -> mediana 1.5 s cai entre grupos
    rng = np.random.default_rng(3)
    n = 80 * audio.SR
    seg_lag = {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 2.0, 6: 3.0}
    drum_t, ref_t = [], []
    for k in range(8):
        ts = np.sort(rng.uniform(10 * k, 10 * k + 10, 25))
        drum_t += list(ts)
        ref_t += list(ts - seg_lag.get(k, 0.0))
    a = estimate_delay(_clicks(drum_t, n), _clicks(ref_t, n), search_s=(-1, 4), window_s=10, step_s=10)
    assert np.isfinite(a.delay_s)
    assert a.delay_s == pytest.approx(1.0, abs=0.01)
    assert a.used == 3
    assert a.windows == 6


def test_pick_cluster_tie_prefers_higher_correlation():
    lags = np.array([100.0, 300.0, 101.0, 301.0])
    corrs = np.array([0.3, 0.6, 0.3, 0.6])
    keep = _pick_cluster(lags, corrs, agree_frames=10)
    assert sorted(lags[keep]) == [300.0, 301.0]


def test_pick_cluster_majority_wins_over_correlation():
    lags = np.array([100.0, 102.0, 104.0, 500.0])
    corrs = np.array([0.2, 0.2, 0.2, 0.9])
    keep = _pick_cluster(lags, corrs, agree_frames=10)
    assert sorted(lags[keep]) == [100.0, 102.0, 104.0]


def test_short_song_uses_one_central_window(tmp_path):
    # 20 s < step_s + window_s do perfil: a grade não cabe, mas a música ainda precisa alinhar
    drums, ref = _drums_and_reference(tmp_path, delay_s=1.5, seconds=20.0)
    a = estimate_delay(drums, ref, search_s=(-5.0, 30.0), window_s=15, step_s=20)
    assert a.delay_s == pytest.approx(1.5, abs=0.01)
    assert a.windows == 1
