import numpy as np
import pytest

from drummix import audio, song
from drummix.audio import SR, rms_db
from drummix.stages.kick_sample import detect_kicks, kick_sample, synth_kick
from tests.synth import make_song


def _kick_stem(tmp_path):
    info = make_song(tmp_path, seconds=20.0, delay_s=1.0)
    stems, _, _ = song.find_stems(info["stems_dir"])
    return audio.load(stems["kick"]), info["kick_times"]


def test_detects_every_kick_slightly_early(tmp_path):
    kick, truth = _kick_stem(tmp_path)
    starts, _ = detect_kicks(kick)
    offset = np.sort(starts / SR) - np.sort(truth)
    assert len(starts) == len(truth)
    assert np.all((offset > -0.010) & (offset < 0.001))  # medido: -6 a -7.5 ms


def test_hits_land_on_true_attack_after_fine_alignment(tmp_path):
    kick, truth = _kick_stem(tmp_path)
    _, hits = kick_sample(kick, tonic_hz=55.0, reference_db=rms_db(kick), level_db=-9.0)
    assert np.max(np.abs(np.array([h.time_s for h in hits]) - np.sort(truth))) < 0.002


def test_synth_kick_is_tuned_to_tonic():
    k = synth_kick(55.0)
    tail = k[int(0.1 * SR):int(0.3 * SR)]
    spectrum = np.abs(np.fft.rfft(tail * np.hanning(len(tail)), n=1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / SR)
    assert freqs[np.argmax(spectrum)] == pytest.approx(55.0, abs=2.0)


def test_kick_sample_level_and_hits(tmp_path):
    kick, truth = _kick_stem(tmp_path)
    sample, hits = kick_sample(kick, tonic_hz=55.0, reference_db=rms_db(kick), level_db=-9.0)
    assert sample.shape == kick.shape
    assert rms_db(sample) - rms_db(kick) == pytest.approx(-9.0, abs=0.1)
    assert len(hits) == len(truth)
    assert all(1 <= h.velocity <= 127 for h in hits)


def test_sample_is_in_phase_with_real_kick(tmp_path):
    kick, _ = _kick_stem(tmp_path)
    sample, _ = kick_sample(kick, tonic_hz=55.0, reference_db=rms_db(kick), level_db=-9.0)
    a, b = kick.mean(1), sample.mean(1)
    lags = range(-int(0.004 * SR), int(0.004 * SR) + 1, 8)
    corr = [np.dot(a[max(0, l):len(a) + min(0, l)], b[max(0, -l):len(b) + min(0, -l)]) for l in lags]
    assert abs(list(lags)[int(np.argmax(corr))]) <= int(0.001 * SR)


def _with_retriggers(kick, truth):
    # "limpo" = cru + golpes fantasmas ~13 dB abaixo, no meio entre golpes reais
    # (o compressor da limpeza redisparava assim na música real: 597 detecções contra 544)
    bump = synth_kick(55.0) * np.max(np.abs(kick)) * 10 ** (-13 / 20)
    out = kick.copy()
    t = np.sort(truth)
    for a, b in zip(t, t[1:]):
        i = int((a + b) / 2 * SR)
        if i + len(bump) <= len(out):
            out[i:i + len(bump)] += bump[:, None]
    return out


def test_detect_on_raw_ignores_retriggers_of_cleaned_kick(tmp_path):
    kick, truth = _kick_stem(tmp_path)
    cleaned = _with_retriggers(kick, truth)
    assert len(detect_kicks(cleaned)[0]) > len(truth)  # o cenário realmente engana o detector
    sample, hits = kick_sample(cleaned, tonic_hz=55.0, reference_db=rms_db(cleaned), level_db=-9.0, detect_on=kick)
    assert len(hits) == len(truth)
    assert sample.shape == cleaned.shape
    assert np.max(np.abs(np.array([h.time_s for h in hits]) - np.sort(truth))) < 0.002
