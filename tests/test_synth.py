import numpy as np

from drummix import audio, song
from tests.synth import make_song


def test_make_song_writes_moises_layout(tmp_path):
    info = make_song(tmp_path, seconds=14.0, delay_s=1.5)
    stems, key, bpm = song.find_stems(info["stems_dir"])
    assert set(stems) == set(song.PIECES)
    assert (key, bpm) == ("A minor", 120)
    band = audio.load(info["band"])
    original = audio.load(info["original"])
    assert band.shape == original.shape
    assert len(info["kick_times"]) > 10
    kick = audio.load(stems["kick"])
    assert np.max(np.abs(kick)) > 0.1
