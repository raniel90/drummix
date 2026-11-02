from pathlib import Path

import pytest
import yaml

from drummix import song


def test_parse_moises_stem_name():
    s = song.parse_stem_name("The Hornet - Audio-Hi-Hat-A minor-131bpm-441hz.wav")
    assert s == song.StemName(piece="hihat", key="A minor", bpm=131)
    assert song.parse_stem_name("The Hornet - Audio-other_kit-A minor-131bpm-441hz.wav").piece == "other_kit"
    assert song.parse_stem_name("qualquer.wav") is None


def test_find_stems(tmp_path):
    for name in ["Kick", "Snare", "Toms", "Hi-Hat", "Cymbals", "Other", "other_kit"]:
        (tmp_path / f"S - Audio-{name}-F# minor-113bpm-443hz.wav").write_bytes(b"")
    stems, key, bpm = song.find_stems(tmp_path)
    assert set(stems) == set(song.PIECES)
    assert (key, bpm) == ("F# minor", 113)


def test_find_stems_requires_kick_and_snare(tmp_path):
    (tmp_path / "S - Audio-Toms-A minor-100bpm-440hz.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="kick"):
        song.find_stems(tmp_path)


def test_tonic_hz():
    assert song.tonic_hz("A minor") == pytest.approx(55.0)
    assert song.tonic_hz("F# minor") == pytest.approx(46.25, abs=0.01)
    assert song.tonic_hz("E minor") == pytest.approx(41.20, abs=0.01)


def test_tonic_hz_below_40_goes_up_an_octave():
    assert song.tonic_hz("C major") == pytest.approx(65.41, abs=0.01)
    assert song.tonic_hz("D minor") == pytest.approx(73.42, abs=0.01)
    assert song.tonic_hz("C# minor") == pytest.approx(69.30, abs=0.01)
    assert song.tonic_hz("Eb major") == pytest.approx(77.78, abs=0.01)


@pytest.mark.parametrize("key, same_as", [("Cb major", "B major"), ("E# minor", "F minor"),
                                          ("Fb major", "E major"), ("B# minor", "C minor")])
def test_tonic_hz_accepts_enharmonics(key, same_as):
    assert song.parse_stem_name(f"S - Audio-Kick-{key}-100bpm-440hz.wav").key == key
    assert song.tonic_hz(key) == pytest.approx(song.tonic_hz(same_as))


def test_deep_merge_overrides_nested_only():
    base = {"clean": {"snare": {"gain_db": 2, "highpass_hz": 100}}, "master": {"target_lufs": -14}}
    merged = song.deep_merge(base, {"clean": {"snare": {"gain_db": 4}}})
    assert merged["clean"]["snare"] == {"gain_db": 4, "highpass_hz": 100}
    assert merged["master"]["target_lufs"] == -14
    assert base["clean"]["snare"]["gain_db"] == 2  # base intacto


def test_load_song_applies_profile_then_overrides(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "fusion.yaml").write_text(yaml.safe_dump({"balance": {"drums_below_band_lu": 2.0}, "x": 1}))
    root = tmp_path / "songs" / "s"
    root.mkdir(parents=True)
    (root / "song.yaml").write_text(yaml.safe_dump(
        {"style": "fusion", "key": "A minor", "bpm": 131, "params": {"balance": {"drums_below_band_lu": 1.0}}}))
    s = song.load_song(root, profiles_dir=profiles)
    assert s.params == {"balance": {"drums_below_band_lu": 1.0}, "x": 1}
    assert (s.key, s.bpm, s.style) == ("A minor", 131, "fusion")
    assert s.stem_path("kick") == root / "input" / "stems" / "kick.wav"
    assert s.band_path == root / "input" / "band.wav"


def _song_dir(tmp_path, profile: dict, cfg_text: str):
    profiles = tmp_path / "profiles"
    profiles.mkdir(exist_ok=True)
    (profiles / "fusion.yaml").write_text(yaml.safe_dump(profile))
    root = tmp_path / "songs" / "s"
    root.mkdir(parents=True, exist_ok=True)
    (root / "song.yaml").write_text(cfg_text)
    return root, profiles


def test_canonical_config_ignores_comments_and_formatting(tmp_path):
    root, profiles = _song_dir(tmp_path, {"x": 1}, "style: fusion\nkey: A minor\nbpm: 131\nparams: {}\n")
    a = song.canonical_config(song.load_song(root, profiles_dir=profiles))
    (root / "song.yaml").write_text("# comentário\nstyle: fusion\nbpm: 131\nkey: A minor   # tom\nparams: {}\n")
    assert song.canonical_config(song.load_song(root, profiles_dir=profiles)) == a


def test_canonical_config_sees_profile_changes(tmp_path):
    root, profiles = _song_dir(tmp_path, {"x": 1}, "style: fusion\nkey: A minor\nbpm: 131\nparams: {}\n")
    a = song.canonical_config(song.load_song(root, profiles_dir=profiles))
    (profiles / "fusion.yaml").write_text(yaml.safe_dump({"x": 2}))
    assert song.canonical_config(song.load_song(root, profiles_dir=profiles)) != a
