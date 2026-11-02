import json

import pytest
import yaml

from drummix import audio, ingest
from tests.synth import make_song


def test_create_song_copies_inputs_and_writes_config(tmp_path):
    info = make_song(tmp_path / "src", seconds=8.0)
    root = ingest.create_song(tmp_path / "songs", "minha", info["stems_dir"], info["band"], info["original"], style="fusion")
    assert root == tmp_path / "songs" / "minha"
    cfg = yaml.safe_load((root / "song.yaml").read_text())
    assert cfg == {"style": "fusion", "key": "A minor", "bpm": 120, "params": {}}
    for piece in ("kick", "snare", "hihat"):
        assert (root / "input" / "stems" / f"{piece}.wav").exists()
    manifest = json.loads((root / "input" / "manifest.json").read_text())
    assert manifest["band.wav"]["sha1"] == audio.sha1(root / "input" / "band.wav")


def test_create_song_refuses_to_overwrite(tmp_path):
    info = make_song(tmp_path / "src", seconds=8.0)
    ingest.create_song(tmp_path / "songs", "s", info["stems_dir"], info["band"], info["original"], style="fusion")
    with pytest.raises(FileExistsError):
        ingest.create_song(tmp_path / "songs", "s", info["stems_dir"], info["band"], info["original"], style="fusion")


def test_non_wav_inputs_are_decoded(tmp_path):
    info = make_song(tmp_path / "src", seconds=8.0)
    mp3 = tmp_path / "band.mp3"
    audio.run(["ffmpeg", "-v", "error", "-y", "-i", str(info["band"]), str(mp3)])
    root = ingest.create_song(tmp_path / "songs", "s", info["stems_dir"], mp3, info["original"], style="fusion")
    assert audio.load(root / "input" / "band.wav").shape[1] == 2
