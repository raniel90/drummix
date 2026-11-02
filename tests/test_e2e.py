import json

import pytest

from drummix import audio, cli
from tests.synth import make_song


def test_new_then_render_twice(tmp_path, capsys):
    info = make_song(tmp_path / "src", seconds=20.0, delay_s=1.5)
    songs = tmp_path / "songs"
    assert cli.main(["new", "synth", "--songs-dir", str(songs), "--stems", str(info["stems_dir"]),
                     "--band", str(info["band"]), "--original", str(info["original"]), "--style", "fusion"]) == 0
    assert cli.main(["render", "synth", "--songs-dir", str(songs)]) == 0
    v1 = songs / "synth" / "output" / "v1"
    rep = json.loads((v1 / "report.json").read_text())
    assert rep["delay_s"] == pytest.approx(1.5, abs=0.01)
    assert rep["final_lufs"] == pytest.approx(-14.0, abs=0.3)
    assert rep["final_true_peak_db"] <= -1.0
    assert rep["drums_vs_band_lu"] == pytest.approx(-2.0, abs=0.5)
    assert rep["kick_hits"] == len(info["kick_times"])
    assert {"kick", "kick_sample", "snare", "room", "band"} <= {p.stem for p in (v1 / "stems").iterdir()}
    assert "other" not in {p.stem for p in (v1 / "stems").iterdir()}
    out = capsys.readouterr().out
    assert "v1" in out and "LUFS" in out

    cfg = (songs / "synth" / "song.yaml").read_text().replace("params: {}", "params:\n  clean:\n    snare:\n      gain_db: 4.0")
    (songs / "synth" / "song.yaml").write_text(cfg)
    assert cli.main(["render", "synth", "--songs-dir", str(songs)]) == 0
    assert (songs / "synth" / "output" / "v2" / "final.wav").exists()
    assert audio.sha1(v1 / "final.wav") == rep["hashes"]["final.wav"]
