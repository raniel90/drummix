"""Real-audio regression against the manual mix approved by ear (2026-10-31)."""
import json
from pathlib import Path

import pytest

from drummix import cli

DL = Path.home() / "Downloads"
STEMS = DL / "The Hornet - Audio-A minor-131bpm-441hz"
BAND = DL / "The Hornet - Sem bateria.wav"
ORIGINAL = DL / "The Hornet - Original.wav"

pytestmark = [pytest.mark.regression,
              pytest.mark.skipif(not STEMS.exists(), reason="arquivos de The Hornet não estão em ~/Downloads")]


def test_hornet_matches_manual_session(tmp_path):
    songs = tmp_path / "songs"
    assert cli.main(["new", "the-hornet", "--songs-dir", str(songs), "--stems", str(STEMS),
                     "--band", str(BAND), "--original", str(ORIGINAL)]) == 0
    assert cli.main(["render", "the-hornet", "--songs-dir", str(songs)]) == 0
    rep = json.loads((songs / "the-hornet" / "output" / "v1" / "report.json").read_text())
    assert rep["delay_s"] == pytest.approx(11.73, abs=0.03)
    assert 490 <= rep["kick_hits"] <= 600  # sessão manual: 544
    assert rep["final_lufs"] == pytest.approx(-14.0, abs=0.3)
    assert rep["final_true_peak_db"] <= -1.0
    assert rep["drums_vs_band_lu"] == pytest.approx(-2.0, abs=0.5)
