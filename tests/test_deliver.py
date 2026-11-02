import json

import numpy as np
import pytest

from drummix import audio, deliver
from drummix.audio import SR
from drummix.dsp.midi import Hit


def _stems():
    rng = np.random.default_rng(0)
    return {"kick": rng.standard_normal((SR * 2, 2)) * 0.05, "band": rng.standard_normal((SR * 2, 2)) * 0.05}


def _final(stems):
    return sum(stems.values()) * 0.9


def test_writes_version_and_verifies(tmp_path):
    stems = _stems()
    v = deliver.deliver(tmp_path, stems, _final(stems), {36: [Hit(0.5, 100)]}, bpm=120,
                        report={"x": 1}, targets={"target_lufs": None, "ceiling_dbtp": 0.0}, config_text="a: 1")
    assert v == tmp_path / "v1"
    assert (v / "final.wav").exists() and (v / "drums.mid").exists()
    assert sorted(p.name for p in (v / "stems").iterdir()) == ["band.wav", "kick.wav"]
    rep = json.loads((v / "report.json").read_text())
    assert rep["x"] == 1 and rep["stems_sum_error_db"] < -90
    assert rep["hashes"]["final.wav"] == audio.sha1(v / "final.wav")
    assert (v / "report.md").exists()
    assert not (tmp_path / "v1.tmp").exists()


def test_next_version_never_overwrites(tmp_path):
    stems = _stems()
    kw = dict(hits={}, bpm=120, report={}, targets={"target_lufs": None, "ceiling_dbtp": 0.0})
    deliver.deliver(tmp_path, stems, _final(stems), config_text="a: 1", **kw)
    first = audio.sha1(tmp_path / "v1" / "final.wav")
    v2 = deliver.deliver(tmp_path, stems, _final(stems) * 0.5, config_text="a: 2", **kw)
    assert v2.name == "v2"
    assert audio.sha1(tmp_path / "v1" / "final.wav") == first


def test_changed_config_with_identical_final_is_an_error(tmp_path):
    stems = _stems()
    kw = dict(hits={}, bpm=120, report={}, targets={"target_lufs": None, "ceiling_dbtp": 0.0})
    deliver.deliver(tmp_path, stems, _final(stems), config_text="a: 1", **kw)
    with pytest.raises(deliver.DeliveryError, match="idêntico"):
        deliver.deliver(tmp_path, stems, _final(stems), config_text="a: 2", **kw)
    assert not (tmp_path / "v2").exists()


def test_failed_check_leaves_no_version(tmp_path):
    stems = _stems()
    with pytest.raises(deliver.DeliveryError, match="LUFS"):
        deliver.deliver(tmp_path, stems, _final(stems), {}, bpm=120, report={},
                        targets={"target_lufs": -14.0, "ceiling_dbtp": -1.0}, config_text="a: 1")
    assert not (tmp_path / "v1").exists() and not (tmp_path / "v1.tmp").exists()


def test_true_peak_over_ceiling_is_strict(tmp_path, monkeypatch):
    stems = _stems()
    monkeypatch.setattr(deliver.audio, "true_peak_db", lambda x: 0.01)
    with pytest.raises(deliver.DeliveryError, match="true peak"):
        deliver.deliver(tmp_path, stems, _final(stems), {}, bpm=120, report={},
                        targets={"target_lufs": None, "ceiling_dbtp": 0.0}, config_text="a: 1")


def test_same_config_and_identical_final_is_allowed(tmp_path):
    stems = _stems()
    kw = dict(hits={}, bpm=120, report={}, targets={"target_lufs": None, "ceiling_dbtp": 0.0})
    deliver.deliver(tmp_path, stems, _final(stems), config_text="a: 1", **kw)
    assert deliver.deliver(tmp_path, stems, _final(stems), config_text="a: 1", **kw).name == "v2"
