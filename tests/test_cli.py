from pathlib import Path

import pytest

from drummix import cli
from drummix.audio import AudioError
from drummix.deliver import DeliveryError
from drummix.stages.align import AlignError
from drummix.stages.master import MasterError

REPO = Path(cli.__file__).resolve().parent.parent


@pytest.mark.parametrize("exc", [AudioError, AlignError, MasterError, DeliveryError, ValueError,
                                 FileNotFoundError, FileExistsError])
def test_known_errors_print_message_and_return_1(monkeypatch, capsys, tmp_path, exc):
    def boom(_):
        raise exc("deu ruim")
    monkeypatch.setattr(cli, "load_song", boom)
    assert cli.main(["render", "x", "--songs-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "erro: deu ruim" in err and "Traceback" not in err


def test_missing_song_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["render", "nao-existe", "--songs-dir", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("erro: ")


def test_unexpected_errors_still_propagate(monkeypatch, tmp_path):
    def boom(_):
        raise KeyError("bug")
    monkeypatch.setattr(cli, "load_song", boom)
    with pytest.raises(KeyError):
        cli.main(["render", "x", "--songs-dir", str(tmp_path)])


def test_style_must_be_a_known_profile(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["new", "s", "--stems", str(tmp_path), "--band", "b.wav", "--original", "o.wav",
                  "--style", "polka", "--songs-dir", str(tmp_path)])
    assert e.value.code == 2
    assert "fusion" in capsys.readouterr().err


def test_default_songs_dir_is_anchored_to_the_repo(monkeypatch, tmp_path):
    seen = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_song", lambda root: seen.append(root) or (_ for _ in ()).throw(FileNotFoundError("x")))
    assert cli.main(["render", "abc"]) == 1
    assert seen == [REPO / "songs" / "abc"]
