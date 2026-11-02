import numpy as np
import pytest

from drummix import audio
from drummix.audio import SR


def _tone(seconds=3.0, amp=0.5, freq=1000.0):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], axis=1)


def test_save_load_roundtrip_24bit(tmp_path):
    x = _tone()
    audio.save(tmp_path / "a.wav", x)
    y = audio.load(tmp_path / "a.wav")
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-6


def test_load_mono_becomes_stereo(tmp_path):
    import soundfile as sf
    sf.write(tmp_path / "m.wav", np.zeros(SR), SR, subtype="PCM_24")
    assert audio.load(tmp_path / "m.wav").shape == (SR, 2)


def test_load_rejects_wrong_samplerate(tmp_path):
    import soundfile as sf
    sf.write(tmp_path / "x.wav", np.zeros((48000, 2)), 48000)
    with pytest.raises(audio.AudioError, match="48000"):
        audio.load(tmp_path / "x.wav")


def test_load_non_wav_goes_through_ffmpeg(tmp_path):
    x = _tone()
    audio.save(tmp_path / "a.wav", x)
    audio.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "a.wav"), "-c:a", "aac", str(tmp_path / "a.m4a")])
    y = audio.load(tmp_path / "a.m4a")
    assert abs(len(y) - len(x)) < SR * 0.1


def test_run_raises_on_failure():
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        audio.run(["ffmpeg", "-v", "error", "-i", "/nao/existe.wav", "-f", "null", "-"])


def test_lufs_tracks_level_and_matches_ffmpeg(tmp_path):
    x = _tone(amp=0.25)
    assert audio.lufs(x * 2) - audio.lufs(x) == pytest.approx(6.02, abs=0.05)
    audio.save(tmp_path / "t.wav", x)
    log = __import__("subprocess").run(["ffmpeg", "-hide_banner", "-i", str(tmp_path / "t.wav"), "-af", "ebur128",
                                        "-f", "null", "-"], capture_output=True, text=True).stderr
    ffmpeg_i = float(log.rsplit("I:", 1)[1].split("LUFS")[0])
    assert audio.lufs(x) == pytest.approx(ffmpeg_i, abs=0.2)


def test_true_peak_detects_intersample_peak():
    # senoide em fs/4 com fase 45°: amostras em ±0.707·A, pico real = A
    n = np.arange(SR)
    x = 0.9 * np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4)
    st = np.stack([x, x], axis=1)
    assert np.max(np.abs(st)) < 0.65
    assert audio.true_peak_db(st) == pytest.approx(20 * np.log10(0.9), abs=0.3)


def test_rms_db_and_gain():
    assert audio.gain(-6.0206) == pytest.approx(0.5, rel=1e-3)
    assert audio.rms_db(np.full((100, 2), 0.5)) == pytest.approx(-6.02, abs=0.01)


def test_sha1_changes_with_content(tmp_path):
    (tmp_path / "a").write_bytes(b"x")
    (tmp_path / "b").write_bytes(b"y")
    assert audio.sha1(tmp_path / "a") != audio.sha1(tmp_path / "b")


def test_lufs_rejects_signal_shorter_than_400ms():
    with pytest.raises(audio.AudioError, match="400 ms"):
        audio.lufs(_tone(seconds=0.3))
    assert np.isfinite(audio.lufs(_tone(seconds=0.4)))
