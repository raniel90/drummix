"""Single place for audio I/O and measurement. Arrays are (samples, 2) float64 at SR."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pyebur128 import MeasurementMode, R128State, get_true_peak

SR = 44100
_NATIVE = {".wav", ".flac", ".aif", ".aiff"}


class AudioError(RuntimeError):
    """An external tool or file did not do what we needed. Never continue silently."""


def run(args: list[str]) -> bytes:
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-800:]
        raise AudioError(f"{args[0]} saiu com código {proc.returncode}: {tail}")
    return proc.stdout


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return np.ascontiguousarray(x[:, :2], dtype=np.float64)


def load(path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in _NATIVE:
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        if sr != SR:
            raise AudioError(f"{path.name}: {sr} Hz, esperado {SR} Hz")
        return _stereo(x)
    # ffmpeg aplica o edit list (ex.: priming AAC de 2112 amostras) ao decodificar
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"])
    return _stereo(np.frombuffer(raw, np.float32).reshape(-1, 2))


def save(path, x: np.ndarray, subtype: str = "PCM_24") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _stereo(x), SR, subtype=subtype)


def lufs(x: np.ndarray) -> float:
    """Integrated loudness (BS.1770). Silence gives -inf; callers check isfinite."""
    x = _stereo(x)
    if len(x) < int(0.4 * SR):
        raise AudioError("sinal curto demais para LUFS (< 400 ms)")
    return float(pyln.Meter(SR).integrated_loudness(x))


def true_peak_db(x: np.ndarray) -> float:
    x = _stereo(x)
    state = R128State(2, SR, MeasurementMode.MODE_TRUE_PEAK)
    state.add_frames(x.reshape(-1), len(x))
    peak = max(get_true_peak(state, c) for c in range(2))
    return float(20 * np.log10(peak)) if peak > 0 else float("-inf")


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x)) + 1e-20))


def gain(db: float) -> float:
    return float(10 ** (db / 20))


def sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
