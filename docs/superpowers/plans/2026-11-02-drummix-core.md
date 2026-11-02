# drummix — Plano 1: núcleo reprodutível — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI `drummix new|render` que reproduz, de forma verificável e versionada, a mixagem manual aprovada de The Hornet (stems Moises → mixagem completa −14 LUFS / −1 dBTP + stems + MIDI + relatório).

**Architecture:** Pacote Python com um único módulo de I/O (`audio.py`), configuração em camadas (`profiles/*.yaml` + `songs/<slug>/song.yaml`), stages puros `(arrays, params) -> arrays` em `drummix/stages/`, orquestrados por `pipeline.py` e gravados por `deliver.py` em `output/vN/` (escrita em `vN.tmp` + rename, verificação de soma/LUFS/TP).

**Tech Stack:** Python 3.12 (uv), numpy, scipy, soundfile, pyloudnorm, pyebur128, pedalboard 0.9.26, pretty_midi, pyyaml, pytest; ffmpeg do sistema.

**Spec:** `docs/superpowers/specs/2026-11-02-drummix-design.md`

**Roadmap (planos seguintes, fora deste):**
- Plano 2 — técnicas avançadas: transientes, sample de caixa (packs CC0, librosa), NY parallel, saturação, de-harsh, IRs Voxengo + plate, EQ dinâmico com chave na banda, EQ-match da master, perfil `rock`.
- Plano 3 — `compare.html` + simulação de celular, troca de áudio no vídeo, skill do Claude Code.

**APIs verificadas (2026-11-02, venv temporário):** `pedalboard.BrickwallLimiter(ceiling_db=-1.0, release_ms=100.0, lookahead_ms=5.0, true_peak=False)` existe e, com `true_peak=True`, entregou TP −1,000 dBTP; pedalboard compensa latência (impulso sai no mesmo índice). `pyebur128.R128State(channels, sr, MeasurementMode.MODE_TRUE_PEAK)` + `add_frames(interleaved_1d, n_frames)` + `get_true_peak(state, ch)`. `pyloudnorm.Meter(sr).integrated_loudness(x)` com x `(n, ch)`; silêncio → `-inf`.

**Convenções:** arrays de áudio são `np.ndarray` `(samples, 2)` float64 a 44 100 Hz. Pedalboard recebe `(2, samples)` float32 — conversão só em `dsp/fx.py`. Nenhum subprocess via shell: sempre lista de argumentos e `audio.run()` (levanta em returncode ≠ 0).

---

## Estrutura de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `pyproject.toml` | deps, entrypoint `drummix`, pytest |
| `drummix/audio.py` | load/save, ffmpeg, LUFS, true peak, RMS, sha1 |
| `drummix/song.py` | nomes Moises, tom/BPM, perfil + overrides, `Song` |
| `drummix/ingest.py` | `create_song`: copia entradas, manifest, `song.yaml` |
| `drummix/dsp/filters.py` | biquads (peaking/HP/LP), `eq`, níveis por oitava |
| `drummix/dsp/fx.py` | ponte pedalboard |
| `drummix/dsp/midi.py` | `Hit`, escrita de MIDI de bateria |
| `drummix/stages/align.py` | atraso bateria↔referência por onsets |
| `drummix/stages/clean.py` | limpeza por peça (HP/LP/EQ/gate/comp/mono/largura/ganho) |
| `drummix/stages/kick_sample.py` | detecção de golpes do bumbo + sample sintético afinado |
| `drummix/stages/tone_match.py` | match de timbre por oitava vs referência (X%) |
| `drummix/stages/space.py` | sala sintética (send) |
| `drummix/stages/band.py` | banda: EQ estático + deslocamento na linha do tempo |
| `drummix/stages/balance.py` | bateria X LU abaixo da banda |
| `drummix/stages/master.py` | glue + ganho + BrickwallLimiter TP até o alvo |
| `drummix/deliver.py` | vN, stems, verificação, MIDI, relatório |
| `drummix/pipeline.py` | `render(song)` orquestra tudo |
| `drummix/cli.py` | `drummix new`, `drummix render` |
| `profiles/fusion.yaml` | parâmetros validados em The Hornet |
| `tests/synth.py` | gerador de música sintética (stems Moises, banda, original) |
| `tests/test_*.py` | um arquivo por módulo + e2e + regressão |

---

### Task 1: Scaffold do projeto

**Files:**
- Create: `pyproject.toml`, `drummix/__init__.py`, `drummix/dsp/__init__.py`, `drummix/stages/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Criar `pyproject.toml`**

```toml
[project]
name = "drummix"
version = "0.1.0"
description = "Mixagem reprodutível de covers de bateria (EAD10 + stems Moises)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=2.1",
    "scipy>=1.14",
    "soundfile>=0.13",
    "pyloudnorm>=0.2",
    "pyebur128>=0.1.1",
    "pedalboard>=0.9.26",
    "pretty_midi>=0.2.11",
    "pyyaml>=6",
]

[project.scripts]
drummix = "drummix.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["regression: usa áudio real de ~/Downloads (lento)"]
addopts = "-m 'not regression'"
```

- [ ] **Step 2: Criar pacotes vazios e teste de fumaça**

`drummix/__init__.py`, `drummix/dsp/__init__.py`, `drummix/stages/__init__.py`, `tests/__init__.py`: arquivos vazios.

`tests/test_smoke.py`:
```python
import pedalboard
import pyebur128
import pyloudnorm


def test_dependencies_import():
    assert hasattr(pedalboard, "BrickwallLimiter")
    assert hasattr(pyebur128, "R128State")
    assert hasattr(pyloudnorm, "Meter")
```

- [ ] **Step 3: Instalar e rodar**

Run: `cd ~/Documents/gitworkspace/personal/music/drummix && uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock drummix tests
git commit -m "chore: scaffold do pacote drummix"
```

---

### Task 2: `audio.py` — I/O e medições

**Files:**
- Create: `drummix/audio.py`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Escrever os testes**

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_audio.py -q`
Expected: FAIL (`ImportError: cannot import name 'audio'`)

- [ ] **Step 3: Implementar `drummix/audio.py`**

```python
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
    return float(pyln.Meter(SR).integrated_loudness(_stereo(x)))


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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_audio.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/audio.py tests/test_audio.py
git commit -m "feat: módulo único de I/O e medições de áudio"
```

---

### Task 3: `song.py` — nomes Moises, tom e configuração em camadas

**Files:**
- Create: `drummix/song.py`
- Test: `tests/test_song.py`

- [ ] **Step 1: Escrever os testes**

```python
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
    assert song.tonic_hz("C major") == pytest.approx(32.70, abs=0.01)
    assert song.tonic_hz("F# minor") == pytest.approx(46.25, abs=0.01)


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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_song.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implementar `drummix/song.py`**

```python
"""Song configuration: Moises file names, musical key, and profile + per-song overrides."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PIECES = ("kick", "snare", "toms", "hihat", "cymbals", "other", "other_kit")
PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

_MOISES = {"Kick": "kick", "Snare": "snare", "Toms": "toms", "Hi-Hat": "hihat",
           "Cymbals": "cymbals", "Other": "other", "other_kit": "other_kit"}
_STEM_RE = re.compile(
    r"-(?P<piece>Kick|Snare|Toms|Hi-Hat|Cymbals|Other|other_kit)"
    r"-(?P<key>[A-G][#b]? (?:major|minor))-(?P<bpm>\d+)bpm-\d+hz\.wav$")
_NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
         "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


@dataclass(frozen=True)
class StemName:
    piece: str
    key: str
    bpm: int


def parse_stem_name(name: str) -> StemName | None:
    m = _STEM_RE.search(name)
    if not m:
        return None
    return StemName(_MOISES[m["piece"]], m["key"], int(m["bpm"]))


def find_stems(folder) -> tuple[dict[str, Path], str, int]:
    found: dict[str, Path] = {}
    keys = set()
    for path in sorted(Path(folder).iterdir()):
        parsed = parse_stem_name(path.name)
        if parsed:
            found[parsed.piece] = path
            keys.add((parsed.key, parsed.bpm))
    for required in ("kick", "snare"):
        if required not in found:
            raise ValueError(f"stem obrigatório ausente em {folder}: {required}")
    if len(keys) != 1:
        raise ValueError(f"stems com tom/BPM divergentes: {sorted(keys)}")
    key, bpm = keys.pop()
    return found, key, bpm


def tonic_hz(key: str, octave: int = 1) -> float:
    note = key.split()[0]
    midi = 12 * (octave + 1) + _NOTE[note]
    return 440.0 * 2 ** ((midi - 69) / 12)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class Song:
    root: Path
    style: str
    key: str
    bpm: int
    params: dict

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def stem_path(self, piece: str) -> Path:
        return self.input_dir / "stems" / f"{piece}.wav"

    @property
    def band_path(self) -> Path:
        return self.input_dir / "band.wav"

    @property
    def original_path(self) -> Path:
        return self.input_dir / "original.wav"

    @property
    def config_path(self) -> Path:
        return self.root / "song.yaml"


def load_song(root, profiles_dir=PROFILES_DIR) -> Song:
    root = Path(root)
    cfg = yaml.safe_load((root / "song.yaml").read_text())
    profile = yaml.safe_load((Path(profiles_dir) / f"{cfg['style']}.yaml").read_text())
    return Song(root, cfg["style"], cfg["key"], int(cfg["bpm"]), deep_merge(profile, cfg.get("params") or {}))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_song.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/song.py tests/test_song.py
git commit -m "feat: configuração da música (Moises, tom, perfil + overrides)"
```

---

### Task 4: Gerador de música sintética para testes

**Files:**
- Create: `tests/synth.py`
- Test: `tests/test_synth.py`

Gera uma "música" curta com stems no padrão Moises, banda e original, e um atraso conhecido entre a bateria gravada e a banda. É a base dos testes de align, kick_sample e e2e.

- [ ] **Step 1: Escrever o teste**

```python
import numpy as np

from drummix import audio, song
from tests.synth import make_song


def test_make_song_writes_moises_layout(tmp_path):
    info = make_song(tmp_path, seconds=12.0, delay_s=1.5)
    stems, key, bpm = song.find_stems(info["stems_dir"])
    assert set(stems) == set(song.PIECES)
    assert (key, bpm) == ("A minor", 120)
    band = audio.load(info["band"])
    original = audio.load(info["original"])
    assert band.shape == original.shape
    assert len(info["kick_times"]) > 10
    kick = audio.load(stems["kick"])
    assert np.max(np.abs(kick)) > 0.1
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_synth.py -q`
Expected: FAIL (`ModuleNotFoundError: tests.synth`)

- [ ] **Step 3: Implementar `tests/synth.py`**

```python
"""Synthetic song for tests: Moises-named drum stems, drumless band, original, known delay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from drummix import audio
from drummix.audio import SR

_NAMES = {"kick": "Kick", "snare": "Snare", "toms": "Toms", "hihat": "Hi-Hat",
          "cymbals": "Cymbals", "other": "Other", "other_kit": "other_kit"}


def _decay(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def _kick(rng):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 55 + 105 * np.exp(-t / 0.02)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.12)


def _noise(rng, n, lo, hi, tau):
    sos = butter(2, [lo, hi], "band", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * _decay(n, tau)


def _place(track, sound, t_s, amp):
    i = int(round(t_s * SR))
    if 0 <= i < len(track):
        seg = sound[: len(track) - i]
        track[i:i + len(seg)] += amp * seg


def make_song(folder, seconds=20.0, delay_s=1.5, bpm=120, seed=0) -> dict:
    """Drums are recorded `delay_s` later than the band timeline (like the EAD10 take)."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    n = int(seconds * SR)
    beat = 60.0 / bpm
    music_len = seconds - delay_s - 0.5
    pieces = {k: np.zeros(n) for k in _NAMES}
    ref = {k: np.zeros(n) for k in _NAMES}  # same drums on the band timeline
    kick_times, snare_times = [], []
    k = 0
    while k * beat < music_len:
        t_band = k * beat
        jitter = rng.uniform(-0.003, 0.003)
        for track, t in ((pieces, t_band + delay_s + jitter), (ref, t_band)):
            if k % 2 == 0:
                _place(track["kick"], _kick(rng), t, 0.8)
            else:
                _place(track["snare"], _noise(rng, int(0.2 * SR), 200, 8000, 0.08), t, 0.6)
            _place(track["hihat"], _noise(rng, int(0.05 * SR), 7000, 15000, 0.02), t, 0.15)
            _place(track["hihat"], _noise(rng, int(0.05 * SR), 7000, 15000, 0.02), t + beat / 2, 0.1)
            if k % 8 == 0:
                _place(track["cymbals"], _noise(rng, int(1.2 * SR), 3000, 16000, 0.6), t, 0.2)
            if k % 2 == 1:  # ghost note no contratempo
                _place(track["snare"], _noise(rng, int(0.1 * SR), 200, 8000, 0.04), t + beat / 2, 0.08)
        (kick_times if k % 2 == 0 else snare_times).append(t_band + delay_s + jitter)
        k += 1
    t = np.arange(n) / SR
    band = np.zeros(n)
    for f in (110.0, 130.81, 164.81):
        band += 0.08 * np.sin(2 * np.pi * f * t)
    band += 0.15 * np.sin(2 * np.pi * 55.0 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / beat)))
    band[int(music_len * SR):] = 0.0
    pieces["toms"] += 1e-4 * rng.standard_normal(n)
    pieces["other"] += 1e-6 * rng.standard_normal(n)
    pieces["other_kit"] += 1e-4 * rng.standard_normal(n)

    stems_dir = folder / "stems"
    for piece, x in pieces.items():
        audio.save(stems_dir / f"Synth - Audio-{_NAMES[piece]}-A minor-{bpm}bpm-440hz.wav", np.stack([x, x], 1))
    ref_drums = sum(ref.values())
    audio.save(folder / "band.wav", np.stack([band, band], 1))
    audio.save(folder / "original.wav", np.stack([band + ref_drums, band + ref_drums], 1))
    return {"stems_dir": stems_dir, "band": folder / "band.wav", "original": folder / "original.wav",
            "delay_s": delay_s, "kick_times": kick_times, "snare_times": snare_times}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_synth.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/synth.py tests/test_synth.py
git commit -m "test: gerador de música sintética com atraso conhecido"
```

---

### Task 5: `dsp/filters.py`, `dsp/fx.py`, `dsp/midi.py`

**Files:**
- Create: `drummix/dsp/filters.py`, `drummix/dsp/fx.py`, `drummix/dsp/midi.py`
- Test: `tests/test_dsp.py`

- [ ] **Step 1: Escrever os testes**

```python
import numpy as np
import pedalboard as pb
import pretty_midi
import pytest

from drummix.audio import SR, rms_db
from drummix.dsp import filters, fx, midi


def _sine(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], 1)


def test_peaking_boosts_at_center_only():
    x = _sine(1000)
    y = filters.eq(x, [[1000, 6.0, 1.0]])
    assert rms_db(y[SR // 2:]) - rms_db(x[SR // 2:]) == pytest.approx(6.0, abs=0.3)
    far = _sine(100)
    assert rms_db(filters.eq(far, [[1000, 6.0, 1.0]])[SR // 2:]) - rms_db(far[SR // 2:]) == pytest.approx(0, abs=0.3)


def test_highpass_attenuates_below_cutoff():
    y = filters.apply_sos(filters.highpass_sos(200, order=2), _sine(50))
    assert rms_db(y[SR // 2:]) - rms_db(_sine(50)[SR // 2:]) < -20


def test_octave_levels_peak_where_energy_is():
    levels = filters.octave_levels(_sine(1000))
    assert filters.OCTAVES[int(np.argmax(levels))] == 1000


def test_fx_process_keeps_shape_and_alignment():
    x = np.zeros((SR, 2))
    x[10000] = 0.5
    y = fx.process(x, [pb.Gain(gain_db=0.0)])
    assert y.shape == x.shape and y.dtype == np.float64
    assert int(np.argmax(np.abs(y[:, 0]))) == 10000


def test_write_drum_midi(tmp_path):
    hits = {36: [midi.Hit(0.5, 100), midi.Hit(1.0, 60)], 38: [midi.Hit(0.75, 90)]}
    midi.write_drum_midi(tmp_path / "d.mid", hits, bpm=131)
    pm = pretty_midi.PrettyMIDI(str(tmp_path / "d.mid"))
    notes = sorted((n.start, n.pitch, n.velocity) for n in pm.instruments[0].notes)
    assert pm.instruments[0].is_drum
    assert [(round(s, 3), p, v) for s, p, v in notes] == [(0.5, 36, 100), (0.75, 38, 90), (1.0, 36, 60)]
    assert pm.get_tempo_changes()[1][0] == pytest.approx(131, abs=0.01)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_dsp.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implementar `drummix/dsp/filters.py`**

```python
"""Biquads (RBJ peaking, Butterworth HP/LP) and octave-band level analysis."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from drummix.audio import SR

OCTAVES = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def peaking_sos(fc: float, gain_db: float, q: float) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * fc / SR
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])[None, :]


def highpass_sos(fc: float, order: int = 2) -> np.ndarray:
    return butter(order, fc, "high", fs=SR, output="sos")


def lowpass_sos(fc: float, order: int = 2) -> np.ndarray:
    return butter(order, fc, "low", fs=SR, output="sos")


def apply_sos(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sosfilt(sos, x, axis=0)


def eq(x: np.ndarray, bands) -> np.ndarray:
    """bands: [[freq_hz, gain_db, q], ...]; bands too close to Nyquist are skipped."""
    sections = [peaking_sos(f, g, q) for f, g, q in bands if abs(g) > 0.05 and f < 0.8 * SR / 2]
    return apply_sos(np.vstack(sections), x) if sections else x


def octave_levels(x: np.ndarray) -> np.ndarray:
    """Level of each octave band in dB relative to the full-band level (mono sum)."""
    m = np.asarray(x).mean(axis=1)
    total = 10 * np.log10(np.mean(m ** 2) + 1e-20)
    out = []
    for fc in OCTAVES:
        lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), 0.95 * SR / 2)
        band = sosfiltfilt(butter(4, [lo, hi], "band", fs=SR, output="sos"), m)
        out.append(10 * np.log10(np.mean(band ** 2) + 1e-20) - total)
    return np.array(out)
```

- [ ] **Step 4: Implementar `drummix/dsp/fx.py`**

```python
"""Bridge to pedalboard: our (samples, 2) float64 <-> its (2, samples) float32."""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from drummix.audio import SR


def process(x: np.ndarray, plugins: list) -> np.ndarray:
    y = pb.Pedalboard(plugins)(np.ascontiguousarray(x.T, dtype=np.float32), SR)
    return np.ascontiguousarray(y.T, dtype=np.float64)
```

- [ ] **Step 5: Implementar `drummix/dsp/midi.py`**

```python
"""Drum hits and General MIDI drum file writing (36 kick, 38 snare)."""
from __future__ import annotations

from dataclasses import dataclass

import pretty_midi

KICK, SNARE = 36, 38


@dataclass(frozen=True)
class Hit:
    time_s: float
    velocity: int


def write_drum_midi(path, hits: dict[int, list[Hit]], bpm: float, note_s: float = 0.1) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="drummix")
    for pitch, items in hits.items():
        for h in items:
            kit.notes.append(pretty_midi.Note(velocity=int(h.velocity), pitch=pitch,
                                              start=h.time_s, end=h.time_s + note_s))
    kit.notes.sort(key=lambda n: n.start)
    pm.instruments.append(kit)
    pm.write(str(path))
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest tests/test_dsp.py -q`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add drummix/dsp tests/test_dsp.py
git commit -m "feat: filtros, ponte pedalboard e escrita de MIDI"
```

---

### Task 6: `stages/align.py` — atraso por correlação de onsets

**Files:**
- Create: `drummix/stages/align.py`
- Test: `tests/test_align.py`

- [ ] **Step 1: Escrever os testes**

```python
import numpy as np
import pytest

from drummix import audio, song
from drummix.stages.align import AlignError, estimate_delay
from tests.synth import make_song


def _drums_and_reference(tmp_path, delay_s, seconds=60.0):
    info = make_song(tmp_path, seconds=seconds, delay_s=delay_s)
    stems, _, _ = song.find_stems(info["stems_dir"])
    drums = sum(audio.load(p) for p in stems.values())
    ref = audio.load(info["original"]) - audio.load(info["band"])
    return drums, ref


def test_recovers_known_delay(tmp_path):
    drums, ref = _drums_and_reference(tmp_path, delay_s=2.37)
    a = estimate_delay(drums, ref, search_s=(-1.0, 5.0), window_s=10, step_s=8)
    assert a.delay_s == pytest.approx(2.37, abs=0.01)
    assert a.spread_s < 0.01
    assert a.used >= 3


def test_raises_when_nothing_correlates():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((audio.SR * 40, 2)) * 1e-9
    with pytest.raises(AlignError):
        estimate_delay(noise, noise * 0, search_s=(0, 2), window_s=10, step_s=8)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_align.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementar `drummix/stages/align.py`**

```python
"""Delay of the recorded drums relative to the band timeline, via onset cross-correlation.

Positive delay_s: the drums happen delay_s later than the reference, so the band must be
shifted later by delay_s to line up (the drum/video timeline is the master).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from drummix.audio import SR

FRAME_SR = 200  # onset frames per second (5 ms)


class AlignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Alignment:
    delay_s: float
    spread_s: float
    mean_corr: float
    windows: int
    used: int


def onset_envelope(x: np.ndarray) -> np.ndarray:
    mono = np.asarray(x).mean(axis=1)
    a = np.abs(resample_poly(mono, 40, 441))  # 44100 -> 4000 Hz
    hop = 4000 // FRAME_SR
    a = np.convolve(a, np.ones(hop) / hop, "same")[::hop]
    return np.maximum(np.diff(a, prepend=a[0]), 0)


def estimate_delay(drums, reference, search_s=(-5.0, 30.0), window_s=15, step_s=20,
                   min_corr=0.05, agree_s=0.05) -> Alignment:
    d, r = onset_envelope(drums), onset_envelope(reference)
    lags = np.arange(int(search_s[0] * FRAME_SR), int(search_s[1] * FRAME_SR) + 1)
    win = int(window_s * FRAME_SR)
    found = []
    windows = 0
    for i0 in range(int(step_s * FRAME_SR), len(d) - win, int(step_s * FRAME_SR)):
        seg = d[i0:i0 + win]
        if seg.std() == 0:
            continue
        windows += 1
        seg = (seg - seg.mean()) / seg.std()
        best_c, best_l = -1.0, 0
        for lag in lags:
            j0 = i0 - lag
            if j0 < 0 or j0 + win > len(r):
                continue
            w = r[j0:j0 + win]
            s = w.std()
            if s == 0:
                continue
            c = float(np.dot(seg, (w - w.mean()) / s) / win)
            if c > best_c:
                best_c, best_l = c, lag
        if best_c >= min_corr:
            found.append((best_c, best_l))
    if not found:
        raise AlignError("nenhuma janela correlacionou bateria e referência")
    corr = np.array([c for c, _ in found])
    lag = np.array([l for _, l in found], dtype=float)
    med = np.median(lag)
    keep = np.abs(lag - med) <= agree_s * FRAME_SR
    return Alignment(delay_s=float(np.median(lag[keep]) / FRAME_SR),
                     spread_s=float(np.std(lag[keep]) / FRAME_SR),
                     mean_corr=float(corr[keep].mean()),
                     windows=windows, used=int(keep.sum()))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_align.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/stages/align.py tests/test_align.py
git commit -m "feat: alinhamento bateria↔banda por correlação de onsets"
```

---

### Task 7: `stages/clean.py` — limpeza por peça

**Files:**
- Create: `drummix/stages/clean.py`
- Test: `tests/test_clean.py`

Parâmetros por peça (todos opcionais): `drop`, `mono`, `highpass_hz`, `lowpass_hz`, `gate` (kwargs de `pb.NoiseGate`), `eq` (`[[f, g, q], ...]`), `compressor` (kwargs de `pb.Compressor`), `width` (0 = mono, 1 = original), `gain_db`.

- [ ] **Step 1: Escrever os testes**

```python
import numpy as np
import pytest

from drummix.audio import SR, rms_db
from drummix.stages.clean import clean


def _stereo_noise(seed=0, seconds=1.0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(seconds * SR), 2)) * 0.1


def test_mono_makes_channels_equal():
    out = clean({"kick": _stereo_noise()}, {"kick": {"mono": True}})
    assert np.allclose(out["kick"][:, 0], out["kick"][:, 1])


def test_width_narrows_side():
    x = _stereo_noise()
    y = clean({"hihat": x}, {"hihat": {"width": 0.5}})["hihat"]
    side = lambda s: rms_db((s[:, 0] - s[:, 1])[:, None] / 2)
    assert side(y) - side(x) == pytest.approx(-6.02, abs=0.05)


def test_gain_and_drop_and_untouched_piece():
    x = _stereo_noise()
    out = clean({"snare": x, "other": x, "toms": x}, {"snare": {"gain_db": 2.0}, "other": {"drop": True}})
    assert "other" not in out
    assert rms_db(out["snare"]) - rms_db(x) == pytest.approx(2.0, abs=0.01)
    assert np.array_equal(out["toms"], x)


def test_gate_silences_low_level_residue():
    x = _stereo_noise() * 0.001  # ~ -70 dBFS
    y = clean({"toms": x}, {"toms": {"gate": {"threshold_db": -36, "ratio": 6, "attack_ms": 1, "release_ms": 200}}})["toms"]
    assert rms_db(y) < rms_db(x) - 10
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_clean.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementar `drummix/stages/clean.py`**

```python
"""Per-piece cleanup of separated stems: filters, gate, EQ, compression, width, gain."""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from drummix.audio import gain
from drummix.dsp import fx
from drummix.dsp.filters import apply_sos, eq, highpass_sos, lowpass_sos


def _width(x: np.ndarray, amount: float) -> np.ndarray:
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2 * amount
    return np.stack([mid + side, mid - side], axis=1)


def clean_piece(x: np.ndarray, p: dict) -> np.ndarray:
    y = x
    if p.get("mono"):
        y = _width(y, 0.0)
    if "highpass_hz" in p:
        y = apply_sos(highpass_sos(p["highpass_hz"]), y)
    if "lowpass_hz" in p:
        y = apply_sos(lowpass_sos(p["lowpass_hz"]), y)
    if "gate" in p:
        y = fx.process(y, [pb.NoiseGate(**p["gate"])])
    if p.get("eq"):
        y = eq(y, p["eq"])
    if "compressor" in p:
        y = fx.process(y, [pb.Compressor(**p["compressor"])])
    if "width" in p:
        y = _width(y, p["width"])
    if "gain_db" in p:
        y = y * gain(p["gain_db"])
    return y


def clean(stems: dict[str, np.ndarray], params: dict) -> dict[str, np.ndarray]:
    out = {}
    for piece, x in stems.items():
        p = params.get(piece) or {}
        if p.get("drop"):
            continue
        out[piece] = clean_piece(x, p) if p else x
    return out
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_clean.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/stages/clean.py tests/test_clean.py
git commit -m "feat: limpeza por peça (filtros, gate, EQ, compressor, largura, ganho)"
```

---

### Task 8: `stages/kick_sample.py` — golpes do bumbo + sample afinado

**Files:**
- Create: `drummix/stages/kick_sample.py`
- Test: `tests/test_kick_sample.py`

Port do detector validado na sessão manual (544 golpes em The Hornet): fluxo de subida em dB do envelope < 150 Hz, pico mínimo de 12 dB, 100 ms entre golpes, piso = percentil 99,5 − 20 dB.

- [ ] **Step 1: Escrever os testes**

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_kick_sample.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementar `drummix/stages/kick_sample.py`**

```python
"""Kick reinforcement: detect hits on the kick stem and lay a tuned synthetic kick under them."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from drummix.audio import SR, gain, rms_db
from drummix.dsp.midi import Hit

_HOP = 64


def _low(kick: np.ndarray) -> np.ndarray:
    return sosfiltfilt(butter(4, 150, "low", fs=SR, output="sos"), kick.mean(axis=1))


def detect_kicks(kick: np.ndarray, min_rise_db: float = 12.0, min_gap_ms: float = 100.0,
                 floor_below_peak_db: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (start sample of each hit, low-band peak amplitude of each hit)."""
    low = _low(kick)
    win = int(0.003 * SR)
    env = np.sqrt(np.convolve(low ** 2, np.ones(win) / win, "same"))
    e = 20 * np.log10(env[::_HOP] + 1e-9)
    lag = max(1, int(0.008 * SR / _HOP))
    flux = np.maximum(e - np.roll(e, lag), 0)
    flux[:lag] = 0
    floor = np.percentile(e, 99.5) - floor_below_peak_db
    pk, _ = find_peaks(flux, height=min_rise_db, distance=max(1, int(min_gap_ms / 1000 * SR / _HOP)))
    pk = pk[e[np.minimum(pk + lag, len(e) - 1)] > floor]
    starts = np.maximum(pk - lag, 0) * _HOP
    peaks = np.array([np.max(np.abs(low[s:s + int(0.04 * SR)])) for s in starts]) if len(starts) else np.zeros(0)
    return starts, peaks


def synth_kick(tonic_hz: float, sweep_from_hz: float = 160.0, decay_s: float = 0.11,
               click: float = 0.25, seed: int = 7) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = tonic_hz + (sweep_from_hz - tonic_hz) * np.exp(-t / 0.018)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay_s)
    rng = np.random.default_rng(seed)
    noise = sosfiltfilt(butter(2, [2500, 7000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    k = body + click * noise * np.exp(-t / 0.0025) / np.max(np.abs(noise))
    fade = int(0.0005 * SR)
    k[:fade] *= np.linspace(0, 1, fade)
    return k / np.max(np.abs(k))


def kick_sample(kick: np.ndarray, tonic_hz: float, reference_db: float, level_db: float = -9.0,
                align_window_ms: tuple[float, float] = (-3.0, 12.0), **synth) -> tuple[np.ndarray, list[Hit]]:
    """Sample track (same length as `kick`) at `level_db` relative to `reference_db` RMS, plus hits."""
    starts, peaks = detect_kicks(kick)
    low = _low(kick)
    sample = synth_kick(tonic_hz, **synth)
    n, L = len(sample), int(0.03 * SR)
    out = np.zeros(len(kick) + n)
    hits = []
    pref = np.percentile(peaks, 95) if len(peaks) else 1.0
    # the flux detector fires ~5-8 ms before the true attack, so the search window is asymmetric
    lo, hi = (int(ms / 1000 * SR) for ms in align_window_ms)
    for s, p in zip(starts, peaks):
        best_c, best_a = -np.inf, int(s)
        for a in range(int(s) + lo, int(s) + hi + 1, 4):
            ref = low[max(a, 0):a + L]
            if a < 0 or len(ref) < L:
                continue
            c = float(np.dot(ref, sample[:L]))
            if c > best_c:
                best_c, best_a = c, a
        g = min(1.0, p / pref)
        out[best_a:best_a + n] += sample * g
        hits.append(Hit(best_a / SR, int(np.clip(40 + 87 * g, 1, 127))))
    mono = out[:len(kick)]
    track = np.stack([mono, mono], axis=1)
    if not hits:
        return track, hits
    return track * gain(reference_db + level_db - rms_db(track)), hits
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_kick_sample.py -q`
Expected: `5 passed`. Se `test_detects_every_kick_slightly_early` falhar por contagem, imprimir `len(found)`, `len(truth)` e ajustar só o gerador sintético (amplitude do bumbo), nunca os defaults do detector — eles são os validados em The Hornet.

- [ ] **Step 5: Commit**

```bash
git add drummix/stages/kick_sample.py tests/test_kick_sample.py
git commit -m "feat: reforço do bumbo com sample sintético afinado e alinhado em fase"
```

---

### Task 9: `stages/tone_match.py`

**Files:**
- Create: `drummix/stages/tone_match.py`
- Test: `tests/test_tone_match.py`

- [ ] **Step 1: Escrever os testes**

```python
import numpy as np
import pytest

from drummix.audio import SR
from drummix.dsp.filters import eq, octave_levels
from drummix.stages.tone_match import tone_match


def _pink(seed, seconds=4.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n)
    x /= np.max(np.abs(x))
    return np.stack([x, x], 1) * 0.3


def test_moves_spectrum_by_requested_amount():
    ref = _pink(0)
    mine = eq(_pink(1), [[1000, 6.0, 1.41]])  # 6 dB a mais em 1 kHz
    out, rep = tone_match({"a": mine * 0.5, "b": mine * 0.5}, ref, amount=0.5)
    i = 5  # banda de 1 kHz
    before = np.array(rep["before"])[i] - np.array(rep["reference"])[i]
    after = np.array(rep["after"])[i] - np.array(rep["reference"])[i]
    assert after == pytest.approx(before * 0.5, abs=0.8)


def test_same_eq_applied_to_every_stem():
    ref = _pink(0)
    a, b = _pink(1), _pink(2)
    out, rep = tone_match({"a": a, "b": b}, ref, amount=0.65)
    assert np.allclose(octave_levels(out["a"]) - octave_levels(a), octave_levels(out["b"]) - octave_levels(b), atol=0.6)


def test_caps_sub_boost_and_top_cut():
    ref = eq(_pink(0), [[31.5, 12.0, 1.41], [16000, -12.0, 1.41]])
    out, rep = tone_match({"a": _pink(1)}, ref, amount=1.0, max_sub_boost_db=3.0, max_top_cut_db=3.0)
    assert rep["eq_db"][0] <= 3.0 + 1e-9
    assert rep["eq_db"][-1] >= -3.0 - 1e-9
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_tone_match.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementar `drummix/stages/tone_match.py`**

```python
"""Move the drum bus's octave-band balance part of the way toward the reference drums.

The same EQ goes on every stem, so the stems still sum to the matched bus.
"""
from __future__ import annotations

import numpy as np

from drummix.dsp.filters import OCTAVES, eq, octave_levels


def tone_match(stems: dict[str, np.ndarray], reference: np.ndarray, amount: float,
               max_step_db: float = 6.0, max_sub_boost_db: float = 3.0, max_top_cut_db: float = 3.0,
               iterations: int = 3, q: float = 1.41):
    ref = octave_levels(reference)
    before = octave_levels(sum(stems.values()))
    target = before + amount * (ref - before)
    total = np.zeros(len(OCTAVES))
    out = dict(stems)
    for _ in range(iterations):
        step = np.clip(target - octave_levels(sum(out.values())), -max_step_db, max_step_db)
        step[0] = min(step[0], max_sub_boost_db - total[0])  # don't chase sub rumble
        step[-1] = max(step[-1], -max_top_cut_db - total[-1])  # keep cymbal air
        total += step
        bands = [[f, g, q] for f, g in zip(OCTAVES, step)]
        out = {k: eq(v, bands) for k, v in out.items()}
    after = octave_levels(sum(out.values()))
    report = {"bands_hz": list(OCTAVES), "reference": ref.round(2).tolist(), "before": before.round(2).tolist(),
              "after": after.round(2).tolist(), "eq_db": total.round(2).tolist(), "amount": amount}
    return out, report
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_tone_match.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/stages/tone_match.py tests/test_tone_match.py
git commit -m "feat: match de timbre por oitava contra a bateria de referência"
```

---

### Task 10: `stages/space.py`, `stages/band.py`, `stages/balance.py`

**Files:**
- Create: `drummix/stages/space.py`, `drummix/stages/band.py`, `drummix/stages/balance.py`
- Test: `tests/test_mix_stages.py`

- [ ] **Step 1: Escrever os testes**

```python
import numpy as np
import pytest

from drummix import audio
from drummix.audio import SR, rms_db
from drummix.stages.balance import balance
from drummix.stages.band import prepare_band
from drummix.stages.space import room


def _clicks(seconds=3.0):
    x = np.zeros((int(seconds * SR), 2))
    x[::SR // 2] = 0.8
    return x


def test_room_level_relative_to_send_and_deterministic():
    stems = {"snare": _clicks(), "kick": _clicks()}
    wet = room(stems, sends={"snare": 1.0, "kick": 0.3}, level_db=-15.0)
    send = stems["snare"] + 0.3 * stems["kick"]
    assert wet.shape == send.shape
    assert rms_db(wet) - rms_db(send) == pytest.approx(-15.0, abs=0.05)
    assert np.array_equal(wet, room(stems, sends={"snare": 1.0, "kick": 0.3}, level_db=-15.0))


def test_prepare_band_shifts_later_and_reports_overlap():
    band = np.ones((SR * 2, 2))
    out, overlap = prepare_band(band, delay_s=0.5, length=SR * 3, eq_bands=[])
    assert out.shape == (SR * 3, 2)
    assert out[: SR // 2].sum() == 0
    assert overlap == slice(SR // 2, SR // 2 + SR * 2)


def test_prepare_band_negative_delay_trims_start():
    band = np.arange(SR * 2, dtype=float)[:, None].repeat(2, 1)
    out, overlap = prepare_band(band, delay_s=-0.5, length=SR, eq_bands=[])
    assert out[0, 0] == SR // 2
    assert overlap == slice(0, SR)


def test_balance_puts_drums_below_band():
    rng = np.random.default_rng(0)
    band = rng.standard_normal((SR * 5, 2)) * 0.1
    drums = {"kick": rng.standard_normal((SR * 5, 2)) * 0.01, "snare": rng.standard_normal((SR * 5, 2)) * 0.02}
    out, g = balance(drums, band, slice(0, SR * 5), below_lu=2.0)
    assert audio.lufs(sum(out.values())) == pytest.approx(audio.lufs(band) - 2.0, abs=0.05)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_mix_stages.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementar `drummix/stages/space.py`**

```python
"""Short synthetic room: decaying decorrelated noise IR fed by a weighted send of the stems."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

from drummix.audio import SR, gain, rms_db


def room_ir(rt60_s: float = 0.45, predelay_ms: float = 10.0, band_hz=(300.0, 6000.0), seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(1.33 * rt60_s * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60_s)[:, None]
    ir = sosfiltfilt(butter(2, list(band_hz), "band", fs=SR, output="sos"), ir, axis=0)
    return np.vstack([np.zeros((int(predelay_ms / 1000 * SR), 2)), ir])


def room(stems: dict[str, np.ndarray], sends: dict[str, float], level_db: float = -15.0, **ir_params) -> np.ndarray:
    send = sum(w * stems[k] for k, w in sends.items() if k in stems)
    ir = room_ir(**ir_params)
    n = len(send)
    wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
    return wet * gain(rms_db(send) + level_db - rms_db(wet))
```

- [ ] **Step 4: Implementar `drummix/stages/band.py`**

```python
"""Backing track: static carve EQ and placement on the drum/video timeline."""
from __future__ import annotations

import numpy as np

from drummix.audio import SR
from drummix.dsp.filters import eq


def prepare_band(band: np.ndarray, delay_s: float, length: int, eq_bands) -> tuple[np.ndarray, slice]:
    y = eq(band, eq_bands) if eq_bands else band
    shift = int(round(delay_s * SR))
    out = np.zeros((length, 2))
    if shift >= 0:
        seg = y[:max(0, length - shift)]
        out[shift:shift + len(seg)] = seg
        return out, slice(shift, shift + len(seg))
    seg = y[-shift:-shift + length]
    out[:len(seg)] = seg
    return out, slice(0, len(seg))
```

- [ ] **Step 5: Implementar `drummix/stages/balance.py`**

```python
"""Set the drums a fixed number of LU under the band, measured where both play."""
from __future__ import annotations

import numpy as np

from drummix.audio import gain, lufs


def balance(drums: dict[str, np.ndarray], band: np.ndarray, overlap: slice, below_lu: float):
    bus = sum(drums.values())
    g = (lufs(band[overlap]) - below_lu) - lufs(bus[overlap])
    return {k: v * gain(g) for k, v in drums.items()}, float(g)
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest tests/test_mix_stages.py -q`
Expected: `4 passed`

- [ ] **Step 7: Commit**

```bash
git add drummix/stages/space.py drummix/stages/band.py drummix/stages/balance.py tests/test_mix_stages.py
git commit -m "feat: sala sintética, posicionamento da banda e equilíbrio bateria×banda"
```

---

### Task 11: `stages/master.py`

**Files:**
- Create: `drummix/stages/master.py`
- Test: `tests/test_master.py`

- [ ] **Step 1: Escrever os testes**

```python
import numpy as np
import pytest

from drummix import audio
from drummix.audio import SR
from drummix.stages.master import MasterError, master


def _music(seed=0, seconds=10.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((int(seconds * SR), 2)) * 0.02
    x[::SR // 4] += 0.5  # transientes
    return x


def test_hits_loudness_and_true_peak():
    y, info = master(_music(), target_lufs=-14.0, ceiling_dbtp=-1.0, tolerance_lu=0.3,
                     glue={"threshold_db": -24, "ratio": 2.5, "attack_ms": 30, "release_ms": 150})
    assert audio.lufs(y) == pytest.approx(-14.0, abs=0.3)
    assert audio.true_peak_db(y) <= -1.0 + 0.05
    assert info["true_peak_db"] <= -0.95


def test_keeps_time_alignment():
    x = np.zeros((SR * 4, 2))
    x[SR] = 0.5
    x += np.random.default_rng(1).standard_normal(x.shape) * 1e-3
    y, _ = master(x, target_lufs=-30.0, ceiling_dbtp=-1.0, tolerance_lu=0.5, glue=None)
    assert int(np.argmax(np.abs(y[:, 0]))) == SR


def test_raises_if_target_unreachable():
    with pytest.raises(MasterError):
        master(np.zeros((SR * 2, 2)), target_lufs=-14.0, ceiling_dbtp=-1.0, tolerance_lu=0.3, glue=None)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_master.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implementar `drummix/stages/master.py`**

```python
"""Glue compression, then gain into a true-peak brickwall limiter until the loudness target is met."""
from __future__ import annotations

import numpy as np
import pedalboard as pb

from drummix.audio import gain, lufs, true_peak_db
from drummix.dsp import fx


class MasterError(RuntimeError):
    pass


def master(premaster: np.ndarray, target_lufs: float = -14.0, ceiling_dbtp: float = -1.0,
           tolerance_lu: float = 0.3, glue: dict | None = None, max_iter: int = 8):
    x = fx.process(premaster, [pb.Compressor(**glue)]) if glue else premaster
    start = lufs(x)
    if not np.isfinite(start):
        raise MasterError("premaster sem conteúdo mensurável (silêncio)")
    g = target_lufs - start
    for _ in range(max_iter):
        y = fx.process(x * gain(g), [pb.BrickwallLimiter(ceiling_db=ceiling_dbtp, true_peak=True)])
        level = lufs(y)
        if abs(level - target_lufs) <= tolerance_lu:
            break
        g += target_lufs - level
    else:
        raise MasterError(f"não atingiu {target_lufs} LUFS em {max_iter} iterações (último {level:.2f})")
    tp = true_peak_db(y)
    if tp > ceiling_dbtp + 0.05:
        raise MasterError(f"true peak {tp:.2f} dBTP acima do teto {ceiling_dbtp}")
    return y, {"gain_db": round(g, 2), "lufs": round(level, 2), "true_peak_db": round(tp, 2)}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_master.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/stages/master.py tests/test_master.py
git commit -m "feat: master com limiter true-peak até o alvo de LUFS"
```

---

### Task 12: `ingest.py` — `create_song`

**Files:**
- Create: `drummix/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Escrever os testes**

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_ingest.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implementar `drummix/ingest.py`**

```python
"""Create a song folder: copy inputs (never touched again), hash them, write song.yaml."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from drummix import audio
from drummix.song import find_stems


def _bring(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        shutil.copy2(src, dst)
    else:
        audio.save(dst, audio.load(src))


def create_song(songs_dir, slug: str, stems_dir, band, original, style: str, take=None) -> Path:
    root = Path(songs_dir) / slug
    if (root / "song.yaml").exists():
        raise FileExistsError(f"música já existe: {root}")
    stems, key, bpm = find_stems(stems_dir)
    inp = root / "input"
    manifest = {}
    for piece, src in stems.items():
        dst = inp / "stems" / f"{piece}.wav"
        _bring(src, dst)
        manifest[f"stems/{piece}.wav"] = {"source": str(src), "sha1": audio.sha1(dst)}
    for name, src in (("band.wav", Path(band)), ("original.wav", Path(original))):
        _bring(src, inp / name)
        manifest[name] = {"source": str(src), "sha1": audio.sha1(inp / name)}
    if take:
        take = Path(take)
        dst = inp / f"take{take.suffix.lower()}"
        shutil.copy2(take, dst)
        manifest[dst.name] = {"source": str(take), "sha1": audio.sha1(dst)}
    (inp / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (root / "song.yaml").write_text(yaml.safe_dump(
        {"style": style, "key": key, "bpm": bpm, "params": {}}, allow_unicode=True, sort_keys=False))
    return root
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_ingest.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/ingest.py tests/test_ingest.py
git commit -m "feat: drummix new — cópia das entradas com hash e song.yaml"
```

---

### Task 13: `deliver.py` — versão, verificação e relatório

**Files:**
- Create: `drummix/deliver.py`
- Test: `tests/test_deliver.py`

- [ ] **Step 1: Escrever os testes**

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_deliver.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implementar `drummix/deliver.py`**

```python
"""Write output/vN atomically (vN.tmp then rename) and verify what was written."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from drummix import audio
from drummix.dsp.midi import Hit, write_drum_midi


class DeliveryError(RuntimeError):
    pass


def _versions(out_dir: Path) -> list[int]:
    return sorted(int(p.name[1:]) for p in out_dir.glob("v*") if p.is_dir() and p.name[1:].isdigit())


def _previous_report(out_dir: Path) -> dict | None:
    vs = _versions(out_dir)
    if not vs:
        return None
    path = out_dir / f"v{vs[-1]}" / "report.json"
    return json.loads(path.read_text()) if path.exists() else None


def _report_md(rep: dict) -> str:
    lines = [f"# drummix — v{rep['version']}", ""]
    for key in ("final_lufs", "final_true_peak_db", "drums_vs_band_lu", "delay_s", "kick_hits"):
        if key in rep:
            lines.append(f"- **{key}**: {rep[key]}")
    if rep.get("warnings"):
        lines += ["", "## Avisos", *[f"- {w}" for w in rep["warnings"]]]
    return "\n".join(lines) + "\n"


def deliver(out_dir, stems: dict[str, np.ndarray], final: np.ndarray, hits: dict[int, list[Hit]], bpm: float,
            report: dict, targets: dict, config_text: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = (_versions(out_dir) or [0])[-1] + 1
    tmp = out_dir / f"v{version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        premaster = sum(stems.values())
        for name, x in stems.items():
            audio.save(tmp / "stems" / f"{name}.wav", x)
        reloaded = sum(audio.load(tmp / "stems" / f"{name}.wav") for name in stems)
        sum_err = audio.rms_db(reloaded - premaster) - audio.rms_db(premaster)
        if sum_err > -90:
            raise DeliveryError(f"soma dos stems difere do premaster ({sum_err:.1f} dB)")
        audio.save(tmp / "final.wav", final)
        written = audio.load(tmp / "final.wav")
        level, tp = audio.lufs(written), audio.true_peak_db(written)
        if targets.get("target_lufs") is not None and abs(level - targets["target_lufs"]) > 0.5:
            raise DeliveryError(f"final em {level:.2f} LUFS, alvo {targets['target_lufs']}")
        if tp > targets["ceiling_dbtp"] + 0.1:
            raise DeliveryError(f"true peak {tp:.2f} dBTP acima de {targets['ceiling_dbtp']}")
        if hits:
            write_drum_midi(tmp / "drums.mid", hits, bpm)
        hashes = {p.relative_to(tmp).as_posix(): audio.sha1(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        config_sha = hashlib.sha1(config_text.encode()).hexdigest()
        prev = _previous_report(out_dir)
        if prev and prev.get("config_sha1") != config_sha and prev["hashes"].get("final.wav") == hashes["final.wav"]:
            raise DeliveryError("configuração mudou mas o final saiu idêntico à versão anterior")
        full = {**report, "version": version, "final_lufs": round(level, 2), "final_true_peak_db": round(tp, 2),
                "stems_sum_error_db": round(sum_err, 1), "config_sha1": config_sha, "hashes": hashes}
        (tmp / "report.json").write_text(json.dumps(full, indent=1, ensure_ascii=False))
        (tmp / "report.md").write_text(_report_md(full))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    final_dir = out_dir / f"v{version}"
    tmp.rename(final_dir)
    return final_dir
```

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest tests/test_deliver.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add drummix/deliver.py tests/test_deliver.py
git commit -m "feat: entrega versionada atômica com verificação de soma, LUFS, TP e hash"
```

---

### Task 14: Perfil `fusion`, `pipeline.py`, `cli.py` e e2e sintético

**Files:**
- Create: `profiles/fusion.yaml`, `drummix/pipeline.py`, `drummix/cli.py`
- Test: `tests/test_e2e.py`

- [ ] **Step 1: Criar `profiles/fusion.yaml`** (valores da sessão manual aprovada)

```yaml
# Jazz fusion — validado de ouvido em The Hornet (Yellowjackets), 2026-10-31.
# Bumbo contido: +2 dB soou "muito pesado". Caixa +2 dB aprovada.
ingest:
  silent_below_lufs: -60
align:
  search_s: [-5.0, 30.0]
  window_s: 15
  step_s: 20
clean:
  kick:
    mono: true
    highpass_hz: 30
    eq: [[400, -4.0, 1.2], [900, -2.0, 1.5]]
    compressor: {threshold_db: -24, ratio: 4, attack_ms: 15, release_ms: 120}
  snare:
    highpass_hz: 100
    eq: [[500, -2.5, 1.2], [1200, -2.0, 1.5]]
    compressor: {threshold_db: -26, ratio: 3, attack_ms: 10, release_ms: 100}
    gain_db: 2.0
  toms:
    gate: {threshold_db: -36.5, ratio: 6, attack_ms: 1, release_ms: 200}
    highpass_hz: 60
    eq: [[500, -3.0, 1.2]]
    compressor: {threshold_db: -22, ratio: 3, attack_ms: 15, release_ms: 150}
    gain_db: -1.0
  hihat:
    highpass_hz: 250
    eq: [[1000, -2.0, 1.2]]
    width: 0.55
    gain_db: -1.0
  cymbals:
    highpass_hz: 200
    eq: [[800, -2.0, 1.0]]
  other:
    drop: true
  other_kit:
    highpass_hz: 100
    gain_db: -6.0
kick_sample:
  level_db: -9.0
  sweep_from_hz: 160
  decay_s: 0.11
  click: 0.25
tone_match:
  amount: 0.65
  max_sub_boost_db: 3.0
  max_top_cut_db: 3.0
space:
  sends: {snare: 1.0, toms: 1.0, cymbals: 0.5, kick: 0.3}
  level_db: -15.0
  rt60_s: 0.45
  predelay_ms: 10
band:
  eq: [[55, -1.5, 4.0], [4000, -1.5, 2.0]]
balance:
  drums_below_band_lu: 2.0
mix:
  headroom_peak_db: -3.0
master:
  target_lufs: -14.0
  ceiling_dbtp: -1.0
  tolerance_lu: 0.3
  glue: {threshold_db: -24, ratio: 2.5, attack_ms: 30, release_ms: 150}
```

- [ ] **Step 2: Escrever o teste e2e**

```python
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
    assert rep["final_true_peak_db"] <= -0.95
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
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: FAIL (`ImportError: cannot import name 'cli'`)

- [ ] **Step 4: Implementar `drummix/pipeline.py`**

```python
"""render(song): run every stage in order and hand the result to deliver."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from drummix import audio
from drummix.deliver import deliver
from drummix.dsp.midi import KICK
from drummix.song import PIECES, Song, tonic_hz
from drummix.stages.align import estimate_delay
from drummix.stages.balance import balance
from drummix.stages.band import prepare_band
from drummix.stages.clean import clean
from drummix.stages.kick_sample import kick_sample
from drummix.stages.master import master
from drummix.stages.space import room
from drummix.stages.tone_match import tone_match


def render(song: Song) -> Path:
    p = song.params
    warnings: list[str] = []
    raw = {k: audio.load(song.stem_path(k)) for k in PIECES if song.stem_path(k).exists()}
    n = min(len(x) for x in raw.values())
    raw = {k: v[:n] for k, v in raw.items()}
    for k in list(raw):
        if k not in ("kick", "snare") and audio.lufs(raw[k]) < p["ingest"]["silent_below_lufs"]:
            warnings.append(f"stem '{k}' praticamente vazio — descartado")
            del raw[k]

    band = audio.load(song.band_path)
    original = audio.load(song.original_path)
    m = min(len(band), len(original))
    ref_drums = original[:m] - band[:m]

    alignment = estimate_delay(sum(raw.values()), ref_drums, **p["align"])
    if alignment.spread_s > 0.03 or alignment.mean_corr < 0.1:
        warnings.append(f"alinhamento fraco/instável (desvio {alignment.spread_s * 1000:.0f} ms, "
                        f"correlação {alignment.mean_corr:.2f}) — confira no ouvido")

    stems = clean(raw, p["clean"])
    sample, kick_hits = kick_sample(raw["kick"], tonic_hz(song.key), reference_db=audio.rms_db(stems["kick"]),
                                    **p["kick_sample"])
    stems["kick_sample"] = sample
    if any(b.time_s - a.time_s < 0.06 for a, b in zip(kick_hits, kick_hits[1:])):
        warnings.append("golpes de bumbo a menos de 60 ms — possível disparo duplo")

    stems, tone = tone_match(stems, ref_drums, **p["tone_match"])
    stems["room"] = room(stems, **p["space"])

    band_al, overlap = prepare_band(band, alignment.delay_s, n, p["band"]["eq"])
    stems, drum_gain = balance(stems, band_al, overlap, p["balance"]["drums_below_band_lu"])
    drums_vs_band = audio.lufs(sum(stems.values())[overlap]) - audio.lufs(band_al[overlap])
    stems["band"] = band_al

    premaster = sum(stems.values())
    head = min(0.0, p["mix"]["headroom_peak_db"] - 20 * np.log10(np.max(np.abs(premaster))))
    stems = {k: v * audio.gain(head) for k, v in stems.items()}
    final, master_info = master(sum(stems.values()), **p["master"])

    report = {"song": song.root.name, "style": song.style, "key": song.key, "bpm": song.bpm,
              "delay_s": round(alignment.delay_s, 4), "align_spread_s": round(alignment.spread_s, 4),
              "align_corr": round(alignment.mean_corr, 3), "kick_hits": len(kick_hits),
              "drums_vs_band_lu": round(drums_vs_band, 2), "drum_gain_db": round(drum_gain, 2),
              "headroom_db": round(head, 2), "tone_match": tone, "master": master_info, "warnings": warnings}
    targets = {"target_lufs": p["master"]["target_lufs"], "ceiling_dbtp": p["master"]["ceiling_dbtp"]}
    return deliver(song.output_dir, stems, final, {KICK: kick_hits}, song.bpm, report, targets,
                   config_text=song.config_path.read_text())
```

- [ ] **Step 5: Implementar `drummix/cli.py`**

```python
"""drummix new | render"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drummix.ingest import create_song
from drummix.pipeline import render
from drummix.song import load_song


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drummix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new", help="cria songs/<slug> a partir dos arquivos do Moises")
    new.add_argument("slug")
    new.add_argument("--stems", required=True, type=Path, help="pasta com os stems de bateria do Moises")
    new.add_argument("--band", required=True, type=Path, help="faixa sem bateria")
    new.add_argument("--original", required=True, type=Path, help="música original (referência)")
    new.add_argument("--take", type=Path, help="take bruto / vídeo do Rec'n'Share (opcional)")
    new.add_argument("--style", default="fusion")
    new.add_argument("--songs-dir", type=Path, default=Path("songs"))
    ren = sub.add_parser("render", help="gera output/vN")
    ren.add_argument("slug")
    ren.add_argument("--songs-dir", type=Path, default=Path("songs"))
    args = ap.parse_args(argv)

    if args.cmd == "new":
        root = create_song(args.songs_dir, args.slug, args.stems, args.band, args.original, args.style, args.take)
        print(f"criado {root}")
        return 0
    out = render(load_song(args.songs_dir / args.slug))
    rep = json.loads((out / "report.json").read_text())
    print(f"{out.name}: {rep['final_lufs']} LUFS, TP {rep['final_true_peak_db']} dBTP, "
          f"bateria {rep['drums_vs_band_lu']} LU vs banda, atraso {rep['delay_s']} s, "
          f"{rep['kick_hits']} golpes de bumbo")
    for w in rep["warnings"]:
        print(f"  aviso: {w}")
    print(f"  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Rodar e ver passar**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: `1 passed`

- [ ] **Step 7: Rodar a suíte inteira**

Run: `uv run pytest -q`
Expected: todos passam (≈ 45 testes).

- [ ] **Step 8: Commit**

```bash
git add profiles/fusion.yaml drummix/pipeline.py drummix/cli.py tests/test_e2e.py
git commit -m "feat: perfil fusion, pipeline completo e CLI new/render"
```

---

### Task 15: Regressão com The Hornet

**Files:**
- Create: `tests/test_regression_hornet.py`

- [ ] **Step 1: Escrever o teste (marcado `regression`, fora da suíte padrão)**

```python
"""Real-audio regression against the manual mix approved by ear (2026-10-31)."""
import json
import shutil
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
    assert rep["final_true_peak_db"] <= -0.95
    assert rep["drums_vs_band_lu"] == pytest.approx(-2.0, abs=0.5)
```

- [ ] **Step 2: Rodar**

Run: `uv run pytest -m regression -q -s`
Expected: `1 passed` (alguns minutos). Se falhar, registrar o valor obtido no relatório antes de mexer em qualquer parâmetro.

- [ ] **Step 3: Criar a música de verdade no projeto para ouvir**

Run:
```bash
uv run drummix new the-hornet --stems "$HOME/Downloads/The Hornet - Audio-A minor-131bpm-441hz" \
  --band "$HOME/Downloads/The Hornet - Sem bateria.wav" --original "$HOME/Downloads/The Hornet - Original.wav" \
  --take "$HOME/Downloads/The Hornet - Audio.m4a"
uv run drummix render the-hornet
```
Expected: linha `v1: -14.0 LUFS, TP ≤ -1 dBTP, bateria ≈ -2 LU vs banda, atraso ≈ 11.73 s, ≈544 golpes`. Ouvir `songs/the-hornet/output/v1/final.wav` contra `~/Downloads/The Hornet - Mix Cover/The Hornet - Cover Final.wav` (baseline aprovado) no mesmo volume — a página A/B vem no Plano 3.

- [ ] **Step 4: Commit**

```bash
git add tests/test_regression_hornet.py songs/the-hornet/song.yaml
git commit -m "test: regressão com The Hornet contra a mixagem manual aprovada"
```

---

## Self-review (feito)

- **Cobertura do spec neste plano:** seções 2 (entradas/saídas, exceto vídeo e compare), 3 (perfil fusion), 4 (arquitetura), 5 stages 0–2, 4 (bumbo), 5, 7 (sala sintética), 8 (cortes estáticos), 9, 10 (sem EQ-match da master), 11 (sem vídeo), 8 (robustez), 9 (testes + regressão). **Fora deste plano, por desenho:** transientes, sample de caixa, NY/saturação/de-harsh, IRs Voxengo + plate, EQ dinâmico com chave, EQ-match da master, perfil rock → Plano 2; compare.html, simulação de celular, vídeo, skill → Plano 3.
- **Placeholders:** nenhum.
- **Consistência de tipos:** `Hit(time_s, velocity)` em `dsp/midi.py` usado por `kick_sample` e `deliver`; `estimate_delay(**p["align"])` aceita `search_s/window_s/step_s` do perfil; `room(stems, **p["space"])` aceita `sends/level_db/rt60_s/predelay_ms`; `kick_sample(..., **p["kick_sample"])` aceita `level_db/sweep_from_hz/decay_s/click`; `master(**p["master"])` aceita `target_lufs/ceiling_dbtp/tolerance_lu/glue`; `deliver(out_dir, stems, final, hits, bpm, report, targets, config_text)` igual em pipeline e testes.
