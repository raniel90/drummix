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
_NOTE.update({"Cb": 11, "E#": 5, "Fb": 4, "B#": 0})  # enarmônicos aceitos pelo regex


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
    hz = 440.0 * 2 ** ((midi - 69) / 12)
    return hz * 2 if hz < 40.0 else hz  # C–D na oitava 1 (32.7–36.7 Hz) são graves demais para o bumbo


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
