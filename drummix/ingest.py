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
