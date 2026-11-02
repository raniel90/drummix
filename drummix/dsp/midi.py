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
    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=float(bpm))
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="drummix")
    for pitch, items in hits.items():
        for h in items:
            kit.notes.append(pretty_midi.Note(velocity=int(h.velocity), pitch=pitch,
                                              start=h.time_s, end=h.time_s + note_s))
    kit.notes.sort(key=lambda n: n.start)
    pm.instruments.append(kit)
    pm.write(str(path))
