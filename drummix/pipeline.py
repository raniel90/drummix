"""render(song): run every stage in order and hand the result to deliver."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from drummix import audio
from drummix.deliver import deliver
from drummix.dsp.midi import KICK
from drummix.song import PIECES, Song, canonical_config, tonic_hz
from drummix.stages.align import estimate_delay
from drummix.stages.balance import balance, scale_in_place
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
    del original  # peak memory: the full-length original is only needed for the reference

    alignment = estimate_delay(sum(raw.values()), ref_drums, **p["align"])
    if alignment.spread_s > 0.03 or alignment.mean_corr < 0.1:
        warnings.append(f"alinhamento fraco/instável (desvio {alignment.spread_s * 1000:.0f} ms, "
                        f"correlação {alignment.mean_corr:.2f}) — confira no ouvido")

    stems = clean(raw, p["clean"])
    raw_kick = raw["kick"]
    del raw  # only the raw kick is needed from here on
    # detect on the *raw* kick (the cleanup compressor causes false retriggers), but align the sample
    # against the *cleaned* kick: its high-pass shifts phase at the tonic (review of Tasks 2-5)
    sample, kick_hits = kick_sample(stems["kick"], tonic_hz(song.key), reference_db=audio.rms_db(stems["kick"]),
                                    detect_on=raw_kick, **p["kick_sample"])
    del raw_kick
    stems["kick_sample"] = sample
    if any(b.time_s - a.time_s < 0.06 for a, b in zip(kick_hits, kick_hits[1:])):
        warnings.append("golpes de bumbo a menos de 60 ms — possível disparo duplo")

    stems, tone = tone_match(stems, ref_drums, **p["tone_match"])
    del ref_drums
    stems["room"] = room(stems, **p["space"])

    band_al, overlap = prepare_band(band, alignment.delay_s, n, p["band"]["eq"])
    del band
    stems, drum_gain = balance(stems, band_al, overlap, p["balance"]["drums_below_band_lu"])
    drums_vs_band = audio.lufs(sum(stems.values())[overlap]) - audio.lufs(band_al[overlap])
    stems["band"] = band_al

    premaster = sum(stems.values())
    head = min(0.0, p["mix"]["headroom_peak_db"] - 20 * np.log10(np.max(np.abs(premaster))))
    del premaster
    scale_in_place(stems, audio.gain(head))
    final, master_info = master(sum(stems.values()), **p["master"])

    report = {"song": song.root.name, "style": song.style, "key": song.key, "bpm": song.bpm,
              "delay_s": round(alignment.delay_s, 4), "align_spread_s": round(alignment.spread_s, 4),
              "align_corr": round(alignment.mean_corr, 3), "kick_hits": len(kick_hits),
              "drums_vs_band_lu": round(drums_vs_band, 2), "drum_gain_db": round(drum_gain, 2),
              "headroom_db": round(head, 2), "tone_match": tone, "master": master_info, "warnings": warnings}
    targets = {"target_lufs": p["master"]["target_lufs"], "ceiling_dbtp": p["master"]["ceiling_dbtp"]}
    return deliver(song.output_dir, stems, final, {KICK: kick_hits}, song.bpm, report, targets,
                   config_text=canonical_config(song))
