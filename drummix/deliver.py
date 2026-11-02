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
    """config_text is the canonical render config (song.canonical_config), not the raw song.yaml."""
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
        if tp > targets["ceiling_dbtp"] + 0.005:
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
