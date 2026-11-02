"""drummix new | render"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drummix.audio import AudioError
from drummix.deliver import DeliveryError
from drummix.ingest import create_song
from drummix.pipeline import render
from drummix.song import PROFILES_DIR, load_song
from drummix.stages.align import AlignError
from drummix.stages.master import MasterError

SONGS_DIR = Path(__file__).resolve().parent.parent / "songs"
# what the user can get wrong (inputs, files, a render that can't meet its targets); anything else is a bug
_EXPECTED = (AudioError, AlignError, MasterError, DeliveryError, ValueError, FileNotFoundError, FileExistsError)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drummix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new", help="cria songs/<slug> a partir dos arquivos do Moises")
    new.add_argument("slug")
    new.add_argument("--stems", required=True, type=Path, help="pasta com os stems de bateria do Moises")
    new.add_argument("--band", required=True, type=Path, help="faixa sem bateria")
    new.add_argument("--original", required=True, type=Path, help="música original (referência)")
    new.add_argument("--take", type=Path, help="take bruto / vídeo do Rec'n'Share (opcional)")
    new.add_argument("--style", default="fusion", choices=sorted(p.stem for p in PROFILES_DIR.glob("*.yaml")))
    new.add_argument("--songs-dir", type=Path, default=SONGS_DIR)
    ren = sub.add_parser("render", help="gera output/vN")
    ren.add_argument("slug")
    ren.add_argument("--songs-dir", type=Path, default=SONGS_DIR)
    args = ap.parse_args(argv)
    try:
        return _run(args)
    except _EXPECTED as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1


def _run(args) -> int:
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
