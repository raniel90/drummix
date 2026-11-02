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
