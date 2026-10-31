"""Diagnostic détaillé : blocs de couleur, runs, coupes, longueurs, sauts."""
import json
import sys
from collections import Counter
from math import hypot

import pyembroidery as pe

MASK = pe.COMMAND_MASK


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    p = pe.EmbPattern()
    p.stitches = [list(s) for s in data["stitches"]]
    for c in data["threads"]:
        p.add_thread({"color": c})
    if data.get("segments"):
        p.img2stitch_segments = [tuple(s) for s in data["segments"]]
    return p


def blocks_of(p):
    """Blocs de couleur : listes de runs (runs = suites de STITCH)."""
    blocks, runs, cur = [], [], []
    for x, y, cmd in p.stitches:
        kind = cmd & MASK
        if kind == pe.STITCH:
            cur.append((x, y))
        else:
            if cur:
                runs.append(cur)
                cur = []
            if kind == pe.COLOR_CHANGE:
                if runs:
                    blocks.append(runs)
                runs = []
    if cur:
        runs.append(cur)
    if runs:
        blocks.append(runs)
    return blocks


def diagnose(p, label=""):
    blocks = blocks_of(p)
    stitches = sum(1 for s in p.stitches if s[2] & MASK == pe.STITCH)
    trims = sum(1 for s in p.stitches if s[2] & MASK == pe.TRIM)
    jumps = sum(1 for s in p.stitches if s[2] & MASK == pe.JUMP)
    ccs = sum(1 for s in p.stitches if s[2] & MASK == pe.COLOR_CHANGE)
    print(f"\n== {label or ''} points={stitches} coupes={trims} sauts={jumps} "
          f"chgt_fil={ccs} blocs_couleur={len(blocks)} fils={len(p.threadlist)}")
    lengths = Counter()
    for bi, runs in enumerate(blocks):
        lens = [hypot(b[0] - a[0], b[1] - a[1]) / 10 for r in runs for a, b in zip(r, r[1:])]
        color = "#%06x" % (p.threadlist[bi].color & 0xFFFFFF) if bi < len(p.threadlist) else "?"
        # séparations : coupes internes au bloc
        print(f"  bloc {bi} {color}: {len(runs)} runs, {sum(len(r) for r in runs)} pts, "
              f"pt moy {sum(lens)/max(1,len(lens)):.1f}mm, <0.5mm: {sum(1 for l in lens if l < 0.5)}, "
              f"3-4.6mm: {sum(1 for l in lens if 2.9 <= l <= 4.6)}, >6mm: {sum(1 for l in lens if l > 6)}")
        # distances de déplacement entre runs consécutifs
        gaps = [hypot(b[0][0] - a[-1][0], b[0][1] - a[-1][1]) / 10
                for a, b in zip(runs, runs[1:])]
        if gaps:
            print(f"      déplacements entre runs: {len(gaps)} total {sum(gaps):.0f}mm, "
                  f"max {max(gaps):.0f}mm, <=2.5mm: {sum(1 for g in gaps if g <= 2.5)}, "
                  f">20mm: {sum(1 for g in gaps if g > 20)}")
    return None


if __name__ == "__main__":
    for path in sys.argv[1:]:
        if path.endswith(".json"):
            diagnose(load(path), path)
        else:
            diagnose(pe.read(path), path)
