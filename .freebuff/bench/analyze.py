"""Analyse d'un motif : blocs, coupes, sauts, points, tissage."""
import sys
from math import hypot

import pyembroidery as pe

MASK = pe.COMMAND_MASK


def parse(pattern):
    blocks, cur, pend = [], [], []
    head = True
    for item in pattern.stitches:
        kind = item[2] & MASK
        if kind == pe.STITCH:
            if head:
                head = False
            cur.append(item)
        else:
            if cur:
                blocks.append(cur)
                cur = []
            if kind == pe.COLOR_CHANGE:
                pass
    if cur:
        blocks.append(cur)
    return blocks


def stats(path):
    p = pe.read(path)
    stitches = trims = jumps = cc = 0
    for x, y, cmd in p.stitches:
        kind = cmd & MASK
        if kind == pe.STITCH:
            stitches += 1
        elif kind == pe.TRIM:
            trims += 1
        elif kind == pe.JUMP:
            jumps += 1
        elif kind == pe.COLOR_CHANGE:
            cc += 1
    blocks = parse(p)
    xs = [s[0] for s in p.stitches]
    ys = [s[1] for s in p.stitches]
    size = ((max(xs) - min(xs)) / 10, (max(ys) - min(ys)) / 10) if xs else (0, 0)
    print(f"{path}")
    print(f"  points={stitches} coupes={trims} sauts={jumps} changements_fil={cc} blocs={len(blocks)}"
          f" fils={len(p.threadlist)} taille={size[0]:.1f}x{size[1]:.1f}mm")
    return dict(stitches=stitches, trims=trims, jumps=jumps, cc=cc, blocks=len(blocks),
                threads=len(p.threadlist), size=size)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        stats(arg)
