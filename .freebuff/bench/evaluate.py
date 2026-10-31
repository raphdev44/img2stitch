"""Banc d'essai : compare l'optimisation simple et l'optimisation radicale."""
import glob
import json
import os
import sys
import time
from math import hypot

import cv2
import numpy as np
import pyembroidery as pe
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
from img2stitch.optimize import optimize_pattern  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

MASK = pe.COMMAND_MASK
THREAD_MM = 0.4


def load(path):
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pattern = pe.EmbPattern()
        pattern.stitches = [list(s) for s in data["stitches"]]
        for color in data["threads"]:
            pattern.add_thread({"color": color})
        if data.get("segments"):
            pattern.img2stitch_segments = [tuple(s) for s in data["segments"]]
        return pattern
    return pe.read(path)


def metrics(pattern):
    blocks = []
    runs, current, starts = [], [], []
    color = 0
    counts = dict(stitches=0, trims=0, jumps=0, colors=0)
    for index, (x, y, cmd) in enumerate(pattern.stitches):
        kind = cmd & MASK
        if kind == pe.STITCH:
            counts["stitches"] += 1
            if not current:
                starts.append(index)
                runs.append(current)
            current.append((x, y))
        else:
            current = []
            if kind == pe.TRIM:
                counts["trims"] += 1
            elif kind == pe.JUMP:
                counts["jumps"] += 1
            elif kind == pe.COLOR_CHANGE:
                counts["colors"] += 1
    if current:
        runs.append(current)
    return counts


def appearance(pattern, grid):
    """Image des couleurs visibles (dernière couverture gagne) + couverture."""
    x0, y0, scale, w, h = grid
    image = Image.new("I", (w, h), -1)
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(THREAD_MM * scale)))
    px = lambda p: ((p[0] - x0) * scale, (p[1] - y0) * scale)
    thread, started, last = 0, False, None
    for x, y, cmd in pattern.stitches:
        kind = cmd & MASK
        color = pattern.threadlist[thread].color & 0xFFFFFF if thread < len(pattern.threadlist) \
            else 0x1000000 + thread
        if kind == pe.STITCH:
            if last is not None:
                draw.line([px(last), px((x, y))], fill=color, width=width)
            else:                    # point isolé : une piqûre
                cx, cy = px((x, y))
                draw.ellipse([cx - width / 2, cy - width / 2,
                              cx + width / 2, cy + width / 2], fill=color)
            last, started = (x, y), True
        else:
            last = None
            # un changement de fil sans piqûre avant (en-tête de fichier) ne peint rien :
            # on l'ignore pour comparer les images de la même façon
            if kind == pe.COLOR_CHANGE and started:
                thread += 1
                started = False
    return np.asarray(image)


def grid_of(patterns):
    xs = [s[0] for p in patterns for s in p.stitches]
    ys = [s[1] for p in patterns for s in p.stitches]
    x0, y0 = min(xs) - 20, min(ys) - 20
    span = max(max(xs) - x0, max(ys) - y0) + 40
    scale = 1.0 / max(2.0, span / 700.0)   # ~0,2 mm/px
    w = int((max(xs) - x0 + 40) * scale) + 2
    h = int((max(ys) - y0 + 40) * scale) + 2
    return (x0, y0, scale, w, h)


def compare(before, after):
    """Part de l'image où une couleur apparaît là où elle n'était pas (tolérance de 2 px)."""
    covered = np.count_nonzero(before >= 0)
    if not covered:
        return 0.0
    labels = set(np.unique(before[before >= 0]).tolist()) | \
        set(np.unique(after[after >= 0]).tolist())
    kernel = np.ones((5, 5), np.uint8)
    mismatch = 0
    for label in labels:
        old = (before == label).astype(np.uint8)
        new = (after == label).astype(np.uint8)
        allowed = cv2.dilate(old, kernel) > 0
        mismatch += int(np.count_nonzero((new > 0) & ~allowed))
    return mismatch / covered


def run(name, pattern, verbose=True):
    grid = grid_of([pattern])
    before_img = appearance(pattern, grid)
    before = metrics(pattern)
    out = {}
    for label, function in (("simple", lambda p: optimize_pattern(p)),
                            ("radical", lambda p: optimize_pattern_radical(p))):
        start = time.perf_counter()
        try:
            optimized, report = function(pattern)
        except Exception as exc:  # noqa: BLE001
            out[label] = dict(error="%s: %s" % (type(exc).__name__, exc))
            continue
        elapsed = time.perf_counter() - start
        after = metrics(optimized)
        after_img = appearance(optimized, grid)
        changed = compare(before_img, after_img)
        out[label] = dict(before=before, after=after, seconds=elapsed, changed=changed,
                          report=report)
    if verbose:
        print("\n== %s" % name)
        print("   avant : %d points, %d coupes, %d sauts, %d chgt fil" % (
            before["stitches"], before["trims"], before["jumps"], before["colors"]))
        for label in ("simple", "radical"):
            data = out[label]
            if "error" in data:
                print("   %-8s ERREUR %s" % (label, data["error"]))
                continue
            after = data["after"]
            print("   %-8s %d points (%+.1f%%), %d coupes (%+.1f%%), %d sauts, %d chgt fil, "
                  "apparence %+.2f%%, %.1fs" % (
                      label, after["stitches"],
                      100 * (after["stitches"] - before["stitches"]) / max(1, before["stitches"]),
                      after["trims"],
                      100 * (after["trims"] - before["trims"]) / max(1, before["trims"]),
                      after["jumps"], after["colors"], 100 * data["changed"],
                      data["seconds"]))
    return out


if __name__ == "__main__":
    paths = sys.argv[1:] or ["examples/exemple.pes"]
    for path in paths:
        try:
            run(os.path.basename(path), load(path))
        except Exception as exc:  # noqa: BLE001
            print("ECHEC", path, type(exc).__name__, exc)
