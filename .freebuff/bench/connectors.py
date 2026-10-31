import os
import sys
from collections import Counter
from math import hypot

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from evaluate import load  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

MASK = pe.COMMAND_MASK


def connectors(pattern):
    runs, pending, current = [], [], []
    for x, y, cmd in pattern.stitches:
        kind = cmd & MASK
        if kind == pe.STITCH:
            if not current:
                runs.append((pending, current))
                pending = []
            current.append((x, y))
        else:
            current = []
            pending.append(kind)
    if current:
        runs.append((pending, current))
    out = []
    for (_, previous), (commands, run) in zip(runs, runs[1:]):
        distance = hypot(run[0][0] - previous[-1][0], run[0][1] - previous[-1][1]) / 10
        if pe.COLOR_CHANGE in commands:
            kind = "color"
        elif pe.TRIM in commands:
            kind = "trim"
        elif pe.JUMP in commands:
            kind = "jump"
        else:
            kind = "stitched"
        out.append((kind, distance))
    return out


for path in sys.argv[1:]:
    optimized, report = optimize_pattern_radical(load(path))
    data = connectors(optimized)
    kinds = Counter(kind for kind, _ in data)
    print("\n==", path, dict(kinds))
    for kind in ("trim", "jump", "stitched", "color"):
        values = sorted(d for k, d in data if k == kind)
        if not values:
            continue
        buckets = Counter()
        for value in values:
            if value <= 3:
                buckets["<=3mm"] += 1
            elif value <= 6:
                buckets["3-6mm"] += 1
            elif value <= 12:
                buckets["6-12mm"] += 1
            elif value <= 30:
                buckets["12-30mm"] += 1
            else:
                buckets[">30mm"] += 1
        print("   %-8s n=%d %s" % (kind, len(values), dict(sorted(buckets.items()))))
