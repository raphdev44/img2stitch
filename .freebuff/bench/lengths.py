import os
import sys
from collections import Counter

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from evaluate import load, metrics  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

MASK = pe.COMMAND_MASK


def lengths(pattern):
    runs, current = [], []
    for x, y, cmd in pattern.stitches:
        if cmd & MASK == pe.STITCH:
            current.append((x, y))
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    values = []
    for run in runs:
        values += [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / 10
                   for a, b in zip(run, run[1:])]
    buckets = Counter()
    for value in values:
        if value < 0.5:
            buckets["<0,5"] += 1
        elif value < 1.5:
            buckets["0,5-1,5"] += 1
        elif value < 2.5:
            buckets["1,5-2,5"] += 1
        elif value < 3.5:
            buckets["2,5-3,5"] += 1
        elif value < 4.6:
            buckets["3,5-4,6"] += 1
        else:
            buckets[">4,6"] += 1
    return len(values), sum(values) / max(1, len(values)), buckets


for path in sys.argv[1:]:
    pattern = load(path)
    optimized, report = optimize_pattern_radical(pattern)
    print("\n==", path)
    for label, item in (("avant", pattern), ("radical", optimized)):
        count, average, buckets = lengths(item)
        print("   %-8s %d points, longueur moyenne %.2f mm, %s" % (
            label, count, average, dict(sorted(buckets.items()))))
    print("   simplified_points=%s removed_micro=%s" % (
        report["simplified_points"], report["removed_micro_stitches"]))
