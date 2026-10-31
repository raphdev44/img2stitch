import os
import sys

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from evaluate import load, metrics  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

MASK = pe.COMMAND_MASK
for path in sys.argv[1:]:
    pattern = load(path)
    print("\n==", path, metrics(pattern))
    print("   fils", [hex(t.color) for t in pattern.threadlist])
    print("   début", pattern.stitches[:6])
    lengths = []
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
    for run in runs:
        lengths += [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / 10
                    for a, b in zip(run, run[1:])]
    print("   runs", len(runs), "longueurs max", round(max(lengths or [0]), 1),
          "moy", round(sum(lengths) / max(1, len(lengths)), 2))
    optimized, report = optimize_pattern_radical(pattern)
    print("   après", metrics(optimized), "split", report["split_long_stitches"],
          "hidden", report["removed_hidden_stitches"])
    print("   avertissements", report["warnings"])
