import glob
import hashlib
import os
import sys

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from corpus import PATHS, key  # noqa: E402
from evaluate import load, metrics, appearance, grid_of, compare  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

rows, seen = [], set()
for path in sorted(PATHS):
    try:
        pattern = load(path)
        signature = key(pattern)
    except Exception:  # noqa: BLE001
        continue
    if signature in seen:
        continue
    seen.add(signature)
    grid = grid_of([pattern])
    before = metrics(pattern)
    before_img = appearance(pattern, grid)
    optimized, report = optimize_pattern_radical(pattern)
    after = metrics(optimized)
    changed = compare(before_img, appearance(optimized, grid))
    rows.append((changed, path, before, after, report))

rows.sort(reverse=True)
for changed, path, before, after, report in rows[:12]:
    print("\n%.2f%% %s" % (100 * changed, path))
    print("   avant", before, "\n   après", after)
    print("   reordered", report["reordered"], "couleurs sauvées", report["color_changes_saved"],
          "hidden", report["removed_hidden_stitches"], "split", report["split_long_stitches"])
    print("   fils avant", [hex(t.color) for t in load(path).threadlist][:12])
    print("   avertissements", report["warnings"])
