"""Passe l'optimisation radicale sur tout un corpus et résume les gains."""
import glob
import hashlib
import os
import sys
import time
from collections import Counter

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from evaluate import load, metrics, appearance, grid_of, compare  # noqa: E402
from img2stitch.optimize import optimize_pattern  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

PATHS = (glob.glob("examples/*.pes")
         + glob.glob("instance/designs/*/pattern.json")
         + glob.glob(".freebuff/bench/dl/samples/*")
         + glob.glob(".freebuff/bench/dl/embmod/*"))


def key(pattern):
    digest = hashlib.sha1()
    for stitch in pattern.stitches:
        digest.update(repr([round(stitch[0]), round(stitch[1]), stitch[2] & 0xFF]).encode())
    return digest.hexdigest()


def main():
    seen, rows, failures = set(), [], []
    totals = Counter()
    for path in sorted(PATHS):
        try:
            pattern = load(path)
            signature = key(pattern)
        except Exception as exc:  # noqa: BLE001
            failures.append((path, "lecture", repr(exc)))
            continue
        if signature in seen:
            continue
        seen.add(signature)
        if not any(stitch[2] & pe.COMMAND_MASK == pe.STITCH for stitch in pattern.stitches):
            continue
        grid = grid_of([pattern])
        before = metrics(pattern)
        before_img = appearance(pattern, grid)
        start = time.perf_counter()
        try:
            optimized, report = optimize_pattern_radical(pattern)
        except Exception as exc:  # noqa: BLE001
            failures.append((path, "radical", repr(exc)))
            continue
        elapsed = time.perf_counter() - start
        after = metrics(optimized)
        changed = compare(before_img, appearance(optimized, grid))
        rows.append((os.path.basename(path), before, after, report, changed, elapsed))
        totals["files"] += 1
        totals["stitches_before"] += before["stitches"]
        totals["stitches_after"] += after["stitches"]
        totals["trims_before"] += before["trims"]
        totals["trims_after"] += after["trims"]
        totals["colors_before"] += before["colors"]
        totals["colors_after"] += after["colors"]
        totals["warnings"] += len(report.get("warnings", []))
        totals["seconds"] += elapsed

    print("%-42s %7s %7s %7s %7s %6s %6s" % (
        "motif", "points", "gain", "coupes", "gain", "coul", "appar"))
    for name, before, after, report, changed, elapsed in rows[:60]:
        print("%-42s %7d %6.1f%% %7d %6.1f%% %3d/%3d %5.2f%% %.1fs" % (
            name[:42], after["stitches"],
            100 * (after["stitches"] - before["stitches"]) / max(1, before["stitches"]),
            after["trims"],
            100 * (after["trims"] - before["trims"]) / max(1, before["trims"]),
            before["colors"], after["colors"], 100 * changed, elapsed))
    print("\nfichiers distincts : %d" % totals["files"])
    print("points : %d -> %d (%.1f %%)" % (
        totals["stitches_before"], totals["stitches_after"],
        -100 * (totals["stitches_before"] - totals["stitches_after"])
        / max(1, totals["stitches_before"])))
    print("coupes : %d -> %d (%.1f %%)" % (
        totals["trims_before"], totals["trims_after"],
        -100 * (totals["trims_before"] - totals["trims_after"]) / max(1, totals["trims_before"])))
    print("changements de fil : %d -> %d" % (totals["colors_before"], totals["colors_after"]))
    print("avertissements : %d, temps total : %.1f s" % (totals["warnings"], totals["seconds"]))
    print("apparence : max %.2f %%, motifs >2 %% : %d" % (
        100 * max((row[4] for row in rows), default=0),
        sum(1 for row in rows if row[4] > 0.02)))
    if failures:
        print("\néchecs :")
        for item in failures[:20]:
            print("  ", item)


if __name__ == "__main__":
    main()
