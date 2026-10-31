import json
import os
import sys

sys.path.insert(0, os.getcwd())
from evaluate import load  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

KEYS = ("removed_duplicates", "removed_micro_stitches", "simplified_points",
        "removed_hidden_stitches", "merged_trims", "hidden_travels_merged",
        "split_long_stitches", "reordered", "blocks_before", "blocks_after",
        "color_changes_saved", "appearance_change_pct", "coverage_iou")

for path in sys.argv[1:]:
    pattern = load(path)
    optimized, report = optimize_pattern_radical(pattern)
    print("\n==", os.path.basename(os.path.dirname(path)) or path)
    print("   " + ", ".join("%s=%s" % (key, report.get(key)) for key in KEYS))
    print("   avant", json.dumps(report["before"]))
    print("   après", json.dumps(report["after"]))
    print("   avertissements:", report["warnings"])
