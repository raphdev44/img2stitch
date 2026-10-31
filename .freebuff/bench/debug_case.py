import json
import os
import sys

sys.path.insert(0, os.getcwd())
from tests.test_optimize_radical import _pattern, _rows, RED, GREEN  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

case = sys.argv[1] if len(sys.argv) > 1 else "merge"
if case == "merge":
    pattern = _pattern([_rows(0, 0, 120, 120), _rows(200, 0, 320, 120),
                        _rows(0, 200, 120, 320)], [RED, GREEN, RED])
else:
    pattern = _pattern([_rows(0, 0, 400, 300), _rows(150, 100, 250, 200)], [RED, GREEN])

optimized, report = optimize_pattern_radical(pattern)
report.pop("after", None)
print(json.dumps(report, indent=1, default=str))
print("threads", [hex(t.color) for t in optimized.threadlist])
print("colors", sum(1 for s in optimized.stitches if s[2] & 0xFF == 5))
