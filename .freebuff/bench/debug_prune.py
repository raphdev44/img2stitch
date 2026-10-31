import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from tests.test_optimize_radical import _pattern, _rows, RED, GREEN  # noqa: E402
from img2stitch import radical  # noqa: E402

pattern = _pattern([_rows(0, 0, 400, 300), _rows(150, 100, 250, 200)], [RED, GREEN])
report = {"split_long_stitches": 0, "warnings": []}
engine = radical._Radical(pattern, dict(radical.DEFAULTS), report)
red, green = engine.blocks
cover = engine.grid.coverage([run.points for run in green.runs], radical.THREAD_MM)
print("couverture verte", int(cover.sum()), "px_mm", engine.grid.px_mm)
dist = cv2.distanceTransform(cover.astype(np.uint8), cv2.DIST_L2, 3)
need = max(1.0, (radical.THREAD_MM / 2 + engine.prm["hidden_margin_mm"]) / engine.grid.px_mm)
print("need", need, "dist max", dist.max())
inside = 0
safe_points = 0
for run in red.runs:
    xy = np.array([engine.grid.px(p) for p in run.points])
    ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, engine.grid.w - 1)
    iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, engine.grid.h - 1)
    safe = dist[iy, ix] >= need
    safe_points += int(safe.sum())
    inside += sum(1 for p in run.points if 175 < p[0] < 225 and 125 < p[1] < 175)
print("points rouges dans la zone verte:", inside, "points sûrs:", safe_points)
pruned, removed = engine._prune([radical._Run(run.points, run.origin) for run in red.runs], cover)
print("retirés:", removed)
for run in red.runs[:1]:
    points = run.points
    xy = np.array([engine.grid.px(p) for p in points])
    print("run exemple", [[round(p[0]), round(p[1])] for p in points])
