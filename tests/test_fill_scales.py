import numpy as np
import pyembroidery as pe

from img2stitch.config import Settings
from img2stitch.fill import fill_scale_paths, zone_fill
from img2stitch.planner import Sewer, plan_fill


def test_scale_paths_are_curved_and_staggered():
    mask = np.ones((120, 120), dtype=bool)
    paths = fill_scale_paths(mask, 10, 0.1)

    assert paths
    by_row = {}
    for path in paths:
        by_row.setdefault(path["row"], []).append(path)

    full_rows = [row for row, items in by_row.items() if len(items) >= 3]
    first, second = full_rows[:2]
    arc = next(item["path"] for item in by_row[first] if len(item["path"]) > 20)
    points = np.asarray(arc)
    assert points[:, 1].max() - points[:, 1].min() > 2.0

    full_first = [item for item in by_row[first] if len(item["path"]) > 20]
    full_second = [item for item in by_row[second] if len(item["path"]) > 20]
    first_x = min(item["p0"][0] for item in full_first)
    second_x = min(item["p0"][0] for item in full_second)
    assert abs(abs(second_x - first_x) - 2.5) < 0.4


def test_scale_fill_plans_background_and_arc_passes():
    mask = np.ones((80, 80), dtype=bool)
    settings = Settings(min_area_mm2=0.1)
    pattern = pe.EmbPattern()
    sewer = Sewer(pattern, settings, 10)
    sewer.new_color(mask, np.zeros_like(mask))

    passes = zone_fill("relief_scales", mask, 10, settings)
    assert len(passes) == 2
    for _, runs in passes:
        assert runs
        plan_fill(sewer, runs)

    assert any((command & pe.COMMAND_MASK) == pe.STITCH for _, _, command in pattern.stitches)
