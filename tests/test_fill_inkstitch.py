"""Remplissages portés d'Ink/Stitch : palette, géométrie et intégration au planificateur."""
import numpy as np
import pyembroidery as pe

from img2stitch import inkstitch
from img2stitch.config import Settings
from img2stitch.fill import FILL_TYPES as ALL_FILLS
from img2stitch.fill import zone_fill
from img2stitch.planner import Sewer, plan_fill

INKSTITCH = list(inkstitch.FILL_TYPES)
SETTINGS = Settings(min_area_mm2=0.1)
TINY_SETTINGS = Settings(min_area_mm2=0.1, row_spacing_mm=2.0)   # écart plus large que la forme


def _disk(n=200):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) < (n / 2 - 2) ** 2


def _band(w=200, h=60):
    mask = np.zeros((h, w), bool)
    mask[10:h - 10, 5:w - 5] = True
    return mask


def test_inkstitch_is_wired_into_the_single_palette():
    assert len(INKSTITCH) == 3
    for key in INKSTITCH:
        assert key in ALL_FILLS, key          # visible dans l'éditeur de zones
        assert key.startswith("inkstitch_")
    # les deux autres sections sont intactes
    assert "tatami" in ALL_FILLS and "relief_cotes" in ALL_FILLS


def test_every_fill_produces_stitches():
    for mask in (_disk(), _band()):
        for key in INKSTITCH:
            passes = zone_fill(key, mask, 10, SETTINGS)
            assert passes and any(data for _, data in passes), key


def test_every_fill_plans_into_a_pattern():
    mask = _disk(120)
    for key in INKSTITCH:
        pattern = pe.EmbPattern()
        pattern.add_thread({"color": 0x902020})
        sewer = Sewer(pattern, SETTINGS, 10)
        sewer.new_color(mask, np.zeros_like(mask))
        for _, data in zone_fill(key, mask, 10, SETTINGS):
            plan_fill(sewer, data)
        pattern.end()
        stitches = sum(1 for _, _, c in pattern.stitches if (c & pe.COMMAND_MASK) == pe.STITCH)
        assert stitches > 100, key


# --- remplissage en contour ------------------------------------------------------------------
def test_contour_fill_is_one_single_spiral():
    # Une spirale, et non des boucles : tout le remplissage ne fait qu'un seul tracé continu.
    passes = zone_fill("inkstitch_contour", _disk(), 10, SETTINGS)
    runs = [r for _, data in passes for r in data]
    assert len(runs) == 1
    assert len(runs[0]["pts"]) > 40


def test_contour_fill_rings_follow_the_shape():
    rings = inkstitch._offset_rings(_disk(), 10, SETTINGS)
    lengths = [inkstitch._length(r) for r in rings]
    assert len(rings) > 5
    assert lengths == sorted(lengths, reverse=True)   # du bord vers le centre


def test_interpolation_between_two_rings_stays_between_them():
    inner = inkstitch._circle(0.0, 0.0, 2.0, 1.0)
    outer = inkstitch._circle(0.0, 0.0, 4.0, 1.0)
    path = inkstitch._interpolate_linear_rings(outer, inner, 2.0)
    radius = np.hypot(path[:, 0], path[:, 1])
    assert radius.min() < 2.2 and radius.max() > 3.8   # le fil glisse d'un anneau à l'autre


# --- remplissage circulaire ------------------------------------------------------------------
def test_circular_fill_double_spiral_covers_the_shape():
    mask = _disk()
    runs = [r for _, data in zone_fill("inkstitch_circular", mask, 10, SETTINGS) for r in data]
    pts = np.array([p for r in runs for p in [r["p0"]] + r["pts"]])
    # les cercles vont jusqu'au bord de la forme, à une demi-rangée près
    assert np.hypot(*pts.T).max() > 9.0


def test_circular_fill_of_a_tiny_shape_is_one_circle():
    # Plus petite que l'écart de rangée : Ink/Stitch pique un seul cercle de cette taille.
    runs = [r for _, data in zone_fill("inkstitch_circular", _disk(24), 10, TINY_SETTINGS) for r in data]
    assert len(runs) == 1
    pts = np.array([runs[0]["p0"]] + runs[0]["pts"])
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    radius = np.hypot(*(pts - centre).T)
    assert 0.9 * TINY_SETTINGS.row_spacing_mm < radius.max() < 1.1 * TINY_SETTINGS.row_spacing_mm


# --- point de croix --------------------------------------------------------------------------
def test_cross_fill_is_a_grid_of_crosses():
    runs = [r for _, data in zone_fill("inkstitch_cross", _disk(), 10, SETTINGS) for r in data]
    pts = np.array([r["p0"] for r in runs])
    assert len(runs) >= 5                                  # une rangée de croix = un tracé
    steps = np.diff(np.unique(np.round(pts[:, 1], 3)))
    assert steps.size and np.allclose(steps, inkstitch.CROSS_CELL_MM, atol=1e-6)


def test_cross_fill_sews_two_diagonals_and_a_bottom_edge():
    runs = [r for _, data in zone_fill("inkstitch_cross", _disk(), 10, SETTINGS) for r in data]
    pts = [runs[0]["p0"]] + runs[0]["pts"]
    cell = inkstitch.CROSS_CELL_MM
    first, second, third = pts[0], pts[1], pts[2]
    assert np.allclose(np.abs(np.subtract(second, first)), (cell, cell))     # diagonale
    assert np.isclose(second[1], third[1])                                   # bord bas
    assert np.isclose(third[0], first[0])


def test_cross_fill_never_flips_a_cross():
    # Toutes les croix sont piquées dans le même ordre : le dessus du croisement ne change jamais.
    runs = [r for _, data in zone_fill("inkstitch_cross", _band(), 10, SETTINGS) for r in data]
    for run in runs:
        a, c, d = ([run["p0"]] + run["pts"])[:3]
        assert c[0] > a[0] and c[1] > a[1]     # A -> C : diagonale descendante
        assert d[1] == c[1] and d[0] == a[0]   # C -> D : bord bas, retour à gauche


def test_cross_fill_thread_count_doubles_the_row():
    runs = [r for _, data in zone_fill("inkstitch_cross", _disk(), 10, SETTINGS) for r in data]
    pts = [runs[0]["p0"]] + runs[0]["pts"]
    if inkstitch.CROSS_THREADS > 1:
        assert pts[0] == pts[-1]   # le second brin revient au point de départ de la rangée
