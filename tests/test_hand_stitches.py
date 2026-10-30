"""Points de broderie à la main : bibliothèque de motifs et classificateur."""
import math

import pytest

from img2stitch import handstitch as hs
from img2stitch.hand import choose_stitch

PATH = [(0.0, 0.0), (20.0, 0.0)]


@pytest.mark.parametrize("name", ["running", "back", "stem", "split", "chain", "couching", "cross_stitch"])
def test_trace_stitches_follow_the_line(name):
    pen = hs.Pen(seed=3)
    getattr(hs, name)(pen, PATH)
    assert pen.strokes
    for stroke in pen.strokes:
        assert len(stroke) >= 2
        for x, y in stroke:
            assert math.isfinite(x) and math.isfinite(y)
            assert -6 < y < 6 and -7 < x < 27      # le motif reste autour du tracé


@pytest.mark.parametrize("name", ["satin", "fishbone", "herringbone", "chevron", "fern", "feather", "cretan"])
def test_band_stitches_reach_both_edges(name):
    pen = hs.Pen(seed=3)
    getattr(hs, name)(pen, PATH, 1.5)
    assert pen.strokes
    ys = [y for stroke in pen.strokes for _, y in stroke]
    assert min(ys) < -0.5 and max(ys) > 0.5


def test_blanket_stitch_runs_a_rail_with_legs_on_one_side():
    pen = hs.Pen(seed=3)
    hs.blanket(pen, PATH, 1.5)
    ys = [y for stroke in pen.strokes for _, y in stroke]
    assert min(ys) == 0 and max(ys) > 1.0


def test_motif_stitches_stay_near_their_centre():
    pen = hs.Pen(seed=1)
    hs.french_knot(pen, (0, 0), 0.9)
    hs.woven_wheel(pen, (10, 0), 3.0)
    hs.spider_web(pen, (20, 0), 3.0)
    hs.lazy_daisy(pen, (30, 0), (30, 6))
    hs.bullion(pen, (40, 0), (46, 0))
    hs.tulip(pen, (50, 0), (50, 6))
    assert len(pen.strokes) >= 10
    xs = [x for stroke in pen.strokes for x, _ in stroke]
    ys = [y for stroke in pen.strokes for _, y in stroke]
    assert -2 < min(xs) and max(xs) < 58
    assert min(ys) > -6 and max(ys) < 12


def test_seed_and_darning_fill_a_mask_without_flooding_it():
    mask = [[False] * 40 for _ in range(20)]
    pen = hs.Pen(seed=2)
    hs.seed(pen, mask, 10)          # masque vide : rien à semer
    assert not pen.strokes
    inside = [[0 <= x < 30 and 0 <= y < 20 for x in range(40)] for y in range(20)]
    import numpy as np
    pen = hs.Pen(seed=2)
    hs.seed(pen, np.array(inside), 10)
    hs.darning(pen, np.array(inside), 10)
    assert 6 <= len(pen.strokes) <= 40 * 20 * 0.9 + 60   # un semis, pas un remplissage machine


def test_choose_stitch_follows_the_shape():
    assert choose_stitch(0.45, 20) == "back"      # trait très fin : ligne continue
    assert choose_stitch(0.9, 20, encloses=True) == "back"
    assert choose_stitch(0.9, 20) == "stem"
    assert choose_stitch(1.8, 20) == "stem"
    assert choose_stitch(3.0, 20) == "chain"
    assert choose_stitch(5.0, 20, creativity=5) == "fishbone"   # arête : cran le plus décoratif
    assert choose_stitch(5.0, 20) == "satin"                     # au cran par défaut, une bande satin
    assert choose_stitch(5.0, 8) == "satin"
    assert choose_stitch(0.8, 0.8, roundness=0.95, area_mm2=0.5) == "french_knot"
    assert choose_stitch(5.0, 5.0, roundness=0.9, area_mm2=40) == "woven_wheel"
    assert choose_stitch(12, 80, roundness=0.85) == "seed"
    assert choose_stitch(12, 12, roundness=0.8, area_mm2=200) == "seed"
    assert choose_stitch(12, 80, roundness=0.2) == "darning"
    assert choose_stitch(3.0, 20, style="chainette") == "chain"
    assert choose_stitch(3.0, 20, style="feuilles") == "fishbone"
    assert choose_stitch(3.0, 20, style="croix") == "cross_stitch"


def test_creativity_gauge_only_moves_how_daring_the_stitches_are():
    leaf = {"taper": 0.7}
    assert choose_stitch(2.4, 14, creativity=1, **leaf) == "satin"      # feuille : satin sobre
    assert choose_stitch(2.4, 14, creativity=2, **leaf) == "satin"
    assert choose_stitch(2.4, 14, creativity=3, **leaf) == "fern"       # fougère à partir du milieu
    ribbon = {"straight": 0.99}
    assert choose_stitch(3.0, 40, creativity=1, **ribbon) == "split"    # ruban : corde classique
    assert choose_stitch(3.0, 40, creativity=2, **ribbon) == "chain"
    assert choose_stitch(3.0, 40, creativity=3, **ribbon) == "chain"
    assert choose_stitch(3.0, 40, creativity=4, **ribbon) == "chevron"
    edge = {"encloses": True, "big_ring": True}
    assert choose_stitch(2.0, 40, creativity=2, **edge) == "back"       # bordure : trait arrière
    assert choose_stitch(2.0, 40, creativity=3, **edge) == "blanket"    # feston au milieu
    wheel = {"roundness": 0.9, "area_mm2": 40}
    assert choose_stitch(5.0, 5.0, creativity=2, **wheel) == "french_knot"
    assert choose_stitch(5.0, 5.0, creativity=3, **wheel) == "woven_wheel"
    # une famille imposée garde le point demandé, quel que soit le cran
    assert choose_stitch(3.0, 20, style="chainette", creativity=1) == "chain"
    assert choose_stitch(2.4, 14, style="feuilles", creativity=1) == "stem"


def test_similar_strokes_share_one_stitch_and_roles_stay_distinct():
    from img2stitch.hand import _assign_families

    def brush(width=1.5, length=12, taper=0.7, straight=0.9, outline=True, encloses=True):
        return {"width": width, "length": length, "taper": taper, "straight": straight,
                "encloses": encloses, "big_ring": False, "outline": outline}

    contour = [brush(length=6), brush(length=25), brush(length=4)]   # le même contour, trois arcs
    stem = [brush(outline=False, encloses=False, straight=0.2, length=30)]
    leaves = [brush(width=2.5, taper=0.85, outline=False, length=7),
              brush(width=2.55, taper=0.8, outline=False, length=18)]
    units = contour + stem + leaves
    _assign_families(units, "auto", 5)
    assert len({u["kind"] for u in contour}) == 1     # une feuille, un seul point
    assert contour[0]["kind"] == "back"
    assert stem[0]["kind"] == "stem"                  # une tige n'est pas un contour
    assert len({u["kind"] for u in leaves}) == 1      # deux feuilles jumelles, le même point
    assert leaves[0]["kind"] in {"fern", "fishbone", "satin"}


def test_creativity_stays_inside_the_five_notches():
    for level in (-3, 0, 1, 3, 5, 9, None):
        kind = choose_stitch(3.0, 40, straight=0.99, creativity=level)
        assert kind in ("split", "chain", "chevron")
