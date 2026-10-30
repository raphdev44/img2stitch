"""Satin bombé : colonne étagée (guides, sous-couche, satin) du remplissage intelligent."""
import math

import numpy as np
import pyembroidery as pe

from img2stitch import satin
from img2stitch.config import Settings
from img2stitch.fill import FILL_TYPES, zone_fill
from img2stitch.planner import Sewer, plan_fill

SETTINGS = Settings(min_area_mm2=0.1)
PPM = 10
N = 200


def _band(width_mm=3.0, length_mm=16.0):
    """Ruban horizontal : largeur et longueur en mm, centré sur l'image."""
    mask = np.zeros((N, N), bool)
    half = round(width_mm * PPM / 2)
    side = round(length_mm * PPM)
    x0 = (N - side) // 2
    mask[N // 2 - half:N // 2 + half, x0:x0 + side] = True
    return mask


def _layer(runs, comp):
    return [r for r in runs if r["comp"] // 100_000 == comp]


def _path(run):
    return [run["p0"]] + list(run.get("pts", []))


def _planned(mask):
    """Motif réellement piqué pour la forme : (pattern, nombre de points)."""
    pattern = pe.EmbPattern()
    pattern.add_thread({"color": 0x902020})
    sewer = Sewer(pattern, SETTINGS, PPM)
    sewer.new_color(mask, np.zeros_like(mask))
    for _, data in zone_fill("satin_relief", mask, PPM, SETTINGS):
        plan_fill(sewer, data)
    pattern.end()
    return pattern


def test_layers_are_sewn_from_the_guides_to_the_satin():
    passes = satin.relief_passes(_band(), PPM, SETTINGS)
    order = []
    for _, runs in passes:
        for r in runs:
            layer = r["comp"] // 100_000
            if not order or order[-1] != layer:
                order.append(layer)
    # guides de bord, guide de centre, sous-couche, satin final — dans cet ordre.
    assert order[-4:] == [2, 3, 4, 5], order


def test_the_underlay_is_narrower_than_the_final_satin():
    # C'est le mécanisme du dôme : la couche du dessus, plus large, se rétracte sur ses bords.
    runs = [r for _, layer in satin.relief_passes(_band(), PPM, SETTINGS) for r in layer]
    narrow = math.dist(_layer(runs, 4)[0]["p0"], _layer(runs, 4)[0]["pts"][0])
    full = math.dist(_layer(runs, 5)[0]["p0"], _layer(runs, 5)[0]["pts"][0])
    assert narrow < 0.85 * full
    # 3 mm de bande + 2 x 0,18 mm de compensation de reprise
    assert 2.8 < full < 3.7


def test_edge_guides_run_on_both_sides():
    runs = [r for _, layer in satin.relief_passes(_band(), PPM, SETTINGS) for r in layer]
    guides = _layer(runs, 2)
    assert len(guides) == 2
    ys = [p[1] for g in guides for p in _path(g)]
    assert min(ys) < 10.0 < max(ys)      # un guide de chaque côté de l'axe médian


def test_underlay_is_flagged_as_covered():
    # Le rendu réaliste ne doit pas creuser les piqûres des sous-couches : elles sont recouvertes.
    runs = [r for _, layer in satin.relief_passes(_band(), PPM, SETTINGS) for r in layer]
    assert all(r["relief"] == "under" for r in _layer(runs, 2) + _layer(runs, 3) + _layer(runs, 4))
    assert all(r["relief"] == "top" for r in _layer(runs, 5))


def test_planned_pattern_carries_the_relief_ranges():
    mask = _band()
    pattern = _planned(mask)
    ranges = getattr(pattern, "img2stitch_relief")
    assert ranges["under"] and ranges["top"]
    covered = np.zeros(len(pattern.stitches), bool)
    for first, end in ranges["under"] + ranges["top"]:
        covered[first:end] = True
    # Le fond ne pique que les jonctions et les bouts : l'essentiel est bien la colonne satin.
    assert covered.sum() > 0.9 * len(pattern.stitches)


def test_no_stitch_exceeds_the_machine_limit():
    for _, layer in satin.relief_passes(_band(width_mm=9.5), PPM, SETTINGS):
        for r in layer:
            steps = [math.dist(a, b) for a, b in zip(_path(r), _path(r)[1:])]
            assert not steps or max(steps) < 12.1


def test_the_fill_plans_into_a_pattern():
    pattern = _planned(_band())
    assert sum(1 for _, _, c in pattern.stitches if (c & pe.COMMAND_MASK) == pe.STITCH) > 100


def test_a_wide_shape_falls_back_to_a_plain_fill():
    # Au-delà de la largeur d'une colonne satin, le module ne pique pas de satin : la surface
    # reste couverte par un remplissage classique, jamais laissée vide.
    mask = np.zeros((N, N), bool)
    mask[40:160, 40:160] = True
    passes = zone_fill("satin_relief", mask, PPM, SETTINGS)
    assert passes and any(data for _, data in passes)
    assert not any("relief" in r for _, layer in passes for r in layer)


def test_registered_in_the_editor_palette():
    assert "satin_relief" in FILL_TYPES
    assert "satin_flow" in FILL_TYPES      # l'ancien satin reste disponible


def test_relief_ranges_survive_the_pattern_file(tmp_path):
    # L'aperçu réaliste est refait depuis pattern.json (changement de tissu) : les plages de
    # relief doivent y survivre, sinon le dôme disparaît au premier changement de tissu.
    from webapp.designs import _load_pattern, _save_pattern

    _save_pattern(_planned(_band()), tmp_path / "pattern.json")
    back = _load_pattern(tmp_path)
    assert getattr(back, "img2stitch_relief")["top"]
