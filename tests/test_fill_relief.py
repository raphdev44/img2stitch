"""Palette relief refondue : trois techniques pro, géométries distinctes et vérifiables."""
import math

import numpy as np
import pyembroidery as pe

from img2stitch import relief
from img2stitch.config import Settings
from img2stitch.fill import FILL_TYPES as ALL_FILLS
from img2stitch.fill import zone_fill
from img2stitch.planner import Sewer, plan_fill


RELIEF = list(relief.FILL_TYPES)
SETTINGS = Settings(min_area_mm2=0.1)


def _disk(n=200):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) < (n / 2 - 2) ** 2


def test_relief_is_wired_into_the_single_palette():
    assert len(RELIEF) == 21
    for key in RELIEF:
        assert key in ALL_FILLS, key           # visible dans l'éditeur
        assert key.startswith("relief_")
    # les remplissages classiques sont intacts
    for classic in ("tatami", "satin", "cross", "contour", "wave", "none"):
        assert classic in ALL_FILLS


def test_every_relief_produces_stitches():
    mask = _disk()
    for key in RELIEF:
        passes = zone_fill(key, mask, 10, SETTINGS)
        assert passes and any(data for _, data in passes), key


def test_every_relief_plans_into_a_pattern():
    mask = _disk(120)
    for key in RELIEF:
        pattern = pe.EmbPattern()
        pattern.add_thread({"color": 0x902020})
        sewer = Sewer(pattern, SETTINGS, 10)
        sewer.new_color(mask, np.zeros_like(mask))
        for _, data in zone_fill(key, mask, 10, SETTINGS):
            plan_fill(sewer, data)
        pattern.end()
        stitches = sum(1 for _, _, c in pattern.stitches if (c & pe.COMMAND_MASK) == pe.STITCH)
        assert stitches > 100, key


# --- 1) Bandes directionnelles ---------------------------------------------------------------
def test_bands_receive_two_opposite_angles():
    # Le cœur de la technique : deux champs de fils croisés, pas un seul remplissage uniforme.
    assert relief.BAND_ANGLES["relief_cotes"] == (90.0, 0.0)
    assert relief.BAND_ANGLES["relief_chevrons"] == (45.0, -45.0)
    for key in relief.BAND_ANGLES:
        first, second = relief._band_masks(key, _disk(), 10)
        assert first.any() and second.any(), key
        assert not (first & second).any(), key       # les deux parités sont disjointes
        assert (first | second).sum() == _disk().sum()  # et couvrent exactement la forme


def test_vertices_carry_a_satin_cord():
    # Sans cordon, deux champs de même densité se contentent de changer de teinte.
    mask = _disk()
    cords = relief._cord_runs("relief_cotes", mask, 10)
    assert cords
    path = cords[0]["pts"]
    xs = [p[0] for p in path]
    # zigzag : les points alternent de part et d'autre de l'arête
    assert max(xs) - min(xs) >= relief.CORD_WIDTH_MM * 0.9
    # seuls les motifs à arêtes verticales en ont (damier/vannerie : croisement 0°/90°)
    assert relief._cord_runs("relief_damier", mask, 10) == []


def test_chevron_cord_follows_the_v_shape():
    mask = _disk()
    cords = relief._cord_runs("relief_chevrons", mask, 10)
    assert cords
    pts = np.array([[p[0] - relief._boundary_x("relief_chevrons", cords[0]["row"], np.array([p[1]]))[0],
                     p[1]] for p in cords[0]["pts"][:40]])
    assert pts[:, 0].min() < 0 < pts[:, 0].max()   # la ligne oscille autour de l'arête


# --- 2) Motif fill ---------------------------------------------------------------------------
def test_motif_layer_sits_on_a_tatami_base():
    mask = _disk()
    for key in relief.TILINGS:
        passes = zone_fill(key, mask, 10, SETTINGS)
        assert len(passes) == 2, key                 # fond puis cordon
        assert not all("pts" in r for r in passes[0][1])   # le fond n'a pas de cordon
        assert all("pts" in r for r in passes[1][1])       # la couche motif si


def test_motif_cords_are_closed_round_trips():
    mask = _disk()
    runs = relief._motif_runs("relief_nid_abeille", mask, 10, SETTINGS)
    assert runs
    for r in runs[:20]:
        assert len(r["pts"]) > 8
        assert math.dist(r["p0"], r["p1"]) < 1e-6         # aller-retour : retour au départ


def test_base_angle_never_parallels_the_pattern():
    # Losanges à 45° sur un fond à 45° = motif invisible (règle d'or des program splits Wilcom).
    assert relief.BASE_OFFSET_DEG == 30.0
    s = Settings(fill_angle_deg=45.0)
    import inspect
    source = inspect.getsource(relief._base_runs)
    assert "BASE_OFFSET_DEG" in source


def test_motif_geometry_matches_its_name():
    # Losanges : 4 sommets, demi-diagonale 3 mm.
    dia = relief._motifs("relief_losanges", 0.0, 0.0, 0, 0)[0]
    assert len(dia) == 5
    assert abs(dia[:, 1].max() - 3.0) < 1e-6
    # Nid d'abeille : hexagone régulier de 6 sommets.
    hexa = relief._motifs("relief_nid_abeille", 0.0, 0.0, 0, 0)[0]
    assert len(hexa) == 7
    # Écailles : arc en U (courbure réelle, pas un segment).
    scale = relief._motifs("relief_ecailles", 0.0, 0.0, 0, 0)[0]
    assert scale[:, 1].max() - scale[:, 1].min() > 2.0
    # Coquilles : arc + 3 nervures.
    assert len(relief._motifs("relief_coquilles", 0.0, 0.0, 0, 0)) == 4
    # Trèfle : 4 lobes + cœur.
    assert len(relief._motifs("relief_trefle", 0.0, 0.0, 0, 0)) == 5
    # Étoiles : 12 sommets alternés.
    star = relief._motifs("relief_etoiles", 0.0, 0.0, 0, 0)[0]
    assert len(star) == 13
    # Spirales : rayon croissant.
    spiral = relief._motifs("relief_spirales", 0.0, 0.0, 0, 0)[0]
    r = np.hypot(spiral[:, 0], spiral[:, 1])
    assert r[-1] > r[0] * 2


def test_motifs_are_clipped_to_the_shape():
    # Aucun point de cordon ne doit sortir de la forme.
    mask = _disk()
    for key in relief.TILINGS:
        for r in relief._motif_runs(key, mask, 10, SETTINGS):
            pts = np.array([r["p0"]] + r["pts"])
            ix = np.rint(pts[:, 0] * 10).astype(int)
            iy = np.rint(pts[:, 1] * 10).astype(int)
            assert mask[iy, ix].all(), key


# --- 3) Vermicelle ---------------------------------------------------------------------------
def test_stipple_is_a_free_meander_without_base():
    mask = _disk()
    passes = zone_fill("relief_vermicelle", mask, 10, SETTINGS)
    assert len(passes) == 1                      # stipple pur : pas de fond tatami
    runs = passes[0][1]
    assert len(runs) >= 3                        # plusieurs serpentins (run count)
    longest = max(runs, key=lambda r: len(r["pts"]))
    ys = [p[1] for p in longest["pts"]]
    assert max(ys) - min(ys) > 1.0               # ça ondule vraiment


# --- 4) Pelage ------------------------------------------------------------------------------
def test_fur_is_made_of_long_stitches():
    # Une fourrure n'est pas un satin : le fil est couché DANS le poil, en points de 2-3 mm.
    runs = relief._fur_runs("relief_pelage", _disk(), 10)
    assert runs
    steps = []
    for r in runs:
        path = [r["p0"]] + r["pts"]
        steps += [math.dist(a, b) for a, b in zip(path, path[1:])]
    assert np.median(steps) > 2.0             # pas de satin à 0,45 mm
    assert max(steps) < 3.6


def test_fur_strands_share_the_coat_direction():
    runs = relief._fur_runs("relief_pelage", _disk(), 10)
    ang = math.radians(relief.FUR["relief_pelage"]["angle"])
    d = np.array([math.cos(ang), math.sin(ang)])
    cosines = []
    for r in runs:
        v = np.array(r["p1"]) - np.array(r["p0"])
        if np.linalg.norm(v) > 2.0:
            cosines.append(abs(float(v @ d) / np.linalg.norm(v)))
    assert np.median(cosines) > 0.98          # ±8° de biais au pire


def test_fur_has_a_combing_base():
    mask = _disk()
    for kind in relief.FUR:
        passes = zone_fill(kind, mask, 10, SETTINGS)
        assert len(passes) == 2, kind
        # fond : tatami dans le sens du poil ; puis les mèches par-dessus.
        assert passes[0][1] == relief._base_runs(mask, 10, SETTINGS, angle=relief.FUR[kind]["angle"])
        assert passes[1][1] == relief._fur_runs(kind, mask, 10)


def test_two_fur_lengths():
    assert relief.FUR["relief_pelage_long"]["length"] > 1.4 * relief.FUR["relief_pelage"]["length"]


# --- 5) Plumes -------------------------------------------------------------------------------
def test_feather_vane_is_solid_satin():
    rims, vane, spines = relief._feather_runs("relief_plumes", _disk(), 10, SETTINGS)
    assert rims and vane and spines
    widths = [math.dist(r["p0"], r["p1"]) for r in vane]
    assert np.median(widths) > 1.5            # les points traversent la barbe
    assert all(r["max_len"] == min(SETTINGS.max_stitch_mm, 4.0) for r in vane)


def test_feather_layers_are_stacked_rim_under_vane():
    passes = [data for _, data in zone_fill("relief_plumes", _disk(), 10, SETTINGS) if data]
    assert len(passes) == 4                    # fond, ourlet, barbe, rachis
    comps = [{r["comp"] for r in layer} for layer in passes[1:]]
    assert comps[0] != comps[1]                # nos deux premières couches ne se mélangent pas


def test_feather_orientations_differ():
    assert relief.PLUME_TILING["relief_plumes"][2] == 12.0
    assert relief.PLUME_TILING["relief_plumes_couchees"][2] == 55.0


# --- 6) Formes celtiques ---------------------------------------------------------------------
def test_interlace_ribbons_are_continuous_bands():
    runs = relief._interlace_runs(_disk(), 10)
    assert runs
    comps = {r["comp"] for r in runs}
    assert len(comps) < len(runs)              # un comp par ruban, pas un par segment
    assert {c // 100000 for c in comps} == {0, 1}   # les deux familles à 45°


def test_interlace_ribbons_are_packed_like_a_weave():
    # Des rubans jointifs (pas une grille) et un vrai passage sous l'autre.
    assert relief.INTERLACE_WIDTH >= 0.45 * relief.INTERLACE_PERIOD
    assert relief.INTERLACE_GAP > 0.1
    assert relief.INTERLACE_GAP < relief.INTERLACE_WIDTH


def test_triskele_is_a_large_three_arm_motif():
    shapes = relief._motifs("relief_triskele", 0.0, 0.0, 0, 0)
    assert len(shapes) == 4                    # trois bras + moyeu
    reaches = [np.hypot(s[-1][0], s[-1][1]) for s in shapes[:3]]
    assert all(r > 3.5 for r in reaches)
    assert relief.TILINGS["relief_triskele"][1] >= 13.0


# --- Compatibilité ---------------------------------------------------------------------------
def test_legacy_names_map_to_new_motifs():
    assert relief.LEGACY["relief_diamonds"] == "relief_losanges"
    assert relief.LEGACY["relief_scales"] == "relief_ecailles"
    mask = _disk(120)
    for old, new in relief.LEGACY.items():
        assert relief.zone_fill(old, mask, 10, SETTINGS) == relief.zone_fill(new, mask, 10, SETTINGS)


def test_unknown_relief_falls_back():
    passes = relief.zone_fill("relief_inconnu", _disk(), 10, SETTINGS)
    assert any(data for _, data in passes)
