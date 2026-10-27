"""Étape 2 : masque binaire -> rangées de remplissage (tatami), SANS ordre de parcours.
L'ordre est décidé ensuite par le planificateur (planner.py) pour limiter les sauts."""
import math
import cv2
import numpy as np
from PIL import Image


def _runs(row):
    """Segments contigus de True dans un tableau 1D -> liste de (x0, x1)."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def _rotated(mask, px_per_mm, angle_deg, min_area_mm2):
    """Masque tourné (rangées horizontales), formes connexes retenues et retour au repère image (mm)."""
    h, w = mask.shape
    side = int(math.hypot(h, w)) + 2
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(Image.fromarray((mask * 255).astype(np.uint8)), ((side - w) // 2, (side - h) // 2))
    rot = np.array(canvas.rotate(-angle_deg, resample=Image.NEAREST)) > 127
    # La rotation "au plus proche voisin" crée des artefacts de 1 px sur les bords (trous, pointes) qui donnent
    # des rangées minuscules : le parcours devrait y revenir par des sauts. On les gomme (fermeture puis ouverture).
    k = np.ones((3, 3), np.uint8)
    rot = cv2.morphologyEx(cv2.morphologyEx(rot.astype(np.uint8), cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k) > 0

    # Formes connexes (dans le repère tourné, où les rangées sont horizontales) ; on écarte le bruit
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rot.astype(np.uint8), connectivity=8)
    min_px = min_area_mm2 * px_per_mm ** 2
    good = {i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px}
    if not good:
        return None

    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    c = side / 2
    ox_, oy_ = (side - w) // 2, (side - h) // 2

    def back(x, y):
        """Repère tourné -> repère image d'origine, en mm."""
        dx, dy = x - c, y - c
        # rotation inverse de celle appliquée au masque (rotate(-angle) = rotation horaire d'angle a)
        ox = dx * ca + dy * sa + c - ox_
        oy = -dx * sa + dy * ca + c - oy_
        return ox / px_per_mm, oy / px_per_mm

    return rot, labels, good, back, side


def fill_runs(mask, px_per_mm, row_spacing_mm, angle_deg, min_area_mm2, min_run_mm=0.3):
    """
    Découpe le masque en rangées parallèles. Retourne une liste de dicts :
        p0, p1 : extrémités (x_mm, y_mm) dans le repère de l'image
        comp   : numéro de la forme connexe à laquelle appartient la rangée
        row    : numéro de rangée (sert à décaler les points)
    """
    prep = _rotated(mask, px_per_mm, angle_deg, min_area_mm2)
    if prep is None:
        return []
    rot, labels, good, back, side = prep
    step = max(1, round(row_spacing_mm * px_per_mm))

    out = []
    for ri, y in enumerate(range(0, side, step)):
        for x0, x1 in _runs(rot[y]):
            if (x1 - x0) / px_per_mm < min_run_mm:   # trop court pour être piqué : le contour couvre le bord
                continue
            comp = int(labels[y, x0])
            if comp not in good:
                continue
            out.append({"p0": back(x0, y), "p1": back(x1, y), "comp": comp, "row": ri})
    return out


def fill_runs_curved(mask, px_per_mm, row_spacing_mm, angle_deg, min_area_mm2, shape="wave",
                     amplitude_mm=0.8, wavelength_mm=5.0, min_run_mm=0.3, sample_mm=0.5):
    """Comme fill_runs, mais chaque rangée suit une onde ("wave") ou un zigzag ("zigzag") ;
    chaque rangée porte en plus `path`, la polyligne à piquer de p0 à p1."""
    prep = _rotated(mask, px_per_mm, angle_deg, min_area_mm2)
    if prep is None:
        return []
    rot, labels, good, back, side = prep
    step = max(1, round(row_spacing_mm * px_per_mm))
    xs = np.arange(side)
    t = xs / (wavelength_mm * px_per_mm)
    wave = np.sin(2 * np.pi * t) if shape == "wave" else 2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1
    off = wave * amplitude_mm * px_per_mm
    sample = max(1, round(sample_mm * px_per_mm))

    out = []
    for ri, y in enumerate(range(0, side, step)):
        yy = np.rint(y + off).astype(int)
        ok = (yy >= 0) & (yy < side)
        inside = np.zeros(side, bool)
        inside[ok] = rot[yy[ok], xs[ok]]
        for x0, x1 in _runs(inside):
            if (x1 - x0) / px_per_mm < min_run_mm:
                continue
            comp = int(labels[yy[x0], x0])
            if comp not in good:
                continue
            path = [back(x, y + off[x]) for x in list(range(x0, x1, sample)) + [x1]]
            out.append({"p0": path[0], "p1": path[-1], "comp": comp, "row": ri, "path": path})
    return out


def contour_loops(mask, px_per_mm, spacing_mm, simplify_mm, min_area_mm2):
    """Remplissage en contours concentriques : boucles successives à `spacing_mm` vers l'intérieur."""
    from .outline import contours_mm
    r = max(1, round(spacing_mm * px_per_mm))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    half = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r + 1, r + 1))
    m = cv2.erode(mask.astype(np.uint8), half)  # 1re boucle à une demi-rangée du bord
    loops = []
    while m.any():
        loops += contours_mm(m, px_per_mm, simplify_mm, min_area_mm2)
        m = cv2.erode(m, kernel)
    return loops


# Motifs de remplissage d'une zone : nom -> libellé. "tatami" est le remplissage par défaut.
FILL_TYPES = {
    "tatami": "Linéaire (diagonal)",
    "tatami_h": "Linéaire horizontal",
    "tatami_v": "Linéaire vertical",
    "cross": "Croisé (quadrillage)",
    "satin": "Satin (zones étroites)",
    "contour": "Contours concentriques",
    "wave": "Ondulé",
    "zigzag": "Zigzag",
    "none": "Contour seul (sans remplissage)",
}


def zone_fill(kind, mask, px_per_mm, s):
    """Passes de remplissage d'une zone selon le motif `kind` : liste de ("runs", rangées) ou ("loops", boucles)."""
    sp, ang, area = s.row_spacing_mm, s.fill_angle_deg, s.min_area_mm2
    if kind == "tatami_h":
        return [("runs", fill_runs(mask, px_per_mm, sp, 0.0, area))]
    if kind == "tatami_v":
        return [("runs", fill_runs(mask, px_per_mm, sp, 90.0, area))]
    if kind == "cross":  # deux couches plus légères, à angle droit
        return [("runs", fill_runs(mask, px_per_mm, sp * 1.75, ang, area)),
                ("runs", fill_runs(mask, px_per_mm, sp * 1.75, ang + 90.0, area))]
    if kind == "satin":  # points d'un bord à l'autre, serrés, sans piqûre intermédiaire (coupés au-delà de 7 mm)
        runs = fill_runs(mask, px_per_mm, max(0.2, sp * 0.6), ang + 90.0, area)
        for r in runs:
            r["max_len"] = 7.0
        return [("runs", runs)]
    if kind == "contour":
        return [("loops", contour_loops(mask, px_per_mm, max(0.45, sp * 1.25), s.outline_simplify_mm, area))]
    if kind in ("wave", "zigzag"):
        return [("runs", fill_runs_curved(mask, px_per_mm, sp, ang - 45.0, area, shape=kind))]
    if kind == "none":
        return []
    return [("runs", fill_runs(mask, px_per_mm, sp, ang, area))]
