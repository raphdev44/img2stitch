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


def fill_runs(mask, px_per_mm, row_spacing_mm, angle_deg, min_area_mm2, min_run_mm=0.3):
    """
    Découpe le masque en rangées parallèles. Retourne une liste de dicts :
        p0, p1 : extrémités (x_mm, y_mm) dans le repère de l'image
        comp   : numéro de la forme connexe à laquelle appartient la rangée
        row    : numéro de rangée (sert à décaler les points)
    """
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
        return []

    step = max(1, round(row_spacing_mm * px_per_mm))
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
