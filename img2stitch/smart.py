"""Détection des formes fines à broder en satin tournant (option « Remplissage intelligent »).

Une bande fine est une forme étroite sur toute sa longueur : lettrage, ruban, tige, trait épais.
On lit sa largeur typique sur l'axe médian (transformée de distance), on borne les renflements, et
on écarte les réseaux et les aplats, qui ne sont pas des bandes même s'ils restent minces par endroits.
"""
import cv2
import numpy as np

from .preprocess import _embroidered_pixels, load_and_quantize

MIN_WIDTH_MM = 0.9      # sous ce seuil, le satin n'est qu'un cordon : le contour s'en charge déjà
# Au-delà, une traversée de satin est trop longue pour rester nette ; et surtout, les éclats de
# forme qui bordent un dessin (un pan de ciel entre deux branches, un tronc large) font 6 mm et
# plus : sans ce plafond, le remplissage intelligent les prend pour des traits. Le satin reste
# applicable à la main au-delà (satin.MAX_WIDTH_MM).
MAX_WIDTH_MM = 5.0
PEAK_FACTOR = 2.0       # un réseau a des nœuds plus larges que ses brins : toléré jusque-là
MIN_LENGTH_MM = 4.0     # une pastille ronde n'est pas une bande
MIN_ELONGATION = 2.2    # surface / largeur² : écarte les formes trapues
WIDTH_PERCENTILE = 90   # largeur lue haut sur l'axe médian : un triangle s'élargit, une bande non


def _fit_width_mm(path, s, probe_px):
    """Largeur de travail de l'image (mm) pour la taille demandée, comme image_to_pattern."""
    if s.max_width_mm is None and s.max_height_mm is None:
        return s.width_mm
    _, keep = _embroidered_pixels(path, probe_px, s.background, s.background_tolerance,
                                  s.background_scope, s.exclude_colors)
    if not keep.any():
        return s.width_mm
    ys, xs = np.nonzero(keep)
    fw, fh = (np.ptp(xs) + 1) / probe_px, (np.ptp(ys) + 1) / probe_px
    max_w = s.max_width_mm - 0.2 if s.max_width_mm else np.inf
    max_h = s.max_height_mm - 0.2 if s.max_height_mm else np.inf
    return min(max_w / fw, max_h / fh)


def _band(comp, px_per_mm):
    """(forme assez fine, point intérieur) d'une forme connexe, ou (False, None)."""
    # Bord de zéro autour de la forme : sinon, une forme qui remplit sa boîte englobante n'a plus
    # de distance au bord fiable.
    pad = cv2.copyMakeBorder(comp.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) > 0
    dist = cv2.distanceTransform(pad.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = pad & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-6)
    ridge &= dist > 0.5  # moins d'un dixième de millimètre : bruit de bord
    if ridge.sum() < 2:
        return False, None
    # Largeur lue haut sur l'axe médian : une bande garde sa largeur sur toute sa longueur, alors
    # qu'une forme qui s'élargit (triangle, coin de ciel, aplat en pointe) a une largeur médiane
    # basse et passerait pour une bande. Le contour noir d'un dessin est un réseau : ses trous et
    # sa surface ne disent rien de sa finesse, seule la largeur des brins compte.
    half_mm = dist[ridge] / px_per_mm
    width_mm = 2 * float(np.percentile(half_mm, WIDTH_PERCENTILE))
    if not MIN_WIDTH_MM <= width_mm <= MAX_WIDTH_MM:
        return False, None
    if 2 * float(half_mm.max()) > PEAK_FACTOR * width_mm:
        return False, None  # renflement local : ce n'est pas une bande
    width_px = width_mm * px_per_mm
    if comp.sum() / max(1.0, width_px ** 2) < MIN_ELONGATION:
        return False, None
    ys, xs = np.nonzero(comp)
    if max(np.ptp(ys), np.ptp(xs)) / px_per_mm < MIN_LENGTH_MM:
        return False, None
    # Point d'ancrage au milieu de l'axe médian : un point au bout de la forme resterait
    # fragile quand la génération la rééchantillonne.
    ys, xs = np.nonzero(ridge)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return True, (int(ys[k]) - 1, int(xs[k]) - 1)


def thin_band_regions(path, settings, probe_px=1000):
    """Zones (u, v, (r,g,b)) des formes assez fines pour un satin tournant.

    `u`, `v` désignent un pixel intérieur à la forme, en fraction de la largeur / hauteur de l'image :
    le même repère que les retouches de zones fermées (meta["regions"])."""
    width_mm = _fit_width_mm(path, settings, probe_px)
    palette, masks = load_and_quantize(str(path), probe_px, settings.n_colors, settings.background,
                                       settings.background_tolerance, settings.background_scope,
                                       settings.exclude_colors, thicken_lines=settings.thicken_lines)
    if not masks:
        return []
    px_per_mm = probe_px / width_mm
    h, w = masks[0].shape
    regions = []
    for rgb, mask in zip(palette, masks):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for lab in range(1, count):
            if stats[lab, cv2.CC_STAT_AREA] / px_per_mm ** 2 < 1.0:
                continue  # bruit et poussières
            x0, y0 = stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP]
            x1, y1 = x0 + stats[lab, cv2.CC_STAT_WIDTH], y0 + stats[lab, cv2.CC_STAT_HEIGHT]
            ok, point = _band(labels[y0:y1, x0:x1] == lab, px_per_mm)
            if ok:
                y, x = point
                regions.append(((x + x0 + 0.5) / w, (y + y0 + 0.5) / h, tuple(rgb)))
    return regions
