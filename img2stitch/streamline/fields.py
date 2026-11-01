"""Champs de densité et de direction d'une région de la photo.

Repris de `embroidery-streamlines` (Zhenyuan Desmond Liu et al., « Directionality-Aware Design of
Embroidery Patterns », Computer Graphics Forum 2023) :

- **densité** : la couleur de la région est paramétrée sur un axe 1-D (leur
  `color_gradient_extraction_with_pca`, ici par les deux couleurs dominantes de la région), puis
  ramenée sur [0.1, 1] par `map_range(1 - t, 0, 1, 0.1, 1)` ; si la moyenne dépasse 0.7, les rôles
  du fond et du motif sont échangés, exactement comme dans `parse_example.py` ;
- **direction** : l'orientation des isophotes (tangente aux niveaux de gris), obtenue par tenseur
  de structure. Elle remplace les annotations de direction tracées à la main sous Labelme : leur
  interface de segmentation et d'annotation n'est pas publiée, la nôtre s'y branche. Là où l'image
  n'a pas de structure, la direction retombe sur l'axe principal de la région.

Les régions, elles, viennent de notre quantification des couleurs (`preprocess.load_and_quantize`),
qui joue le rôle des polygones annotés du dépôt.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ..preprocess import _embroidered_pixels, load_and_quantize

GRID = 210          # côté de la grille normalisée (leur pipeline : 501 x 500)
WORK_PX = 520       # largeur de l'image de travail : les champs sont calculés à cette résolution


@dataclass
class Patch:
    """Une région brodée : sa géométrie, son champ de densité et son champ de direction."""

    color: tuple                 # couleur du fil dense (rgb)
    color_bg: tuple              # couleur du fond de la région (le fil « clair »)
    mask: np.ndarray             # bool, résolution de travail
    inside: np.ndarray           # (GRID, GRID) bool
    density: np.ndarray          # (GRID, GRID) float sur [0.1, 1]
    direction: np.ndarray        # (GRID, GRID) float, radians
    boundary: np.ndarray         # contour de la plus grande composante
    holes: list                  # trous de TOUTES les composantes
    boundaries: list = None      # contours extérieurs de TOUTES les composantes


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _counterclockwise(poly):
    """Comme `ensure_counterclockwise` : le bord tourne dans un sens connu, une seule fois."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 3:
        return poly
    return poly if _signed_area(poly) > 0 else poly[::-1]


def _smooth_closed(poly, sigma):
    n = len(poly)
    if n < 8 or sigma <= 0:
        return poly
    radius = int(min(3 * sigma, n // 2 - 1))
    if radius < 1:
        return poly
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.empty_like(poly)
    for axis in range(2):
        ext = np.concatenate((poly[-radius:, axis], poly[:, axis], poly[:radius, axis]))
        out[:, axis] = np.convolve(ext, k, mode="valid")
    return out


MIN_OUTER_PX2 = 12
MIN_HOLE_PX2 = 4


def _outline(mask):
    """Contours extérieurs de TOUTES les composantes + trous, normalisés par le plus grand côté."""
    h, w = mask.shape
    side = float(max(w, h))
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not len(contours):
        return [], []
    hierarchy = hierarchy.reshape(-1, 4) if hierarchy is not None else np.zeros((len(contours), 4), int)

    def to_norm(c):
        pts = np.asarray(c, float).reshape(-1, 2) + 0.5          # centre du pixel
        return _counterclockwise(_smooth_closed(pts, 2.0) / side)

    outers, holes = [], []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if hierarchy[i][3] < 0:
            if area >= MIN_OUTER_PX2 and len(c) >= 6:
                outers.append((area, to_norm(c)))
        elif area >= MIN_HOLE_PX2 and len(c) >= 6:
            holes.append(to_norm(c))
    outers.sort(key=lambda t: -t[0])
    return [o for _, o in outers], holes


def _dominant_colors(pixels):
    """Deux couleurs dominantes de la région (le premier axe de leur analyse en composantes)."""
    data = np.ascontiguousarray(pixels, dtype=np.float32)
    if len(data) < 2 or np.allclose(data.min(axis=0), data.max(axis=0)):
        mean = data.mean(axis=0) if len(data) else np.zeros(3, np.float32)
        return mean, mean
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 0.5)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=2)
    order = np.argsort(counts)[::-1]
    return centers[order[0]], centers[order[1]]


def _resize_grid(field, edge=False):
    """Image (h, w) -> grille (GRID, GRID) SANS déformer : l'image occupe le coin haut-gauche
    du carré de côté max(h, w), comme `nor2img` du dépôt d'origine."""
    field = np.ascontiguousarray(field, dtype=np.float32)
    h, w = field.shape[:2]
    side = max(h, w)
    if (h, w) != (side, side):
        pad = ((0, side - h), (0, side - w))
        field = np.pad(field, pad, mode="edge" if edge else "constant")
    return cv2.resize(field, (GRID, GRID), interpolation=cv2.INTER_AREA)


DENSITY_SIGMA = 3.0     # en cases de grille


def _masked_blur(field, weight, sigma):
    num = cv2.GaussianBlur((field * weight).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), field).astype(np.float32)


def _density(patch_rgb, mask):
    """Densité de fils : 1 là où la région est dense (couleur du motif), 0.1 vers le fond."""
    c_main, c_other = _dominant_colors(patch_rgb[mask])
    axis = c_other.astype(float) - c_main.astype(float)
    norm = float(axis @ axis)
    if norm <= 1e-6:
        t = np.zeros(mask.shape, np.float32)
    else:
        t = (((patch_rgb.reshape(-1, 3).astype(float) - c_main) @ axis) / norm)
        t = np.clip(t, 0.0, 1.0).astype(np.float32).reshape(mask.shape)
    weight = _resize_grid(mask.astype(np.float32))
    t = _resize_grid(t * mask) / np.maximum(weight, 1e-3)    # moyenne masquée
    inside = weight > 0.5
    t = _masked_blur(t, weight, DENSITY_SIGMA)
    density = 0.1 + 0.9 * (1.0 - np.clip(t, 0.001, 0.999))
    if inside.any() and float(density[inside].mean()) > 0.7:   # leur échange fond / motif
        density = 1.0 - density
        c_main, c_other = c_other, c_main
    return density.astype(np.float32), inside, c_main, c_other


def _direction(gray, mask, inside):
    """Direction des streamlines : tangent aux isophotes, lissée à l'échelle de la région.

    Le champ est calculé directement sur la grille : c'est un champ lisse, le calculer à la
    résolution de travail ne ferait que dupliquer les mêmes valeurs.
    """
    gray = _resize_grid(gray, edge=True)          # remplace cv2.resize(...)
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(2.0, GRID / 45.0)
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), sigma)
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), sigma)
    # L'isophote est perpendiculaire au gradient : c'est la direction que suit le fil.
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy, dtype=np.float32) + np.float32(np.pi / 2.0)
    confidence = (jxx + jyy).astype(np.float32)
    weight = inside.astype(np.float32)
    norm = np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)
    su = cv2.GaussianBlur(np.cos(theta) * confidence, (0, 0), sigma) / norm
    sv = cv2.GaussianBlur(np.sin(theta) * confidence, (0, 0), sigma) / norm
    field = np.arctan2(sv, su, dtype=np.float32)
    # Région sans structure (aplat uniforme) : on suit son allongement, jamais du bruit.
    ys, xs = np.nonzero(_resize_grid(mask * 255) > 127)   # remplace le cv2.resize du repli
    if len(xs) > 2:
        cov = np.cov(np.vstack([xs.astype(float), ys.astype(float)]))
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, int(np.argmax(values))]
        fallback = float(np.arctan2(major[1], major[0]))
    else:
        fallback = 0.0
    flat = np.hypot(su, sv) < 1e-3
    field = np.where(flat, np.float32(fallback), field)
    return field.astype(np.float32)


def _nearest_palette(rgb, palette, fallback):
    """Couleur de la palette la plus proche : aucun fil hors de la limite demandée."""
    if not len(palette):
        return fallback
    colors = np.asarray(palette, float).reshape(-1, 3)
    distance = ((colors - np.asarray(rgb, float)) ** 2).sum(axis=1)
    return tuple(int(round(v)) for v in colors[int(np.argmin(distance))])


def load_patches(path, n_colors, background, background_tolerance, background_scope,
                 exclude=(), width_px=WORK_PX):
    """Image -> liste de régions prêtes à tracer.

    La quantification et la suppression du fond sont celles de l'application (mêmes réglages que
    les autres moteurs) : chaque masque de couleur devient une région à broder en streamlines.
    """
    rgb, keep = _embroidered_pixels(path, width_px, background=background, bg_tol=background_tolerance,
                                    bg_scope=background_scope, exclude=exclude)
    palette, masks = load_and_quantize(path, width_px, n_colors, background, background_tolerance,
                                       background_scope, exclude)
    patches = []
    for mask, color in zip(masks, palette):
        mask = np.asarray(mask, bool) & keep
        if mask.sum() < 40:
            continue
        boundaries, holes = _outline(mask)
        if not boundaries:
            continue
        gray = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        density, inside, c_main, c_other = _density(np.asarray(rgb, np.float32), mask)
        if inside.sum() < 40:
            continue
        main = tuple(int(round(v)) for v in color)          # couleur quantifiée de la région
        # Le fond croisé reste lui aussi dans la palette : la limite de couleurs du formulaire
        # est tenue, au lieu d'ajouter deux teintes par région.
        shades = _nearest_palette(c_other, palette, main)
        patches.append(Patch(
            color=main,
            color_bg=shades,
            mask=mask,
            inside=inside,
            density=density,
            direction=_direction(gray, mask, inside),
            boundary=boundaries[0],
            holes=holes,
            boundaries=boundaries,
        ))
    return patches
