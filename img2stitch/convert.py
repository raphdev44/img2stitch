"""Assemblage du motif et écriture du fichier via pyembroidery."""
import math
from dataclasses import replace

import cv2
import numpy as np
import pyembroidery as pe
from .config import Settings
from .preprocess import load_and_quantize, _embroidered_pixels
from .fill import fill_runs
from .outline import overlap_masks, contours_mm
from .planner import Sewer, plan_fill, plan_outline

SIZE_MARGIN_MM = 0.2  # les formats arrondissent les coordonnées au 1/10 mm : marge gardée dans le cadre


def _center_on_origin(pattern):
    """Translate le motif pour que le centre de sa boîte englobante soit en (0,0)."""
    pts = [(x, y) for x, y, c in pattern.stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)]
    if not pts:
        return
    xs, ys = zip(*pts)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for st in pattern.stitches:
        st[0] -= cx
        st[1] -= cy


def _stitch_size_mm(pattern):
    pts = np.array([(x, y) for x, y, c in pattern.stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)])
    if not len(pts):
        return 0.0, 0.0
    return tuple(float(v) / 10 for v in pts.max(axis=0) - pts.min(axis=0))


def _scale(pattern, f):
    for st in pattern.stitches:
        st[0] *= f
        st[1] *= f


def natural_size_mm(src, settings: Settings = None):
    """(largeur, hauteur) en mm du motif que le moteur produit sans contrainte de cadre (taille par défaut)."""
    s = replace(settings or Settings(), max_width_mm=None, max_height_mm=None)
    return _stitch_size_mm(_build_pattern(src, s))


def image_to_pattern(path, settings: Settings = None):
    """Image -> motif. Avec max_width_mm / max_height_mm, le motif est le plus grand possible
    tout en tenant strictement dans ce rectangle."""
    s = settings or Settings()
    if s.max_width_mm is None and s.max_height_mm is None:
        return _build_pattern(path, s)
    margin = SIZE_MARGIN_MM
    max_w = s.max_width_mm - margin if s.max_width_mm else math.inf
    max_h = s.max_height_mm - margin if s.max_height_mm else math.inf

    # 1re estimation : part de l'image occupée par la zone à broder (fond exclu)
    probe_px = 1000
    _, keep = _embroidered_pixels(path, probe_px, s.background, s.background_tolerance, s.background_scope,
                                  s.exclude_colors)
    if not keep.any():
        return _build_pattern(path, s)
    ys, xs = np.nonzero(keep)
    fw, fh = (np.ptp(xs) + 1) / probe_px, (np.ptp(ys) + 1) / probe_px
    width = min(max_w / fw, max_h / fh)

    # Ajustement sur les points réellement piqués (contours, chevauchements, petites zones ignorées...)
    for _ in range(4):
        pattern = _build_pattern(path, replace(s, width_mm=width))
        w, h = _stitch_size_mm(pattern)
        if not w and not h:
            return pattern
        ratio = min(max_w / w if w else math.inf, max_h / h if h else math.inf)
        if 0.98 < ratio < 1.02:  # à 2 % près : un léger dépassement est corrigé ci-dessous
            break
        width *= ratio * 0.998
    if ratio < 1:  # écart résiduel infime : on ramène strictement dans le cadre
        _scale(pattern, ratio * 0.999)
    return pattern


def _build_pattern(path, s: Settings):
    px_per_mm = 10  # résolution de travail : 10 px par mm
    palette, masks = load_and_quantize(
        path, round(s.width_mm * px_per_mm), s.n_colors, s.background, s.background_tolerance, s.background_scope,
        s.exclude_colors
    )
    pattern = pe.EmbPattern()
    if not masks:
        pattern.end()
        return pattern

    fill_masks = overlap_masks(masks, px_per_mm, s.overlap_mm)

    # later[i] = tout ce qui sera piqué APRÈS la couleur i : un fil qui passe là sera recouvert
    later = [None] * len(masks)
    acc = np.zeros_like(masks[0])
    for i in range(len(masks) - 1, -1, -1):
        later[i] = acc.copy()
        acc |= masks[i]

    sw = Sewer(pattern, s, px_per_mm)
    first = True
    tol = np.ones((3, 3), np.uint8)  # tolérance d'1 px : les contours sont sur le bord des formes
    for i, (rgb, mask, fmask) in enumerate(zip(palette, masks, fill_masks)):
        runs = fill_runs(fmask, px_per_mm, s.row_spacing_mm, s.fill_angle_deg, s.min_area_mm2)
        loops = contours_mm(mask, px_per_mm, s.outline_simplify_mm, s.min_area_mm2) if s.outline else []
        if not runs and not loops:
            continue
        if not first:
            pattern.add_command(pe.COLOR_CHANGE)
        first = False
        pattern.add_thread({"color": (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]})

        hidden = cv2.dilate((fmask | later[i]).astype(np.uint8), tol).astype(bool)
        sw.new_color(hidden, later[i])
        plan_fill(sw, runs)
        if loops:
            plan_outline(sw, loops, s.outline_passes)
    pattern.end()
    if s.center_on_origin:
        _center_on_origin(pattern)
    return pattern


def pattern_stats(pattern):
    """Chiffres utiles pour juger de la qualité du parcours."""
    st, n = pattern.stitches, {"stitches": 0, "jumps": 0, "trims": 0, "color_changes": 0}
    prev = None
    for _, _, c in st:
        c &= pe.COMMAND_MASK
        if c == pe.STITCH:
            n["stitches"] += 1
        elif c == pe.JUMP and prev not in (pe.COLOR_CHANGE, None):
            n["jumps"] += 1
        elif c == pe.TRIM:
            n["trims"] += 1
        elif c == pe.COLOR_CHANGE:
            n["color_changes"] += 1
        prev = c
    return n


def convert(src, dst, settings: Settings = None):
    """Convertit `src` (png/jpg) en `dst` ; le format est déduit de l'extension (.dst, .pes, .jef, ...)."""
    s = settings or Settings()
    pattern = image_to_pattern(src, s)
    opts = {"tie_on": True, "tie_off": True} if s.lock_stitches else {}
    pe.write(pattern, dst, opts)
    return pattern
