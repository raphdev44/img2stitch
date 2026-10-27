"""Assemblage du motif et écriture du fichier via pyembroidery."""
import cv2
import numpy as np
import pyembroidery as pe
from .config import Settings
from .preprocess import load_and_quantize
from .fill import fill_runs
from .outline import overlap_masks, contours_mm
from .planner import Sewer, plan_fill, plan_outline


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


def image_to_pattern(path, settings: Settings = None):
    s = settings or Settings()
    px_per_mm = 10  # résolution de travail : 10 px par mm
    palette, masks = load_and_quantize(
        path, round(s.width_mm * px_per_mm), s.n_colors, s.background, s.background_tolerance
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
