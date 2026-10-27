"""Assemblage du motif et écriture du fichier via pyembroidery."""
import math
from dataclasses import replace

import cv2
import numpy as np
import pyembroidery as pe
from .config import Settings
from .preprocess import load_and_quantize, _embroidered_pixels
from .fill import fill_runs, zone_fill
from .outline import overlap_masks, contours_mm
from .planner import Sewer, plan_fill, plan_outline

SIZE_MARGIN_MM = 0.2  # les formats arrondissent les coordonnées au 1/10 mm : marge gardée dans le cadre


def _center_on_origin(pattern):
    """Translate le motif pour que le centre de sa boîte englobante soit en (0,0). Retourne la translation."""
    pts = [(x, y) for x, y, c in pattern.stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)]
    if not pts:
        return 0.0, 0.0
    xs, ys = zip(*pts)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for st in pattern.stitches:
        st[0] -= cx
        st[1] -= cy
    return cx, cy


def _stitch_size_mm(pattern):
    pts = np.array([(x, y) for x, y, c in pattern.stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)])
    if not len(pts):
        return 0.0, 0.0
    return tuple(float(v) / 10 for v in pts.max(axis=0) - pts.min(axis=0))


def _scale(pattern, f):
    for st in pattern.stitches:
        st[0] *= f
        st[1] *= f
    frame = getattr(pattern, "img2stitch_frame", None)
    if frame:
        frame["scale"] *= f


def pattern_to_image(frame, x, y):
    """Point du motif (1/10 mm) -> position (u, v) dans l'image source, en fraction de sa largeur / hauteur."""
    px = (x / frame["scale"] + frame["cx"]) * frame["ppm"] / 10
    py = (y / frame["scale"] + frame["cy"]) * frame["ppm"] / 10
    return px / frame["w"], py / frame["h"]


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

    frame = {"w": masks[0].shape[1], "h": masks[0].shape[0], "ppm": px_per_mm, "cx": 0.0, "cy": 0.0, "scale": 1.0}
    palette, masks, regions = _split_regions(palette, masks, s.region_overrides, px_per_mm)
    fill_masks = overlap_masks(masks, px_per_mm, s.overlap_mm)

    # later[i] = tout ce qui sera piqué APRÈS la couleur i : un fil qui passe là sera recouvert
    later = [None] * len(masks)
    acc = np.zeros_like(masks[0])
    for i in range(len(masks) - 1, -1, -1):
        later[i] = acc.copy()
        acc |= masks[i]

    sw = Sewer(pattern, s, px_per_mm)
    first = True
    zones = []  # (couleur d'origine, index de la retouche de zone ou None) de chaque bloc, dans l'ordre de piqué
    tol = np.ones((3, 3), np.uint8)  # tolérance d'1 px : les contours sont sur le bord des formes
    for i, (rgb, mask, fmask) in enumerate(zip(palette, masks, fill_masks)):
        if regions[i] is None:
            color, kind = _zone_override(rgb, s.zone_overrides)
        else:
            _, _, r_color, r_kind = s.region_overrides[regions[i]]
            color, kind = tuple(r_color or rgb), r_kind or "tatami"
        passes = zone_fill(kind, fmask, px_per_mm, s)
        loops = contours_mm(mask, px_per_mm, s.outline_simplify_mm, s.min_area_mm2) if s.outline else []
        if not any(data for _, data in passes) and not loops:
            continue
        if not first:
            pattern.add_command(pe.COLOR_CHANGE)
        first = False
        pattern.add_thread({"color": (color[0] << 16) | (color[1] << 8) | color[2]})
        zones.append((tuple(rgb), regions[i]))

        hidden = cv2.dilate((fmask | later[i]).astype(np.uint8), tol).astype(bool)
        sw.new_color(hidden, later[i])
        for what, data in passes:
            if what == "runs":
                plan_fill(sw, data)
            elif data:
                plan_outline(sw, data, 1)
        if loops:
            plan_outline(sw, loops, s.outline_passes)
    pattern.end()
    pattern.img2stitch_zones = zones
    if s.center_on_origin:
        frame["cx"], frame["cy"] = _center_on_origin(pattern)
    pattern.img2stitch_frame = frame
    return pattern


def _split_regions(palette, masks, region_overrides, px_per_mm, tol=40.0, search_mm=0.8):
    """Isole chaque zone fermée visée par une retouche de zone ((u, v), couleur d'origine, fil, motif) :
    la forme connexe de cette couleur qui contient le point (u, v) devient un bloc à part, piqué juste après
    le reste de sa couleur. Retourne (palette, masques, index de retouche ou None pour chaque bloc)."""
    if not region_overrides:
        return palette, masks, [None] * len(masks)
    h, w = masks[0].shape
    rest = [m.copy() for m in masks]
    extra = [[] for _ in masks]
    r = max(1, round(search_mm * px_per_mm))
    for ri, ((u, v), source, _, _) in enumerate(region_overrides):
        d = [math.dist(source, c) for c in palette]
        k = int(np.argmin(d))
        if d[k] > tol:
            continue
        x, y = min(w - 1, max(0, int(u * w))), min(h - 1, max(0, int(v * h)))
        _, labels = cv2.connectedComponents(rest[k].astype(np.uint8), connectivity=8)
        lab = labels[y, x]
        if not lab:  # point cliqué entre deux rangées ou sur un bord : forme la plus proche, tout près
            win = labels[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            ys, xs = np.nonzero(win)
            if not len(ys):
                continue
            j = int(np.argmin((ys - min(y, r)) ** 2 + (xs - min(x, r)) ** 2))
            lab = win[ys[j], xs[j]]
        comp = labels == lab
        rest[k] &= ~comp
        extra[k].append((comp, ri))
    out_p, out_m, out_r = [], [], []
    for k, rgb in enumerate(palette):
        for m, ri in [(rest[k], None)] + extra[k]:
            if m.any():
                out_p.append(rgb)
                out_m.append(m)
                out_r.append(ri)
    return out_p, out_m, out_r


def _zone_override(rgb, overrides, tol=40.0):
    """(couleur de fil, motif de remplissage) d'une zone, d'après la retouche dont la couleur d'origine est la plus proche."""
    best, best_d = None, tol
    for source, color, kind in overrides:
        d = math.dist(source, rgb)
        if d <= best_d:
            best, best_d = (color, kind), d
    if best is None:
        return tuple(rgb), "tatami"
    return tuple(best[0] or rgb), best[1] or "tatami"


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
