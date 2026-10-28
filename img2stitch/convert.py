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
        s.exclude_colors, thicken_lines=s.thicken_lines
    )
    pattern = pe.EmbPattern()
    if not masks:
        pattern.end()
        return pattern

    frame = {"w": masks[0].shape[1], "h": masks[0].shape[0], "ppm": px_per_mm, "cx": 0.0, "cy": 0.0, "scale": 1.0}
    palette, masks, regions = _split_regions(palette, masks, s.region_overrides, px_per_mm)
    # (couleur de fil, motif de remplissage, point de contour ou None = réglage global)
    looks = [_zone_override(rgb, s.zone_overrides) if regions[i] is None
             else _region_look(s.region_overrides[regions[i]], rgb)
             for i, rgb in enumerate(palette)]
    if s.merge_same_colors:
        order = _merge_order([c for c, _, _ in looks], masks, px_per_mm, s.overlap_mm)
        palette, masks, regions, looks = ([lst[k] for k in order] for lst in (palette, masks, regions, looks))
    fill_masks = overlap_masks(masks, px_per_mm, s.overlap_mm)

    # later[i] = tout ce qui sera piqué APRÈS la couleur i : un fil qui passe là sera recouvert
    later = [None] * len(masks)
    acc = np.zeros_like(masks[0])
    for i in range(len(masks) - 1, -1, -1):
        later[i] = acc.copy()
        acc |= masks[i]

    sw = Sewer(pattern, s, px_per_mm)
    first, thread = True, None
    zones = []  # (couleur d'origine, index de la retouche de zone ou None) de chaque zone, dans l'ordre de piqué
    segments = []  # (index du premier point, index de zone) : plusieurs zones peuvent partager un même fil
    tol = np.ones((3, 3), np.uint8)  # tolérance d'1 px : les contours sont sur le bord des formes
    for i, (rgb, mask, fmask) in enumerate(zip(palette, masks, fill_masks)):
        color, kind, outline = looks[i]
        passes = zone_fill(kind, fmask, px_per_mm, s)
        # Le contour se règle zone par zone (None : on suit le réglage global)
        loops = (contours_mm(mask, px_per_mm, s.outline_simplify_mm, s.min_area_mm2)
                 if (s.outline if outline is None else outline) else [])
        # La zone entre au catalogue même si elle ne pique rien (remplissage ET contour coupés) :
        # elle reste modifiable dans l'éditeur, qui doit retrouver une entrée par couleur.
        zone_index = len(zones)
        zones.append((tuple(rgb), regions[i]))
        if not any(data for _, data in passes) and not loops:
            continue
        # Un bloc qui reprend le fil de la zone précédente n'a pas besoin d'arrêt machine ; c'est le cas
        # des bandes fines isolées en zones (lettrage, rubans) et des couleurs réunies par _merge_order.
        same = color == thread
        if not same:
            if not first:
                pattern.add_command(pe.COLOR_CHANGE)
            pattern.add_thread({"color": (color[0] << 16) | (color[1] << 8) | color[2]})
        first, thread = False, color
        segments.append((len(pattern.stitches), zone_index))

        hidden = cv2.dilate((fmask | later[i]).astype(np.uint8), tol).astype(bool)
        sw.new_color(hidden, later[i])
        if same:
            sw.fresh = False  # vrai déplacement vers cette zone : piqué à couvert, ou coupe + saut
        for what, data in passes:
            if what == "runs":
                plan_fill(sw, data)
            elif data:
                plan_outline(sw, data, 1)
        if loops:
            plan_outline(sw, loops, s.outline_passes)
    pattern.end()
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    if s.center_on_origin:
        frame["cx"], frame["cy"] = _center_on_origin(pattern)
    pattern.img2stitch_frame = frame
    return pattern


def _merge_order(colors, masks, px_per_mm, overlap_mm):
    """Ordre de piqué qui rapproche les zones de même fil, sans changer la superposition : une zone n'est
    avancée que si elle ne touche aucune des zones qu'elle dépasse (sinon le dessus et le dessous s'inverseraient)."""
    r = max(1, round(max(overlap_mm, 0.1) * px_per_mm)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks]
    touch = lambda a, b: bool((grown[a] & masks[b]).any())
    remaining, order = list(range(len(masks))), []
    while remaining:
        order.append(remaining.pop(0))
        j = 0
        while j < len(remaining):
            k = remaining[j]
            if colors[k] == colors[order[-1]] and not any(touch(k, m) for m in remaining[:j]):
                order.append(remaining.pop(j))
            else:
                j += 1
    return order


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
    for ri, override in enumerate(region_overrides):
        (u, v), source = override[0], override[1]
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


def _override_parts(override):
    """(source, fil, motif, contour) d'une retouche ; le contour est optionnel (anciens projets)."""
    return (override[0], override[1], override[2], override[3] if len(override) > 3 else None)


def _zone_override(rgb, overrides, tol=40.0):
    """(couleur de fil, motif de remplissage, point de contour) d'une zone, d'après la retouche dont la
    couleur d'origine est la plus proche. Le contour vaut None quand la zone suit le réglage global."""
    best, best_d = None, tol
    for override in overrides:
        source, color, kind, outline = _override_parts(override)
        d = math.dist(source, rgb)
        if d <= best_d:
            best, best_d = (color, kind, outline), d
    if best is None:
        return tuple(rgb), "tatami", None
    return tuple(best[0] or rgb), best[1] or "tatami", best[2]


def _region_look(override, rgb):
    """(couleur, motif, contour) d'une zone fermée isolée : ((u, v), source, fil, motif[, contour])."""
    return (tuple(override[2] or rgb), override[3] or "tatami", override[4] if len(override) > 4 else None)


THREAD_LOOP_FACTOR = 1.3  # boucles, tension et reprise du fil au ras du tissu
THREAD_SETUP_MM = 80      # amorçage et coupe, pour chaque fil


def thread_lengths_mm(pattern, factor=THREAD_LOOP_FACTOR, setup_mm=THREAD_SETUP_MM):
    """Longueur de fil estimée (mm) par fil, dans l'ordre de threadlist.

    Somme des segments réellement piqués par ce fil, majorée des boucles et de la tension,
    plus l'amorçage et la coupe. Les sauts et les coupes ne sont pas comptés : le fil y est
    tiré puis coupé, sans être posé sur le tissu.
    """
    count = len(pattern.threadlist)
    if not count:
        return []
    lengths = [0.0] * count
    index, last = 0, None
    for x, y, command in pattern.stitches:
        kind = command & pe.COMMAND_MASK
        if kind == pe.COLOR_CHANGE:
            index = min(index + 1, count - 1)
            last = None
        elif kind == pe.STITCH:
            if last is not None:
                lengths[index] += math.hypot(x - last[0], y - last[1]) / 10.0
            last = (x, y)
        else:
            last = (x, y)
    return [round(value * factor + setup_mm) for value in lengths]


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
