"""Assemblage d'un motif brodé à partir des streamlines d'une photo.

Reprend la post-production de `embroidery-streamlines` : pour chaque région, un passage
« fond » croisé à 90° puis le passage principal qui suit la direction du dessin
(`run_multiple_patches.py`), les lignes rééchantillonnées au pas de `scale_and_resample`, puis
mises bout à bout (`connecting.connect_lines`).

Leur graphe de connexion (arbre couvrant minimal puis circuit eulérien) sert à obtenir UN seul
tracé continu. Ici, les lignes sont simplement enchaînées au plus proche et le fil est coupé quand
le raccord dépasserait `CONNECT_MM` : le parcours reste propre, et les coupes restent lisibles
dans l'aperçu.
"""
import math

import numpy as np
import pyembroidery as pe

from . import fields, relax, tracing

UNIT = 10               # pyembroidery travaille en 1/10 de mm
STITCH_MM = 1.3         # leur `subsample(..., 13)` : un point tous les 1,3 mm
SPACING_FACTOR = 4.0    # `scale_and_resample(..., relative_line_width * 4)`
CONNECT_MM = 3.0        # au-delà, coupe de fil + saut au lieu d'un raccord piqué
DEFAULT_SIZE_MM = 99.0  # leur `target_total_physical_size_mm`
LINE_WIDTH_MM = 0.4     # leur `physical_line_width_mm`
MAX_LINES = 4000


def _pass_lines(patch, relative, step, step_size, iterations, density=None, direction=None):
    """Un passage complet sur une région : sources, tracé, coupe, lissage, rééchantillonnage."""
    density = patch.density if density is None else density
    direction = patch.direction if direction is None else direction
    z = tracing.field_vectors(density, direction)
    sources, sinks = tracing.sources_and_sinks(density, direction, patch.inside, patch.boundary,
                                               patch.holes, relative)
    lines = tracing.trace(sources, z, patch.inside, step=step, step_size=step_size)
    lines = tracing.crop_inside(tracing.cut_at_sinks(lines, sinks), patch.inside)
    lines = tracing.drop_short(lines)
    lines = relax.smooth(lines, direction, patch.inside, density, relative, iterations)
    return tracing.resample(lines, relative * SPACING_FACTOR)


def _order(lines):
    """Enchaîne les lignes au plus proche, en les retournant quand c'est plus court."""
    rest = [np.asarray(line, float) for line in lines]
    ordered, current = [], None
    while rest:
        best = None
        for index, line in enumerate(rest):
            for flip, point in ((False, line[0]), (True, line[-1])):
                distance = 0.0 if current is None else float(np.hypot(*(point - current)))
                if best is None or distance < best[0]:
                    best = (distance, index, flip)
        _, index, flip = best
        line = rest.pop(index)
        if flip:
            line = line[::-1]
        ordered.append(line)
        current = line[-1]
    return ordered


def _emit(pattern, lines, previous):
    """Pique les lignes en mm dans le motif ; retourne la dernière position de l'aiguille."""
    for line in _order(lines):
        start = line[0]
        if previous is None:
            pattern.add_stitch_absolute(pe.JUMP, start[0] * UNIT, start[1] * UNIT)
        elif math.hypot(start[0] - previous[0], start[1] - previous[1]) > CONNECT_MM:
            pattern.add_command(pe.TRIM)
            pattern.add_stitch_absolute(pe.JUMP, start[0] * UNIT, start[1] * UNIT)
        elif math.hypot(start[0] - previous[0], start[1] - previous[1]) > 0.05:
            pattern.add_stitch_absolute(pe.STITCH, start[0] * UNIT, start[1] * UNIT)
        for x, y in line[1:]:
            pattern.add_stitch_absolute(pe.STITCH, x * UNIT, y * UNIT)
        previous = line[-1]
    return previous


def streamline_pattern(path, *, n_colors, background="auto", background_tolerance=40,
                       background_scope="all", exclude=(), line_width_mm=LINE_WIDTH_MM,
                       target_mm=None, background_fill=True, step=1800, step_size=0.001,
                       relax_iterations=relax.ITERATIONS):
    """Photo -> motif de broderie en streamlines. Lève ValueError si rien n'est brodable."""
    patches = fields.load_patches(path, n_colors, background, background_tolerance,
                                  background_scope, exclude)
    if not patches:
        raise ValueError("Aucune zone à broder : l'image est vide une fois le fond retiré.")

    height_px, width_px = patches[0].mask.shape
    side = float(max(width_px, height_px))     # leur normalisation par le plus grand côté
    width_n, height_n = width_px / side, height_px / side
    if target_mm:
        limit_w, limit_h = (max(1.0, float(v)) for v in target_mm)
        size = min(limit_w / width_n, limit_h / height_n)
    else:
        size = DEFAULT_SIZE_MM
    size = max(10.0, float(size))
    line_width_mm = min(2.0, max(0.15, float(line_width_mm)))
    relative = line_width_mm / size           # leur `relative_line_width`

    groups = []                               # [(couleur, lignes normalisées)] dans l'ordre de piqué
    for patch in patches:
        if background_fill:
            crossed = _pass_lines(patch, relative, step, step_size, relax_iterations,
                                  density=np.ones_like(patch.density),
                                  direction=patch.direction + np.float32(np.pi / 2.0))
            if crossed:
                groups.append((patch.color_bg, crossed))
        lines = _pass_lines(patch, relative, step, step_size, relax_iterations)
        if lines:
            groups.append((patch.color, lines))
    if not groups:
        raise ValueError("Les streamlines n'ont produit aucune ligne : élargissez le motif ou "
                         "diminuez la largeur du fil.")

    pattern = pe.EmbPattern()
    zones, segments, previous = [], [], None
    total_lines = 0
    for color, lines in groups:
        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES]
        mm = [(np.asarray(line, float) * size) for line in lines]
        total_lines += len(mm)
        rgb = tuple(int(max(0, min(255, c))) for c in color)
        pattern.add_thread({"color": (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]})
        if zones:   # add_thread ne pique rien : sans ce changement, tout sortirait d'un seul fil
            pattern.add_command(pe.COLOR_CHANGE)
        zones.append((rgb, None))
        segments.append((len(pattern.stitches), len(zones) - 1))
        previous = _emit(pattern, mm, previous)
    pattern.end()

    _center_on_origin(pattern)
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_streamline = {"patches": len(patches), "lines": total_lines,
                                     "size_mm": round(size, 1),
                                     "line_width_mm": round(line_width_mm, 2)}
    pattern.img2stitch_frame = {"w": width_px, "h": height_px, "ppm": side / size,
                                "cx": 0.0, "cy": 0.0, "scale": 1.0}
    return pattern


def _center_on_origin(pattern, keep_jumps=True):
    """Centre le motif sur (0,0), centre du cerceau, sans toucher aux commandes."""
    points = [(x, y) for x, y, c in pattern.stitches
              if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)]
    if not points:
        return 0.0, 0.0
    xs, ys = zip(*points)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for stitch in pattern.stitches:
        stitch[0] -= cx
        stitch[1] -= cy
    return cx, cy


def thread_count(patches):
    return len(patches) * 2
