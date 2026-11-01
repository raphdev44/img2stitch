"""Assemblage d'un motif brodé à partir des streamlines d'une photo.

Reprend la post-production de `embroidery-streamlines` : pour chaque région, un passage
« fond » croisé à 90° puis le passage principal qui suit la direction du dessin
(`run_multiple_patches.py`), les lignes rééchantillonnées au pas de `scale_and_resample`, puis
mises bout à bout (`connecting.connect_lines`).

 Leur graphe de connexion (arbre couvrant minimal puis circuit eulérien) sert à obtenir UN seul
 tracé continu. Ici, les lignes sont simplement enchaînées au plus proche et le fil est coupé quand
 le raccord dépasserait `CONNECT_MM` : le parcours reste propre, et les coupes restent lisibles
 dans l'aperçu.

Le dépôt d'origine s'appuie sur `triangle`, `scipy.sparse`, `networkx`, `numba`, `scikit-image`
et des annotations Labelme ; ici tout est réécrit avec numpy et OpenCV, les seules dépendances
déjà nécessaires à l'application, et la segmentation vient de notre quantification de couleurs.
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
PASS_UNITS = 6          # unités de progression d'un passage, hors itérations de relaxation


def _pass_lines(patch, relative, step, step_size, iterations, density=None, direction=None,
                on_progress=None):
    """Un passage complet sur une région : sources, tracé, coupe, lissage, rééchantillonnage.

    `on_progress(fraction, étape)` suit le passage ; la relaxation, qui prend l'essentiel du
    temps, progresse itération par itération.
    """
    density = patch.density if density is None else density
    direction = patch.direction if direction is None else direction
    iterations = max(1, int(iterations))
    units = PASS_UNITS + iterations
    done = 0

    def report(unit, stage):
        nonlocal done
        done = max(done, unit)
        if on_progress:
            on_progress(done / units, stage)

    report(0, "sources du champ")
    z = tracing.field_vectors(density, direction)
    sources, sinks = tracing.sources_and_sinks(density, direction, patch.inside,
                                               patch.boundaries or patch.boundary,
                                               patch.holes, relative)
    report(1, "tracé des lignes de courant")
    lines = tracing.trace(sources, z, patch.inside, step=step, step_size=step_size)
    report(3, "coupe aux puits")
    lines = tracing.crop_inside(tracing.cut_at_sinks(lines, sinks), patch.inside)
    lines = tracing.drop_short(lines)
    report(5, "lissage par ressorts")
    lines = relax.smooth(
        lines, direction, patch.inside, density, relative, iterations,
        on_progress=(lambda k, n: report(5 + k, f"lissage par ressorts ({k}/{n})"))
        if on_progress else None)
    lines = tracing.resample(lines, relative * SPACING_FACTOR)
    report(units, "terminé")
    return lines


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
            # add_command place la commande en (0,0) : l'aiguille doit couper sur place, sinon
            # le tracé et les lecteurs repartent du coin du motif (long fil en diagonale).
            pattern.add_stitch_absolute(pe.TRIM, previous[0] * UNIT, previous[1] * UNIT)
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
                       relax_iterations=relax.ITERATIONS, color_overrides=(),
                       progress_callback=None):
    """Photo -> motif de broderie en streamlines. Lève ValueError si rien n'est brodable.

    `progress_callback` est un appel progress_callback(current_step, total_steps, message)
    appelé souvent pendant le calcul : à chaque passe et à chaque itération de relaxation.
    """

    def report(fraction, message):
        if progress_callback:
            progress_callback(int(round(1000 * min(max(fraction, 0.0), 1.0))), 1000, message)

    report(0.01, "Chargement des patches et quantification des couleurs")
    patches = fields.load_patches(path, n_colors, background, background_tolerance,
                                  background_scope, exclude)
    if not patches:
        raise ValueError("Aucune zone à broder : l'image est vide une fois le fond retiré.")

    report(0.04, "Construction des champs de direction et de densité")
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

    # Passage « fond » croisé puis passage « motif » de chaque région. Chaque passe pèse selon sa
    # surface : la relaxation, l'essentiel du temps, travaille sur tous les points de la région.
    iterations = max(1, int(relax_iterations))
    passes = []
    for index, patch in enumerate(patches, 1):
        label = f"Région {index}/{len(patches)}"
        if background_fill:
            passes.append((patch, f"{label} · fond", np.ones_like(patch.density),
                           patch.direction + np.float32(np.pi / 2.0)))
        passes.append((patch, f"{label} · motif", None, None))
    # Le passage « motif » ne remplit que selon sa densité : il produit donc bien moins de points
    # que le passage « fond », et doit peser proportionnellement moins dans la barre.
    weights = []
    for patch, _, dens, _ in passes:
        share = 1.0 if dens is not None else max(0.15, float(patch.density[patch.inside].mean()))
        weights.append((PASS_UNITS + iterations) * max(1.0, float(patch.mask.sum()) * share))
    total_weight = sum(weights) or 1.0

    groups = []                               # [(couleur, lignes normalisées)] dans l'ordre de piqué
    done_weight = 0.0
    for (patch, label, dens, direc), weight in zip(passes, weights):
        def on_progress(fraction, stage, label=label, base=done_weight, weight=weight):
            report(0.05 + 0.80 * (base + fraction * weight) / total_weight, f"{label} · {stage}")

        lines = _pass_lines(patch, relative, step, step_size, iterations,
                            density=dens, direction=direc, on_progress=on_progress)
        done_weight += weight
        if lines:
            groups.append((patch.color_bg if dens is not None else patch.color, lines))

    # Une seule couleur de fil par teinte : la limite demandée est respectée, et deux passages
    # de même couleur ne déclenchent pas un changement de fil inutile.
    merged, slot = [], {}
    for color, lines in groups:
        if color not in slot:
            slot[color] = len(merged)
            merged.append((color, list(lines)))
        else:
            merged[slot[color]][1].extend(lines)
    groups = merged
    if not groups:
        raise ValueError("Les streamlines n'ont produit aucune ligne : élargissez le motif ou "
                         "diminuez la largeur du fil.")

    report(0.88, "Enchaînement des lignes au plus proche")
    pattern = pe.EmbPattern()
    zones, segments, previous = [], [], None
    total_lines = 0
    for color, lines in groups:
        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES]
        mm = [(np.asarray(line, float) * size) for line in lines]
        total_lines += len(mm)
        rgb = tuple(int(max(0, min(255, c))) for c in color)
        rgb = dict(color_overrides).get(rgb, rgb)     # couleur de fil choisie par l'utilisateur
        pattern.add_thread({"color": (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]})
        if zones:   # add_thread ne pique rien : sans ce changement, tout sortirait d'un seul fil
            pattern.add_stitch_absolute(pe.COLOR_CHANGE, previous[0] * UNIT, previous[1] * UNIT)
        zones.append((rgb, None))
        segments.append((len(pattern.stitches), len(zones) - 1))
        previous = _emit(pattern, mm, previous)
    pattern.end()

    report(0.97, "Centrage et finition du motif")
    _center_on_origin_pattern(pattern)
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_streamline = {"patches": len(patches), "lines": total_lines,
                                     "size_mm": round(size, 1),
                                     "line_width_mm": round(line_width_mm, 2)}
    pattern.img2stitch_frame = {"w": width_px, "h": height_px, "ppm": side / size,
                                "cx": 0.0, "cy": 0.0, "scale": 1.0}
    report(1.0, "Terminé")
    return pattern


def _center_on_origin_pattern(pattern, keep_jumps=True):
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