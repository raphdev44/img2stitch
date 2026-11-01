"""Numérisation de grilles pixel art, sans remplissage ni redimensionnement."""
from collections import Counter

import cv2
import numpy as np
import pyembroidery as pe
from PIL import Image, ImageOps

from .algorithm_standard import _center_on_origin
from .preprocess import _nearest, _reduce_colors, _to_lab

CELL_MM = 2.0
MAX_CELLS = 40_000
MAX_SIDE_CELLS = 200
MAX_COLORS = 256
MAX_SOURCE_COLORS = 20


def _grid_lines(rgb, axis):
    ink = (255 - rgb.astype(np.float32).mean(axis=2)).mean(axis=axis)
    ink = cv2.morphologyEx(ink.astype(np.float32)[None, :], cv2.MORPH_TOPHAT,
                           np.ones((1, 7), np.uint8))[0]
    low, high = np.percentile(ink, [20, 98])
    if high - low < 8:
        return None
    indices = np.flatnonzero(ink > low + (high - low) * 0.45)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    centers = np.array([group.mean() for group in groups if 0 < len(group) <= 4])
    if len(centers) < 5:
        return None
    gaps = np.diff(centers)
    pitch = float(np.median(gaps))
    if pitch < 5:
        return None
    origin = float(centers[0])
    for _ in range(3):
        positions = np.rint((centers - origin) / pitch)
        residuals = np.abs(centers - (origin + positions * pitch))
        aligned = residuals <= max(1.5, pitch * 0.15)
        if len(np.unique(positions[aligned])) < 5:
            return None
        pitch, origin = np.polyfit(positions[aligned], centers[aligned], 1)
    positions = np.rint((centers - origin) / pitch)
    aligned = np.abs(centers - (origin + positions * pitch)) <= max(1.5, pitch * 0.1)
    matched = np.unique(positions[aligned])
    if len(matched) < 5 or aligned.mean() < 0.65:
        return None
    count = int(matched[-1] - matched[0]) + 1
    if len(matched) / count < 0.5:
        return None
    lines = origin + np.arange(matched[0], matched[-1] + 1) * pitch
    return lines, float(pitch)


def _block_lines(rgba, axis):
    pixels = rgba.astype(np.float32)
    pixels[:, :, :3] *= pixels[:, :, 3:] / 255
    transitions = np.max(np.abs(np.diff(pixels, axis=1 - axis)), axis=2) > 24
    strength = transitions.sum(axis=axis)
    indices = np.flatnonzero(strength >= max(3, strength.max(initial=0) * 0.04))
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    groups = [group for group in groups if 0 < len(group) <= 5]
    if not 4 <= len(groups) <= MAX_SIDE_CELLS:
        return None
    centers = np.array([np.average(group + 1, weights=strength[group]) for group in groups])
    weights = np.array([strength[group].sum() for group in groups])
    gaps = np.diff(centers)
    candidates = sorted({round(float(gap / divisor), 3) for gap in gaps
                         for divisor in range(1, 9) if gap / divisor >= 3}, reverse=True)
    for candidate in candidates:
        pitch, origin = candidate, centers[np.argmax(weights)]
        for _ in range(3):
            positions = np.rint((centers - origin) / pitch)
            aligned = np.abs(centers - origin - positions * pitch) <= max(0.8, pitch * 0.06)
            if len(np.unique(positions[aligned])) < 4:
                break
            pitch, origin = np.polyfit(positions[aligned], centers[aligned], 1,
                                      w=np.sqrt(weights[aligned]))
        else:
            positions = np.rint((centers - origin) / pitch)
            aligned = np.abs(centers - origin - positions * pitch) <= max(0.8, pitch * 0.06)
            if pitch < 3 - 1e-6 or aligned.mean() < 0.9 or weights[aligned].sum() < weights.sum() * 0.95:
                continue
            size = rgba.shape[1 - axis]
            start = origin + np.floor(-origin / pitch + 0.08) * pitch
            count = int(np.ceil((size - start) / pitch - 0.08))
            lines = start + np.arange(count + 1) * pitch
            lines[0], lines[-1] = 0, size
            return lines, float(pitch)
    return None


def read_cells(source):
    """Extrait les cases d'une grille, les blocs agrandis ou les pixels natifs."""
    if not isinstance(source, Image.Image):
        with Image.open(source) as image:
            return read_cells(image)
    rgba = np.array(ImageOps.exif_transpose(source).convert("RGBA"))
    rgb = rgba[:, :, :3]
    vertical, horizontal = _grid_lines(rgb, 0), _grid_lines(rgb, 1)
    if vertical is None or horizontal is None:
        vertical, horizontal = _block_lines(rgba, 0), _block_lines(rgba, 1)
    if vertical is not None and horizontal is not None:
        xs, dx = vertical
        ys, dy = horizontal
        if abs(dx - dy) > max(2, min(dx, dy) * 0.15):
            raise ValueError("Les cases du quadrillage doivent être carrées.")
        columns, rows = len(xs) - 1, len(ys) - 1
        if max(columns, rows) > MAX_SIDE_CELLS:
            raise ValueError("La grille ne doit pas dépasser 200 cases par côté (400 mm).")
        cells = np.empty((rows, columns, 4), dtype=np.uint8)
        for row in range(rows):
            for col in range(columns):
                cx, cy = (xs[col] + xs[col + 1]) / 2, (ys[row] + ys[row + 1]) / 2
                rx = max(0, int((xs[col + 1] - xs[col]) * 0.2))
                ry = max(0, int((ys[row + 1] - ys[row]) * 0.2))
                patch = rgba[max(0, round(cy) - ry):round(cy) + ry + 1,
                             max(0, round(cx) - rx):round(cx) + rx + 1]
                cells[row, col] = np.median(patch, axis=(0, 1)).astype(np.uint8)
    else:
        cells = rgba
        if cells.shape[0] * cells.shape[1] > MAX_CELLS:
            raise ValueError("Quadrillage ou blocs de pixels non détectés. Envoyez une grille régulière, un pixel art à blocs nets ou à sa résolution native.")
    if cells.shape[0] * cells.shape[1] > MAX_CELLS:
        raise ValueError("Le motif dépasse la limite de 40 000 cases.")
    if max(cells.shape[:2]) > MAX_SIDE_CELLS:
        raise ValueError("La grille ne doit pas dépasser 200 cases par côté (400 mm).")
    return cells


def _limit_cell_colors(cells):
    visible = cells[:, :, 3] >= 128
    colors = cells[:, :, :3][visible]
    if len(np.unique(colors, axis=0)) <= MAX_SOURCE_COLORS:
        return cells
    centers, groups, palette = _reduce_colors(colors, MAX_SOURCE_COLORS, min_share=0)
    reduced = np.asarray(palette, dtype=np.uint8)[groups[_nearest(_to_lab(colors), centers)]]
    cells = cells.copy()
    cells[visible, :3] = reduced
    return cells


def pixelate_for_cross_stitch(source, cell_px=None):
    """Pixelise une image quelconque en blocs réguliers, avec une palette réduite.

    Sert de repli quand une image n'est pas reconnue comme pixel art : chaque bloc prend la couleur
    moyenne de sa zone, puis la palette est ramenée à MAX_SOURCE_COLORS. Renvoie une image RGBA.
    """
    if not isinstance(source, Image.Image):
        with Image.open(source) as image:
            return pixelate_for_cross_stitch(image, cell_px)
    image = ImageOps.exif_transpose(source).convert("RGBA")
    if cell_px is None:
        # Une grille d'au plus 150 cases par côté, sous la limite de 40 000 cases.
        cell_px = max(1, -(-max(image.size) // 150))
    cols = max(1, round(image.width / cell_px))
    rows = max(1, round(image.height / cell_px))
    small = image.resize((cols, rows), Image.Resampling.BOX)
    cells = _limit_cell_colors(np.array(small, dtype=np.uint8))
    return Image.fromarray(cells, "RGBA")


def _cell_palette(cells, tolerance=0):
    visible = cells[:, :, 3] >= 128
    colors = Counter(map(tuple, cells[:, :, :3][visible].tolist()))
    palette = []
    mapping = {}
    for color, _count in colors.most_common():
        match = next((index for index, rgb in enumerate(palette)
                      if max(abs(int(a) - int(b)) for a, b in zip(color, rgb)) <= tolerance), None)
        if match is None:
            match = len(palette)
            palette.append(color)
        mapping[color] = match
    labels = np.full(visible.shape, -1, dtype=np.int32)
    for row, col in zip(*np.nonzero(visible)):
        labels[row, col] = mapping[tuple(cells[row, col, :3].tolist())]
    return palette, labels


def _prepare_cells(source, background, source_format=None, exclude=()):
    if not isinstance(source, Image.Image):
        with Image.open(source) as image:
            return _prepare_cells(image, background, source_format, exclude)
    tolerance = 18 if (source_format or source.format) in {"JPEG", "WEBP", "AVIF"} else 0
    cells = _limit_cell_colors(read_cells(source))
    palette, labels = _cell_palette(cells, tolerance)
    if background not in {"outside", "all", "keep"}:
        raise ValueError("Option d'arrière-plan invalide.")
    if background != "keep" and palette:
        border = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        counts = Counter(border.tolist())
        if counts:
            bg = counts.most_common(1)[0][0]
            mask = (labels == bg).astype(np.uint8)
            if background == "all":
                labels[mask.astype(bool)] = -1
            else:
                _, components = cv2.connectedComponents(mask, connectivity=4)
                outside = np.unique(np.concatenate((components[0], components[-1],
                                                    components[:, 0], components[:, -1])))
                labels[np.isin(components, outside[outside != 0])] = -1
    if exclude and (labels >= 0).any():
        active = np.unique(labels[labels >= 0])
        lab = _to_lab(np.array(palette)[active])
        removed = np.zeros(len(active), dtype=bool)
        for color, others in exclude:
            distance = np.linalg.norm(lab - _to_lab(np.array([color])), axis=1)
            other_distance = (np.linalg.norm(lab[:, None] - _to_lab(np.array(others))[None], axis=2).min(axis=1)
                              if len(others) else np.inf)
            removed |= distance < other_distance
        available = active[~removed]
        if not len(available):
            raise ValueError("Conservez au moins une couleur pour broder le motif.")
        remap = np.arange(len(palette))
        if removed.any():
            remap[active[removed]] = available[_nearest(lab[removed], lab[~removed])]
        keep = labels >= 0
        labels[keep] = remap[labels[keep]]
    return cells, palette, labels


def _reduce_palette(palette, labels, n_colors):
    keep = labels >= 0
    if n_colors is not None and len(np.unique(labels[keep])) > n_colors:
        rgb = np.array(palette, dtype=np.uint8)[labels[keep]]
        reduced = Image.fromarray(rgb[None, :, :]).quantize(
            colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        colors, inverse = np.unique(np.array(reduced.convert("RGB"))[0], axis=0, return_inverse=True)
        palette = [tuple(color.tolist()) for color in colors]
        labels = labels.copy()
        labels[keep] = inverse
    return palette, labels


def cross_stitch_palettes(source, background="outside", *, source_format=None, max_colors=12, exclude=()):
    _, palette, labels = _prepare_cells(source, background, source_format, exclude)
    count = len(np.unique(labels[labels >= 0]))
    palettes = {}
    for limit in range(1, max_colors + 1):
        colors, reduced = _reduce_palette(palette, labels, limit)
        palettes[limit] = [rgb for index, rgb in enumerate(colors) if (reduced == index).any()]
    return {"colors": max(1, min(max_colors, count)), "palettes": palettes}


def cross_stitch_pattern(source, background="outside", *, source_format=None, n_colors=None, color_overrides=(),
                         exclude=(), double=False):
    if n_colors is not None and (isinstance(n_colors, bool) or not isinstance(n_colors, int)
                                 or not 1 <= n_colors <= MAX_COLORS):
        raise ValueError("Le nombre de couleurs doit être compris entre 1 et 256.")
    cells, palette, labels = _prepare_cells(source, background, source_format, exclude)
    palette, labels = _reduce_palette(palette, labels, n_colors)
    keep = labels >= 0
    source_palette = [rgb for index, rgb in enumerate(palette) if (labels == index).any()]
    overrides = dict(color_overrides)
    grouped, remap = {}, []
    for rgb in palette:
        color = overrides.get(rgb, rgb)
        if color not in grouped:
            grouped[color] = len(grouped)
        remap.append(grouped[color])
    if keep.any():
        labels[keep] = np.array(remap)[labels[keep]]
    palette = list(grouped)
    pattern = pe.EmbPattern()
    zones, segments = [], []
    unit = CELL_MM * 10
    for index, rgb in enumerate(palette):
        rows, cols = np.nonzero(labels == index)
        if not len(rows):
            continue
        if zones:
            pattern.add_command(pe.COLOR_CHANGE)
        pattern.add_thread({"color": (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]})
        segments.append((len(pattern.stitches), len(zones)))
        zones.append((rgb, None))
        previous = None
        for row in np.unique(rows):
            row_cols = cols[rows == row]
            if row % 2:
                row_cols = row_cols[::-1]
            for col in row_cols:
                left, top = float(col * unit), float(row * unit)
                if previous is not None and (left - previous[0]) ** 2 + (top - previous[1]) ** 2 > 30 ** 2:
                    pattern.add_command(pe.TRIM)
                pattern.move_abs(left, top)
                pattern.stitch_abs(left + unit, top + unit)
                pattern.move_abs(left + unit, top)
                pattern.stitch_abs(left, top + unit)
                if double:
                    # Point de croix double : la croix droite s'ajoute à la croix en diagonale,
                    # soit une étoile à huit branches sur quatre segments.
                    half = unit / 2
                    pattern.move_abs(left, top + half)
                    pattern.stitch_abs(left + unit, top + half)
                    pattern.move_abs(left + half, top)
                    pattern.stitch_abs(left + half, top + unit)
                previous = left, top + unit
    pattern.end()
    cx, cy = _center_on_origin(pattern)
    pattern.img2stitch_frame = {"w": cells.shape[1], "h": cells.shape[0], "ppm": 1 / CELL_MM,
                               "cx": cx, "cy": cy, "scale": 1.0}
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_source_palette = source_palette
    pattern.img2stitch_grid = {"columns": cells.shape[1], "rows": cells.shape[0], "cell_mm": CELL_MM,
                             "crosses": int(np.count_nonzero(labels >= 0)), "double": bool(double)}
    return pattern