"""Passes d'optimisation prudentes pour motifs de broderie existants (pyembroidery).

Coordonnées pyembroidery : dixièmes de millimètre.

Modèle de travail
-----------------
Le motif est découpé en *passages* (``run``) : suites de points piqués entre deux
commandes non piquées (TRIM, COLOR_CHANGE, JUMP...). Le premier point d'un passage
est la destination du déplacement qui suit la coupe, il n'est pas relié au point
précédent. Les passes ne piquent jamais ce déplacement et ne le découpent jamais.

Passes (dans l'ordre)
---------------------
1. doublons et micro-points (par passage)
2. simplification des points colinéaires (longueur de point plafonnée)
3. contrôle de couverture (IoU) avec repli automatique si le rendu change trop
4. réordonnancement des passages d'une même couleur (contraintes de recouvrement)
5. suppression des coupes : déplacement court, ou déplacement caché sous une zone
   qui sera brodée ensuite (ou déjà brodée dans la même couleur)
6. découpe des points trop longs (à l'intérieur d'un passage uniquement)

numpy et Pillow sont optionnels : sans eux, les passes 3 et 4 et le test « caché »
de la passe 5 sont désactivés et un avertissement est ajouté au rapport.
"""
from bisect import bisect_right
from copy import deepcopy
from math import ceil, hypot

import pyembroidery as pe

try:  # contrôles géométriques optionnels
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    np = None

MASK = pe.COMMAND_MASK
UNITS_PER_MM = 10.0

LEVELS = {
    "safe": dict(min_stitch_mm=0.3, simplify_tol_mm=0.05, max_merged_mm=3.0,
                 merge_travel_mm=0.0, hidden_travel_mm=0.0, reorder=False, min_iou=0.97),
    "standard": dict(min_stitch_mm=0.5, simplify_tol_mm=0.10, max_merged_mm=3.5,
                     merge_travel_mm=2.5, hidden_travel_mm=20.0, reorder=True, min_iou=0.94),
    "aggressive": dict(min_stitch_mm=0.7, simplify_tol_mm=0.15, max_merged_mm=4.0,
                       merge_travel_mm=3.5, hidden_travel_mm=30.0, reorder=True, min_iou=0.90),
}
THREAD_MM = 0.4          # largeur de fil utilisée pour rasteriser la couverture
OVERLAP_MM = 0.6         # largeur utilisée pour détecter les recouvrements entre passages
MAX_RUNS_FOR_REORDER = 600


def optimize_pattern(pattern, *, level="standard", min_stitch_mm=None, max_stitch_mm=12.0,
                     merge_travel_mm=None, simplify_tol_mm=None, max_merged_mm=None,
                     hidden_travel_mm=None, reorder=None, min_iou=None,
                     travel_piece_mm=3.0):
    """Retourne ``(motif_optimisé, rapport)`` sans modifier ``pattern``.

    ``reorder=None`` : automatique (actif si le niveau le demande, si numpy/Pillow sont
    présents et si le motif n'a pas de zones ``img2stitch_segments``, car réordonner
    casserait leur découpage).
    """
    if level not in LEVELS:
        raise ValueError("level doit être l'un de %s" % sorted(LEVELS))
    prm = dict(LEVELS[level])
    for key, value in (("min_stitch_mm", min_stitch_mm), ("merge_travel_mm", merge_travel_mm),
                       ("simplify_tol_mm", simplify_tol_mm), ("max_merged_mm", max_merged_mm),
                       ("hidden_travel_mm", hidden_travel_mm), ("min_iou", min_iou)):
        if value is not None:
            prm[key] = value

    optimized = _copy_pattern(pattern)
    original = [list(stitch) + [index] for index, stitch in enumerate(pattern.stitches)]
    segments = getattr(pattern, "img2stitch_segments", None) or []
    object_starts = {int(start) for start, _ in segments}
    report = {
        "level": level, "removed_duplicates": 0, "removed_micro_stitches": 0,
        "simplified_points": 0, "reordered": False, "merged_trims": 0,
        "hidden_travels_merged": 0, "split_long_stitches": 0, "coverage_iou": None,
        "warnings": [],
    }

    head, runs, seps, tail = _parse(original)
    report["before"] = _summary(original, runs, seps, pattern)
    if not runs:
        optimized.stitches = [item[:3] for item in original]
        report["after"] = report["before"]
        return optimized, report

    have_raster = np is not None
    if not have_raster:
        report["warnings"].append(
            "numpy/Pillow absents : contrôle de couverture, réordonnancement et test de "
            "déplacement caché désactivés.")
    grid = _Grid(original) if have_raster else None
    reference = grid.coverage(runs, THREAD_MM) if grid else None

    # --- passes 1-3 : géométrie interne des passages, avec contrôle de couverture -------
    attempts = [
        (prm["min_stitch_mm"], prm["simplify_tol_mm"]),
        (min(prm["min_stitch_mm"], 0.3), 0.0),     # repli 1 : sans simplification
        (0.0, 0.0),                                 # repli 2 : doublons seulement
    ]
    for number, (micro, tol) in enumerate(attempts):
        counts = dict(dup=0, micro=0, simp=0)
        cleaned = [_clean_run(run, micro * UNITS_PER_MM, tol * UNITS_PER_MM,
                              prm["max_merged_mm"] * UNITS_PER_MM, object_starts, counts)
                   for run in runs]
        if grid is None:
            break
        iou = _iou(reference, grid.coverage(cleaned, THREAD_MM))
        if iou >= prm["min_iou"] or number == len(attempts) - 1:
            report["coverage_iou"] = round(float(iou), 4)
            if number:
                report["warnings"].append(
                    "Couverture trop modifiée par le réglage demandé : repli sur le réglage %d/3 "
                    "(IoU %.3f, seuil %.2f)." % (number + 1, iou, prm["min_iou"]))
            break
    runs = cleaned
    report["removed_duplicates"] = counts["dup"]
    report["removed_micro_stitches"] = counts["micro"]
    report["simplified_points"] = counts["simp"]

    # --- passe 4 : réordonnancement -------------------------------------------------------
    do_reorder = prm["reorder"] if reorder is None else reorder
    if do_reorder and reorder is None and segments:
        do_reorder = False
        report["warnings"].append(
            "Réordonnancement ignoré : le motif contient des zones img2stitch_segments.")
    if do_reorder and grid is None:
        do_reorder = False
    if do_reorder:
        runs, seps = _reorder(runs, seps, grid, report)

    # --- passe 5 : coupes --------------------------------------------------------------------
    runs, seps = _merge_trims(runs, seps, head, grid, prm, travel_piece_mm * UNITS_PER_MM,
                              pattern, report)

    # --- passe 6 : points trop longs à l'intérieur des passages ----------------------------
    runs = [_split_long(run, max_stitch_mm * UNITS_PER_MM, report) for run in runs]

    output = list(head)
    for index, run in enumerate(runs):
        output.extend(run)
        if index < len(seps):
            output.extend(seps[index])
    output.extend(tail)

    optimized.stitches = [item[:3] for item in output]
    if segments:
        optimized.img2stitch_segments = _remap_segments(segments, output)
    report["after"] = _summary(output, runs, seps, pattern)
    report["after"]["coverage_iou"] = report["coverage_iou"]
    report["min_stitch_mm"] = prm["min_stitch_mm"]
    report["max_stitch_mm"] = max_stitch_mm
    report["merge_travel_mm"] = prm["merge_travel_mm"]
    return optimized, report


# ---------------------------------------------------------------------------------------------
# Découpage / reconstruction
# ---------------------------------------------------------------------------------------------
def _copy_pattern(pattern):
    optimized = pattern.copy()
    optimized.stitches = deepcopy(pattern.stitches)
    optimized.threadlist = deepcopy(pattern.threadlist)
    optimized.extras = deepcopy(pattern.extras)
    for name, value in pattern.__dict__.items():
        if name not in {"stitches", "threadlist", "extras", "_previousX", "_previousY"}:
            setattr(optimized, name, deepcopy(value))
    return optimized


def _parse(items):
    """Découpe en (commandes de tête, passages, séparateurs, commandes de queue)."""
    head, runs, seps, pending, current = [], [], [], [], None
    for item in items:
        if item[2] & MASK == pe.STITCH:
            if current is None:
                current = []
                runs.append(current)
                if len(runs) == 1:
                    head = pending
                else:
                    seps.append(pending)
                pending = []
            current.append(item)
        else:
            current = None
            pending.append(item)
    return head, runs, seps, pending


def _sep_kind(sep):
    commands = {item[2] & MASK for item in sep}
    if pe.TRIM in commands and commands <= {pe.TRIM, pe.JUMP}:
        return "trim"
    if pe.COLOR_CHANGE in commands:
        return "color"
    return "other"


def _color_keys(head, seps, count, pattern):
    """Clé de couleur (RGB si connu) de chaque passage, d'après les COLOR_CHANGE."""
    block = sum(1 for item in head if item[2] & MASK == pe.COLOR_CHANGE)
    blocks = [block]
    for sep in seps:
        block += sum(1 for item in sep if item[2] & MASK == pe.COLOR_CHANGE)
        blocks.append(block)
    threads = pattern.threadlist

    def key(index):
        return threads[index].color if index < len(threads) else ("block", index)
    return [key(block) for block in blocks[:count]]


def _distance(first, second):
    return hypot(second[0] - first[0], second[1] - first[1])


def _travel_total(runs):
    return sum(_distance(a[-1], b[0]) for a, b in zip(runs, runs[1:])) / UNITS_PER_MM


def _summary(items, runs, seps, pattern):
    stitches = sum(1 for item in items if item[2] & MASK == pe.STITCH)
    trims = sum(1 for item in items if item[2] & MASK == pe.TRIM)
    colors = sum(1 for item in items if item[2] & MASK == pe.COLOR_CHANGE)
    return {
        "stitches": stitches, "trims": trims, "runs": len(runs),
        "travel_mm": round(_travel_total(runs), 1) if runs else 0.0,
        # estimation indicative : 600 points/min, 4 s par coupe, 30 s par changement de couleur
        "est_time_min": round(stitches / 600.0 + trims * 4 / 60.0 + colors * 0.5, 1),
    }


# ---------------------------------------------------------------------------------------------
# Passes 1-2 : doublons, micro-points, simplification
# ---------------------------------------------------------------------------------------------
def _clean_run(run, min_units, tol_units, cap_units, object_starts, counts):
    unique = []
    for item in run:
        if unique and _distance(unique[-1], item) <= 0.5:
            counts["dup"] += 1
            continue
        unique.append(item)

    budget = len(unique) // 4           # jamais plus de 25 % de points retirés par passage
    for _ in range(3):                  # jusqu'au point fixe : la passe devient idempotente
        before = len(unique)
        if min_units > 0 and budget > 0:
            kept = []
            for position, item in enumerate(unique):
                endpoint = (position == 0 or position == len(unique) - 1
                            or item[3] in object_starts)
                if (not endpoint and budget > 0 and kept
                        and _distance(kept[-1], item) < min_units):
                    budget -= 1
                    counts["micro"] += 1
                    continue
                kept.append(item)
            unique = kept
        if tol_units > 0 and len(unique) > 2:
            simplified = _simplify(unique, tol_units, cap_units, object_starts)
            counts["simp"] += len(unique) - len(simplified)
            unique = simplified
        if len(unique) == before:
            break
    return unique


def _segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = dx * dx + dy * dy
    if length == 0:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length))
    return hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def _simplify(points, tol, cap, protected):
    """Fenêtre glissante : retire les points intermédiaires à moins de ``tol`` de la corde."""
    out = [points[0]]
    i, n = 0, len(points)
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            k = j + 1
            if points[j][3] in protected or _distance(points[i], points[k]) > cap:
                break
            if any(_segment_distance(points[m], points[i], points[k]) > tol
                   for m in range(i + 1, k)):
                break
            j = k
        out.append(points[j])
        i = j
    return out


# ---------------------------------------------------------------------------------------------
# Rasterisation (couverture, recouvrements, déplacements cachés)
# ---------------------------------------------------------------------------------------------
class _Grid:
    def __init__(self, items, max_px=1600, min_px_mm=0.1):
        xs = [item[0] for item in items if item[2] & MASK == pe.STITCH]
        ys = [item[1] for item in items if item[2] & MASK == pe.STITCH]
        self.x0, self.y0 = min(xs) - 20, min(ys) - 20
        span = max(max(xs) - self.x0, max(ys) - self.y0) + 40
        self.px_mm = max(min_px_mm, span / UNITS_PER_MM / max_px)
        self.scale = 1.0 / (self.px_mm * UNITS_PER_MM)
        self.w = int((max(xs) - self.x0 + 40) * self.scale) + 2
        self.h = int((max(ys) - self.y0 + 40) * self.scale) + 2

    def px(self, point):
        return ((point[0] - self.x0) * self.scale, (point[1] - self.y0) * self.scale)

    def _width(self, mm):
        return max(1, int(round(mm / self.px_mm)))

    def coverage(self, runs, width_mm):
        image = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(image)
        width = self._width(width_mm)
        for run in runs:
            for a, b in zip(run, run[1:]):
                draw.line([self.px(a), self.px(b)], fill=255, width=width)
        return np.asarray(image) > 0

    def index_map(self, runs, width_mm, last=True, colors=None, key=None):
        """Index (position globale) du dernier / premier point couvrant chaque pixel."""
        empty = -1 if last else 2 ** 30
        image = Image.new("I", (self.w, self.h), empty)
        draw = ImageDraw.Draw(image)
        width = self._width(width_mm)
        flat, segments = 0, []
        for number, run in enumerate(runs):
            for position, point in enumerate(run):
                if position and (colors is None or colors[number] == key):
                    segments.append((flat, run[position - 1], point))
                flat += 1
        for index, a, b in (segments if last else reversed(segments)):
            draw.line([self.px(a), self.px(b)], fill=index, width=width)
        return np.asarray(image)

    def run_mask(self, run, width_mm):
        xs = [self.px(p)[0] for p in run]
        ys = [self.px(p)[1] for p in run]
        pad = self._width(width_mm) + 1
        left, top = int(min(xs)) - pad, int(min(ys)) - pad
        right, bottom = int(max(xs)) + pad + 1, int(max(ys)) + pad + 1
        image = Image.new("L", (right - left, bottom - top), 0)
        draw = ImageDraw.Draw(image)
        for a, b in zip(run, run[1:]):
            (ax, ay), (bx, by) = self.px(a), self.px(b)
            draw.line([(ax - left, ay - top), (bx - left, by - top)], fill=255,
                      width=self._width(width_mm))
        return (left, top, right, bottom), np.asarray(image) > 0


def _iou(first, second):
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else np.count_nonzero(first & second) / union


# ---------------------------------------------------------------------------------------------
# Passe 4 : réordonnancement des passages d'une même couleur
# ---------------------------------------------------------------------------------------------
def _reorder(runs, seps, grid, report):
    """Réordonne chaque chaîne (passages d'une même couleur reliés par des coupes).

    Une chaîne n'est remplacée que si le plus proche voisin raccourcit ses déplacements.
    """
    kinds = [_sep_kind(sep) for sep in seps]
    new_runs, new_seps, start, changed = [], [], 0, False
    while start < len(runs):
        end = start
        while end < len(seps) and kinds[end] == "trim":
            end += 1
        chain = runs[start:end + 1]
        if 2 < len(chain) <= MAX_RUNS_FOR_REORDER:
            order = _chain_order(chain, grid)
            candidate = [chain[i] for i in order]
            if _travel_total(candidate) < _travel_total(chain) * 0.99:
                chain, changed = candidate, True
        new_runs.extend(chain)
        for a in chain[:-1]:
            new_seps.append([[a[-1][0], a[-1][1], pe.TRIM, a[-1][3]]])   # TRIM canonique
        if end < len(seps):
            new_seps.append(seps[end])
        start = end + 1
    if not changed:
        return runs, seps
    report["reordered"] = True
    return new_runs, _restore_trim_flags(new_seps, seps)


def _restore_trim_flags(new_seps, old_seps):
    flags = next((item[2] for sep in old_seps for item in sep if item[2] & MASK == pe.TRIM),
                 pe.TRIM)
    for sep in new_seps:
        for item in sep:
            if item[2] & MASK == pe.TRIM:
                item[2] = flags
    return new_seps


def _chain_order(chain, grid):
    """Plus proche voisin sous contraintes : deux passages qui se recouvrent gardent leur ordre."""
    boxes, masks = [], []
    for run in chain:
        box, mask = grid.run_mask(run, OVERLAP_MM)
        boxes.append(box)
        masks.append(mask)
    count = len(chain)
    preds = [set() for _ in range(count)]
    for j in range(count):
        for i in range(j):
            left, top = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1])
            right, bottom = min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            if left >= right or top >= bottom:
                continue
            a = masks[i][top - boxes[i][1]:bottom - boxes[i][1], left - boxes[i][0]:right - boxes[i][0]]
            b = masks[j][top - boxes[j][1]:bottom - boxes[j][1], left - boxes[j][0]:right - boxes[j][0]]
            if np.any(a & b):
                preds[j].add(i)
    order, done, remaining = [0], {0}, set(range(1, count))
    while remaining:
        end = chain[order[-1]][-1]
        available = [j for j in remaining if preds[j] <= done]
        best = min(available, key=lambda j: (_distance(end, chain[j][0]), j))
        order.append(best)
        done.add(best)
        remaining.discard(best)
    return order


# ---------------------------------------------------------------------------------------------
# Passe 5 : suppression des coupes
# ---------------------------------------------------------------------------------------------
def _merge_trims(runs, seps, head, grid, prm, piece_units, pattern, report):
    merge_units = prm["merge_travel_mm"] * UNITS_PER_MM
    hidden_units = prm["hidden_travel_mm"] * UNITS_PER_MM
    if merge_units <= 0 and hidden_units <= 0:
        return runs, seps
    kinds = [_sep_kind(sep) for sep in seps]
    ends, flat = [], 0
    for run in runs:
        flat += len(run)
        ends.append(flat - 1)
    last_map = first_maps = colors = None
    if grid is not None and hidden_units > 0:
        last_map = grid.index_map(runs, THREAD_MM, last=True)
        colors = _color_keys(head, seps, len(runs), pattern)
        first_maps = {}

    out_runs, out_seps, current = [], [], list(runs[0])
    for index, sep in enumerate(seps):
        nxt = runs[index + 1]
        merge = False
        if kinds[index] == "trim":
            travel = _distance(current[-1], nxt[0])
            if 0 < merge_units and travel <= merge_units:
                merge = True
                report["merged_trims"] += 1
            elif (last_map is not None and travel <= hidden_units
                  and _hidden(grid, last_map, first_maps, runs, colors, index, ends[index])):
                merge = True
                report["hidden_travels_merged"] += 1
        if merge:
            current.extend(_travel_points(current[-1], nxt[0], piece_units))
            current.extend(nxt)
        else:
            out_runs.append(current)
            out_seps.append(sep)
            current = list(nxt)
    out_runs.append(current)
    return out_runs, out_seps


def _travel_points(start, end, piece_units):
    pieces = max(1, ceil(_distance(start, end) / piece_units))
    return [[round(start[0] + (end[0] - start[0]) * k / pieces),
             round(start[1] + (end[1] - start[1]) * k / pieces),
             pe.STITCH, end[3]] for k in range(1, pieces)]


def _hidden(grid, last_map, first_maps, runs, colors, index, end_index,
            ratio=0.97, max_gap_mm=0.8):
    """Vrai si le trajet reste sous des points brodés plus tard ou déjà brodés (même couleur)."""
    key = colors[index]
    if key not in first_maps:
        first_maps[key] = grid.index_map(runs, THREAD_MM, last=False, colors=colors, key=key)
    start, end = runs[index][-1], runs[index + 1][0]
    steps = max(2, int(_distance(start, end) * grid.scale / 0.5) + 1)
    t = np.linspace(0.0, 1.0, steps)
    px = np.array([grid.px(start), grid.px(end)])
    xs = np.clip((px[0, 0] + (px[1, 0] - px[0, 0]) * t).astype(int), 0, grid.w - 1)
    ys = np.clip((px[0, 1] + (px[1, 1] - px[0, 1]) * t).astype(int), 0, grid.h - 1)
    covered = (last_map[ys, xs] > end_index) | (first_maps[key][ys, xs] <= end_index)
    if covered.mean() < ratio:
        return False
    gap = longest = 0
    for flag in covered:
        gap = 0 if flag else gap + 1
        longest = max(longest, gap)
    return longest * (_distance(start, end) / UNITS_PER_MM / (steps - 1)) <= max_gap_mm


# ---------------------------------------------------------------------------------------------
# Passe 6 : points trop longs (jamais le déplacement qui suit une coupe)
# ---------------------------------------------------------------------------------------------
def _split_long(run, max_units, report):
    out = [run[0]]
    for item in run[1:]:
        previous = out[-1]
        distance = _distance(previous, item)
        pieces = max(1, ceil(distance / max_units))
        if pieces > 1:
            report["split_long_stitches"] += pieces - 1
            for part in range(1, pieces):
                ratio = part / pieces
                out.append([round(previous[0] + (item[0] - previous[0]) * ratio),
                            round(previous[1] + (item[1] - previous[1]) * ratio),
                            item[2], item[3]])
        out.append(item)
    return out


def _remap_segments(segments, output):
    starts = sorted({int(start) for start, _ in segments})
    first_output = {}
    for position, item in enumerate(output):
        zone = bisect_right(starts, item[3]) - 1
        if zone >= 0 and starts[zone] not in first_output and item[2] & MASK == pe.STITCH:
            first_output[starts[zone]] = position
    return [(first_output[int(start)], zone_index) for start, zone_index in segments
            if int(start) in first_output]
