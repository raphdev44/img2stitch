"""Lissage des streamlines par ressorts (leur « spring energy »).

Le dépôt assemble cinq matrices creuses — alignement, espacement, régularité de direction,
régularisation, points fixes — et résout le système directement (`weigh_matrices_and_solve`).
Sans solveur creux ici, ce sont les MÊMES termes d'énergie qui sont appliqués par relaxation :
chaque point est poussé contre ses ressorts, une quinzaine de fois. Les poids relatifs du dépôt
sont conservés (espacement ≪ alignement ≪ régularité de direction), et l'effet est celui qu'on
cherche : des lignes lisses, également espacées, qui suivent le dessin.

- ressort d'alignement : deux points consécutifs d'une même ligne gardent leur écart, ce qui
  redresse la ligne sans l'écourter ;
- ressort d'espacement : deux points de lignes voisines sont ramenés à `largeur de fil / densité`
  le long de la normale au champ, ce qui égalise la couverture ;
- régularité de direction : un point ne glisse pas le long de sa ligne (il y reste à la même
  abscisse), ce qui empêche les lignes de se ratatiner aux extrémités ;
- points sortis de la forme : ramenés à leur position d'origine (leur terme « fix »).
"""
import numpy as np

from .tracing import GRID, _bilinear

# Poids du dépôt (`main_pipeline` : w_align=1e3, w_space=1e1, w_dirreg=1e5, w_reg=1e-7, w_fix=1e5).
# Ils ne servent ici qu'à fixer les gains relatifs de la relaxation.
WEIGHTS = {"align": 1e3, "space": 1e1, "dirreg": 1e5, "reg": 1e-7, "fix": 1e5}
ITERATIONS = 14
ADJACENT = 1.5   # deux lignes sont voisines si elles sont à moins de 1,5 x l'écart voulu
MAX_NEIGHBORS = 6  # arêtes par point : l'équivalent creux des arêtes de Delaunay du dépôt


def _gains():
    total = WEIGHTS["align"] + WEIGHTS["space"] + WEIGHTS["dirreg"]
    return {key: WEIGHTS[key] / total for key in ("align", "space", "dirreg")}


def _pairs(points, curve_of, radius):
    """Paires de points appartenant à deux lignes différentes et distants de moins de `radius`.

    C'est l'équivalent du maillage de Delaunay du dépôt, restreint aux arêtes entre courbes. Sur
    une grille dense, toutes les paires d'un rayon en donnent des dizaines de millions et la
    relaxation ne finit plus : chaque point ne garde donc que ses `MAX_NEIGHBORS` plus proches.
    """
    if len(points) < 2:
        return np.empty((0, 2), np.intp)
    cell = max(radius, 1e-6)
    keys = np.floor(points / cell).astype(np.int64)
    buckets = {}
    for index, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(index)
    buckets = {key: np.asarray(value, np.intp) for key, value in buckets.items()}
    r2 = radius * radius
    best = {}

    def keep(index, distances, neighbours):
        good = np.isfinite(distances)
        if good.any():
            best.setdefault(int(index), []).extend(
                zip(distances[good].tolist(), neighbours[good].tolist()))

    # Chaque paire de cases n'est visitée qu'une fois (moitié des 8 voisines), et par tranches
    # pour borner la mémoire des matrices de distances.
    for (cx, cy), a in buckets.items():
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1)):
            b = buckets.get((cx + dx, cy + dy))
            if b is None:
                continue
            chunk = max(1, 2_000_000 // max(1, len(b)))
            for start in range(0, len(a), chunk):
                rows = a[start:start + chunk]
                d2 = ((points[rows][:, None, :] - points[b][None, :, :]) ** 2).sum(axis=2)
                ok = (d2 <= r2) & (curve_of[rows][:, None] != curve_of[b][None, :])
                if dx == 0 and dy == 0:
                    ok &= np.triu(np.ones(ok.shape, bool), 1)
                d2 = np.where(ok, d2, np.inf)
                k = min(MAX_NEIGHBORS, len(b))
                idx = (np.argpartition(d2, k - 1, axis=1)[:, :k] if k < len(b)
                       else np.broadcast_to(np.arange(len(b)), d2.shape).copy())
                picked = np.take_along_axis(d2, idx, axis=1)
                for row, cols, dists in zip(rows, idx, picked):
                    keep(row, dists, b[cols])

    found = set()
    for index, partners in best.items():
        partners.sort()
        found.update((min(index, int(other)), max(index, int(other)))
                     for _, other in partners[:MAX_NEIGHBORS])
    return np.asarray(sorted(found), np.intp).reshape(-1, 2) if found else np.empty((0, 2), np.intp)


def smooth(lines, direction, inside, density, line_width, iterations=ITERATIONS, on_progress=None):
    """Relâche les ressorts entre les points de toutes les lignes d'une région."""
    lines = [np.asarray(line, float) for line in lines if len(line) >= 2]
    if not lines:
        return []
    sizes = [len(line) for line in lines]
    points = np.vstack(lines)
    origin = points.copy()
    curve_of = np.repeat(np.arange(len(lines)), sizes)
    step = np.arange(len(points) - 1)
    same = step[curve_of[:-1] == curve_of[1:]]
    values = _bilinear(density, points)
    d_min = float(np.clip(np.percentile(values, 5), 0.25, 1.0))
    near = _pairs(points, curve_of, radius=ADJACENT * line_width / d_min)
    if len(near):
        gap = np.hypot(*(points[near[:, 1]] - points[near[:, 0]]).T)
        wanted = line_width / np.maximum(0.5 * (values[near[:, 0]] + values[near[:, 1]]), 0.1)
        near = near[gap <= ADJACENT * wanted]
    tangent = np.zeros_like(points)
    start_of = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    for first, size in zip(start_of, sizes):
        tangent[first:first + size] = np.gradient(origin[first:first + size], axis=0)
    tnorm = np.hypot(tangent[:, 0], tangent[:, 1])
    tangent /= np.where(tnorm < 1e-12, 1.0, tnorm)[:, None]
    gains = _gains()
    # Le ressort d'alignement doit dominer l'espacement : sinon les lignes se collent aux voisines.
    align_gain = 0.5 * gains["align"] / max(gains["align"] + gains["space"], 1e-9)
    space_gain = 0.5
    damping = 0.55
    fixed = np.zeros(len(points), bool)

    count = max(1, int(iterations))
    for index in range(count):
        move = np.zeros_like(points)
        weight = np.zeros(len(points))
        # --- alignement : l'écart entre deux points consécutifs reste celui d'origine
        if len(same):
            i, j = same, same + 1
            err = (points[j] - points[i]) - (origin[j] - origin[i])
            np.add.at(move, i, err * align_gain)
            np.add.at(move, j, -err * align_gain)
            np.add.at(weight, i, align_gain)
            np.add.at(weight, j, align_gain)
        # --- espacement : distance normale entre lignes voisines ramenée à la largeur de fil
        if len(near):
            i, j = near[:, 0], near[:, 1]
            delta = points[j] - points[i]
            angle = 0.5 * (_bilinear(direction, points[i]) + _bilinear(direction, points[j]))
            normal = np.column_stack((np.sin(angle), -np.cos(angle)))   # rotate90 de la direction
            local = 0.5 * (_bilinear(density, points[i]) + _bilinear(density, points[j]))
            target = np.sign(np.einsum("ij,ij->i", origin[j] - origin[i], normal) + 1e-12)
            target = target * line_width / np.maximum(local, 0.1)
            error = np.einsum("ij,ij->i", delta, normal) - target
            correction = normal * error[:, None] * (0.5 * space_gain)
            np.add.at(move, i, correction)
            np.add.at(move, j, -correction)
            np.add.at(weight, i, space_gain)
            np.add.at(weight, j, space_gain)
        weight = np.maximum(weight, 1e-9)
        points = points + (move / weight[:, None]) * damping
        # pas de glissement le long de la ligne, par rapport à la position d'origine
        if len(same):
            shift = points - origin
            points = points - np.einsum("ij,ij->i", shift, tangent)[:, None] * tangent
        # --- points sortis de la forme : ils reviennent à leur place (leur terme « fix »)
        outside = ~(np.asarray(_bilinear(inside.astype(np.float32), points)) > 0.5)
        fixed |= outside
        points[fixed] = origin[fixed]
        if on_progress:
            on_progress(index + 1, count)

    out, start = [], 0
    for size in sizes:
        out.append(points[start:start + size])
        start += size
    return out
