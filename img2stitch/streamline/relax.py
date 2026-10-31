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


def _gains():
    total = WEIGHTS["align"] + WEIGHTS["space"] + WEIGHTS["dirreg"]
    return {key: WEIGHTS[key] / total for key in ("align", "space", "dirreg")}


def _pairs(points, curve_of, radius):
    """Paires de points appartenant à deux lignes différentes et distants de moins de `radius`.

    C'est l'équivalent du maillage de Delaunay du dépôt, restreint aux arêtes entre courbes.
    """
    if len(points) < 2:
        return np.empty((0, 2), np.intp)
    cell = max(radius, 1e-6)
    keys = np.floor(points / cell).astype(np.int64)
    buckets = {}
    for index, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(index)
    found = []
    for (cx, cy), members in buckets.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                others = buckets.get((cx + dx, cy + dy))
                if not others:
                    continue
                for i in members:
                    for j in others:
                        if j <= i or curve_of[i] == curve_of[j]:
                            continue
                        if np.hypot(*(points[j] - points[i])) <= radius:
                            found.append((i, j))
    return np.asarray(found, np.intp).reshape(-1, 2) if found else np.empty((0, 2), np.intp)


def smooth(lines, direction, inside, density, line_width, iterations=ITERATIONS):
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
    near = _pairs(points, curve_of, radius=max(4.0 * line_width, 0.012))
    gains = _gains()
    # Le ressort d'alignement doit dominer l'espacement : sinon les lignes se collent aux voisines.
    align_gain = 0.5 * gains["align"] / max(gains["align"] + gains["space"], 1e-9)
    space_gain = 0.5
    damping = 0.55
    fixed = np.zeros(len(points), bool)

    for _ in range(int(iterations)):
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
        # --- régularité de direction : pas de glissement le long de la ligne
        slip = np.zeros_like(points)
        if len(same):
            i, j = same, same + 1
            tangent = origin[j] - origin[i]
            norm = np.hypot(tangent[:, 0], tangent[:, 1])
            tangent = tangent / np.where(norm < 1e-12, 1.0, norm)[:, None]
            delta = points[j] - points[i]
            tangential = np.einsum("ij,ij->i", delta, tangent)
            slip[i] += tangent * (0.5 * tangential)[:, None]
            slip[j] += tangent * (0.5 * tangential)[:, None]
            points = points - slip
        # --- points sortis de la forme : ils reviennent à leur place (leur terme « fix »)
        outside = ~(np.asarray(_bilinear(inside.astype(np.float32), points)) > 0.5)
        fixed |= outside
        points[fixed] = origin[fixed]

    out, start = [], 0
    for size in sizes:
        out.append(points[start:start + size])
        start += size
    return out
