"""Sources, tracé et coupe des streamlines.

Porté de `embroidery-streamlines` : `streamlines.trace_streamlines_from_sources`,
`cut_streamlines_assignment`, `crop_streamlines_outside_boundary`, `scale_and_resample` et
l'échantillonnage par intégrale de `kdsampling.py`.

Le principe : le champ vectoriel z = densité × direction a une divergence. Là où elle est
positive, des lignes doivent naître (sources) ; là où elle est négative, elles doivent finir
(puits). On trace depuis les sources en suivant la direction du champ (Runge-Kutta d'ordre 2,
comme leur `trace_streamline`), on coupe chaque ligne au puits le plus proche, puis on jette les
lignes trop courtes.

Seule différence assumée : les sources sont tirées par un échantillonnage aléatoire pondéré par
l'intégrale du signal (leur `jitter_sampling`), au lieu de la subdivision récursive par bissection
de leur intégrateur — même loi de probabilité, sans splines.
"""
import numpy as np

GRID = 210


def _bilinear(field, pts):
    """Échantillonnage bilinéaire d'un champ (GRID, GRID) ou (GRID, GRID, k) aux points normalisés."""
    x = np.clip(pts[:, 0], 0.0, 1.0) * (GRID - 1)
    y = np.clip(pts[:, 1], 0.0, 1.0) * (GRID - 1)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, GRID - 2)
    y0 = np.clip(np.floor(y).astype(np.intp), 0, GRID - 2)
    fx = (x - x0).reshape((-1,) + (1,) * (field.ndim - 2))
    fy = (y - y0).reshape((-1,) + (1,) * (field.ndim - 2))
    v00, v10 = field[y0, x0], field[y0, x0 + 1]
    v01, v11 = field[y0 + 1, x0], field[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy


def _inside(inside_grid, pts):
    return _bilinear(inside_grid.astype(np.float32), pts) > 0.5


def field_vectors(density, direction):
    """z = densité × direction, sous forme de champ (GRID, GRID, 2)."""
    return np.dstack((density * np.cos(direction), density * np.sin(direction))).astype(np.float32)


def divergence(density, direction):
    """Divergence de z sur la grille (différences centrées, période non bouclée)."""
    z = field_vectors(density, direction)
    step = float(GRID - 1)   # la grille couvre [0, 1] : ramène les différences en unités normalisées
    du = np.gradient(z[..., 0], axis=1) * step
    dv = np.gradient(z[..., 1], axis=0) * step
    return (du + dv).astype(np.float32)


def _sample_signal(signal, inside, relative_line_width, rng):
    """Un point source par unité d'intégrale du signal (leur `sample_points_on_interior`)."""
    positive = np.maximum(signal, 0.0) * inside
    cell = 1.0 / (GRID - 1) ** 2
    expected = positive * cell / max(relative_line_width, 1e-6)
    count = np.floor(expected).astype(np.intp) + (rng.random(expected.shape) < (expected % 1.0))
    ys, xs = np.nonzero(count)
    if not len(xs):
        return np.empty((0, 2))
    blocks = []
    for y, x in zip(ys, xs):
        n = int(count[y, x])
        if n:
            blocks.append((np.array([x, y], float) + rng.random((n, 2))) / (GRID - 1))
    return np.vstack(blocks) if blocks else np.empty((0, 2))


def _boundary_sources(boundary, z, inside, relative_line_width, rng):
    """Sources posées sur le bord, là où le flux entre (leur `sample_sources_on_closed_boundary`)."""
    closed = np.vstack((boundary, boundary[:1]))
    seg = np.diff(closed, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    arc = np.concatenate(([0.0], np.cumsum(length)))
    total = float(arc[-1])
    if total <= 1e-9:
        return np.empty((0, 2))
    step = max(total / 4000.0, 0.002)
    ts = np.arange(0.0, total, step)
    pts = np.column_stack((np.interp(ts, arc, closed[:, 0]), np.interp(ts, arc, closed[:, 1])))
    tangents = np.column_stack((np.gradient(closed[:, 0]), np.gradient(closed[:, 1])))
    tangents = np.column_stack((np.interp(ts, arc, tangents[:, 0]),
                                np.interp(ts, arc, tangents[:, 1])))
    tn = np.hypot(tangents[:, 0], tangents[:, 1])
    tangents /= np.where(tn < 1e-9, 1.0, tn)[:, None]
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))   # rotate90 de leur code
    # On garde la normale qui rentre dans la forme : le sens du contour suffit à le dire.
    probe = pts + normals * (2.0 * step)
    flip = ~_inside(inside, probe)
    normals[flip] *= -1.0
    inflow = np.einsum("ij,ij->i", normals, _bilinear(z, pts))
    weight = np.maximum(inflow, 0.0) * step / max(relative_line_width, 1e-6)
    count = np.floor(weight).astype(np.intp) + (rng.random(len(weight)) < (weight % 1.0))
    taken = np.flatnonzero(count)
    if not len(taken):
        return np.empty((0, 2))
    blocks = [pts[i] + normals[i] * 0.002 for i in taken for _ in range(int(count[i]))]
    return np.asarray(blocks, float).reshape(-1, 2) if blocks else np.empty((0, 2))


def sources_and_sinks(density, direction, inside, boundary, holes, relative_line_width, seed=7):
    """Sources (bord + intérieur) et puits (intérieur) du champ, comme dans leur pipeline."""
    rng = np.random.default_rng(seed)
    z = field_vectors(density, direction)
    div = divergence(density, direction)
    positive = _sample_signal(div, inside, relative_line_width, rng)
    negative = _sample_signal(-div, inside, relative_line_width, rng)
    edges = [_boundary_sources(boundary, z, inside, relative_line_width, rng)]
    edges += [_boundary_sources(np.asarray(h, float), z, inside, relative_line_width, rng) for h in holes]
    edges = [e for e in edges if len(e)]
    sources = np.vstack([positive] + edges) if edges else positive
    return sources, negative


def trace(sources, z, inside, step=1800, step_size=0.001):
    """Tracé simultané de toutes les lignes : RK2 (point milieu), arrêt à la sortie de la forme.

    Leur `trace_streamline` fait la même chose ligne par ligne ; ici toutes les sources avancent
    ensemble, ce qui tient en quelques secondes sur une grille de 210 x 210.
    """
    n = len(sources)
    if not n:
        return []
    pts = np.zeros((step, n, 2), np.float32)
    pos = np.asarray(sources, float).copy()
    alive = _inside(inside, pos)
    count = np.zeros(n, np.intp)
    for i in range(step):
        active = np.flatnonzero(alive)
        if not len(active):
            break
        p = pos[active]
        d = _bilinear(z, p)
        norm = np.hypot(d[:, 0], d[:, 1])
        moving = norm > 1e-9                       # direction nulle : la ligne s'arrête (leur `break`)
        if not moving.any():
            break
        d[moving] /= norm[moving, None]
        mid = p + d * (step_size * 0.5)
        dm = _bilinear(z, mid)
        norm_mid = np.hypot(dm[:, 0], dm[:, 1])
        good = norm_mid > 1e-9
        dm[good] /= norm_mid[good, None]
        dm[~good] = d[~good]
        pts[i, active] = p
        count[active] = i + 1
        pos[active] = p + dm * step_size
        alive[active[~_inside(inside, pos[active])]] = False
        alive[active[~moving]] = False
    return [pts[:count[k], k].astype(float) for k in range(n) if count[k] >= 2]


def crop_inside(lines, inside):
    """Garde les tronçons réellement dans la forme (leur `crop_streamlines_outside_boundary`)."""
    kept = []
    for line in lines:
        mask = _inside(inside, line)
        start = None
        for i, ok in enumerate(mask):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                if i - start >= 2:
                    kept.append(line[start:i])
                start = None
        if start is not None and len(line) - start >= 2:
            kept.append(line[start:])
    return kept


def cut_at_sinks(lines, sinks, candidates=8):
    """Coupe chaque ligne au puits le plus proche (leur appartement biparti, en glouton).

    Seuls les `candidates` puits les plus proches du centre de la ligne sont comparés point par
    point : la matrice complète (points x puits) serait très coûteuse et ne changerait pas la coupe.
    """
    if not len(sinks):
        return lines
    out = []
    for line in lines:
        centre = line.mean(axis=0)
        near = np.argsort(((sinks - centre) ** 2).sum(axis=1))[:candidates]
        distance = ((line[:, None, :] - sinks[near][None, :, :]) ** 2).sum(axis=2)
        point_index, _ = np.unravel_index(int(np.argmin(distance)), distance.shape)
        if point_index <= 1:          # le puits tombe sur le début : la ligne entière disparaît
            continue
        out.append(line[:point_index + 1])
    return out


def drop_short(lines):
    """Leur heuristique : écarte les lignes plus courtes que 30 % de la moyenne (plafonnée à 2)."""
    if not lines:
        return []
    threshold = min(2, float(np.mean([len(l) for l in lines])) * 0.3)
    kept = [l for l in lines if len(l) > round(threshold)]
    return kept or lines


def resample(lines, spacing):
    """Points régulièrement espacés le long de chaque ligne (leur `scale_and_resample`)."""
    out = []
    for line in lines:
        length = float(np.hypot(*np.diff(line, axis=0).T).sum()) if len(line) > 1 else 0.0
        if length <= 1e-9:
            continue
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))
        t = np.linspace(0.0, length, max(2, int(round(length / max(spacing, 1e-6)))))
        out.append(np.column_stack((np.interp(t, arc, line[:, 0]), np.interp(t, arc, line[:, 1]))))
    return out
