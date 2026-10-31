"""Remplissages portés d'Ink/Stitch : circulaire, contour et point de croix.

Repris de `lib/stitches/circular_fill.py`, `lib/stitches/contour_fill.py` et
`lib/stitches/cross_stitch.py` (Ink/Stitch, GPL-3.0), avec la même construction mais sans
Shapely ni networkx : les anneaux sont extraits du masque de la zone et les sommets sont en
millimètres, comme partout ailleurs dans le projet.

1. REMPLISSAGE EN CONTOUR (contour fill). Des isocontours concentriques, du bord vers le
   centre, reliés en UNE seule spirale : deux anneaux voisins sont rééchantillonnés sur le même
   nombre de points puis interpolés (`_interpolate_linear_rings`), donc le fil ne s'arrête jamais
   au milieu de la forme, il tourne. C'est exactement `single_spiral` d'Ink/Stitch.

2. REMPLISSAGE CIRCULAIRE (circular fill). La même spirale, mais les anneaux sont des cercles
   autour d'un centre : le rayon part de l'écart de rangée et croît jusqu'à la distance de
   Hausdorff du centre au bord. C'est la DOUBLE spirale de Fermat d'Ink/Stitch : une spirale sur
   les anneaux pairs de l'extérieur vers le centre, une seconde sur les impairs qui revient, ce
   qui évite de finir coincé au milieu.

3. POINT DE CROIX (cross stitch fill). Une grille de croix, chacune piquée dans le même ordre
   (diagonale, bord bas, diagonale) : le dessus du croisement ne change donc jamais d'une croix à
   l'autre. Deux croix voisines partagent un coin, si bien qu'une rangée entière tient dans un
   seul tracé ; le fil repasse une seconde fois (deux brins), comme le `thread_count` d'Ink/Stitch.

Ce qui change : Ink/Stitch ordonne les segments avec un graphe et des points de départ/arrivée
explicites. Ici c'est le planificateur du projet (`planner.py`) qui joue ce rôle : chaque tracé
est rendu avec ses extrémités et c'est lui qui décide des déplacements et des coupes.
"""
import math

import cv2
import numpy as np

from .outline import contours_mm
from .relief import _clip, _resample   # découpe au masque et rééchantillonnage : mêmes règles

CROSS_CELL_MM = 2.0      # côté d'une croix, comme le pixel art et le point de croix du projet
CROSS_THREADS = 2        # deux brins : le fil repasse sur lui-même
MIN_SEGMENT_MM = 1.2     # en dessous, un bout de spirale n'apporte que des points parasites
SIMPLIFY_MM = 0.1        # même tolérance que le `simplify(0.1)` d'Ink/Stitch
INTERPOLATION_FACTOR = 20.0   # points par longueur de point, comme `_interpolate_linear_rings`


# --- outillage commun ------------------------------------------------------------------------
def _closed(ring):
    ring = np.asarray(ring, dtype=float)
    return np.vstack((ring, ring[:1])) if len(ring) else ring


def _signed_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient(ring):
    """Tous les anneaux dans le même sens : sinon l'interpolation entre deux anneaux se croise et
    la spirale fait des nœuds au lieu de tourner."""
    return ring[::-1] if _signed_area(ring) < 0 else ring


def _length(poly):
    return float(np.hypot(*np.diff(poly, axis=0).T).sum()) if len(poly) > 1 else 0.0


def _centroid(ring):
    return float(ring[:, 0].mean()), float(ring[:, 1].mean())


def _resample_count(poly, count):
    """`count` points régulièrement répartis sur une polyligne fermée."""
    if len(poly) < 2 or count < 2:
        return poly[:1] if len(poly) else poly
    seg = np.hypot(*np.diff(poly, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(poly[:1], count, axis=0)
    t = np.linspace(0.0, total, count, endpoint=False)
    return np.column_stack((np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])))


def _reorder(ring, start):
    """Fait commencer l'anneau au point le plus proche de `start` (roll_linear_ring)."""
    start = np.asarray(start, dtype=float)
    d = np.hypot(ring[:, 0] - start[0], ring[:, 1] - start[1])
    return np.roll(ring, -int(np.argmin(d)), axis=0)


def _simplify(poly, tol_mm):
    """Douglas-Peucker sur une polyligne ouverte (l'équivalent de `LineString.simplify`)."""
    if len(poly) < 4:
        return poly
    pts = np.ascontiguousarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(pts, tol_mm, False).reshape(-1, 2).astype(float)


# --- spirales (contour_fill.py) ---------------------------------------------------------------
def _interpolate_linear_rings(ring1, ring2, max_stitch_length, start=None):
    """Anneau 1 → anneau 2 en tournant.

    Les deux anneaux sont rééchantillonnés sur le même nombre de points, puis chaque point du
    trajet est interpolé linéairement de l'un à l'autre : le fil glisse d'une rangée à la suivante
    au lieu de sauter. C'est le cœur du remplissage en contour.
    """
    num = int(INTERPOLATION_FACTOR * _length(ring1) / max(1e-6, max_stitch_length))
    if num <= 1:
        return np.empty((0, 2))
    a = _resample_count(ring1, num)
    b = _resample_count(ring2, num)
    if start is not None:
        a, b = _reorder(a, start), _reorder(b, start)
    w = np.linspace(0.0, 1.0, num).reshape(-1, 1)
    return _simplify(a * (1.0 - w) + b * w, SIMPLIFY_MM)


def _make_spiral(rings, stitch_length, starting_point):
    """Spirale simple : on enchaîne les anneaux deux à deux, de l'extérieur vers le centre."""
    path, part, start = [], None, starting_point
    for ring1, ring2 in zip(rings[:-1], rings[1:]):
        part = _interpolate_linear_rings(ring1, ring2, stitch_length, start)
        if len(part) >= 2:
            path.extend(part[:-1])
            start = part[-1]   # l'anneau suivant se raccorde là où le fil s'est arrêté
    if part is not None and len(part):
        path.append(part[-1])
    return np.array(path, dtype=float) if path else np.empty((0, 2))


def _make_fermat_spiral(rings, stitch_length, starting_point):
    """Double spirale : une spirale sur les anneaux pairs, une sur les impairs parcourue à
    l'envers. Le tracé ressort du centre au lieu de s'y arrêter, et le remplissage circulaire se
    termine là où il a commencé."""
    forward = _make_spiral(rings[::2], stitch_length, starting_point)
    back = _make_spiral(rings[1::2], stitch_length, starting_point)[::-1]
    if len(forward) and len(back):
        return np.vstack((forward, back))
    return forward if len(forward) else back


# --- anneaux ----------------------------------------------------------------------------------
def _offset_rings(mask, px_per_mm, s):
    """Isocontours du masque, du bord vers le centre (l'équivalent d'`offset_polygon`).

    L'érosion par un disque donne des joints ronds, comme le `join_style` d'Ink/Stitch ; le
    premier anneau est posé à une demi-rangée du bord pour ne pas repasser sur le contour.
    """
    r = max(1, round(s.row_spacing_mm * px_per_mm))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    half = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r + 1, r + 1))
    m = cv2.erode(mask.astype(np.uint8), half)
    rings, prev = [], None
    while m.any():
        found = [_orient(_closed(c)) for c in contours_mm(m, px_per_mm, s.outline_simplify_mm, s.min_area_mm2)
                 if len(c) >= 3]
        if not found:
            break
        # Forme qui se casse en plusieurs morceaux : on suit celui dont le centre est le plus
        # proche du précédent, comme `_get_spiral_rings` qui garde l'enfant le plus central.
        if prev is None:
            ring = found[0]
        else:
            cx, cy = _centroid(prev)
            ring = min(found, key=lambda c: math.dist(_centroid(c), (cx, cy)))
        rings.append(ring)
        prev = ring
        m = cv2.erode(m, kernel)
    return rings


def _level_rings(level, px_per_mm, s):
    """Courbes de niveau : bords extérieurs ET bords de trous, tous orientés pareil.

    Une forme percée a deux frontières. N'en garder qu'une laissait toute la couronne
    intérieure vide (56 % de couverture sur un anneau) : les deux sont donc des anneaux à part
    entière, et la spirale de chacun avance vers le milieu de la matière jusqu'à rejoindre
    l'autre — c'est exactement ce que fait l'offset de polygones d'Ink/Stitch.
    """
    found, _ = cv2.findContours(level.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    eps = max(0.5, s.outline_simplify_mm * px_per_mm)
    rings = []
    for c in found:
        if cv2.contourArea(c) / px_per_mm ** 2 < s.min_area_mm2 or len(c) < 4:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = np.asarray([((x + 0.5) / px_per_mm, (y + 0.5) / px_per_mm) for x, y in c], float)
        rings.append(_orient(_closed(pts)))
    return rings


def _poly_distance(a, b):
    """Distance minimale entre deux polylignes (points sous-échantillonnés)."""
    pa = np.asarray(a, float)[::max(1, len(a) // 120)]
    pb = np.asarray(b, float)[::max(1, len(b) // 120)]
    d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
    return float(d.min())


def _ring_chains(mask, px_per_mm, s):
    """Anneaux du bord vers le centre, groupés en chaînes : une chaîne par lobe de la forme.

    Les anneaux sont les isocontours de la transformée de distance. Contrairement à une érosion
    répétée, ils épousent AUSSI le bord d'un trou et ne s'arrêtent pas quand la forme s'amincit :
    une forme percée est remplie des deux côtés, jusqu'à ce que les deux fronts se rejoignent.

    Chaque anneau rejoint la chaîne dont le dernier anneau est le plus proche ; quand un lobe se
    scinde en deux, la chaîne se duplique : chaque morceau reçoit sa propre spirale au lieu d'être
    abandonné, et la bande entre le dernier anneau du parent et le premier de l'enfant est
    interpolée comme les autres.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) / px_per_mm
    spacing = max(0.2, s.row_spacing_mm)
    top = float(distance.max())
    if top <= 0:
        return []
    limit = 1.6 * spacing

    chains, open_chains = [], []      # open_chains : [(index de chaîne, dernier anneau)]
    level = spacing / 2.0
    while level <= top + spacing / 2.0:
        region = (distance >= level).astype(np.uint8)
        rings = _level_rings(region, px_per_mm, s)
        if not rings:
            break
        targets = {}
        for ring in rings:
            best = None
            for index, previous in open_chains:
                found = _poly_distance(ring, previous)
                if best is None or found < best[0]:
                    best = (found, index)
            if best is None or best[0] > limit:
                chains.append([ring])
                targets.setdefault(len(chains) - 1, []).append((0.0, ring))
            else:
                targets.setdefault(best[1], []).append((best[0], ring))
        following = []
        for index, candidates in targets.items():
            candidates.sort(key=lambda c: c[0])
            chains[index].append(candidates[0][1])        # le plus proche continue la chaîne
            following.append((index, candidates[0][1]))
            for _, extra in candidates[1:]:               # les autres bifurquent
                chains.append(list(chains[index][:-1]) + [extra])
                following.append((len(chains) - 1, extra))
        open_chains = following
        level += spacing
    return chains


def _circle(cx, cy, radius, stitch_length):
    """Anneau circulaire, assez fin pour que l'interpolation reste fidèle à l'arrondi."""
    n = max(24, int(2.0 * math.pi * radius / max(0.35, stitch_length * 0.35)))
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return _closed(np.column_stack((cx + radius * np.cos(t), cy + radius * np.sin(t))))


# --- les trois remplissages -------------------------------------------------------------------
def _contour_runs(mask, px_per_mm, s):
    """Remplissage en contour : une spirale par lobe, du bord vers le centre."""
    stitch = max(0.6, min(s.max_stitch_mm, 2.0))
    runs = []
    for index, chain in enumerate(_ring_chains(mask, px_per_mm, s)):
        if not chain:
            continue
        path = (_make_spiral(chain, stitch, chain[0][0]) if len(chain) > 1 else chain[0])
        # `comp` distinct : le planificateur entre dans chaque lobe par son bord au lieu de
        # relier deux lobes éloignés par un déplacement piqué au travers du tissu.
        runs += _to_runs(_clip(path, mask, px_per_mm, MIN_SEGMENT_MM), s, comp=100 + index)
    return runs


def _circular_runs(mask, px_per_mm, s):
    """Remplissage circulaire : cercles concentriques reliés en double spirale de Fermat."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    cx, cy = float(xs.mean()) / px_per_mm, float(ys.mean()) / px_per_mm
    distance = float(np.hypot(xs / px_per_mm - cx, ys / px_per_mm - cy).max())
    spacing = max(0.2, s.row_spacing_mm)
    stitch = max(0.6, min(s.max_stitch_mm, 2.0))
    if spacing > distance:
        # Forme plus petite qu'une rangée : Ink/Stitch pique un unique cercle de la taille de
        # l'écart de rangée plutôt que de laisser la zone vide.
        return _to_runs([_circle(cx, cy, spacing, stitch)], s)
    circles = [_circle(cx, cy, 0.1, stitch)]   # petit cercle final, pour finir près du centre
    radius, stop = spacing, distance + spacing * 2.0
    while radius < stop:
        circles.append(_circle(cx, cy, radius, stitch))
        radius += spacing
    circles.reverse()   # de l'extérieur vers le centre, comme l'original
    path = _make_fermat_spiral(circles, stitch, circles[0][0])
    return _to_runs(_clip(path, mask, px_per_mm, MIN_SEGMENT_MM), s)


def _cross_runs(mask, px_per_mm, s):
    """Point de croix : grille de 2 mm, une rangée = un seul tracé continu.

    Une croix de côté `cell` a quatre coins A (haut gauche), B (haut droit), C (bas droit) et
    D (bas gauche). Elle se pique A → C → D → B : deux diagonales séparées par le bord bas, et
    le fil ressort au coin B, qui est justement le coin A de la croix suivante. La rangée se
    déroule donc sans jamais revenir sur ses pas ni changer le sens du croisement.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    h, w = mask.shape
    cell = max(2, round(CROSS_CELL_MM * px_per_mm))
    x0 = int(xs.min()) // cell * cell
    y0 = int(ys.min()) // cell * cell
    runs = []
    for row, y in enumerate(range(y0, int(ys.max()) + 1, cell)):
        mid_y = min(h - 1, y + cell // 2)
        # Une croix est posée dès que le centre de sa case tombe dans la zone : le bord du
        # remplissage reste crénelé, comme le veut le point de croix.
        cols = [x for x in range(x0, int(xs.max()) + 1, cell) if mask[mid_y, min(w - 1, x + cell // 2)]]
        groups = []
        for x in cols:
            if groups and x == groups[-1][-1] + cell:
                groups[-1].append(x)
            else:
                groups.append([x])
        for g, line in enumerate(groups):
            pts = [(line[0] / px_per_mm, y / px_per_mm)]
            for x in line:
                pts += [((x + cell) / px_per_mm, (y + cell) / px_per_mm),   # C, diagonale
                        (x / px_per_mm, (y + cell) / px_per_mm),           # D, bord bas
                        ((x + cell) / px_per_mm, y / px_per_mm)]           # B, diagonale
            if CROSS_THREADS > 1:   # deux brins : le fil refait la rangée à l'envers
                pts = pts + pts[-2::-1]
            runs.append({"p0": pts[0], "p1": pts[-1], "row": row,
                         "comp": 1_000_000 + row * 1000 + g,
                         "pts": [tuple(p) for p in pts[1:]]})
    return runs


def _to_runs(paths, s, comp=0):
    """Polylignes -> rangées du planificateur (p0, p1, pts), rééchantillonnées au pas des points.

    Une spirale est une courbe : le pas suit la convention des autres passages courbes du projet
    (`Sewer.sew_path`), soit 2 mm au plus. Avec la longueur maximale du remplissage (3 mm), un
    petit motif ne recevrait que deux ou trois points et le tracé partirait en cordes.
    """
    step = max(0.4, min(s.max_stitch_mm, 2.0))
    runs = []
    for n, path in enumerate(paths):
        pts = _resample(np.asarray(path, dtype=float), step)
        if len(pts) < 2:
            continue
        runs.append({"p0": tuple(pts[0]), "p1": tuple(pts[-1]), "comp": comp, "row": n,
                     "pts": [tuple(p) for p in pts[1:]]})
    return runs


# --- Interface -------------------------------------------------------------------------------
FILL_TYPES = {
    "inkstitch_contour": "Ink/Stitch : remplissage en contour",
    "inkstitch_circular": "Ink/Stitch : remplissage circulaire",
    "inkstitch_cross": "Ink/Stitch : point de croix",
}


def zone_fill(kind, mask, px_per_mm, s):
    """Passes de remplissage Ink/Stitch : même contrat que fill.zone_fill."""
    if mask is None or not mask.any():
        return []
    if kind == "inkstitch_contour":
        runs = _contour_runs(mask, px_per_mm, s)
    elif kind == "inkstitch_cross":
        runs = _cross_runs(mask, px_per_mm, s)
    else:
        runs = _circular_runs(mask, px_per_mm, s)
    return [("runs", runs)] if runs else []
