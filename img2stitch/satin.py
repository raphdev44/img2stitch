"""Satin bombé : la colonne de satin des lettrages et des rubans brodés.

Ce que l'on voit sur une lettre brodée n'est pas un remplissage : c'est une colonne satin
étagée. Le dôme ne vient pas d'un effet de rendu, il vient de la tension du fil :

1. GUIDES DE BORD (edge walk). Un tracé fin à 0,55 mm du contour, de chaque côté. Il fixe le
   tissu là où la couche finale s'arrêtera ; sans lui, le satin tire le bord et la lettre
   s'élargit de travers.
2. GUIDE DE CENTRE (center walk). Un tracé le long de l'axe médian : il remonte le centre.
3. SOUS-COUCHE SATIN à 75 % de la largeur. Elle donne le volume.
4. SATIN FINAL à 100 % de la largeur, avec compensation de reprise. Plus large que la
   sous-couche, il se rétracte sur ses bords : les bords s'enfoncent et le centre bombe.
   C'est exactement ce qui distingue un satin aplati d'un satin bombé.

Les trois premières couches sont marquées « under » : le rendu réaliste sait qu'elles sont
recouvertes par la couche finale et ne doit donc pas creuser leurs piqûres (sinon le dôme se
couvre de points sombres). La couche finale est marquée « top » : c'est elle qui porte le dôme.

Le module est indépendant de flow.py (satin qui suit la forme, sans relief) : ajouter le relief
ne doit rien changer aux motifs qui ne le demandent pas.
"""
import math

import cv2
import numpy as np

from .flow import _branches, _prune, _ray, _resample, _thin

DENSITY_MM = 0.30          # pas de la couche finale, le long de la bande (serré : l'arête est nette)
UNDERLAY_MM = 0.50         # pas de la sous-couche satin (plus aérée : elle doit rester dessous)
UNDERLAY_WIDTH = 0.75      # largeur de la sous-couche, en fraction de la bande
WALK_MM = 2.4              # pas des tracés de guidage
EDGE_INSET_MM = 0.55       # distance du guide de bord au contour
PULL_MM = 0.18             # compensation de reprise : le satin déborde pour contrer la tension
MAX_WIDTH_MM = 10.0        # au-delà, ce n'est plus une bande : le remplissage classique reprend
SPLIT_MM = 9.0             # un point plus long est coupé (les formats plafonnent à 12,1 mm)
MIN_BRANCH_MM = 1.5        # sous cette longueur, une branche du squelette est du bruit de bord

# Étiquette de couche portée par `relief` sur chaque rangée : les passes sont piquées séparément,
# mais le rendu doit pouvoir distinguer ce qui est recouvert de ce qui porte le dôme.
COMP_EDGE, COMP_CENTRE, COMP_UNDER, COMP_TOP = 200_000, 300_000, 400_000, 500_000


def _column(path, mask, px_per_mm, step, max_width_mm=MAX_WIDTH_MM):
    """Croisements perpendiculaires d'une branche : ([(point, bord a, bord b)], largeur médiane en px).

    Chaque croisement va d'un bord à l'autre de la bande. La branche est prolongée tout droit à
    ses deux bouts : le squelette s'arrête toujours avant le bord de la forme.
    """
    q = _resample(path, step)
    if len(q) < 2:
        return None
    half = max(1, round(1.0 * px_per_mm / step))  # tangente estimée sur ±1 mm
    core = []
    for i in range(len(q)):
        t = q[min(len(q) - 1, i + half)] - q[max(0, i - half)]
        n = np.hypot(*t)
        if n == 0:
            continue
        nrm = np.array([-t[1], t[0]]) / n
        a = _ray(mask, q[i], nrm, max_width_mm * px_per_mm)
        b = _ray(mask, q[i], -nrm, max_width_mm * px_per_mm)
        if np.hypot(*(a - b)) >= 0.3 * px_per_mm:
            core.append((q[i], a, b))
    if not core:
        return None
    width = float(np.median([np.hypot(*(a - b)) for _, a, b in core]))
    # Un croisement qui se resserre (bout de branche, éperon du squelette) fait converger le satin
    # en éventail : on l'écarte, la couche de fond s'en charge.
    core = [c for c in core if np.hypot(*(c[1] - c[2])) >= 0.5 * width]
    if not core:
        return None

    def clip(p, e):
        """Dans un angle vif, la perpendiculaire file le long de l'autre bras : on borne le point."""
        v = e - p
        n = np.hypot(*v)
        return p + v * min(1.0, 0.75 * width / n) if n > 0 else e

    core = [(p, clip(p, a), clip(p, b)) for p, a, b in core]

    def extend(p0, direction, nrm):
        """Prolonge la branche tout droit jusqu'au bout de la forme."""
        out, h, w = [], *mask.shape
        for j in range(1, int((width + 2.5 * px_per_mm) / step) + 1):
            p = p0 + direction * step * j
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < w and 0 <= y < h and mask[y, x]):
                break
            a = _ray(mask, p, nrm, max_width_mm * px_per_mm)
            b = _ray(mask, p, -nrm, max_width_mm * px_per_mm)
            wj = np.hypot(*(a - b))
            if wj < 0.3 * px_per_mm or wj > 1.6 * width:  # sortie de la forme, ou autre branche
                break
            if np.dot((a - b) / wj, nrm) < 0.8:  # le croisement a tourné : c'est l'autre bras
                break
            out.append((p, a, b))
        return out

    k = min(len(core) - 1, 2 * half)
    t0 = core[0][0] - core[k][0]
    t1 = core[-1][0] - core[-1 - k][0]
    crossings = core
    if np.hypot(*t0) > 0 and np.hypot(*t1) > 0:
        t0, t1 = t0 / np.hypot(*t0), t1 / np.hypot(*t1)
        n0 = (core[0][1] - core[0][2]) / max(1e-6, np.hypot(*(core[0][1] - core[0][2])))
        n1 = (core[-1][1] - core[-1][2]) / max(1e-6, np.hypot(*(core[-1][1] - core[-1][2])))
        crossings = extend(core[0][0], t0, n0)[::-1] + core + extend(core[-1][0], t1, n1)
    return crossings, width


def _row(points, comp, layer, px_per_mm):
    """Polyligne en pixels -> rangée du planificateur (p0, pts, p1 en mm)."""
    if points is None or len(points) < 2:
        return None
    mm = [(float(x) / px_per_mm, float(y) / px_per_mm) for x, y in points]
    return {"p0": mm[0], "p1": mm[-1], "comp": comp, "row": 0, "pts": mm[1:], "relief": layer}


def _crossing_stitches(crossings, px_per_mm, split_mm=SPLIT_MM):
    """Suite des piqûres d'une colonne : a0, b0, a1, b1… (un point par bord, comme un satin).

    Un point plus long que `split_mm` est coupé en deux : les formats de machine plafonnent à
    12,1 mm et une bande large dépasserait la limite d'encodage.
    """
    pts = []
    for _, a, b in crossings:
        for e in (a, b):
            if pts:
                n = max(1, math.ceil(np.hypot(*(e - pts[-1])) / px_per_mm / split_mm))
                pts += [pts[-1] + (e - pts[-1]) * j / n for j in range(1, n)]
            pts.append(e)
    return pts


def _inset(p, e, distance):
    """Point à `distance` du bord `e`, vers l'axe médian `p` (négatif : vers l'extérieur)."""
    v = p - e
    n = np.hypot(*v)
    return e + v * (distance / n) if n > 1e-6 else e


def relief_passes(mask, px_per_mm, s):
    """Passes du satin bombé, dans l'ordre de couture (fond, guides, sous-couche, satin).

    Même contrat que fill.zone_fill : liste de ("runs", rangées). Liste vide quand la forme n'a
    pas de bande exploitable ; l'appelant retombe alors sur un remplissage classique.
    """
    from .fill import fill_runs, zone_fill

    smooth = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.6 * px_per_mm) > 0.5
    skel = _prune(_thin(smooth & mask), round(2.5 * px_per_mm))
    branches = _branches(skel, max(3, round(MIN_BRANCH_MM * px_per_mm)))
    step = DENSITY_MM * px_per_mm
    covered = np.zeros(mask.shape, np.uint8)
    edges, centre, under, top = [], [], [], []
    for bi, path in enumerate(branches):
        column = _column(path, mask, px_per_mm, step)
        if column is None:
            continue
        crossings, width_px = column
        width_mm = width_px / px_per_mm
        if width_mm > MAX_WIDTH_MM:
            continue  # aplat : le satin n'est pas le bon outil, et la couverture revient au fond
        for _, a, b in crossings:
            cv2.line(covered, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), 1,
                     max(2, round(step * 2)))

        # 1) guides de bord : un tracé de chaque côté, à EDGE_INSET_MM du contour
        inset = min(EDGE_INSET_MM, 0.18 * width_mm) * px_per_mm
        for side in (0, 1):
            line = [_inset(c[0], c[side + 1], inset) for c in crossings
                    if np.hypot(*(c[0] - c[side + 1])) > inset * 1.3]
            row = _row(_resample(np.array(line), WALK_MM * px_per_mm) if len(line) > 1 else None,
                       COMP_EDGE + 100 * bi + side, "under", px_per_mm)
            if row:
                edges.append(row)

        # 2) guide de centre : le long de l'axe médian
        spine = np.array([p for p, _, _ in crossings])
        row = _row(_resample(spine, WALK_MM * px_per_mm), COMP_CENTRE + 100 * bi, "under", px_per_mm)
        if row:
            centre.append(row)

        # 3) sous-couche satin : 75 % de la largeur, resserrée vers l'axe
        shrink = 1.0 - UNDERLAY_WIDTH
        every = max(1, round(UNDERLAY_MM / DENSITY_MM))
        narrow = [(p, a + (p - a) * shrink, b + (p - b) * shrink) for p, a, b in crossings[::every]]
        row = _row(np.array(_crossing_stitches(narrow, px_per_mm)), COMP_UNDER + 100 * bi, "under", px_per_mm)
        if row:
            under.append(row)

        # 4) satin final : toute la largeur, débordant de la compensation de reprise
        # Compensation bornée par la largeur : sur un cordon fin, un débord fixe dépasserait le
        # contour et ferait des épines.
        pull = min(PULL_MM, 0.15 * width_mm) * px_per_mm
        full = [(p, _inset(p, a, -pull), _inset(p, b, -pull)) for p, a, b in crossings]
        row = _row(np.array(_crossing_stitches(full, px_per_mm)), COMP_TOP + 100 * bi, "top", px_per_mm)
        if row:
            top.append(row)

    # Coins et fourches : ce qui reste entre deux colonnes est couvert en satin, jamais en tatami.
    # Deux textures différentes se voient justement là où l'œil se pose : au coin d'un tronc.
    gaps = mask & ~cv2.dilate(covered, np.ones((3, 3), np.uint8)).astype(bool)
    passes = list(zone_fill("satin", gaps, px_per_mm, s)) if gaps.any() else []
    if not passes:
        passes = [("runs", fill_runs(gaps, px_per_mm, s.row_spacing_mm, s.fill_angle_deg, s.min_area_mm2))]
    passes += [("runs", layer) for layer in (edges, centre, under, top) if layer]
    return [p for p in passes if p[1]]
