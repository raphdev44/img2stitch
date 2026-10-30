"""Moteur « industriel » : un parcours pensé pour la machine avant d'être pensé pour l'œil.

Objectif : le motif qui passe à tous les coups (peu de casses de fil, pas de nœuds, pas de plis) et qui
coûte le moins cher à produire (temps machine, coupes, arrêts couleur, fil). Les trois moteurs partagent
la même entrée (zones de couleur quantifiées) et le même contrat de sortie ; seul le parcours change.

Principes (chacun répond à un coût ou à un risque réel en production) :

1. **Ordre des couleurs sous contraintes** (``schedule_zones``) : seules les zones de couleurs différentes
   qui se touchent imposent un ordre (le dessus doit être piqué après le dessous). Tout le reste est
   libre : on regroupe les zones de même fil pour minimiser les arrêts couleur (~20-30 s chacun).
2. **Déplacements couverts, pas de coupes** (``Cover``) : un déplacement est piqué en point droit quand il
   reste dans une zone que le fil recouvrira (zone courante ou couleurs ultérieures). Quand la ligne
   droite sort, un A* cherche un chemin couvert. Une coupe (~3-5 s + points d'arrêt + risque de nœud)
   n'est faite qu'en dernier recours ; un saut court sans coupe est toléré sous ``short_jump_mm``.
3. **Remplissage continu en serpentin** : les rangées d'une forme sont enchaînées par de petits liens le
   long du bord (pas de saut entre rangées). Les points de pénétration sont décalés de rangée en rangée
   (rapport d'or) : pas de « fermeture éclair » qui perfore le tissu, et des points longs (3,6 mm) qui
   divisent le nombre de points par ~3 par rapport à un remplissage à 1 mm.
4. **Angle choisi au coût** : pour chaque forme, on compare quelques angles de hachure et on garde celui
   qui minimise points + déplacements non couverts. Les formes allongées (lettres, rubans) sont
   piquées en travers (effet satin) : rangées courtes, aucun saut, pas de rangée de 40 mm qui accroche.
5. **Formes fines = une ligne** : sous ``thin_width_mm``, une rangée n'a pas de sens (trop dense, fil qui
   casse). On pique une ligne médiane en triple point (aller-retour-aller : finit à l'autre bout).
6. **Sous-couche proportionnée** : aucune sous-couche sous 10 mm², contour seul jusqu'à 90 mm², puis
   contour + treillis perpendiculaire. La sous-couche sert aussi de chemin vers le début du remplissage.
7. **Garde-fous de fiabilité** : densité plancher (0,40 mm), longueur de point bornée
   [``min_stitch_mm``, ``max_stitch_mm``], aucun point plus court que le seuil (cause n°1 de nids et de
   casses d'aiguille), compensation de tension (``pull_mm``). ``check_pattern`` contrôle le résultat.

Le moteur pose lui-même ses points (aucune dépendance au ``Sewer`` raster) : il réutilise la géométrie
de ``algorithm_vectorize`` et le découpage en zones de ``algorithm_standard``.
"""
import heapq
import math
from collections import Counter
from dataclasses import dataclass, replace

import cv2
import numpy as np
import pyembroidery as pe

from .outline import overlap_masks
from .preprocess import load_and_quantize
from .algorithm_vectorize import explode, lines_of, polygons_of

ENGINE = "industrial"
PHI = 0.6180339887


@dataclass
class IndustrialParams:
    """Réglages du moteur. Les défauts visent le polo / casquette en 40 wt sur machine ~700-850 pts/min."""
    pull_mm: float = 0.25            # compensation de tension (les formes débordent sous le fil voisin)
    spacing_floor_mm: float = 0.40   # densité plancher : en dessous, casse de fil et plaque rigide
    fill_stitch_mm: float = 3.6      # pas du remplissage (pénétrations décalées)
    max_stitch_mm: float = 4.5       # plafond des liens, contours et déplacements
    min_stitch_mm: float = 0.35      # aucun point plus court : nids, casses d'aiguille
    min_run_mm: float = 0.8          # rangée plus courte : ignorée (sera recouverte)
    travel_stitch_mm: float = 3.0    # pas des déplacements couverts
    max_travel_mm: float = 60.0      # au-delà, une coupe est préférable à un long déplacement piqué
    short_jump_mm: float = 2.0       # saut sans coupe toléré en dessous (flottant court, invisible)
    trim_penalty: float = 45.0       # coût d'une coupe, en « points équivalents » (≈ 3,5 s à 800 pts/min)
    max_link_mm: float = 3.0         # lien maximal entre deux rangées voisines
    underlay: str = "auto"           # "auto" | "edge" | "none"
    underlay_inset_mm: float = 0.5
    underlay_step_mm: float = 3.0
    edge_min_mm2: float = 10.0       # sous-couche contour à partir de cette surface
    lattice_min_mm2: float = 90.0    # treillis perpendiculaire à partir de cette surface
    lattice_spacing_mm: float = 2.4
    thin_width_mm: float = 1.0       # largeur sous laquelle la forme devient une ligne
    thin_passes: int = 3             # 1 ou 3 (3 = triple point, finit à l'autre extrémité)
    thin_step_mm: float = 2.0
    satin_like_max_mm: float = 6.0   # petite dimension max pour piquer « en travers »
    angle_gain: float = 0.08         # un autre angle doit gagner 8 % pour détrôner l'angle réglé
    outline_step_mm: float = 2.2
    tiny_mm2: float = 0.8            # en dessous : traité comme une ligne (ou ignoré)
    force_color_grouping: bool = True


def get_params(settings):
    p = getattr(settings, "industrial", None)
    if isinstance(p, IndustrialParams):
        return p
    if isinstance(p, dict):
        return replace(IndustrialParams(), **p)
    return IndustrialParams()


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- Planification des couleurs ----------------------------------------------------------------
def schedule_zones(threads, masks, px_per_mm, overlap_mm):
    """Ordre de piqué qui minimise les arrêts couleur sans inverser un dessus et un dessous.

    Contrainte : a avant b si a précède b dans l'ordre d'origine ET que leurs fils diffèrent ET qu'elles se
    touchent (au rayon de chevauchement près). Ordonnancement topologique glouton : on reste sur le fil
    courant tant qu'une zone prête l'utilise ; sinon on change pour le fil qui a le plus de zones prêtes.
    """
    n = len(masks)
    r = max(1, round(max(overlap_mm, 0.1) * px_per_mm)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    boxes = [cv2.boundingRect(m.astype(np.uint8)) for m in masks]
    grown = [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks]

    def near(a, b):
        xa, ya, wa, ha = boxes[a]
        xb, yb, wb, hb = boxes[b]
        m = r + 1
        return not (xa + wa + m < xb or xb + wb + m < xa or ya + ha + m < yb or yb + hb + m < ya)

    pred = [set() for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if threads[a] != threads[b] and near(a, b) and (grown[a] & masks[b]).any():
                pred[b].add(a)
    done, order, thread = set(), [], None
    while len(order) < n:
        ready = [i for i in range(n) if i not in done and pred[i] <= done]
        same = [i for i in ready if threads[i] == thread]
        if same:
            pick = same[0]
        else:
            count = Counter(threads[i] for i in ready)
            best = max(count.values())
            pick = next(i for i in ready if count[threads[i]] == best)
        order.append(pick)
        done.add(pick)
        thread = threads[pick]
    return order


# --- Zones couvertes : où un fil flottant sera recouvert -----------------------------------------
class Cover:
    """Zones où un déplacement piqué est invisible (zone courante + couleurs piquées ensuite)."""

    def __init__(self, hidden, px_per_mm, cell_mm=0.3):
        self.m = hidden
        self.ppm = px_per_mm
        self.f = max(1, round(cell_mm * px_per_mm))
        self.cs = self.f / px_per_mm
        self._coarse = None
        self._labels = None

    def covered(self, p, q):
        n = max(1, int(_dist(p, q) / 0.4) + 1)
        xs = np.rint(np.linspace(p[0], q[0], n + 1) * self.ppm).astype(int)
        ys = np.rint(np.linspace(p[1], q[1], n + 1) * self.ppm).astype(int)
        h, w = self.m.shape
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= w or ys.max() >= h:
            return False
        return bool(self.m[ys, xs].all())

    def _grid(self):
        if self._coarse is None:
            f = self.f
            h, w = self.m.shape
            hh, ww = h // f, w // f
            if hh == 0 or ww == 0:
                self._coarse = np.zeros((1, 1), bool)
            else:
                self._coarse = self.m[:hh * f, :ww * f].reshape(hh, f, ww, f).all(axis=(1, 3))
            _, self._labels = cv2.connectedComponents(self._coarse.astype(np.uint8), connectivity=8)
        return self._coarse

    def _cell(self, p):
        return int(p[1] / self.cs), int(p[0] / self.cs)

    def _snap(self, p, radius=3):
        grid = self._grid()
        r0, c0 = self._cell(p)
        h, w = grid.shape
        best = None
        for r in range(r0 - radius, r0 + radius + 1):
            for c in range(c0 - radius, c0 + radius + 1):
                if 0 <= r < h and 0 <= c < w and grid[r, c]:
                    d = (r - r0) ** 2 + (c - c0) ** 2
                    if best is None or d < best[0]:
                        best = (d, r, c)
        return None if best is None else (best[1], best[2])

    def same_region(self, p, q):
        """Estimation rapide : p et q touchent la même zone couverte connexe (un chemin couvert existe)."""
        self._grid()
        a, b = self._snap(p), self._snap(q)
        return a is not None and b is not None and self._labels[a] == self._labels[b] and self._labels[a] > 0

    def route(self, p, q, max_len):
        """Chemin couvert p -> q (A* sur grille de 0,3 mm, puis tendu), ou None."""
        if _dist(p, q) > max_len:
            return None
        grid = self._grid()
        s, g = self._snap(p), self._snap(q)
        if s is None or g is None or self._labels[s] != self._labels[g]:
            return None
        h, w = grid.shape
        cs = self.cs
        hq = lambda r, c: math.hypot(r - g[0], c - g[1]) * cs
        heap = [(hq(*s), 0.0, s)]
        cost, parent = {s: 0.0}, {s: None}
        while heap:
            _, gc, node = heapq.heappop(heap)
            if node == g:
                break
            if gc > cost.get(node, 1e18) or gc > max_len:
                continue
            r, c = node
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not (dr or dc):
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w and grid[nr, nc]):
                        continue
                    ng = gc + (cs if not (dr and dc) else cs * 1.4142)
                    if ng < cost.get((nr, nc), 1e18):
                        cost[(nr, nc)] = ng
                        parent[(nr, nc)] = node
                        heapq.heappush(heap, (ng + hq(nr, nc), ng, (nr, nc)))
        if g not in parent:
            return None
        cells, node = [], g
        while node is not None:
            cells.append(node)
            node = parent[node]
        pts = [p] + [((c + 0.5) * cs, (r + 0.5) * cs) for r, c in reversed(cells)] + [q]
        out, i = [pts[0]], 0   # tendre le chemin : plus long segment couvert possible
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.covered(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        length = sum(_dist(out[k], out[k + 1]) for k in range(len(out) - 1))
        return out if length <= max_len else None


# --- Géométrie de remplissage -------------------------------------------------------------------
def hatch_runs(poly, angle_deg, spacing, min_run):
    """Rangées parallèles sur une grille globale (les formes voisines restent alignées).
    Retourne [(k, t0, t1)] : ligne k, abscisses t0 < t1 le long de la direction de rangée."""
    from shapely.geometry import LineString, MultiLineString

    a = math.radians(angle_deg)
    d, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    minx, miny, maxx, maxy = poly.bounds
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    ss = [x * n[0] + y * n[1] for x, y in corners]
    ts = [x * d[0] + y * d[1] for x, y in corners]
    k0, k1 = math.ceil(min(ss) / spacing), math.floor(max(ss) / spacing)
    if k1 < k0:
        return []
    t0, t1 = min(ts) - 1.0, max(ts) + 1.0
    lines = [LineString([(n[0] * k * spacing + d[0] * t0, n[1] * k * spacing + d[1] * t0),
                         (n[0] * k * spacing + d[0] * t1, n[1] * k * spacing + d[1] * t1)])
             for k in range(k0, k1 + 1)]
    runs = []
    for part in lines_of(poly.intersection(MultiLineString(lines))):
        c = list(part.coords)
        (x0, y0), (x1, y1) = c[0], c[-1]
        ta, tb = x0 * d[0] + y0 * d[1], x1 * d[0] + y1 * d[1]
        if ta > tb:
            ta, tb = tb, ta
        if tb - ta < min_run:
            continue
        k = round(((x0 + x1) / 2 * n[0] + (y0 + y1) / 2 * n[1]) / spacing)
        runs.append((k, ta, tb))
    return runs


def _build_strokes(runs, angle, spacing, stitch_mm, min_gap, link_ok, trans, hint, max_link):
    """Enchaîne les rangées en traits continus (serpentin). Retourne (traits, coût).

    Un trait est une liste de points à piquer d'un seul tenant. Un nouveau trait ne commence que si
    aucune rangée voisine n'est atteignable par un lien : c'est un « cul-de-sac », qui coûte un
    déplacement (couvert ou, au pire, une coupe). Coût = points piqués + déplacements.
    """
    a = math.radians(angle)
    d, nrm = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    pt = lambda k, t: (nrm[0] * k * spacing + d[0] * t, nrm[1] * k * spacing + d[1] * t)
    by_k = {}
    for i, (k, _, _) in enumerate(runs):
        by_k.setdefault(k, []).append(i)
    unvisited = set(range(len(runs)))
    has = lambda k: any(j in unvisited for j in by_k.get(k, ()))

    def run_points(i, forward):
        k, ta, tb = runs[i]
        lo, hi = (ta, tb)
        off = ((k * PHI) % 1.0) * stitch_mm
        m0, m1 = math.ceil((lo + min_gap - off) / stitch_mm), math.floor((hi - min_gap - off) / stitch_mm)
        mids = [m * stitch_mm + off for m in range(m0, m1 + 1)]
        ts = [lo] + mids + [hi]
        if not forward:
            ts.reverse()
        return [pt(k, t) for t in ts]

    strokes, cost, cur = [], 0.0, hint
    while unvisited:
        ends = [i for i in unvisited if not (has(runs[i][0] - 1) and has(runs[i][0] + 1))]
        best = None
        for i in (ends or unvisited):
            k, ta, tb = runs[i]
            for fwd, q in ((True, pt(k, ta)), (False, pt(k, tb))):
                c = trans(cur, q) if cur is not None else q[1] * 1e-3 + q[0] * 1e-6
                if best is None or c < best[0]:
                    best = (c, i, fwd)
        c, i, fwd = best
        cost += c if cur is not None else 0.0
        k = runs[i][0]
        up = sum(1 for j in unvisited if runs[j][0] > k)
        down = sum(1 for j in unvisited if runs[j][0] < k)
        dk = 1 if up >= down else -1
        pts = []
        while True:
            unvisited.discard(i)
            pts.extend(run_points(i, fwd))
            k = runs[i][0]
            e = pts[-1]
            nxt = None
            for step in (dk, -dk):
                bestj = None
                for j in by_k.get(k + step, ()):
                    if j not in unvisited:
                        continue
                    kj, ta, tb = runs[j]
                    for fj, q in ((True, pt(kj, ta)), (False, pt(kj, tb))):
                        dd = _dist(e, q)
                        if dd <= max_link and (bestj is None or dd < bestj[0]) and link_ok(e, q):
                            bestj = (dd, j, fj)
                if bestj:
                    nxt = (bestj, step)
                    break
            if nxt is None:
                break
            (_, i, fwd), dk = nxt
        strokes.append(pts)
        cost += len(pts)
        cur = pts[-1]
    return strokes, cost


def _principal(poly):
    """(angle de l'axe long en degrés, longueur, largeur) du rectangle minimal."""
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return None
    c = list(rect.exterior.coords)
    e1, e2 = (c[1][0] - c[0][0], c[1][1] - c[0][1]), (c[2][0] - c[1][0], c[2][1] - c[1][1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    v, long_, short = (e1, l1, l2) if l1 >= l2 else (e2, l2, l1)
    return math.degrees(math.atan2(v[1], v[0])), long_, short


def _seg_dev(p, a, b):
    """Distance du point p au segment [a, b]."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _decimate(pts, step, tol=0.15):
    """Allège une ligne brisée : points espacés d'au plus `step`, et jamais de corde qui s'écarte de
    plus de `tol` mm du tracé (sinon la sous-couche ou le contour sortirait de la forme dans les virages)."""
    n = len(pts)
    if n < 3:
        return list(pts)
    out, s, i = [pts[0]], 0, 2
    while i < n:
        a, b = pts[s], pts[i]
        if _dist(a, b) > step or max(_seg_dev(pts[j], a, b) for j in range(s + 1, i)) > tol:
            out.append(pts[i - 1])
            s = i - 1
            i = s + 2
        else:
            i += 1
    out.append(pts[-1])
    return out


# --- Machine : émission des points -----------------------------------------------------------------
class Stitcher:
    """Pose les points dans le motif (coordonnées en mm -> 1/10 mm) et décide de chaque déplacement."""

    def __init__(self, pattern, params):
        self.pt, self.P = pattern, params
        self.cur = None
        self.fresh = False   # vrai juste après un changement de couleur : le prochain déplacement est un saut libre
        self.cover = None
        self.fill_stitch = params.fill_stitch_mm

    # émission
    def _emit(self, cmd, p):
        self.pt.add_stitch_absolute(cmd, p[0] * 10, p[1] * 10)

    def stitch(self, p):
        self._emit(pe.STITCH, p)
        self.cur = p

    def jump(self, p):
        self._emit(pe.JUMP, p)
        self.cur = p

    def trim(self):
        self._emit(pe.TRIM, self.cur)

    def sew(self, pts, max_len):
        """Pique la ligne brisée ; scinde les segments > max_len, ignore ceux < min_stitch_mm."""
        for q in pts:
            d = _dist(self.cur, q)
            if d < self.P.min_stitch_mm:
                continue
            n = math.ceil(d / max_len - 1e-9)
            p = self.cur
            for i in range(1, n + 1):
                self.stitch((p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n))

    def start_at(self, q):
        """Premier point d'une couleur (l'arrêt machine a déjà coupé le fil)."""
        self.jump(q)
        self.stitch(q)

    # déplacements
    def trans_cost(self, p, q):
        """Coût approché d'un déplacement p -> q, en points équivalents (sert aux choix, pas à l'émission)."""
        if p is None:
            return 0.0
        d = _dist(p, q)
        if d < 1e-3:
            return 0.0
        P = self.P
        if d <= P.max_travel_mm and (self.cover.covered(p, q) or self.cover.same_region(p, q)):
            return 1.2 * d / P.travel_stitch_mm
        if d <= P.short_jump_mm:
            return 1.0
        return P.trim_penalty + 0.01 * d

    def go_to(self, q):
        """Amène l'aiguille en q, avec le moyen le moins cher et le plus sûr :
        point couvert > chemin couvert > saut court > coupe + saut."""
        p, P = self.cur, self.P
        if self.fresh:   # le changement de couleur a coupé le fil : saut libre, sans coupe
            self.fresh = False
            self.jump(q)
            self.stitch(q)
            return
        d = _dist(p, q)
        if d < P.min_stitch_mm:
            return
        if d <= P.max_travel_mm and self.cover.covered(p, q):
            self.sew([q], P.travel_stitch_mm)
            return
        if d <= P.short_jump_mm:
            self.jump(q)
            self.stitch(q)
            return
        path = self.cover.route(p, q, P.max_travel_mm)
        if path:
            self.sew(path[1:], P.travel_stitch_mm)
            return
        self.trim()
        self.jump(q)
        self.stitch(q)

    # formes
    def fill_strokes(self, poly, prepared, angle, spacing, stitch_mm, hint, runs=None):
        from shapely.geometry import LineString

        P = self.P
        runs = hatch_runs(poly, angle, spacing, P.min_run_mm) if runs is None else runs
        if not runs:
            return [], 0.0, runs

        def link_ok(e, q):
            return _dist(e, q) <= 1.2 or prepared.contains(LineString([e, q])) or self.cover.covered(e, q)

        strokes, cost = _build_strokes(runs, angle, spacing, stitch_mm, P.min_stitch_mm + 0.05,
                                       link_ok, self.trans_cost, hint, P.max_link_mm)
        return strokes, cost, runs

    def emit_strokes(self, strokes):
        for pts in strokes:
            if self.cur is None:
                self.start_at(pts[0])
            else:
                self.go_to(pts[0])
            self.sew(pts[1:], self.P.max_stitch_mm)

    def sew_loop(self, ring, step, passes=1):
        """Boucle fermée piquée depuis son sommet le plus proche de l'aiguille."""
        if len(ring) < 3:
            return
        if self.cur is not None:
            i = min(range(len(ring)), key=lambda k: _dist(self.cur, ring[k]))
            ring = ring[i:] + ring[:i]
        pts = _decimate(ring + [ring[0]], step)
        if self.cur is None:
            self.start_at(pts[0])
        else:
            self.go_to(pts[0])
        for _ in range(max(1, passes)):
            self.sew(pts[1:], step)

    def sew_line(self, pts, passes):
        """Ligne médiane : triple point (aller, retour, aller) pour une ligne nette et solide."""
        if len(pts) < 2:
            return
        pts = _decimate(pts, self.P.thin_step_mm)
        if self.cur is None:
            self.start_at(pts[0])
        else:
            self.go_to(pts[0])
        seq = pts
        for k in range(max(1, passes)):
            if k:
                seq = seq[::-1]
            self.sew(seq[1:], self.P.thin_step_mm)

    def choose_angle(self, poly, prepared, kind, base_angle, spacing):
        """Angle de hachure : réglage global, sauf gain net ou forme allongée (piquée en travers)."""
        P = self.P
        if kind in ("tatami_h", "tatami_v"):
            return base_angle, None
        axis = _principal(poly) if not list(poly.interiors) else None
        if axis and axis[1] / max(axis[2], 1e-6) >= 2.2 and axis[2] <= P.satin_like_max_mm:
            return axis[0] + 90.0, None
        cands = [base_angle, base_angle + 90.0]
        if axis and axis[1] / max(axis[2], 1e-6) >= 1.4:
            cands += [axis[0], axis[0] + 90.0]
        best = None
        for ang in cands:
            strokes, cost, runs = self.fill_strokes(poly, prepared, ang, spacing, self.fill_stitch, self.cur)
            if not strokes:
                continue
            score = cost * (1.0 if ang == base_angle else 1.0 + P.angle_gain)
            if best is None or score < best[0]:
                best = (score, ang, runs)
        return (best[1], best[2]) if best else (base_angle, None)

    def sew_thin(self, poly):
        """Forme fine : ligne médiane. Retourne False si rien n'a pu être piqué."""
        P = self.P
        lines = []
        if list(poly.interiors):   # anneau fin : la ligne médiane est l'extérieur rentré d'une demi-largeur
            from shapely.geometry import Polygon
            w = 2 * poly.area / max(poly.length, 1e-6)
            for g in explode(Polygon(poly.exterior).buffer(-w / 2, join_style=1, quad_segs=4)):
                lines.append(list(g.exterior.coords))
        else:
            axis = _principal(poly)
            ang = (axis[0] if axis else 0.0) + 90.0
            runs = sorted(hatch_runs(poly, ang, 0.7, 0.0))
            a = math.radians(ang)
            d, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
            cur = []
            for k, ta, tb in runs:
                t = (ta + tb) / 2
                q = (n[0] * k * 0.7 + d[0] * t, n[1] * k * 0.7 + d[1] * t)
                if cur and _dist(cur[-1], q) > 2.5:
                    lines.append(cur)
                    cur = []
                cur.append(q)
            if cur:
                lines.append(cur)
        done = False
        for pts in lines:
            if sum(_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) >= 1.0:
                self.sew_line(pts, P.thin_passes)
                done = True
        return done

    def underlay(self, poly, angle):
        P = self.P
        if P.underlay == "none" or poly.area < P.edge_min_mm2:
            return
        inner = poly.buffer(-P.underlay_inset_mm, join_style=1, quad_segs=4)
        from shapely.prepared import prep
        for g in explode(inner):
            if g.area < P.edge_min_mm2 * 0.4:
                continue
            self.sew_loop(list(g.exterior.coords)[:-1], P.underlay_step_mm)
            if P.underlay == "auto" and g.area >= P.lattice_min_mm2:
                strokes, _, _ = self.fill_strokes(g, prep(g.buffer(0.3)), angle + 90.0, P.lattice_spacing_mm,
                                                  self.fill_stitch, self.cur)
                self.emit_strokes(strokes)

    def sew_job(self, job, kind, base_angle, spacing, outline_passes):
        from shapely.prepared import prep

        P = self.P
        poly, loops = job["poly"], job["loops"]
        if poly is not None and kind not in ("none", "contour"):
            prepared = prep(poly.buffer(0.3))
            w = 2 * poly.area / max(poly.length, 1e-6)
            if poly.area < P.tiny_mm2 or w < P.thin_width_mm:
                if self.sew_thin(poly) or poly.area < P.tiny_mm2:
                    return                # la ligne médiane tient lieu de contour
            angle, runs = self.choose_angle(poly, prepared, kind, base_angle, spacing)
            self.underlay(poly, angle)
            strokes, _, _ = self.fill_strokes(poly, prepared, angle, spacing, self.fill_stitch, self.cur, runs)
            if not strokes:               # forme trop étroite pour une rangée
                self.sew_thin(poly)
            self.emit_strokes(strokes)
        elif poly is not None and kind == "contour":
            for g in explode(poly.buffer(-max(0.4, spacing * 0.6), join_style=1, quad_segs=4)):
                for ring in [g.exterior] + list(g.interiors):
                    self.sew_loop(list(ring.coords)[:-1], P.outline_step_mm)
        for ring in loops:
            self.sew_loop(ring, P.outline_step_mm, outline_passes)


# --- Assemblage ------------------------------------------------------------------------------------
def _jobs(fill_polys, shape_polys, want_outline):
    """Une tâche par forme à piquer : son remplissage et les contours qui lui appartiennent."""
    jobs = [{"poly": p, "loops": [], "bounds": p.bounds} for p in fill_polys]
    if want_outline:
        for sp in shape_polys:
            rings = [list(sp.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in sp.interiors]
            rep = sp.representative_point()
            owner = next((j for j in jobs if j["poly"] is not None and j["poly"].contains(rep)), None)
            if owner is None:
                xs, ys = zip(*rings[0])
                owner = {"poly": None, "loops": [], "bounds": (min(xs), min(ys), max(xs), max(ys))}
                jobs.append(owner)
            owner["loops"].extend(rings)
    return jobs


def _job_anchor(job, cur):
    """Point de la tâche le plus proche de l'aiguille."""
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    if job["poly"] is not None:
        q = nearest_points(job["poly"], Point(cur))[0]
        return (q.x, q.y)
    return min((v for ring in job["loops"] for v in ring), key=lambda v: _dist(cur, v))


def _next_job(jobs, cur, stitcher):
    if cur is None:
        return min(jobs, key=lambda j: (j["bounds"][1], j["bounds"][0]))
    def box_dist(j):
        x0, y0, x1, y1 = j["bounds"]
        return math.hypot(max(x0 - cur[0], 0, cur[0] - x1), max(y0 - cur[1], 0, cur[1] - y1))
    near = sorted(jobs, key=box_dist)[:6]
    def score(j):
        q = _job_anchor(j, cur)
        return stitcher.trans_cost(cur, q) + 0.02 * _dist(cur, q)
    return min(near, key=score)


def build_industrial_pattern(path, settings):
    """Image -> motif, parcours industriel. Même contrat que ``build_vector_pattern``."""
    from .algorithm_standard import _center_on_origin, _region_look, _split_regions, _zone_override
    from .algorithm_vectorize import _hatch_angle

    P = get_params(settings)
    px_per_mm = 10
    palette, masks = load_and_quantize(
        path, round(settings.width_mm * px_per_mm), settings.n_colors, settings.background,
        settings.background_tolerance, settings.background_scope, settings.exclude_colors,
        thicken_lines=settings.thicken_lines)
    pattern = pe.EmbPattern()
    if not masks:
        pattern.end()
        return pattern

    frame = {"w": masks[0].shape[1], "h": masks[0].shape[0], "ppm": px_per_mm, "cx": 0.0, "cy": 0.0, "scale": 1.0}
    palette, masks, regions = _split_regions(palette, masks, settings.region_overrides, px_per_mm)
    looks = [_zone_override(rgb, settings.zone_overrides) if regions[i] is None
             else _region_look(settings.region_overrides[regions[i]], rgb)
             for i, rgb in enumerate(palette)]
    if settings.merge_same_colors or P.force_color_grouping:
        order = schedule_zones([c for c, _, _ in looks], masks, px_per_mm, settings.overlap_mm)
        palette, masks, regions, looks = ([lst[k] for k in order] for lst in (palette, masks, regions, looks))
    fill_masks = overlap_masks(masks, px_per_mm, settings.overlap_mm)

    later = [None] * len(masks)
    acc = np.zeros_like(masks[0])
    for i in range(len(masks) - 1, -1, -1):
        later[i] = acc.copy()
        acc |= masks[i]

    sw = Stitcher(pattern, P)
    first, thread = True, None
    zones, segments = [], []
    reach = 2 * int(round(P.pull_mm * px_per_mm)) + 1
    tol = np.ones((reach, reach), np.uint8)
    min_area = max(0.5, settings.min_area_mm2)
    spacing = max(P.spacing_floor_mm, settings.row_spacing_mm)

    for i, (rgb, mask, fmask) in enumerate(zip(palette, masks, fill_masks)):
        color, kind, outline = looks[i]
        zone_index = len(zones)
        zones.append((tuple(rgb), regions[i]))
        want_outline = settings.outline if outline is None else outline
        fill_polys = polygons_of(fmask, px_per_mm, P.pull_mm, min_area) if kind != "none" else []
        shape_polys = polygons_of(mask, px_per_mm, 0.0, min_area) if (want_outline or kind == "none") else []
        jobs = _jobs(fill_polys, shape_polys, want_outline or kind == "none")
        if not jobs:
            continue

        same = color == thread
        if not same:
            if not first:
                pattern.add_command(pe.COLOR_CHANGE)
            pattern.add_thread({"color": (color[0] << 16) | (color[1] << 8) | color[2]})
        first, thread = False, color
        segments.append((len(pattern.stitches), zone_index))

        sw.cover = Cover(cv2.dilate((fmask | later[i]).astype(np.uint8), tol).astype(bool), px_per_mm)
        angle = _hatch_angle(kind, settings)
        remaining = list(jobs)
        if not same and sw.cur is not None:
            sw.fresh = True
        while remaining:
            job = _next_job(remaining, sw.cur, sw)
            remaining.remove(job)
            sw.sew_job(job, kind, angle, spacing, settings.outline_passes)

    pattern.end()
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_engine = ENGINE
    if settings.center_on_origin:
        frame["cx"], frame["cy"] = _center_on_origin(pattern)
    pattern.img2stitch_frame = frame
    return pattern


# --- Mesure : rentabilité et fiabilité ---------------------------------------------------------------
def production_report(pattern, spm=750, trim_s=4.0, color_change_s=25.0, jump_s=0.4):
    """Chiffres qui comptent pour le coût de production (à confronter aux trois moteurs)."""
    n = {"stitches": 0, "jumps": 0, "trims": 0, "color_changes": 0}
    length_mm, last = 0.0, None
    for x, y, c in pattern.stitches:
        c &= pe.COMMAND_MASK
        if c == pe.STITCH:
            n["stitches"] += 1
            if last is not None:
                length_mm += math.hypot(x - last[0], y - last[1]) / 10
            last = (x, y)
        elif c == pe.JUMP:
            n["jumps"] += 1
            last = (x, y)
        elif c == pe.TRIM:
            n["trims"] += 1
            last = None
        elif c == pe.COLOR_CHANGE:
            n["color_changes"] += 1
            last = None
    n["thread_m"] = round(length_mm * 1.3 / 1000 + 0.08 * (n["color_changes"] + 1), 2)
    n["est_minutes"] = round((n["stitches"] / spm * 60 + n["trims"] * trim_s
                              + n["color_changes"] * color_change_s + n["jumps"] * jump_s) / 60, 1)
    return n


def check_pattern(pattern, long_mm=6.5, short_mm=0.3, cell_mm=1.0, dense_per_mm2=6.0):
    """Contrôle de fiabilité : points trop longs (accrochent), trop courts (nids, casses d'aiguille),
    et densité de piqûres (plaque rigide, perforation du tissu)."""
    seg, pts, last = [], [], None
    for x, y, c in pattern.stitches:
        c &= pe.COMMAND_MASK
        if c == pe.STITCH:
            pts.append((x / 10, y / 10))
            if last is not None:
                seg.append(math.hypot(x - last[0], y - last[1]) / 10)
            last = (x, y)
        else:
            last = None
    if not pts:
        return {}
    a = np.array(pts)
    sx, sy = np.floor(a[:, 0] / cell_mm).astype(int), np.floor(a[:, 1] / cell_mm).astype(int)
    _, counts = np.unique(np.stack([sx, sy], 1), axis=0, return_counts=True)
    seg = np.array(seg) if seg else np.zeros(1)
    peak = counts.max() / (cell_mm * cell_mm)
    return {"max_stitch_mm": round(float(seg.max()), 2), "min_stitch_mm": round(float(seg.min()), 2),
            "stitches_too_long": int((seg > long_mm).sum()), "stitches_too_short": int((seg < short_mm).sum()),
            "peak_penetrations_per_mm2": round(float(peak), 1),
            "dense_cells": int((counts / (cell_mm * cell_mm) > dense_per_mm2).sum())}
