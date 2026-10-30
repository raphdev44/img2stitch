"""Étape 3 : ordre de parcours. Objectif : le moins de sauts / coupes de fil possible.

Idée centrale : un déplacement entre deux points n'a PAS besoin d'être un saut s'il peut être
piqué "à couvert", c'est-à-dire dans une zone qui sera de toute façon recouverte (la forme
elle-même, ou une couleur piquée plus tard). Le fil ne se voit pas, on évite saut + coupe.
"""
import math
import cv2
import numpy as np
import pyembroidery as pe

UNIT = 10  # pyembroidery travaille en 1/10 de mm


class Sewer:
    """Écrit les commandes dans le motif et mémorise la position de l'aiguille."""

    def __init__(self, pattern, settings, px_per_mm):
        self.p, self.s, self.ppm = pattern, settings, px_per_mm
        self.cur = None
        self.fresh = True      # vrai juste après un changement de couleur : le premier déplacement est un saut "gratuit"
        self.region = None     # masque bool : zone où un déplacement piqué est acceptable (forme + couleurs suivantes)
        self.later = None      # masque bool : zone qui sera recouverte par une couleur piquée plus tard
        self.sewn = None       # masque uint8 : ce qui est déjà piqué dans la couleur en cours
        self.line_px = max(1, round(settings.row_spacing_mm * px_per_mm))
        self.jumps = 0         # sauts en cours de couleur (hors positionnement après changement de couleur)
        self.trims = 0

    # -- contexte ---------------------------------------------------------------------------
    def new_color(self, region, later):
        self.fresh = True
        self.region, self.later = region, later
        self.sewn = np.zeros(region.shape, np.uint8)

    # -- tests de déplacement ---------------------------------------------------------------
    def sewable(self, a, b):
        """Peut-on relier a et b en piquant, sans que ça se voie ?
        Oui si le trajet reste dans la zone acceptable ET ne passe pas sur plus de
        `max_visible_travel_mm` de fil déjà posé et non recouvert par une couleur suivante."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if d <= self.s.short_travel_mm:
            return True
        n = max(2, int(d * self.ppm / 2))
        xs = np.linspace(a[0], b[0], n) * self.ppm
        ys = np.linspace(a[1], b[1], n) * self.ppm
        h, w = self.region.shape
        ix = np.clip(np.rint(xs).astype(int), 0, w - 1)
        iy = np.clip(np.rint(ys).astype(int), 0, h - 1)
        if not self.region[iy, ix].all():
            return False
        visible = (self.sewn[iy, ix] > 0) & ~self.later[iy, ix]
        return visible.sum() * (d / n) <= self.s.max_visible_travel_mm

    def cost(self, a, b):
        """Coût d'un déplacement : sa longueur + une pénalité s'il faut sauter."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        return d if self.sewable(a, b) else d + self.s.jump_penalty_mm

    # -- écriture ---------------------------------------------------------------------------
    def _stitch(self, x, y):
        if self.cur is not None:
            cv2.line(self.sewn, (round(self.cur[0] * self.ppm), round(self.cur[1] * self.ppm)),
                     (round(x * self.ppm), round(y * self.ppm)), 1, self.line_px)
        self.p.add_stitch_absolute(pe.STITCH, x * UNIT, y * UNIT)
        self.cur = (x, y)

    def _line(self, a, b, max_len):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, math.ceil(d / max_len))
        for i in range(1, n + 1):
            t = i / n
            self._stitch(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def move_to(self, pt):
        """Amène l'aiguille en pt : piqûres à couvert si possible, sinon (coupe +) saut."""
        a = self.cur
        if a is None or self.fresh:
            self.p.add_stitch_absolute(pe.JUMP, pt[0] * UNIT, pt[1] * UNIT)
            self.cur, self.fresh = pt, False
        elif self.sewable(a, pt):
            self._line(a, pt, self.s.max_stitch_mm)
        else:
            if math.hypot(pt[0] - a[0], pt[1] - a[1]) > self.s.trim_threshold_mm:
                self.p.add_command(pe.TRIM)
                self.trims += 1
            self.p.add_stitch_absolute(pe.JUMP, pt[0] * UNIT, pt[1] * UNIT)
            self.jumps += 1
            self.cur = pt

    def sew_run(self, b, phase):
        """Pique de la position courante jusqu'à b, en décalant la grille de points selon `phase` (0..1)."""
        a, L = self.cur, math.hypot(b[0] - self.cur[0], b[1] - self.cur[1])
        m = self.s.max_stitch_mm
        t = m * (0.35 + 0.65 * phase) if self.s.stagger else m
        while t < L - 0.8:  # on garde un dernier point d'au moins 0,8 mm
            self._stitch(a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L)
            t += m
        self._stitch(*b)

    def sew_polyline(self, pts):
        for q in pts:
            self._line(self.cur, q, self.s.outline_stitch_mm)

    def sew_path(self, pts, step=None):
        """Pique le long d'une polyligne (rangée courbe) avec des points réguliers mesurés le long du tracé."""
        step = step or min(self.s.max_stitch_mm, 2.0)
        prev, acc = self.cur, 0.0
        for q in pts:
            seg = math.hypot(q[0] - prev[0], q[1] - prev[1])
            while seg > 0 and acc + seg >= step:
                t = (step - acc) / seg
                prev = (prev[0] + (q[0] - prev[0]) * t, prev[1] + (q[1] - prev[1]) * t)
                self._stitch(*prev)
                seg = math.hypot(q[0] - prev[0], q[1] - prev[1])
                acc = 0.0
            acc += seg
            prev = q
        if math.hypot(pts[-1][0] - self.cur[0], pts[-1][1] - self.cur[1]) > 1e-6:
            self._stitch(*pts[-1])


# ---------------------------------------------------------------------------------------------
def _nearest_candidates(cur, arr0, arr1, k=6):
    """Indices des k rangées dont une extrémité est la plus proche de cur."""
    d0 = np.hypot(arr0[:, 0] - cur[0], arr0[:, 1] - cur[1])
    d1 = np.hypot(arr1[:, 0] - cur[0], arr1[:, 1] - cur[1])
    d = np.minimum(d0, d1)
    k = min(k, len(d))
    return np.argpartition(d, k - 1)[:k] if k < len(d) else np.arange(len(d))


def plan_fill(sw: Sewer, runs):
    """Remplit toutes les rangées, forme par forme, en minimisant les sauts."""
    if not runs:
        return
    comps = {}
    for r in runs:
        comps.setdefault(r["comp"], []).append(r)
    todo = list(comps)

    while todo:
        # 1) Quelle forme piquer ensuite, et par quelle extrémité ? On n'entre que par le bord haut ou bas :
        #    partir du milieu obligerait à revenir plus tard (= un saut de plus).
        best = None
        for cid in todo:
            rs = comps[cid]
            rmin, rmax = min(r["row"] for r in rs), max(r["row"] for r in rs)
            for i, r in enumerate(rs):
                if r["row"] in (rmin, rmax):
                    for e, pt in enumerate((r["p0"], r["p1"])):
                        c = 0.0 if sw.cur is None else sw.cost(sw.cur, pt)
                        if best is None or c < best[0]:
                            best = (c, cid, i, e)
        _, cid, start, ent = best
        todo.remove(cid)

        # 2) Parcours glouton à l'intérieur de la forme
        rs = comps[cid]
        a0 = np.array([r["p0"] for r in rs])
        a1 = np.array([r["p1"] for r in rs])
        left = np.ones(len(rs), bool)
        i, e = start, ent
        while True:
            r = rs[i]
            entry, exit_ = (r["p0"], r["p1"]) if e == 0 else (r["p1"], r["p0"])
            sw.move_to(entry)
            if "path" in r:  # rangée courbe (ondulé, zigzag)
                sw.sew_path((r["path"] if e == 0 else r["path"][::-1])[1:])
            elif "pts" in r:  # piqûres imposées (remplissage en relief)
                layer, first = r.get("relief"), len(sw.p.stitches)
                for q in (r["pts"] if e == 0 else r["pts"][-2::-1] + [r["p0"]]):
                    sw._stitch(*q)
                if layer:  # le rendu réaliste a besoin de savoir ce qui porte le dôme
                    ranges = getattr(sw.p, "img2stitch_relief", None) or {}
                    ranges.setdefault(layer, []).append((first, len(sw.p.stitches)))
                    sw.p.img2stitch_relief = ranges
            elif "max_len" in r:  # satin : un seul point d'un bord à l'autre
                sw._line(sw.cur, exit_, r["max_len"])
            else:
                sw.sew_run(exit_, (r["row"] * 0.618034) % 1.0)
            left[i] = False
            if not left.any():
                break
            idx = np.where(left)[0]
            cand = idx[_nearest_candidates(sw.cur, a0[idx], a1[idx])]
            best_c = None
            for j in cand:
                for ee, pt in enumerate((rs[j]["p0"], rs[j]["p1"])):
                    c = sw.cost(sw.cur, pt)
                    if best_c is None or c < best_c[0]:
                        best_c = (c, j, ee)
            _, i, e = best_c


def plan_outline(sw: Sewer, loops, passes):
    """Pique les contours, boucle la plus proche d'abord."""
    left = list(range(len(loops)))
    arrs = [np.array(l) for l in loops]
    while left:
        best = None
        for li in left:
            d = np.hypot(arrs[li][:, 0] - sw.cur[0], arrs[li][:, 1] - sw.cur[1]) if sw.cur else np.zeros(len(arrs[li]))
            vi = int(d.argmin())
            c = 0.0 if sw.cur is None else sw.cost(sw.cur, loops[li][vi])
            if best is None or c < best[0]:
                best = (c, li, vi)
        _, li, vi = best
        left.remove(li)
        loop = loops[li][vi:] + loops[li][:vi]
        ring = loop + [loop[0]]
        sw.move_to(ring[0])
        sw.sew_polyline(ring[1:])
        for k in range(max(0, passes - 1)):  # retour sur le même tracé : trait plus dense
            sw.sew_polyline(list(reversed(ring[:-1])) if k % 2 == 0 else ring[1:])
