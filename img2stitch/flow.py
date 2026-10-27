"""Satin qui suit la forme ("satin tournant", comme sur les lettres brodées).

La ligne médiane de la zone (squelette) donne le sens du mouvement ; chaque point traverse la forme d'un
bord à l'autre perpendiculairement à ce mouvement, en alternant les côtés comme un satin classique.
Les parties que le satin ne couvre pas (embranchements, bouts arrondis) sont complétées en linéaire.
"""
import math
import cv2
import numpy as np


def _thin(mask):
    """Squelette d'un masque (amincissement de Zhang-Suen)."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            c = img[1:-1, 1:-1]
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(n.astype(np.int16) for n in nb[:8])
            A = sum(((nb[i] == 0) & (nb[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                m = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                m = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            rem = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & m
            if rem.any():
                c[rem] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _neighbours(skel):
    k = np.ones((3, 3), np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - skel.astype(np.uint8)


def _prune(skel, length_px):
    """Supprime les petites barbes du squelette (crans du bord) sans raccourcir les vraies extrémités."""
    sk = skel.copy()
    removed = np.zeros_like(sk)
    for _ in range(length_px):
        ends = sk & (_neighbours(sk) == 1)
        if not ends.any():
            break
        sk &= ~ends
        removed |= ends
    k = np.ones((3, 3), np.uint8)
    for _ in range(length_px):  # on regagne la longueur perdue, depuis les extrémités restantes seulement
        ends = sk & (_neighbours(sk) == 1)
        grow = cv2.dilate(ends.astype(np.uint8), k).astype(bool) & removed & ~sk
        if not grow.any():
            break
        sk |= grow
        removed &= ~grow
    return sk


def _crossings(skel):
    """Nombre de branches qui partent de chaque pixel (transitions 0->1 autour de lui)."""
    img = np.pad(skel.astype(np.uint8), 1)
    ring = [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2],
            img[:-2, :-2]]
    return sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))


def _branches(skel, min_len_px):
    """Branches du squelette (chemins de pixels ordonnés), coupées aux embranchements."""
    junction = skel & (_crossings(skel) >= 3)
    simple = skel & ~cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n, labels = cv2.connectedComponents(simple.astype(np.uint8), connectivity=8)
    out = []
    for lab in range(1, n):
        ys, xs = np.nonzero(labels == lab)
        if len(ys) < min_len_px:
            continue
        path = _longest_path(set(zip(xs.tolist(), ys.tolist())))
        if len(path) >= min_len_px:
            out.append(np.array(path, np.float32))
    return out


def _longest_path(pts):
    """Chemin le plus long d'un bout à l'autre d'une branche (double parcours en largeur) : les marches
    d'escalier du squelette ne coupent plus le tracé en morceaux."""
    def bfs(start):
        parent, queue, last = {start: None}, [start], start
        for cur in queue:
            last = cur
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (cur[0] + dx, cur[1] + dy)
                    if (dx or dy) and q in pts and q not in parent:
                        parent[q] = cur
                        queue.append(q)
        return last, parent

    u, _ = bfs(next(iter(pts)))
    v, parent = bfs(u)
    path = [v]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path


def _resample(path, step):
    """Lisse la polyligne puis la rééchantillonne à pas constant (pixels)."""
    if len(path) >= 5:
        kern = np.ones(5) / 5
        path = np.stack([np.convolve(np.pad(path[:, i], 2, mode="edge"), kern, "valid") for i in (0, 1)], 1)
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return path[[0, -1]] if len(path) > 1 else path
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])], 1)


def _ray(mask, p, d, max_px):
    """Dernier point du masque en partant de p dans la direction d."""
    ts = np.arange(0, max_px, 0.5)
    xs = np.rint(p[0] + ts * d[0]).astype(int)
    ys = np.rint(p[1] + ts * d[1]).astype(int)
    h, w = mask.shape
    ok = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    inside = np.zeros(len(ts), bool)
    inside[ok] = mask[ys[ok], xs[ok]]
    out = np.flatnonzero(~inside)
    t = ts[out[0] - 1] if len(out) and out[0] > 0 else (0.0 if len(out) else ts[-1])
    return p + t * d


def satin_flow_runs(mask, px_per_mm, spacing_mm=0.4, max_width_mm=25.0, split_mm=7.0, min_branch_mm=1.5):
    """Rangées de satin tournant : une par branche du squelette, avec `pts` = suite des piqûres (zigzag
    d'un bord à l'autre). Retourne (rangées, masque des parties non couvertes)."""
    # Bords lissés avant l'amincissement : chaque cran du contour ferait pousser une barbe au squelette
    smooth = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.6 * px_per_mm) > 0.5
    skel = _prune(_thin(smooth & mask), round(2.5 * px_per_mm))
    branches = _branches(skel, max(3, round(min_branch_mm * px_per_mm)))
    step = spacing_mm * px_per_mm
    half = max(1, round(1.0 * px_per_mm / step))  # tangente estimée sur ±1 mm
    covered = np.zeros(mask.shape, np.uint8)
    runs = []
    for bi, path in enumerate(branches):
        q = _resample(path, step)
        if len(q) < 2:
            continue
        # Traversées du cœur de la branche : (point, a, b) perpendiculaires au tracé
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
            continue
        width = float(np.median([np.hypot(*(a - b)) for _, a, b in core]))

        def clip(p, e):
            """Dans un angle vif, la perpendiculaire file le long de l'autre bras : on borne le point à la largeur."""
            v = e - p
            n = np.hypot(*v)
            return p + v * min(1.0, 0.75 * width / n) if n > 0 else e

        core = [(p, clip(p, a), clip(p, b)) for p, a, b in core]

        def extend(p0, direction, nrm):
            """Prolonge la branche tout droit jusqu'au bout de la forme (le squelette s'arrête toujours avant)."""
            out, h, w = [], *mask.shape
            for j in range(1, int((width + 2.5 * px_per_mm) / step) + 1):
                p = p0 + direction * step * j
                x, y = int(round(p[0])), int(round(p[1]))
                if not (0 <= x < w and 0 <= y < h and mask[y, x]):
                    break
                a = _ray(mask, p, nrm, max_width_mm * px_per_mm)
                b = _ray(mask, p, -nrm, max_width_mm * px_per_mm)
                wj = np.hypot(*(a - b))
                if wj < 0.3 * px_per_mm or wj > 1.6 * width:  # sortie de la forme, ou entrée dans une autre branche
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

        pts = []
        for _, a, b in crossings:
            cv2.line(covered, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), 1,
                     max(2, round(step * 2)))
            for e in (a, b):  # a toujours du même côté : a0 b0 a1 b1... zigzag ; point trop long coupé
                if pts:
                    n_seg = max(1, math.ceil(np.hypot(*(e - pts[-1])) / px_per_mm / split_mm))
                    pts += [pts[-1] + (e - pts[-1]) * k / n_seg for k in range(1, n_seg)]
                pts.append(e)
        if len(pts) < 2:
            continue
        mm = [(float(x) / px_per_mm, float(y) / px_per_mm) for x, y in pts]
        runs.append({"p0": mm[0], "p1": mm[-1], "comp": 10_000 + bi, "row": 0, "pts": mm[1:]})
    gaps = mask & ~cv2.dilate(covered, np.ones((3, 3), np.uint8)).astype(bool)
    return runs, gaps
