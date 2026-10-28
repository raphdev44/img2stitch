"""Remplissages en relief : module dédié, indépendant des remplissages classiques (fill.py).

Trois techniques, celles des logiciels professionnels (Wilcom, Hatch, BERNINA, PE-Design) :

1. BANDES DIRECTIONNELLES (« turning fill » / facettes). La forme est découpée en bandes de
   5-7 mm ; les rangées de remplissage de chaque bande sont piquées à un angle opposé
   (+45° / -45°, 0° / 90°). Le fil change de direction de part et d'autre de la frontière :
   arête nette, réflexion de la lumière opposée, effet de facette en volume. C'est exactement
   le principe des « turning stitches » de Hatch et des program splits Wilcom.

2. MOTIF FILL (cordon de motif par-dessus un fond tatami). Un réseau de motifs fermés
   (losanges, alvéoles, écailles, coquilles, pastilles, trèfles, étoiles, feuilles, spirales)
   est brodé PAR-DESSUS un fond tatami, en aller-retour. Le cordon est physiquement plus haut
   que le fond : il porte sa propre ombre et dessine le motif. C'est le « motif fill » de
   PE-Design / Hatch (Brother : motif patterns, Hatch : embossed fill).

3. VERMICELLE (stipple matelassé). Plusieurs tracés continus qui serpentent dans la forme avec
   amplitude et longueur d'onde variables : rendu matelassé aléatoire, très utilisé pour les
   fonds texturés (BERNINA stipple fill).

4. PELAGE (fur fill). Une nappe de points longs couchés dans le sens du poil, puis des mèches
éparses plus longues et légèrement incurvées : la lumière glisse le long des mèches, la
surface lit comme une fourrure. Deux longueurs (pelage ras, pelage long). Le fond suit le
poil : croisé à 90°, il dessinerait un cannage sous la fourrure.

5. PLUMES (feather fill). Chaque plume est une surface de satin dont les points traversent la
barbe, avec un ourlet piqué SOUS la barbe : c'est le gradin qui sépare deux plumes, pas un
contour d'autocollant. Les rangées se recouvrent comme des tuiles, et l'orientation alterne
de ±10° (plumes dressées) ou suit une trame à 55° (plumes couchées).

6. FORMES CELTIQUES. Triskèles : trois spirales en cordon satin autour d'un moyeu, sur un
pavage large (14,5 x 13 mm) pour rester lisibles. Entrelacs : deux familles de rubans satin
à 45° qui se croisent en alternance dessus/dessous ; chaque ruban est coupé au passage sous
l'autre, ce qui donne un vrai tissage en relief et non un simple quadrillage.

Toutes les cotes sont en millimètres ; l'échelle des motifs (5-7 mm) est celle des
bibliothèques pro : lisible de près sans devenir une bouillie de points.
"""
import math

import numpy as np

# --- Textures directionnelles (pelage) --------------------------------------------------------
# kind -> géométrie des brins. Toutes les cotes en mm.
FUR = {
    "relief_pelage": dict(angle=-78.0, across=1.7, along=3.0, length=9.0, wave=0.5, split=3.0),
    "relief_pelage_long": dict(angle=-52.0, across=2.4, along=3.8, length=13.0, wave=0.9, split=3.4),
}

# --- Plumes-----------------------------------------------------------------------------------
# kind -> (pas horizontal, pas vertical, angle de base). Les plumes sont couchées comme des
# tuiles : chaque rangée recouvre la précédente, et l'orientation alterne de ±10° autour de l'axe.
PLUME_TILING = {
    "relief_plumes": (6.2, 5.6, 12.0),
    "relief_plumes_couchees": (6.4, 5.8, 55.0),
}
FEATHER_L = 11.5
FEATHER_W = 4.0

# --- Bandes directionnelles ------------------------------------------------------------------
# kind -> (angle de la parité paire, angle de la parité impaire)
# Côtes : le fil court LE LONG de la nervure une bande sur deux, et EN TRAVERS sur l'autre.
# Deux textures franchement différentes (flûte / satin croisé) : la nervure se lit en relief.
BAND_ANGLES = {
    "relief_cotes": (90.0, 0.0),
    "relief_chevrons": (45.0, -45.0),
    "relief_vagues": (45.0, -45.0),
    "relief_damier": (45.0, -45.0),
    "relief_vannerie": (0.0, 90.0),
}

# --- Motif fill ------------------------------------------------------------------------------
# kind -> (mode de pavage, pas horizontal, pas vertical)   modes : "grid", "brick", "hex"
TILINGS = {
    "relief_losanges": ("grid", 6.0, 6.0),
    "relief_nid_abeille": ("hex", 4.35, 5.02),
    "relief_ecailles": ("brick", 5.4, 2.7),
    "relief_coquilles": ("brick", 6.4, 3.2),
    "relief_pastilles": ("brick", 5.2, 5.2),
    "relief_trefle": ("grid", 6.0, 6.0),
    "relief_etoiles": ("brick", 6.8, 6.8),
    "relief_feuilles": ("brick", 6.0, 6.0),
    "relief_spirales": ("brick", 6.0, 6.0),
    "relief_triskele": ("brick", 14.5, 13.0),
}

MOTIF_STEP_MM = 0.45   # pas du zigzag le long du cordon (satin serré)
MOTIF_MIN_MM = 0.8     # en dessous, le bout de motif n'apporte rien (le contour couvre)
MOTIF_CORD_MM = 1.15   # largeur du cordon : un fil de satin, pas un simple trait
MOTIF_CORD_MIN_MM = 0.55
MOTIF_CORD_SCALE = {"relief_triskele": 0.85}
BASE_OFFSET_DEG = 30.0  # le fond ne doit JAMAIS être parallèle aux lignes du motif


def _band_masks(kind, mask, px_per_mm):
    """Deux masques complémentaires (parité de bande) découpés par le champ de phase du motif.

    Chaque famille de bandes reçoit ensuite un angle de remplissage différent : c'est la
    frontière entre les deux qui se lit en relief.
    """
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx / px_per_mm, yy / px_per_mm
    if kind == "relief_cotes":
        phase = np.floor(x / 5.0)                       # côtes verticales de 5 mm
    elif kind == "relief_chevrons":
        tri = 2.0 * np.abs(2.0 * (y / 6.0 - np.floor(y / 6.0 + 0.5))) - 1.0
        phase = np.floor((x + 2.5 * tri) / 5.0)          # frontières en V (pointe 5 mm)
    elif kind == "relief_vagues":
        phase = np.floor((x + 2.2 * np.sin(2.0 * np.pi * y / 8.0)) / 5.0)
    elif kind == "relief_damier":
        phase = np.floor(x / 6.0) + np.floor(y / 6.0)    # damier de blocs 6 x 6 mm
    else:  # vannerie : blocs 5 x 5 mm alternés (chaîne / trame)
        phase = np.floor(x / 5.0) + np.floor(y / 5.0)
    odd = (phase.astype(np.int64) % 2) == 1
    return mask & odd, mask & ~odd


# --- Cordon d'arête (matelassage) --------------------------------------------------------------
# Une frontière de bande n'est lisible que si elle porte un cordon en relief : sans lui, deux
# champs de fils de même densité se contentent de changer de teinte. On pique donc un satin
# étroit (0,95 mm) en zigzag le long de chaque arête : fil deux fois plus épais, arête franche,
# ombre portée. C'est le principe des remplissages « quilted » / satin-in-tatami.
CORD_PERIOD = {"relief_cotes": 5.2, "relief_chevrons": 5.0, "relief_vagues": 5.0}
CORD_WIDTH_MM = 0.95
CORD_STEP_MM = 0.45


def _boundary_x(kind, k, yy):
    """Abscisse de l'arête n° k à hauteur yy (les arêtes sont des courbes quasi verticales)."""
    period = CORD_PERIOD[kind]
    if kind == "relief_cotes":
        return np.full(len(yy), k * period)
    if kind == "relief_chevrons":
        tri = 2.0 * np.abs(2.0 * (yy / 6.0 - np.floor(yy / 6.0 + 0.5))) - 1.0
        return k * period - 2.5 * tri
    return k * period - 2.2 * np.sin(2.0 * np.pi * yy / 8.0)


def _cord_runs(kind, mask, px_per_mm):
    """Satin en zigzag le long des arêtes : les points alternent de part et d'autre de la ligne.

    Seuls les motifs à arêtes quasi verticales (côtes, chevrons, vagues) en profitent : damier et
    vannerie ont des frontières en quadrillage, déjà franches par le croisement 0° / 90°.
    """
    if kind not in CORD_PERIOD:
        return []
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    y0, y1 = ys.min() / px_per_mm - 0.5, ys.max() / px_per_mm + 0.5
    x0, x1 = xs.min() / px_per_mm - 2.0, xs.max() / px_per_mm + 2.0
    period, half = CORD_PERIOD[kind], CORD_WIDTH_MM / 2.0
    yy = np.arange(y0, y1, CORD_STEP_MM)
    sign = np.where(np.arange(len(yy)) % 2 == 0, -1.0, 1.0)

    runs = []
    for k in range(int(math.floor(x0 / period)) - 1, int(math.ceil(x1 / period)) + 2):
        bx = _boundary_x(kind, k, yy) + sign * half
        poly = np.column_stack((bx, yy))
        for seg in _clip(poly, mask, px_per_mm, 2.0):
            if len(seg) < 3:
                continue
            runs.append({"p0": tuple(seg[0]), "p1": tuple(seg[-1]), "comp": 7,
                         "row": k, "pts": [tuple(p) for p in seg[1:]]})
    return runs


def _band_fill(kind, mask, px_per_mm, s):
    """Deux champs de fils croisés, puis le cordon d'arête par-dessus."""
    from .fill import fill_runs   # remplissage classique réutilisé tel quel

    out = []
    for k, (sub, angle) in enumerate(zip(_band_masks(kind, mask, px_per_mm), BAND_ANGLES[kind])):
        if not sub.any():
            continue
        runs = fill_runs(sub, px_per_mm, s.row_spacing_mm, angle, s.min_area_mm2)
        for r in runs:
            r["comp"] += k * 1000003   # deux espaces d'étiquettes distincts
        out += runs
    if not out:
        return []
    cords = _cord_runs(kind, mask, px_per_mm)
    return [("runs", out), ("runs", cords)] if cords else [("runs", out)]


# --- Géométrie des motifs --------------------------------------------------------------------
def _closed(poly):
    """Referme une polyligne (le premier point est répété en fin)."""
    return np.vstack((poly, poly[:1]))


def _circle(cx, cy, r, n=16):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))


def _arc(cx, cy, r, a0, a1, n=15):
    t = np.linspace(a0, a1, n)
    return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))


def _bezier(p0, ctrl, p1, n=10):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def _rotate(poly, cx, cy, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    x, y = poly[:, 0] - cx, poly[:, 1] - cy
    return np.column_stack((cx + x * ca - y * sa, cy + x * sa + y * ca))


def _triskele(cx, cy, flip):
    """Triskèle : trois spirales rayonnantes et un cœur, tournant dans un sens ou l'autre."""
    arms = []
    for k in range(3):
        t = np.linspace(0.0, 1.0, 30)
        a = flip * (2.0 * np.pi * 1.55 * t + k * 2.0 * np.pi / 3.0)
        r = 0.4 + 4.4 * t
        arms.append(np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a))))
    return arms + [_closed(_circle(cx, cy, 0.8, 12))]


def _motifs(kind, cx, cy, i, j):
    """Motifs d'une cellule (centre cx, cy) : polylignes en mm, refermées si fermées."""
    if kind == "relief_losanges":
        h = 3.0
        return [_closed(np.array([(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy)]))]
    if kind == "relief_nid_abeille":
        r = 2.9                                             # alvéole 5.8 x 5.02 mm
        a = np.arange(6) * np.pi / 3.0
        return [_closed(np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a))))]
    if kind == "relief_ecailles":
        return [_closed(_arc(cx, cy, 2.7, 0.0, np.pi))]     # écaille en U, rangées imbriquées
    if kind == "relief_coquilles":
        r = 3.2
        arc = _arc(cx, cy, r, np.pi, 2.0 * np.pi)           # éventail ouvert vers le haut
        ribs = [np.array([(cx, cy), (cx + r * math.cos(t), cy + r * math.sin(t))])
                for t in (7 * np.pi / 6, 3 * np.pi / 2, 11 * np.pi / 6)]
        return [_closed(arc)] + ribs
    if kind == "relief_pastilles":
        return [_closed(_circle(cx, cy, 1.85))]
    if kind == "relief_trefle":
        lobes = [_circle(cx + dx, cy + dy, 1.05)
                 for dx, dy in ((1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5))]
        return [_closed(p) for p in lobes] + [_closed(_circle(cx, cy, 0.42, 8))]
    if kind == "relief_etoiles":
        a = np.arange(12) * np.pi / 6.0
        rr = np.where(np.arange(12) % 2 == 0, 2.7, 1.05)
        return [_closed(np.column_stack((cx + rr * np.cos(a), cy + rr * np.sin(a))))]
    if kind == "relief_feuilles":
        a, b = 2.3, 1.15
        p0, p2 = np.array([cx, cy - a]), np.array([cx, cy + a])
        loop = np.vstack((_bezier(p0, np.array([cx + b * 1.5, cy]), p2),
                          _bezier(p2, np.array([cx - b * 1.5, cy]), p0)))
        vein = np.array([[cx, cy - a * 0.85], [cx, cy + a * 0.85]])
        ang = 35.0 if (i + j) % 2 == 0 else -35.0
        return [_closed(_rotate(loop, cx, cy, ang)), _rotate(vein, cx, cy, ang)]
    if kind == "relief_triskele":
        return _triskele(cx, cy, 1.0 if (i + j) % 2 == 0 else -1.0)
    if kind == "relief_spirales":
        t = np.linspace(0.0, 1.0, 34)
        ang = 2.0 * np.pi * 1.4 * t
        r = 0.45 + 1.95 * t
        sign = 1.0 if (i + j) % 2 == 0 else -1.0
        return [np.column_stack((cx + sign * r * np.cos(ang), cy + r * np.sin(ang)))]
    return []


def _tile_positions(mode, dx, dy, x0, y0, x1, y1):
    """Centres des cellules couvrant le rectangle, avec le décalage du mode de pavage."""
    nx = int(math.ceil((x1 - x0) / dx)) + 1
    ny = int(math.ceil((y1 - y0) / dy)) + 1
    out = []
    for j in range(-1, ny + 2):
        cy = y0 + j * dy
        shift = dx / 2.0 if (mode == "brick" and j % 2) else 0.0
        for i in range(-1, nx + 2):
            cx = x0 + i * dx + shift
            out.append((i, j, cx, cy + (dy / 2.0 if (mode == "hex" and i % 2) else 0.0)))
    return out


def _clip(poly, mask, px_per_mm, min_len_mm):
    """Découpe une polyligne aux limites de la forme et écarte les petits bouts."""
    h, w = mask.shape
    ix = np.rint(poly[:, 0] * px_per_mm).astype(int)
    iy = np.rint(poly[:, 1] * px_per_mm).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    inside = np.zeros(len(poly), bool)
    inside[ok] = mask[iy[ok], ix[ok]]

    segs, start = [], None
    for k, ins in enumerate(inside):
        if ins and start is None:
            start = k
        elif not ins and start is not None:
            segs.append(poly[start:k])
            start = None
    if start is not None:
        segs.append(poly[start:])

    out = []
    for seg in segs:
        if len(seg) < 2:
            continue
        length = np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])).sum()
        if length >= min_len_mm:
            out.append(seg)
    return out


def _resample(poly, step):
    """Points régulièrement espacés le long du tracé (le cordon doit être homogène)."""
    d = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    cum = np.concatenate(([0.0], np.cumsum(d)))
    total = float(cum[-1])
    if total <= 1e-6:
        return poly
    n = max(2, int(math.ceil(total / step)) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack((np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])))


def _clip_segment(a, b, mask, px_per_mm, n=14):
    """Partie utile d'un point qui sort de la forme : on garde le plus long tronçon intérieur.

    Une plume ou une mèche posée sur le bord doit s'arrêter net au contour, pas dépasser de la
    zone. On échantillonne le segment et on renvoie ses extrémités recadrées.
    """
    pts = np.linspace(a, b, n)
    h, w = mask.shape
    ix = np.rint(pts[:, 0] * px_per_mm).astype(int)
    iy = np.rint(pts[:, 1] * px_per_mm).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    inside = np.zeros(len(pts), bool)
    inside[ok] = mask[iy[ok], ix[ok]]
    if not inside.any():
        return None
    best, k = (0, 0), 0
    while k < n:
        if not inside[k]:
            k += 1
            continue
        m = k
        while m + 1 < n and inside[m + 1]:
            m += 1
        if m - k > best[1] - best[0]:
            best = (k, m)
        k = m + 1
    return pts[best[0]], pts[best[1]]


def _feather_runs(kind, mask, px_per_mm, s):
    """Plumes pleines : satin en travers de la barbe, puis rachis en points longs.

    Une plume n'est pas un contour : c'est une surface de satin dont les points traversent la
    barbe. Le sens du fil est donc perpendiculaire au rachis ; quand on incline la plume, la
    lumière accroche la barbe et la plume se détache de sa voisine. C'est la construction des
    palettes « feather » (Brother motif fill, Hatch feather pattern), sans la lourdeur d'un
    cordon d'ourlet qui ferait un autocollant.
    """
    dx, dy, base_deg = PLUME_TILING[kind]
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [], []
    x0, y0 = xs.min() / px_per_mm, ys.min() / px_per_mm
    x1, y1 = xs.max() / px_per_mm, ys.max() / px_per_mm
    L, W = FEATHER_L, FEATHER_W
    pitch = 0.45
    rims, vane, spines = [], [], []
    for i, j, cx, cy in _tile_positions("brick", dx, dy, x0, y0, x1, y1):
        deg = math.radians(base_deg + (10.0 if (i + j) % 2 == 0 else -10.0))
        ca, sa = math.cos(deg), math.sin(deg)

        def place(lx, ly):
            return (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

        # Ourlet de la plume, piqué SOUS la barbe : il ne dessine pas un contour d'autocollant,
        # mais le gradin qui sépare deux plumes et fait l'ombre de la tuile.
        t = np.linspace(0.0, 1.0, 22)
        y = L / 2.0 - L * t
        bend = 0.32 * np.sin(np.pi * t)
        half = W / 2.0 * np.sin(np.pi * t) ** 0.55
        loop = _closed(np.vstack((np.column_stack((bend - half, y)),
                                  np.column_stack((bend + half, y))[::-1])))
        loop = np.array([place(q[0], q[1]) for q in loop])
        for seg in _clip(loop, mask, px_per_mm, 1.2):
            pts = _resample(seg, 0.55)
            zig = _satin_cord(pts, 0.9)
            if zig is not None:
                rims.append({"p0": tuple(zig[0]), "p1": tuple(zig[-1]),
                             "pts": [tuple(q) for q in zig[1:]],
                             "comp": 20000 + j, "row": j})

        for n, tt in enumerate(np.arange(0.05, 0.96, pitch / L)):
            y = L / 2.0 - L * tt
            bend = 0.32 * math.sin(math.pi * tt)
            h = W / 2.0 * math.sin(math.pi * tt) ** 0.55
            seg = _clip_segment(place(bend - h, y), place(bend + h, y), mask, px_per_mm, 8)
            if seg is None:
                continue
            vane.append({"p0": tuple(seg[0]), "p1": tuple(seg[1]),
                         "max_len": min(s.max_stitch_mm, 4.0),
                         "comp": 30000 + j, "row": j, "n": n})
        t = np.linspace(0.02, 0.98, 12)
        spine = np.array([place(0.32 * math.sin(math.pi * k), L / 2.0 - L * k) for k in t])
        for seg in _clip(spine, mask, px_per_mm, 0.8):
            pts = _resample(seg, 2.6)
            if len(pts) >= 2:
                spines.append({"p0": tuple(pts[0]), "p1": tuple(pts[-1]),
                               "pts": [tuple(q) for q in pts[1:]],
                               "comp": 40000 + j, "row": j})
    return rims, vane, spines


def _satin_cord(pts, width):
    """Transforme une ligne médiane en cordon de satin : les points alternent de part et d'autre.

    Sans cela, un motif n'est qu'un fil d'un seul brin posé sur le fond : il ne fait aucune
    ombre et ne se lit pas en relief. Avec deux points par pas, le fil couvre 1,15 mm de large
    et le motif devient une pièce bombée, comme un « embossed fill » de Hatch.
    """
    if len(pts) < 3:
        return None
    d = np.gradient(pts, axis=0)
    nrm = np.column_stack((-d[:, 1], d[:, 0]))
    ln = np.hypot(nrm[:, 0], nrm[:, 1])
    ln[ln == 0] = 1.0
    nrm = nrm / ln[:, None]
    side = np.where(np.arange(len(pts)) % 2 == 0, 1.0, -1.0)[:, None]
    return pts + nrm * side * (width / 2.0)


def _cord_width(poly):
    """Largeur du cordon adaptée à la taille du motif : un petit lobe ne porte pas 1,15 mm."""
    span = poly.max(axis=0) - poly.min(axis=0)
    return float(np.clip(0.55 * min(span), MOTIF_CORD_MIN_MM, MOTIF_CORD_MM))


def _motif_runs(kind, mask, px_per_mm, s):
    """Cordon de satin du motif, brodé en aller-retour par-dessus le fond."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    mode, dx, dy = TILINGS[kind]
    x0, y0 = xs.min() / px_per_mm, ys.min() / px_per_mm
    x1, y1 = xs.max() / px_per_mm, ys.max() / px_per_mm

    runs = []
    for i, j, cx, cy in _tile_positions(mode, dx, dy, x0, y0, x1, y1):
        for poly in _motifs(kind, cx, cy, i, j):
            for seg in _clip(poly, mask, px_per_mm, MOTIF_MIN_MM):
                pts = _resample(seg, MOTIF_STEP_MM)
                if len(pts) < 3:
                    continue
                # cordon satin (points alternés de part et d'autre), puis aller-retour :
                # le fil repasse sur lui-même et gagne en hauteur.
                zig = _satin_cord(pts, _cord_width(poly) * MOTIF_CORD_SCALE.get(kind, 1.0))
                if zig is None:
                    continue
                # Les points de cordon qui dépassent la forme reviennent sur la ligne médiane :
                # le motif s'arrête au contour au lieu de baver hors de la zone.
                ix = np.rint(zig[:, 0] * px_per_mm).astype(int)
                iy = np.rint(zig[:, 1] * px_per_mm).astype(int)
                inb = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
                ok = np.zeros(len(zig), bool)
                ok[inb] = mask[iy[inb], ix[inb]]
                zig[~ok] = pts[~ok]
                pts = np.vstack((zig, zig[-2::-1]))
                runs.append({"p0": tuple(pts[0]), "p1": tuple(pts[-1]), "comp": 0,
                             "row": j, "pts": [tuple(p) for p in pts[1:]]})
    return runs


def _base_runs(mask, px_per_mm, s, angle=None):
    """Fond tatami sous le cordon.

    L'angle est décalé de 30° par rapport à celui de l'utilisateur : quand les rangées du fond
    sont parallèles aux lignes du motif (losanges à 45° sur un fond à 45°), le cordon ne se
    distingue plus du fond et le motif disparaît. C'est la règle d'or des program splits Wilcom.
    `angle` permet aux textures directionnelles de choisir un fond franchement perpendiculaire.
    """
    from .fill import fill_runs

    ang = s.fill_angle_deg + BASE_OFFSET_DEG if angle is None else angle
    return fill_runs(mask, px_per_mm, s.row_spacing_mm, ang, s.min_area_mm2)


# --- Pelage : mèches longues dans un champ directeur ------------------------------------------
def _fur_runs(kind, mask, px_per_mm):
    """Mèches couchées dans le sens du poil, en points longs.

    Un poil n'est PAS un satin : un satin court sur 0,6 mm de large n'a que deux points et
    son grain perpendiculaire transforme la fourrure en cannage. La technique des remplissages
    « fur » (Wilcom, Hatch) est l'inverse : des points longs (2-3 mm) posés DANS le sens de la
    mèche, sur un fond fin qui bouche les trous. Le fil se couche dans le poil, la lumière
    glisse le long des mèches et la surface lit comme une fourrure.
    """
    cfg = FUR[kind]
    ang = math.radians(cfg["angle"])
    d = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-d[1], d[0]])
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    x0, x1 = xs.min() / px_per_mm - 2.0, xs.max() / px_per_mm + 2.0
    y0, y1 = ys.min() / px_per_mm - 2.0, ys.max() / px_per_mm + 2.0
    corners = np.array([(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
    pd, pn = corners @ d, corners @ nrm
    rng = np.random.default_rng(761 + (17 if kind.endswith("long") else 0))

    runs = []
    j0 = int(math.floor(pd.min() / cfg["along"])) - 1
    i0 = int(math.floor(pn.min() / cfg["across"])) - 1
    for j in range(j0, int(math.ceil(pd.max() / cfg["along"])) + 2):
        s0 = j * cfg["along"] + (cfg["along"] / 2.0 if j % 2 else 0.0) + rng.uniform(-0.35, 0.35)
        for i in range(i0, int(math.ceil(pn.max() / cfg["across"])) + 2):
            n0 = i * cfg["across"] + rng.uniform(-0.2, 0.2) * cfg["across"]
            base = d * s0 + nrm * n0
            if not (x0 <= base[0] <= x1 and y0 <= base[1] <= y1):
                continue
            length = cfg["length"] * (0.78 + 0.44 * rng.random())
            # Chaque mèche prend un léger biais (±7°) et une courbure propre : sans cela, les
            # mèches forment un peigne trop régulier et la surface lit comme une grille.
            tilt = math.radians(rng.uniform(-7.0, 7.0))
            dt = np.array([math.cos(ang + tilt), math.sin(ang + tilt)])
            nt = np.array([-dt[1], dt[0]])
            bow = cfg["wave"] * (0.5 + 0.8 * rng.random())
            n = max(5, int(length / 0.6) + 1)
            t = np.linspace(0.0, 1.0, n)
            centre = (base + dt * (length * (t - 0.5))[:, None]
                      + nt * (bow * np.sin(np.pi * t))[:, None])
            ix = np.rint(centre[:, 0] * px_per_mm).astype(int)
            iy = np.rint(centre[:, 1] * px_per_mm).astype(int)
            ok = ((ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0]))
            inside = np.zeros(n, bool)
            inside[ok] = mask[iy[ok], ix[ok]]
            k = 0
            while k < n:
                if not inside[k]:
                    k += 1
                    continue
                m = k
                while m + 1 < n and inside[m + 1]:
                    m += 1
                if m - k >= 1:
                    pts = _resample(centre[k:m + 1], cfg["split"])
                    if len(pts) >= 2:
                        runs.append({"p0": tuple(pts[0]), "p1": tuple(pts[-1]),
                                     "pts": [tuple(p) for p in pts[1:]],
                                     "comp": 5000 + j, "row": j})
                k = m + 1
    return runs


# --- Entrelacs celtique : rubans satin tissés -------------------------------------------------
INTERLACE_PERIOD = 3.9
INTERLACE_WIDTH = 1.85   # rubans larges et jointifs : c'est un tissage, pas un quadrillage
INTERLACE_STEP = 0.42
INTERLACE_GAP = 0.22   # le ruban qui passe dessous s'arrête sous le bord du ruban du dessus


def _interlace_runs(mask, px_per_mm):
    """Deux familles de rubans à 45° qui se croisent en alternance dessus/dessous.

    On travaille dans le repère (u = x+y, v = y-x) : les deux familles y sont des droites
    u = const et v = const, donc les croisements et leur parité se calculent exactement. Le
    ruban perdant est coupé de part et d'autre du croisement, ce qui donne un vrai tissage.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    x0, x1 = xs.min() / px_per_mm - 2.0, xs.max() / px_per_mm + 2.0
    y0, y1 = ys.min() / px_per_mm - 2.0, ys.max() / px_per_mm + 2.0
    corners = np.array([(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
    U, V = corners[:, 0] + corners[:, 1], corners[:, 1] - corners[:, 0]
    P = INTERLACE_PERIOD
    u_start = math.floor(U.min() / P) * P + P / 2.0
    v_start = math.floor(V.min() / P) * P + P / 2.0
    u_lines = np.arange(u_start, U.max() + P, P)
    v_lines = np.arange(v_start, V.max() + P, P)
    nrm_u = np.array([1.0, 1.0]) / math.sqrt(2.0)   # normale des droites u = const
    nrm_v = np.array([1.0, -1.0]) / math.sqrt(2.0)  # normale des droites v = const
    half = INTERLACE_WIDTH / 2.0
    gap = half + INTERLACE_GAP

    runs = []
    for family, fixed_values, lines in ((0, u_lines, v_lines), (1, v_lines, u_lines)):
        for k, fixed in enumerate(fixed_values):
            lo, hi = (lines.min() - P, lines.max() + P)
            par = np.arange(lo, hi, INTERLACE_STEP)
            if family == 0:       # u fixe, on parcourt v
                x = (fixed + par) / 2.0
                y = (fixed - par) / 2.0
                nrm, cross, under = nrm_u, lines, (k + np.arange(len(lines))) % 2 == 0
            else:                 # v fixe, on parcourt u
                x = (par + fixed) / 2.0
                y = (par - fixed) / 2.0
                nrm, cross, under = nrm_v, lines, (np.arange(len(lines)) + k) % 2 == 1
            centres = np.column_stack((x, y))
            ea, eb = centres + nrm * half, centres - nrm * half
            ok = np.ones(len(par), bool)
            for pts in (centres, ea, eb):
                ix = np.rint(pts[:, 0] * px_per_mm).astype(int)
                iy = np.rint(pts[:, 1] * px_per_mm).astype(int)
                good = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
                inside = np.zeros(len(pts), bool)
                inside[good] = mask[iy[good], ix[good]]
                ok &= inside
            if len(cross):
                ok &= np.abs(par[:, None] - cross[None, :]).min(axis=1) >= gap
            m = 0
            while m < len(par):
                if not ok[m]:
                    m += 1
                    continue
                end = m
                while end + 1 < len(par) and ok[end + 1]:
                    end += 1
                if end - m >= 2:
                    zig = [ea[j] if (m + j) % 2 == 0 else eb[j] for j in range(m, end + 1)]
                    runs.append({"p0": tuple(zig[0]), "p1": tuple(zig[-1]),
                                 "pts": [tuple(p) for p in zig[1:]],
                                 "comp": 9000 + family * 100000 + k, "row": k})
                m = end + 1
    return runs


# --- Vermicelle ------------------------------------------------------------------------------
def _stipple(mask, px_per_mm, s, passes=5, row_mm=2.2):
    """Serpentins continus à amplitude et longueur d'onde variables (matelassé aléatoire)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    x0, x1 = xs.min() / px_per_mm - 1.0, xs.max() / px_per_mm + 1.0
    y0, y1 = ys.min() / px_per_mm - 1.0, ys.max() / px_per_mm + 1.0
    span = x1 - x0
    n_rows = int(math.ceil((y1 - y0) / row_mm)) + 1
    rng = np.random.default_rng(20261028)

    runs = []
    for k in range(passes):
        rows = []
        for r in range(n_rows):
            y = y0 + r * row_mm + (k * row_mm / passes)
            amp = 1.05 * (0.8 + 0.35 * rng.random())
            wave = 7.5 * (0.8 + 0.4 * rng.random())
            phase = rng.random()
            xm = np.linspace(x0, x1, max(2, int(span / 0.55)))
            row = np.column_stack((xm, y + amp * np.sin(2.0 * np.pi * (xm / wave + phase))))
            rows.append(row[::-1] if r % 2 else row)
        poly = np.vstack(rows)   # lignes raccordées : un seul tracé continu
        for seg in _clip(poly, mask, px_per_mm, 1.5):
            pts = _resample(seg, 0.8)
            if len(pts) < 3:
                continue
            runs.append({"p0": tuple(pts[0]), "p1": tuple(pts[-1]), "comp": k,
                         "row": k, "pts": [tuple(p) for p in pts[1:]]})
    return runs


# --- Interface -------------------------------------------------------------------------------
FILL_TYPES = {
    "relief_cotes": "Relief : côtes en facettes",
    "relief_chevrons": "Relief : chevrons",
    "relief_vagues": "Relief : vagues",
    "relief_damier": "Relief : damier de facettes",
    "relief_vannerie": "Relief : vannerie",
    "relief_losanges": "Relief : losanges",
    "relief_nid_abeille": "Relief : nid d'abeille",
    "relief_ecailles": "Relief : écailles",
    "relief_coquilles": "Relief : coquilles",
    "relief_pastilles": "Relief : pastilles",
    "relief_trefle": "Relief : trèfle",
    "relief_etoiles": "Relief : étoiles",
    "relief_feuilles": "Relief : feuilles",
    "relief_spirales": "Relief : spirales",
    "relief_pelage": "Relief : pelage ras",
    "relief_pelage_long": "Relief : pelage long",
    "relief_plumes": "Relief : plumes dressées",
    "relief_plumes_couchees": "Relief : plumes couchées",
    "relief_triskele": "Relief : triskèles celtiques",
    "relief_entrelacs": "Relief : entrelacs celtique",
    "relief_vermicelle": "Relief : vermicelle",
}

# Anciens noms (projets créés avant la refonte) -> motif équivalent.
LEGACY = {
    "relief_stripes": "relief_cotes",
    "relief_waves": "relief_vagues",
    "relief_chevron": "relief_chevrons",
    "relief_herringbone": "relief_chevrons",
    "relief_diamonds": "relief_losanges",
    "relief_honeycomb": "relief_nid_abeille",
    "relief_basket": "relief_vannerie",
    "relief_waffle": "relief_losanges",
    "relief_bricks": "relief_vannerie",
    "relief_maze": "relief_cotes",
    "relief_circles": "relief_pastilles",
    "relief_quatrefoil": "relief_trefle",
    "relief_scales": "relief_ecailles",
    "relief_shells": "relief_coquilles",
}


def zone_fill(kind, mask, px_per_mm, s):
    """Passes de remplissage relief : même contrat que fill.zone_fill."""
    kind = LEGACY.get(kind, kind)
    if kind in BAND_ANGLES:
        return _band_fill(kind, mask, px_per_mm, s)
    if kind in FUR:
        # Deux couches : une nappe de points longs dans le sens du poil (le corps de la
        # fourrure), puis des mèches éparses plus longues qui cassent la régularité et
        # accrochent la lumière. Un fond croisé à 90° dessinerait un cannage sous la
        # fourrure et casserait l'effet de nappe.
        base = _base_runs(mask, px_per_mm, s, angle=FUR[kind]["angle"])
        strands = _fur_runs(kind, mask, px_per_mm)
        return [("runs", base), ("runs", strands)] if strands else [("runs", base)]
    if kind in PLUME_TILING:
        base = _base_runs(mask, px_per_mm, s)
        rims, vane, spines = _feather_runs(kind, mask, px_per_mm, s)
        out = [("runs", base)] if base else []
        for layer in (rims, vane, spines):
            if layer:
                out.append(("runs", layer))
        return out
    if kind == "relief_entrelacs":
        runs = _interlace_runs(mask, px_per_mm)
        base = _base_runs(mask, px_per_mm, s)
        return [("runs", base), ("runs", runs)] if runs else [("runs", base)]
    if kind == "relief_vermicelle":
        # Stipple pur (BERNINA, Tajima) : PAS de fond tatami. Les serpentins se lisent parce
        # qu'ils sont seuls ; un fond, même léger, écrase le tracé et fait apparaître ses
        # rangées au lieu du vermicelle. 5 passes serrées pour garder de la couverture.
        runs = _stipple(mask, px_per_mm, s)
        return [("runs", runs)] if runs else []
    if kind not in TILINGS:
        kind = "relief_losanges"
    runs = _motif_runs(kind, mask, px_per_mm, s)
    base = _base_runs(mask, px_per_mm, s)
    return [("runs", base), ("runs", runs)] if runs else [("runs", base)]
