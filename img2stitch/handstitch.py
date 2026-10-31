"""Points de broderie à la main, émulés par des points machine.

Chaque fonction dessine un motif dans un `Pen`, qui accumule des traits en millimètres.
Un trait est une polyligne parcourue d'un seul tenant : le fil y reste posé sur le tissu.
Deux traits séparés par un vide donnent un saut (coupe ou déplacement), comme au passage
d'une aiguillée à l'autre.

Les géométries suivent les points réels : la tige est une torsade, la chaînette une suite
de boucles, la plume des V alternés de part et d'autre de l'axe, la feuille un arête de
poisson, etc. Aucune de ces fonctions ne connaît l'image : `hand.py` les choisit et les
place le long des axes médians.
"""
import math

import numpy as np


class Pen:
    """Accumule les traits d'un motif, avec un léger hasard reproductible."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.strokes = []

    def put(self, points):
        """Ajoute un trait ; les points confondus sont retirés, un point seul est ignoré."""
        pts = [(float(x), float(y)) for x, y in points]
        out = []
        for p in pts:
            if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
                out.append(p)
        if len(out) >= 2:
            self.strokes.append(out)

    # --- hasard de la main ---------------------------------------------------------------
    def length(self, value, spread=0.18):
        """Longueur d'un point : la main ne fait jamais deux points exactement pareils."""
        return float(value * (1.0 + self.rng.uniform(-spread, spread)))

    def wobble(self, amount=0.06):
        return self.rng.normal(0.0, amount, 2)

    def choice(self, options):
        return options[int(self.rng.integers(len(options)))]


# --- géométrie de base -------------------------------------------------------------------

def resample(path, step):
    """Points régulièrement espacés (extrémités comprises) le long d'une polyligne."""
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2 or step <= 0:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    total = float(seg.sum())
    if total <= 1e-9:
        return pts[:1]
    count = max(1, int(round(total / step)))
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    out = []
    for s in np.linspace(0.0, total, count + 1):
        i = min(len(seg) - 1, max(0, int(np.searchsorted(cum, s, side="right")) - 1))
        t = 0.0 if seg[i] <= 1e-9 else (s - cum[i]) / seg[i]
        out.append(pts[i] + (pts[i + 1] - pts[i]) * t)
    return np.array(out)


def tangents(pts):
    """Tangente unitaire en chaque point."""
    t = np.gradient(pts, axis=0) if len(pts) > 2 else np.tile(pts[-1] - pts[0], (len(pts), 1))
    norm = np.hypot(t[:, 0], t[:, 1])[:, None]
    return t / np.where(norm < 1e-9, 1.0, norm)


def normals(pts):
    t = tangents(pts)
    return np.stack((-t[:, 1], t[:, 0]), axis=1)


def point_at(pts, index, offset=0.0):
    """Point du chemin décalé de `offset` mm sur la normale (négatif = autre côté)."""
    n = normals(pts)[index]
    return pts[index] + n * offset


def offset_path(pts, distance):
    return pts + normals(pts) * distance


def _length(pts):
    """Longueur d'une polyligne."""
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _at(pts, s):
    """Point de la polyligne à l'abscisse curviligne `s`."""
    if len(pts) < 2:
        return pts[0]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    s = float(np.clip(s, 0.0, cum[-1]))
    i = min(len(seg) - 1, max(0, int(np.searchsorted(cum, s, side="right")) - 1))
    t = 0.0 if seg[i] <= 1e-9 else (s - cum[i]) / seg[i]
    return pts[i] + (pts[i + 1] - pts[i]) * t


def _tangent_at(pts, s):
    """Tangente unitaire à l'abscisse curviligne `s` (lissée sur 0,8 mm)."""
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    d = _at(pts, s + 0.4) - _at(pts, max(0.0, s - 0.4))
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def arc(center, radius, start, stop, count=12):
    """Arc de cercle en coordonnées mm."""
    angles = np.linspace(start, stop, count)
    return np.stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)), axis=1)


# --- points de trait ---------------------------------------------------------------------

def running(pen, path, length=2.6, gap=0.35, spread=0.22):
    """Point avant : petits traits séparés par un vide, le long du tracé."""
    pts = resample(path, 0.2)
    if len(pts) < 2:
        return
    step = length + gap
    for s in np.arange(0.0, _length(pts), step):
        a = _at(pts, s)
        b = _at(pts, min(s + pen.length(length, spread), _length(pts)))
        pen.put([a, b])


def back(pen, path, length=2.2, spread=0.2):
    """Point arrière : traits qui se chevauchent, ligne continue et nette."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    step = max(0.4, pen.length(length, spread) * 0.72)  # recouvrement : pas de trou dans la ligne
    # Une seule polyligne : le fil reste posé d'un point à l'autre. Des traits séparés
    # obligeraient un saut entre chaque point, et la ligne partirait en tirets.
    line = [_at(pts, s) for s in np.arange(0.0, total, step)]
    line.append(_at(pts, total))
    pen.put(line)


def stem(pen, path, width=1.8, length=1.9, spread=0.12):
    """Point de tige : points inclinés qui se recouvrent, formant une torsade."""
    pts = resample(path, 0.15)
    total = _length(pts)
    if total <= 1e-6:
        return
    # La torsade vient du recouvrement : chaque point repart en arrière du précédent, en
    # changeant de côté. Une simple alternance sans recouvrement ferait une vague, pas une tige.
    amplitude = min(0.28, max(0.15, width * 0.18))
    advance = length * 0.62
    side = 1.0
    line = []   # polyligne unique : la torsade est un aller-retour du fil, pas des traits isolés
    for s in np.arange(advance, total, pen.length(advance, spread)):
        a = _at(pts, max(0.0, s - advance * 0.72))
        b = _at(pts, min(s + advance * 0.35, total))
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        line.extend([a + normal * amplitude * side, b - normal * amplitude * side])
        side = -side
    if line:
        pen.put(line)


def split(pen, path, width=2.0, length=1.8, spread=0.1):
    """Point fendu : chaque point sort au milieu du précédent, ligne continue et texturée."""
    pts = resample(path, 0.15)
    total = _length(pts)
    if total <= 1e-6:
        return
    side = 1.0
    line = []
    for s in np.arange(0.0, total - 0.1, length * 0.62):
        a = _at(pts, s)
        b = _at(pts, min(s + length, total))
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        line.extend([a + normal * width * 0.18 * side, b - normal * width * 0.12 * side])
        side = -side
    if line:
        pen.put(line)


def _loop(base, tip, normal, bulge, inner, count=8):
    """Boucle arrondie : aller bombé d'un côté, retour plus plat de l'autre."""
    points = []
    for i in range(1, count):
        t = i / count
        points.append(base + (tip - base) * t + normal * bulge * math.sin(math.pi * t))
    points.append(tip)
    for i in range(1, count):
        t = i / count
        points.append(tip + (base - tip) * t - normal * inner * math.sin(math.pi * t))
    points.append(base)
    return points


def chain(pen, path, width=2.4, pitch=2.6, spread=0.12):
    """Point de chaînette : boucles arrondies successives, chacune attrapée par la suivante."""
    pts = resample(path, 0.15)
    total = _length(pts)
    if total <= 1e-6:
        return
    base = _at(pts, 0.0)
    s = 0.0
    while s < total - 0.2:
        nxt = min(s + pen.length(pitch, spread), total)
        tip = _at(pts, nxt)
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        pen.put(_loop(base, tip, normal, width * 0.62, width * 0.22))
        base, s = tip, nxt


def satin(pen, path, half_width, spacing=0.55, long_short=True, spread=0.08):
    """Point passé à la main : points serrés d'un bord à l'autre d'une bande.

    `half_width` : demi-largeur en mm à chaque étape (nombre ou fonction de l'abscisse).
    En long-and-short, le point démarre plus ou moins loin du bord d'une rangée à l'autre :
    c'est ce désordre régulier qui distingue le satin main du satin machine.
    """
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    row = 0
    for s in np.arange(0.0, total, spacing):
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        center = _at(pts, s)
        h = float(half(s))
        if h <= 0.05:
            continue
        inset_a = inset_b = 0.0
        if long_short:
            phase = (row % 6) / 6.0
            inset_a = h * 0.45 * phase
            inset_b = h * 0.45 * ((row + 3) % 6) / 6.0
        a = center - normal * max(0.0, h - inset_a) + pen.wobble(0.05)
        b = center + normal * max(0.0, h - inset_b) + pen.wobble(0.05)
        pen.put([a, b])
        row += 1


def fishbone(pen, path, half_width, spacing=0.8):
    """Point d'arête : chaque point traverse la nervure, d'un bord à l'autre en alternance."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    for index, s in enumerate(np.arange(0.0, total, spacing)):
        center = _at(pts, s)
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        h = float(half(s))
        ahead = tangent * min(spacing * 0.55, 1.2)
        side = 1.0 if index % 2 == 0 else -1.0
        a = center + normal * h * side - ahead
        b = center - normal * h * 0.9 * side + ahead
        pen.put([a, b])


def herringbone(pen, path, half_width, spacing=1.6):
    """Point de filet : deux diagonales croisées par pas, en épi dans la bande."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    for s in np.arange(0.0, total - spacing, spacing):
        a, b = _at(pts, s), _at(pts, min(s + spacing, total))
        ta, tb = _tangent_at(pts, s), _tangent_at(pts, min(s + spacing, total))
        na = np.array([-ta[1], ta[0]])
        nb = np.array([-tb[1], tb[0]])
        h = float(half(s))
        pen.put([a - na * h, b + nb * h])
        pen.put([a + na * h, b - nb * h])


def chevron(pen, path, half_width, spacing=1.9):
    """Point chevron : zigzag franc d'un bord à l'autre."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    current = []
    side = 1.0
    for s in np.arange(0.0, total + 1e-6, spacing):
        center = _at(pts, min(s, total))
        tangent = _tangent_at(pts, min(s, total))
        normal = np.array([-tangent[1], tangent[0]])
        current.append(center + normal * float(half(s)) * side)
        side = -side
    if len(current) >= 2:
        pen.put(current)


def fern(pen, path, half_width, spacing=2.2, arm_deg=52.0):
    """Point de fougère : une tige droite et deux bras qui partent de la même base."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    for s in np.arange(0.0, total, spacing):
        base = _at(pts, s)
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        pen.put([base - tangent * spacing * 0.45, base + tangent * spacing * 0.55])
        arm = float(half(s))
        if arm <= 0.2:
            continue
        for side in (-1.0, 1.0):
            angle = math.radians(arm_deg) * side
            direction = tangent * math.cos(angle) + normal * math.sin(angle)
            tip = base + direction * arm
            pen.put([base + tangent * spacing * 0.35, tip])


def feather(pen, path, half_width, spacing=3.4):
    """Point de plume : des V alternés de part et d'autre de l'axe."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    step = spacing / 2
    stroke = []
    side = 1.0
    for index, s in enumerate(np.arange(0.0, total + 1e-6, step)):
        center = _at(pts, min(s, total))
        tangent = _tangent_at(pts, min(s, total))
        normal = np.array([-tangent[1], tangent[0]])
        h = float(half(min(s, total)))
        if index % 2 == 0:          # pointe d'un côté
            stroke.append(center + normal * h * side)
            side = -side
        else:                        # retour sur l'axe
            stroke.append(center + tangent * 0.1)
    if len(stroke) >= 2:
        pen.put(stroke)


def cretan(pen, path, half_width, spacing=2.4):
    """Point crétois : bande à deux bords, traversée en alternance, jambes verticales."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    step = spacing / 2
    for index, s in enumerate(np.arange(0.0, total, step)):
        a = _at(pts, s)
        b = _at(pts, min(s + step, total))
        ta, tb = _tangent_at(pts, s), _tangent_at(pts, min(s + step, total))
        na, nb = np.array([-ta[1], ta[0]]), np.array([-tb[1], tb[0]])
        h = float(half(s))
        if index % 2 == 0:
            pen.put([a - na * h, a + na * h])
            pen.put([b + nb * h, b - nb * h * 0.15])
        else:
            pen.put([a + na * h, a - na * h])
            pen.put([b - nb * h, b + nb * h * 0.15])


def blanket(pen, path, half_width, spacing=2.0):
    """Point de feston : un rail et des jambes perpendiculaires régulières."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda u: half_width)
    rail = offset_path(pts, 0.0)
    pen.put(rail)
    for s in np.arange(0.0, total, spacing):
        base = _at(pts, s)
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        h = float(half(s))
        pen.put([base, base + normal * h + tangent * 0.25, base + tangent * spacing * 0.55])


def couching(pen, path, half_width=1.1, spacing=2.0):
    """Point posé : un fil épais le long du tracé, retenu par de petits points en travers."""
    pts = resample(path, 0.25)
    total = _length(pts)
    if total <= 1e-6:
        return
    pen.put(pts)
    pen.put(offset_path(pts, 0.28))  # deuxième passage : le fil paraît plus épais
    for s in np.arange(spacing * 0.5, total, spacing):
        center = _at(pts, s)
        tangent = _tangent_at(pts, s)
        normal = np.array([-tangent[1], tangent[0]])
        pen.put([center - normal * half_width, center + normal * half_width])


def cross_stitch(pen, path, size=2.0, spacing=2.4):
    """Croix : deux diagonales, motif de base du point de croix."""
    pts = resample(path, 0.2)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = size / 2
    for s in np.arange(0.0, total + 1e-6, spacing):
        center = _at(pts, min(s, total))
        tangent = _tangent_at(pts, min(s, total))
        normal = np.array([-tangent[1], tangent[0]])
        a = center - tangent * half - normal * half
        b = center + tangent * half + normal * half
        c = center - tangent * half + normal * half
        d = center + tangent * half - normal * half
        pen.put([a, b, c, d])


def overlock(pen, path, half_width=1.2, length=1.8, loop_ratio=0.4):
    """Point de surjet : un rail, et des boucles qui enjambent le bord.

    Repris du motif « overlock_stitch » de stitch_generator : la boucle ne revient pas au même
    point, elle déborde du bord (`loop_ratio` de la jambe) avant de retomber sur le rail. C'est ce
    débord qui ourle le tissu, et ce qui distingue le surjet d'un simple point de feston.
    """
    pts = resample(path, 0.25)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda _s: half_width)
    pen.put(pts)
    for s in np.arange(0.0, total, length):
        base = _at(pts, min(s, total))
        tangent = _tangent_at(pts, min(s, total))
        normal = np.array([-tangent[1], tangent[0]])
        h = max(0.3, float(half(s)))
        tip = base + normal * h
        pen.put([base - tangent * 0.35, tip - tangent * 0.5,
                 tip + tangent * h * loop_ratio, base + tangent * 0.35])


def e_stitch(pen, path, half_width=1.2, spacing=2.0, bar=0.55):
    """Point en E : un rail, des jambes vers un bord et une petite barre à leur bout.

    « e_stitch » de stitch_generator : la ligne du tracé, puis de petits traits qui pointent de
    côté ; la barre ajoutée au bout de chaque jambe est ce qui fait lire un E.
    """
    pts = resample(path, 0.25)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda _s: half_width)
    pen.put(pts)
    for s in np.arange(0.0, total, spacing):
        base = _at(pts, min(s, total))
        tangent = _tangent_at(pts, min(s, total))
        normal = np.array([-tangent[1], tangent[0]])
        tip = base + normal * max(0.3, float(half(s)))
        pen.put([base, tip])
        pen.put([tip - tangent * bar, tip + tangent * bar])


def arrow_chain(pen, path, half_width=1.6, pitch=2.2, height=1.6):
    """Chaînette fléchée : des flèches posées le long du tracé.

    « arrow_chain » de stitch_generator, dont le motif est un V répété : la pointe est sur le
    tracé, les deux bras partent vers l'arrière, si bien que la suite se lit dans le sens du
    tracé. Le V est symétrique, le motif répété en miroir par l'original donne le même dessin.
    """
    pts = resample(path, 0.25)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda _s: half_width)
    for s in np.arange(0.0, total + 1e-6, pitch):
        apex = _at(pts, min(s, total))
        tangent = _tangent_at(pts, min(s, total))
        normal = np.array([-tangent[1], tangent[0]])
        h = max(0.5, float(half(s)))
        pen.put([apex - tangent * height - normal * h, apex, apex - tangent * height + normal * h])


def lattice(pen, path, half_width=1.2, span=2.6):
    """Point de résille : deux familles de diagonales tendues d'un bord à l'autre.

    Le treillis de stitch_generator est une seule ligne qui va et vient entre les deux bords. À
    la main, la résille se brode en diagonales séparées, croisées deux fois : c'est leur
    croisement qui dessine les losanges du filet, pas le va-et-vient.
    """
    pts = resample(path, 0.25)
    total = _length(pts)
    if total <= 1e-6:
        return
    half = half_width if callable(half_width) else (lambda _s: half_width)

    def edge(s, side):
        base = _at(pts, min(max(s, 0.0), total))
        tangent = _tangent_at(pts, min(max(s, 0.0), total))
        normal = np.array([-tangent[1], tangent[0]])
        return base + normal * max(0.3, float(half(s))) * side

    for flip in (1.0, -1.0):
        s = 0.0
        while s < total - 1e-6:
            end = min(s + span, total)
            pen.put([edge(s, flip), edge(end, -flip)])
            s = end


def darning(pen, mask, px_per_mm, spacing=1.3, length=3.2):
    """Point de reprisage : rangées de points avant, décalées comme des briques."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return
    step = max(2, round(spacing * px_per_mm))
    every = max(2, round(length * px_per_mm))
    for index, y in enumerate(range(int(rows.min()), int(rows.max()) + 1, step)):
        xs = np.flatnonzero(mask[y])
        if not len(xs):
            continue
        start = int(xs.min()) + (every // 2 if index % 2 else 0)
        for x in range(start, int(xs.max()), every):
            if mask[y, min(x + every // 2, mask.shape[1] - 1)]:
                pen.put([(x / px_per_mm, y / px_per_mm),
                         ((min(x + every // 2, mask.shape[1] - 1)) / px_per_mm, y / px_per_mm)])


def seed(pen, mask, px_per_mm, density=0.9, length=1.5):
    """Point de graine : petits points dispersés, dans tous les sens."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return
    area = len(rows) / px_per_mm ** 2
    count = int(min(600, max(6, area * density)))   # un semis reste un semis, même sur 50 cm²
    for _ in range(count):
        i = int(pen.rng.integers(len(rows)))
        x, y = cols[i] / px_per_mm, rows[i] / px_per_mm
        angle = pen.rng.uniform(0, 2 * math.pi)
        size = pen.length(length, 0.3)
        pen.put([(x, y), (x + math.cos(angle) * size, y + math.sin(angle) * size)])


# --- motifs (fleurs, feuilles, nœuds) ----------------------------------------------------

def french_knot(pen, center, radius=0.85, turns=2.4):
    """Nœud français : un point enroulé qui forme une petite bosse ronde."""
    points = []
    steps = 26
    for i in range(steps + 1):
        t = i / steps
        angle = 2 * math.pi * turns * t
        r = radius * (1.0 - 0.55 * t)
        points.append((center[0] + math.cos(angle) * r, center[1] + math.sin(angle) * r))
    pen.put(points)
    pen.put([center, (center[0] + 0.35, center[1] + 0.25)])


def bullion(pen, start, end, radius=0.75, pitch=0.4):
    """Point de poste : un barreau enroulé, fait d'une spirale serrée autour d'un axe."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.hypot(*(end - start)))
    if length <= 1e-6:
        return
    axis = (end - start) / length
    normal = np.array([-axis[1], axis[0]])
    points = []
    wobble = math.sin(math.pi * min(1.0, length / 6.0)) * radius * 0.5
    for t in np.arange(0.0, 1.0, pitch / max(length, 1e-6)):
        side = 1.0 if int(t / (pitch / max(length, 1e-6))) % 2 == 0 else -1.0
        points.append(start + axis * (t * length) + normal * radius * side
                      + axis * wobble * math.sin(math.pi * t))
    pen.put(points)
    pen.put([start - normal * radius * 0.4, end + normal * radius * 0.4])


def lazy_daisy(pen, base, tip, bulge=0.55):
    """Point de marguerite : une boucle détachée qui forme un pétale."""
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    length = float(np.hypot(*(tip - base)))
    if length <= 1e-6:
        return
    axis = (tip - base) / length
    normal = np.array([-axis[1], axis[0]])
    pen.put(_loop(base, tip, normal, length * bulge, length * bulge * 0.9))  # pétale arrondi
    pen.put([tip - axis * 0.25, tip + axis * 0.45])  # petit point d'arrêt au bout du pétale


def woven_wheel(pen, center, radius, spokes=8, rounds=2.6, wavy=0.0):
    """Roue enroulée : rayons puis fil qui passe dessus-dessous, en spirale."""
    center = np.asarray(center, float)
    for i in range(spokes):
        angle = 2 * math.pi * i / spokes
        tip = center + np.array([math.cos(angle), math.sin(angle)]) * radius
        pen.put([center, tip])
    points = []
    steps = 90
    for i in range(steps + 1):
        t = i / steps
        angle = 2 * math.pi * rounds * t
        r = radius * (0.22 + 0.78 * t)
        if wavy:
            r += radius * wavy * math.sin(spokes * angle)
        points.append((center[0] + math.cos(angle) * r, center[1] + math.sin(angle) * r))
    pen.put(points)


def spider_web(pen, center, radius, spokes=6, rounds=3.0):
    """Toile d'araignée : rayons droits, puis un fil tendu de rayon en rayon."""
    center = np.asarray(center, float)
    for i in range(spokes):
        angle = 2 * math.pi * i / spokes
        pen.put([center, center + np.array([math.cos(angle), math.sin(angle)]) * radius])
    for ring in range(1, int(round(rounds)) + 1):
        r = radius * ring / (rounds + 0.5)
        points = []
        for i in range(spokes * 2 + 1):
            angle = math.pi * i / spokes
            points.append((center[0] + math.cos(angle) * r, center[1] + math.sin(angle) * r))
        pen.put(points)


def tulip(pen, base, tip, width=1.6):
    """Tulipe : une boucle de chaînette et deux boucles latérales plus courtes."""
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    axis = tip - base
    length = float(np.hypot(*axis))
    if length <= 1e-6:
        return
    axis /= length
    normal = np.array([-axis[1], axis[0]])
    mid = base + axis * length * 0.55
    lazy_daisy(pen, base, tip, bulge=0.45)
    for side in (-1.0, 1.0):
        side_tip = mid + normal * width * side + axis * length * 0.18
        pen.put([base, mid + normal * width * 0.35 * side, side_tip,
                 mid - normal * width * 0.1 * side, base])


# --- catalogue (interface) ---------------------------------------------------------------

# family : "trait" (ligne), "bande" (surface étroite), "motif" (fleur, feuille, nœud).
STITCH_TYPES = {
    "running": ("Point avant", "Petits traits espacés, le plus simple des points de main.", "trait"),
    "back": ("Point arrière", "Traits qui se chevauchent : ligne continue et nette.", "trait"),
    "stem": ("Point de tige", "Points inclinés et resserrés, en torsade.", "trait"),
    "split": ("Point fendu", "Chaque point sort du milieu du précédent.", "trait"),
    "chain": ("Point de chaînette", "Boucles successives accrochées l'une à l'autre.", "trait"),
    "couching": ("Point posé", "Fil épais retenu par de petits points en travers.", "trait"),
    "cross_stitch": ("Point de croix", "Deux diagonales croisées, en file.", "trait"),
    "darning": ("Point de reprisage", "Rangées de points avant décalées en briques.", "trait"),
    "satin": ("Point passé", "Points serrés d'un bord à l'autre, long-and-short.", "bande"),
    "fishbone": ("Point d'arête", "Points en arête de poisson autour d'une nervure.", "bande"),
    "herringbone": ("Point de filet", "Diagonales croisées en épi.", "bande"),
    "chevron": ("Point chevron", "Zigzag franc d'un bord à l'autre.", "bande"),
    "fern": ("Point de fougère", "Tige et bras alternés, feuille légère.", "bande"),
    "feather": ("Point de plume", "V alternés de part et d'autre de l'axe.", "bande"),
    "cretan": ("Point crétois", "Bande à deux bords, traversée en alternance.", "bande"),
    "blanket": ("Point de feston", "Rail et jambes perpendiculaires.", "bande"),
    "overlock": ("Point de surjet", "Rail et boucles qui enjambent le bord.", "bande"),
    "e_stitch": ("Point en E", "Rail, jambes vers un bord et barre au bout.", "bande"),
    "arrow_chain": ("Chaînette fléchée", "Des flèches posées le long du tracé.", "trait"),
    "lattice": ("Point de résille", "Diagonales croisées d'un bord à l'autre : un filet.", "bande"),
    "seed": ("Point de graine", "Petits points dispersés, dans tous les sens.", "bande"),
    "french_knot": ("Nœud français", "Petite bosse ronde, faite d'un point enroulé.", "motif"),
    "bullion": ("Point de poste", "Barreau enroulé, en relief.", "motif"),
    "lazy_daisy": ("Point de marguerite", "Boucle détachée : un pétale.", "motif"),
    "woven_wheel": ("Roue enroulée", "Rayons et fil tissé en spirale.", "motif"),
    "spider_web": ("Toile d'araignée", "Rayons droits et fils tendus en cercles.", "motif"),
    "tulip": ("Tulipe", "Chaînette et boucles latérales.", "motif"),
}
