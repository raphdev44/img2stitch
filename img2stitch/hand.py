"""Numérisation « broderie à la main » : chaque forme reçoit le point de main qui lui convient.

Le principe est l'inverse d'un remplissage machine : on cherche l'axe médian de chaque
forme, on mesure sa largeur et sa longueur, puis on choisit un point dont la géométrie
raconte cette forme (trait fin = point avant ou arrière, torsade = point de tige, bande =
chaînette ou arête de poisson, galet = nœud français ou roue enroulée). Les grandes
surfaces ne sont pas remplies à la machine : elles reçoivent un semis de points à la main.

`style` permet de forcer une famille de points ; "auto" laisse le classificateur décider.
"""
import math
from collections import Counter

import cv2
import numpy as np
import pyembroidery as pe

from . import handstitch as hs
from .algorithm_standard import _center_on_origin
from .flow import _branches, _prune, _thin
from .preprocess import load_and_quantize

WORK_PX_PER_MM = 10      # résolution de travail, comme le moteur standard
TARGET_MM = 100.0        # largeur de travail : une broderie main se lit à une dizaine de cm
MAX_END_MM = 110.0       # au-delà, on réduit le motif pour garder l'échelle des points
LINK_MM = 2.6            # deux traits plus proches que ça sont reliés par un point piqué
TRIM_MM = 4.0            # au-delà, le fil est coupé avant le déplacement
MIN_AREA_MM2 = 0.35
WOVEN_MAX_AREA = 150.0   # une roue enroulée n'a de sens que sur un petit galet (cœur de fleur)
GREEDY_LIMIT = 160       # au-delà, on range les traits par bandes plutôt que par proximité
# Le squelette doit être calculé à la résolution de travail : à 192 px, un pixel vaut 0,8 mm, le
# tracé revient en escalier et l'élagage de 2 mm avale les traits fins (moitié du dessin perdue).
# À 10 px/mm, `_thin` coûte ~2 s par grande forme : c'est le prix d'un axe médian fidèle.
SKELETON_MAX_PX = 1000   # au-delà seulement, le squelette est calculé sur une copie réduite

# mode du formulaire -> (couleur de fond, portée)
BACKGROUND_MODES = {"outside": ("auto", "outside"), "all": ("auto", "all"), "keep": ("none", "all")}

# Familles proposées à l'utilisateur : "auto" laisse le classificateur choisir.
STYLES = {
    "auto": "Automatique (le système choisit)",
    "tige": "Point de tige pour les tracés",
    "chainette": "Point de chaînette pour les tracés",
    "feuilles": "Feuilles au point d'arête",
    "fleurs": "Fleurs et nœuds pour les petites surfaces",
    "croix": "Point de croix pour les tracés",
}


# Jauge de créativité : 1 = les points les plus sobres et les plus solides, 5 = le plus décoratif.
# Le classificateur propose toujours le point le plus juste pour la forme ; ce rang dit à partir de
# quel cran on accepte qu'il s'écarte d'un tracé classique. Un point trop décoratif est remplacé par
# son équivalent sobre : la fidélité au dessin ne dépend jamais du curseur, seulement l'audace.
CREATIVITY = {
    1: "Sobre",
    2: "Discret",
    3: "Équilibré",
    4: "Expressif",
    5: "Très décoratif",
}
DECORATIVE = {
    "chain": 2, "couching": 2,
    "fern": 3, "woven_wheel": 3, "blanket": 3,
    "chevron": 4, "fishbone": 4,
    "feather": 5, "cretan": 5, "herringbone": 5, "spider_web": 5, "lazy_daisy": 5, "cross_stitch": 5,
}
# Le point classique qui raconte la même forme, quand le cran demandé refuse le point décoratif.
PLAINER = {
    "chain": "split",
    "couching": "stem",
    "fern": "satin",
    "woven_wheel": "french_knot",
    "blanket": "back",
    "chevron": "chain",
    "fishbone": "satin",
    "feather": "satin",
    "cretan": "satin",
    "herringbone": "satin",
    "spider_web": "woven_wheel",
    "lazy_daisy": "satin",
    "cross_stitch": "back",
}


def _temper(kind, level):
    """Ramène un point au cran de créativité demandé (1 sobre … 5 décoratif)."""
    level = max(1, min(5, int(level if level is not None else 3)))
    seen = set()
    while DECORATIVE.get(kind, 1) > level and kind in PLAINER and kind not in seen:
        seen.add(kind)
        kind = PLAINER[kind]
    return kind


def _classify(width_mm, length_mm, *, roundness=0.0, encloses=False, area_mm2=0.0, style="auto",
              taper=0.0, straight=0.0, big_ring=False):
    """Point de main le plus adapté à une forme mesurée.

    La largeur ne suffit pas : `taper` (effilement aux deux bouts) distingue une feuille
    d'une bande, `straight` (corde / longueur d'arc) distingue une bordure droite d'une courbe,
    `big_ring` dit qu'on longe le bord d'une grande forme.
    """
    width = max(float(width_mm), 1e-3)
    ratio = float(length_mm) / width
    if style == "croix":
        return "cross_stitch"
    if style == "chainette":
        return "chain"
    if style == "tige":
        return "stem"
    if style == "feuilles":
        return "fishbone" if width >= 2.6 else "stem"
    if style == "fleurs":
        if area_mm2 <= 3.0:
            return "french_knot"
        if roundness >= 0.6 and area_mm2 <= WOVEN_MAX_AREA:
            return "woven_wheel"
        return "lazy_daisy" if ratio >= 2.2 else "satin"
    # --- automatique : le cas le plus général d'abord, puis les formes étroites ---
    if 0 < area_mm2 <= 1.2 and ratio < 1.9:   # un vrai point rond, sans axe médian
        return "french_knot"
    if roundness >= 0.72 and 3.0 < area_mm2 <= WOVEN_MAX_AREA:
        return "woven_wheel"
    if roundness >= 0.72 and ratio < 2.0 and area_mm2 > WOVEN_MAX_AREA:
        return "seed"      # grande surface ronde et courte : un semis, pas un satin en éventail
    # --- la forme avant la largeur : feuille, bordure, bande ---
    if taper >= 0.35 and 1.4 <= width <= 7.5 and ratio >= 2.0 and length_mm >= 5.0:
        return "fern" if width <= 3.2 else "fishbone"   # feuille : nervure et bras, ou arête
    if big_ring and encloses and 1.2 <= width <= 3.4:
        return "blanket"      # on longe le bord d'une grande forme : point de feston
    if width <= 0.7:      # trait très fin : une ligne continue, jamais des tirets
        return "back"
    if width <= 2.2:      # trait fin : torsade au point de tige, arrière si c'est un liseré
        return "back" if encloses else "stem"
    if width <= 3.6:
        # Bande large : chaînette pour un ruban qui court, chevron s'il est bien droit (bordure),
        # satin pour une plaque courte — des points fendus y feraient de grands zigzags.
        if ratio >= 6.0:
            return "chevron" if straight >= 0.93 else "chain"
        return "satin"
    if width <= 7.5:
        return "fishbone" if ratio >= 2.0 else "satin"
    if width <= 9.0:
        return "satin"
    # Au-delà de 9 mm, un satin n'est plus un point de main : on sème, ou on reprise si c'est allongé.
    return "seed" if roundness >= 0.45 else "darning"


def choose_stitch(width_mm, length_mm, *, roundness=0.0, encloses=False, area_mm2=0.0, style="auto",
                  taper=0.0, straight=0.0, big_ring=False, creativity=3):
    """Point de main le plus adapté à une forme, à un cran de créativité donné.

    `creativity` (1 à 5) ne change jamais la géométrie : il fixe seulement jusqu'où le moteur a le
    droit de s'écarter des points classiques. À 1, feston, fougère et roue enroulée sont remplacés
    par point arrière, satin et nœud ; à 5, tout est permis. Une famille imposée (`style`) n'est
    jamais tempérée, puisque l'utilisateur l'a demandée.
    """
    kind = _classify(width_mm, length_mm, roundness=roundness, encloses=encloses, area_mm2=area_mm2,
                     style=style, taper=taper, straight=straight, big_ring=big_ring)
    return kind if style != "auto" else _temper(kind, creativity)


def _half_width_fn(path_px, dist, origin, ppm):
    """Demi-largeur de la forme le long de son axe médian, en mm, indexée par l'abscisse curviligne."""
    pts = np.asarray(path_px, float)
    if len(pts) == 0:
        return lambda _u: 0.0
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T)))) / ppm if len(pts) > 1 else np.zeros(1)
    height, width = dist.shape
    values = []
    for x, y in pts:
        ix = int(np.clip(round(x) - origin[0], 0, width - 1))
        iy = int(np.clip(round(y) - origin[1], 0, height - 1))
        values.append(float(dist[iy, ix]))
    values = np.asarray(values)
    if len(arc) < 2:
        return lambda _u: float(values[0])
    return lambda u: float(np.interp(u, arc, values))


def _shape(comp, ppm):
    """Mesures d'une forme : axe médian, largeur, longueur, rondeur, trous."""
    ys, xs = np.nonzero(comp)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    sub = comp[y0:y1, x0:x1]
    dist = cv2.distanceTransform(np.pad(sub, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1] / ppm
    # 80e centile : la largeur d'un trait, pas celle de ses extrémités effilées. Au 60e, un
    # trait de 1 mm était lu à 0,7 mm et recevait des points espacés au lieu d'une ligne continue.
    half = float(np.percentile(dist[dist > 0], 80)) if (dist > 0).any() else 0.0
    area = float(sub.sum()) / ppm ** 2
    # Squelette sur une copie réduite quand la forme est grande : l'axe médian reste juste au
    # dixième de millimètre près, et l'amincissement (coûteux en Python) va plusieurs fois plus vite.
    # Une forme minuscule n'a pas besoin d'axe médian : elle sera un nœud, posé à son centre.
    paths = []
    if not (area <= 3.5 and max(sub.shape) <= round(4.0 * ppm)):
        shrink, small = 1, sub
        while max(small.shape) > SKELETON_MAX_PX:
            small = cv2.resize(small.astype(np.uint8) * 255, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA) > 127
            shrink *= 2
        # Sans élagage, chaque irrégularité du bord laisse une barbe de 1 à 2 px qui coupe le
        # tracé : la ligne part alors en dizaines de branches courtes, donc en tirets.
        skel = _prune(_thin(small), max(2, round(2.0 * ppm / shrink)))
        for branch in _branches(skel, max(2, round(1.0 * ppm / shrink))):
            paths.append(np.stack([branch[:, 0] * shrink + x0, branch[:, 1] * shrink + y0], axis=1))
    contours, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) / ppm
    roundness = 4 * math.pi * area / perimeter ** 2 if perimeter > 1e-6 else 0.0
    length = max((x1 - x0), (y1 - y0)) / ppm
    for path in paths:   # un tracé en diagonale dépasse sa boîte englobante
        if len(path) > 1:
            length = max(length, float(np.hypot(*np.diff(path, axis=0).T).sum()) / ppm)
    holes = (~sub).astype(np.uint8)
    count, labels = cv2.connectedComponents(holes, connectivity=4)
    border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
    encloses = any(i not in border and int((labels == i).sum()) >= 4 for i in range(1, count))
    outlines = [(contour.reshape(-1, 2).astype(float) + np.array([x0, y0])) / ppm for contour in contours]
    return {"paths": paths, "outlines": outlines, "half": half, "dist": dist, "origin": (x0, y0), "area": area,
            "roundness": roundness, "length": length, "encloses": encloses,
            "center": ((x0 + x1) / 2 / ppm, (y0 + y1) / 2 / ppm)}


def _order(strokes):
    """Ordonne les traits bout à bout pour limiter les sauts (et les retourne au besoin)."""
    rest = [list(s) for s in strokes if len(s) >= 2]
    if len(rest) > GREEDY_LIMIT:   # semis dense : rangement par bandes, en va-et-vient
        band = 7.0
        rest.sort(key=lambda s: (int(s[0][1] // band), int(s[0][0] // band) * (1 if int(s[0][1] // band) % 2 == 0 else -1)))
        return rest
    ordered, cur = [], None
    while rest:
        best = None
        for index, stroke in enumerate(rest):
            for flip, point in ((False, stroke[0]), (True, stroke[-1])):
                d = 0.0 if cur is None else math.hypot(point[0] - cur[0], point[1] - cur[1])
                if best is None or d < best[0]:
                    best = (d, index, flip)
        _, index, flip = best
        stroke = rest.pop(index)
        if flip:
            stroke.reverse()
        ordered.append(stroke)
        cur = stroke[-1]
    return ordered


def _draw_motif(pen, info, kind):
    """Motifs ronds : nœud, roue, toile, pétale."""
    cx, cy = info["center"]
    if kind == "french_knot":
        hs.french_knot(pen, (cx, cy), max(0.5, min(1.4, info["half"])))
    elif kind == "woven_wheel":
        hs.woven_wheel(pen, (cx, cy), max(1.0, info["half"]))
    elif kind == "spider_web":
        hs.spider_web(pen, (cx, cy), max(1.0, info["half"]))
    else:
        hs.lazy_daisy(pen, (cx, cy), (cx, cy + max(3.0, info["length"])))


def _draw_one(pen, pts, half_fn, half, kind):
    """Un passage du point choisi le long d'un axe médian."""
    width = max(0.3, 2 * half)
    if kind == "stem":
        hs.stem(pen, pts, width=width)
    elif kind == "split":
        hs.split(pen, pts, width=width)
    elif kind == "chain":
        hs.chain(pen, pts, width=width)
    elif kind == "couching":
        hs.couching(pen, pts, half_width=max(0.6, half))
    elif kind == "cross_stitch":
        hs.cross_stitch(pen, pts, size=max(1.4, min(3.0, width)))
    elif kind in ("satin", "fishbone", "herringbone", "chevron", "fern", "feather", "cretan", "blanket"):
        getattr(hs, kind)(pen, pts, half_fn)
    else:
        hs.back(pen, pts)


def _draw_path(pen, pts, half_fn, half, kind):
    """Trace une branche ; un trait plus fin qu'un fil est brodé en deux brins parallèles.

    Un dessin au trait fait souvent 0,3 mm : piqué d'un seul fil, il reste plus maigre que
    l'original. Deux passages décalés de part et d'autre donnent l'épaisseur d'une aiguillée.
    """
    if kind in ("satin", "fishbone", "herringbone", "chevron", "fern", "feather",
                "cretan", "blanket", "cross_stitch"):
        _draw_one(pen, pts, half_fn, half, kind)
        return
    # Autant de brins qu'il faut pour remplir l'épaisseur mesurée du trait (un fil = 0,42 mm).
    # Les brins se chevauchent : espacés de leur largeur, ils se liraient comme autant de lignes.
    width = max(0.35, 2 * half)
    strands = int(max(1, min(4, math.ceil(width / 0.36))))
    span = max(0.0, (width - 0.42) / 2)
    for i in range(strands):
        shift = 0.0 if strands == 1 else -span + 2 * span * i / (strands - 1)
        # Plusieurs brins se posent en lignes droites : trois torsades côte à côte font une corde.
        _draw_one(pen, pts + hs.normals(pts) * shift, half_fn, half,
                  kind if strands == 1 else "back")


def _draw(pen, info, ppm, kind):
    """Trace toutes les branches d'une forme avec un même point."""
    if kind in ("french_knot", "woven_wheel", "spider_web", "lazy_daisy"):
        _draw_motif(pen, info, kind)
        return
    for path in info["paths"]:
        pts = path.astype(float) / ppm
        if len(pts) >= 2:
            _draw_path(pen, pts, _half_width_fn(path, info["dist"], info["origin"], ppm),
                       info["half"], kind)


def _draw_area(pen, comp, info, ppm, kind):
    """Grande surface : semis ou reprisage, jamais la densité d'un tatami machine."""
    if kind == "seed":
        hs.seed(pen, comp, ppm, density=0.9, length=1.6)
    else:
        hs.darning(pen, comp, ppm, spacing=1.4, length=3.4)
    for outline in info["outlines"]:   # le tracé de la main suit le bord, pas l'axe médian
        if len(outline) >= 3:
            hs.running(pen, outline, length=3.0, gap=1.3)


def _draw_short_stroke(pen, comp, ppm):
    """Petite marque allongée : un seul point de la longueur de son axe principal."""
    ys, xs = np.nonzero(comp)
    pts = np.stack([xs, ys], axis=1).astype(float) / ppm
    mean = pts.mean(axis=0)
    centered = pts - mean
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    along = centered @ axis
    pen.put([mean + axis * float(along.min()), mean + axis * float(along.max())])


def _branch_measures(info, path, ppm):
    """Longueur et demi-largeur d'une branche, mesurées le long de son axe."""
    pts = path.astype(float) / ppm
    arc = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
    half_fn = _half_width_fn(path, info["dist"], info["origin"], ppm)
    # 80e centile et non médiane : la médiane d'une branche courte sous-estime la largeur du
    # tracé (elle tombe sur les bouts effilés) et le fil reste plus maigre que l'original.
    samples = [half_fn(s) for s in np.linspace(0.0, max(arc, 0.01), 9)]
    half = float(np.percentile(samples, 80))
    # Deux descripteurs de forme, pour ne pas choisir un point sur la seule largeur : une bande
    # qui s'effile aux deux bouts est une feuille, une bande droite et longue est une bordure.
    peak = max(samples) or 1e-6
    chord = float(np.hypot(*(pts[-1] - pts[0]))) if len(pts) > 1 else 0.0
    shape = {"taper": min(samples[0], samples[-1]) / peak,
             "straight": chord / arc if arc > 1e-6 else 0.0}
    return arc, half, half_fn, shape


def _component_strokes(comp, ppm, style, seed, creativity=3):
    """Traits et point retenu pour une forme connexe.

    La largeur se mesure **branche par branche** : sur un dessin au trait, toutes les lignes
    forment une seule forme de plusieurs millimètres de large, et une mesure globale ferait
    broder un remplissage à la place du tracé.
    """
    info = _shape(comp, ppm)
    pen = hs.Pen(seed=seed)
    # Une pastille ronde et compacte est un nœud colonial, pas un trait : à cette taille, l'axe
    # médian la réduit à un point ou à une roue minuscule. C'est LE point classique de la main.
    # Un anneau (œil, boucle) est exclu : il doit rester un contour, pas devenir une boule.
    if (info["roundness"] >= 0.72 and info["length"] <= 4.0 and info["area"] <= 9.0
            and not info["encloses"]):
        hs.french_knot(pen, info["center"], max(0.6, min(1.5, info["length"] / 2)))
        return pen.strokes, "french_knot"
    if info["paths"]:
        kinds = []
        for path in info["paths"]:
            arc, half, half_fn, shape = _branch_measures(info, path, ppm)
            if arc < 0.8:
                continue
            width = 2 * half if half > 0.05 else 0.6
            # area_mm2=0 : une branche est un trait, jamais un point. Avec une surface estimée,
            # un petit trait de 2 mm passait sous le seuil du nœud et devenait une pastille.
            kind = choose_stitch(width, arc, roundness=0.0, encloses=info["encloses"],
                                 area_mm2=0.0, style=style, taper=shape["taper"],
                                 straight=shape["straight"], big_ring=info["area"] >= 40,
                                 creativity=creativity)
            _draw_path(pen, path.astype(float) / ppm, half_fn, half, kind)
            kinds.append(kind)
        if kinds:
            return pen.strokes, Counter(kinds).most_common(1)[0][0]
    width = 2 * info["half"] if info["half"] > 0.05 else info["length"]
    kind = choose_stitch(width, info["length"], roundness=info["roundness"], encloses=info["encloses"],
                         area_mm2=info["area"], style=style, creativity=creativity)
    if kind in ("seed", "darning"):
        _draw_area(pen, comp, info, ppm, kind)
    elif not info["paths"] and kind not in ("french_knot", "woven_wheel", "spider_web", "lazy_daisy"):
        if info["length"] >= 1.6 * max(2 * info["half"], 0.2):
            _draw_short_stroke(pen, comp, ppm)      # petite marque allongée : un point, pas une pastille
            kind = "back"
        else:
            hs.french_knot(pen, info["center"], max(0.6, min(1.4, info["half"])))
            kind = "french_knot"
    else:
        _draw(pen, info, ppm, kind)
    return pen.strokes, kind


def hand_pattern(source, background="outside", *, n_colors=None, exclude=(), style="auto", seed=0,
                 creativity=3):
    """Image -> motif « broderie à la main » (EmbPattern pyembroidery)."""
    bg, scope = BACKGROUND_MODES.get(background, BACKGROUND_MODES["outside"])
    width_px = int(WORK_PX_PER_MM * TARGET_MM)
    # thicken_lines : sans lui, le filtre médian du prétraitement efface les traits de 1 à 3 px
    # et laisse des tirets. Une broderie main a besoin du tracé entier, pas d'un contour érodé.
    palette, masks = load_and_quantize(source, width_px, n_colors or 8, bg, 40, scope, exclude,
                                       thicken_lines=True)

    pattern = pe.EmbPattern()
    height = masks[0].shape[0] if masks else 0
    if not masks:
        pattern.end()
        pattern.img2stitch_zones, pattern.img2stitch_segments = [], []
        pattern.img2stitch_hand = {"stitches": {}, "shapes": 0}
        pattern.img2stitch_frame = {"w": width_px, "h": height, "ppm": float(WORK_PX_PER_MM),
                                    "cx": 0.0, "cy": 0.0, "scale": 1.0}
        return pattern

    # Les grandes surfaces d'abord, les détails ensuite : c'est l'ordre du travail à la main.
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    zones, segments, tally, shapes = [], [], {}, 0
    first = True
    for color_index in order:
        rgb, mask = palette[color_index], masks[color_index]
        strokes, chosen = [], []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] / WORK_PX_PER_MM ** 2 < MIN_AREA_MM2:
                continue
            made, kind = _component_strokes(labels == index, WORK_PX_PER_MM, style,
                                            seed=seed + 7919 * (index + 1), creativity=creativity)
            strokes.extend(made)
            chosen.append(kind)
            shapes += 1
        if not strokes:
            continue
        for kind in chosen:
            tally[kind] = tally.get(kind, 0) + 1
        color = tuple(int(v) for v in rgb)
        if not first:
            pattern.add_command(pe.COLOR_CHANGE)
        pattern.add_thread({"color": (color[0] << 16) | (color[1] << 8) | color[2]})
        first = False
        segments.append((len(pattern.stitches), len(zones)))
        zones.append((color, None))
        cur = None
        for stroke in _order(strokes):
            start = stroke[0]
            near = cur is not None and math.hypot(start[0] - cur[0], start[1] - cur[1]) <= LINK_MM
            if cur is not None and not near and math.hypot(start[0] - cur[0], start[1] - cur[1]) > TRIM_MM:
                pattern.add_command(pe.TRIM)
            if not near:
                pattern.add_stitch_absolute(pe.JUMP, start[0] * 10, start[1] * 10)
            for x, y in stroke:
                pattern.add_stitch_absolute(pe.STITCH, x * 10, y * 10)
            cur = stroke[-1]

    # Cadrage : au-delà de 11 cm, les points perdent l'échelle de la main.
    xs = [s[0] for s in pattern.stitches]
    ys = [s[1] for s in pattern.stitches]
    if xs:
        longest = max(max(xs) - min(xs), max(ys) - min(ys)) / 10
        if longest > MAX_END_MM:
            factor = MAX_END_MM / longest
            for stitch in pattern.stitches:
                stitch[0] *= factor
                stitch[1] *= factor
    pattern.end()
    cx, cy = _center_on_origin(pattern)
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_hand = {"stitches": tally, "shapes": shapes}
    pattern.img2stitch_frame = {"w": width_px, "h": height, "ppm": float(WORK_PX_PER_MM),
                                "cx": cx, "cy": cy, "scale": 1.0}
    return pattern
