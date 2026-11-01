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
    "bandes": "Bandes décoratives (résille, E, surjet)",
}


# Jauge de créativité : 1 = sobre / sobre, 5 = très décoratif.
# Cette carte contrôle quelle famille de points l'algorithme peut piocher,
# du sobre (1) au expressif/décoratif (5). Le point classique de départ est le point avant.
CREATIVITY = {
    1: "Sobre",
    2: "Discret",
    3: "Équilibré",
    4: "Expressif",
    5: "Très décoratif",
}

# Mapping créativité -> familles de points autorisées, dans l'ordre croissant.
# À chaque cran on ajoute une nouvelle famille ; on ne retire jamais ce qui a été autorisé auparavant.
# Niveau 1-2 : sobre, seulement les points de base
# Niveau 3 : ajout des points de structure (tige, fendu)
# Niveau 4 : accès presque complet aux points décoratifs et de remplissage
# Niveau 5 : tout le palette + légère aléatoire dans le placement
CREATIVITY_STITCH_MAP = {
    1: {"allowed_kinds": ["running", "french_knot"]},
    2: {"allowed_kinds": ["running", "split", "french_knot"]},
    3: {"allowed_kinds": ["running", "stem", "split", "french_knot"]},
    4: {"allowed_kinds": ["running", "stem", "split", "chain", "french_knot", "seed", "laid", "tatami",
                          "satin", "fishbone", "herringbone", "chevron", "fern", "feather", "cretan",
                          "blanket", "overlock", "e_stitch", "arrow_chain", "lattice", "couching",
                          "cross_stitch", "woven_wheel", "spider_web", "lazy_daisy", "bullion", "tulip"]},
    5: {"allowed_kinds": ["running", "stem", "split", "chain", "french_knot", "seed", "laid", "tatami", "back",
                          "satin", "fishbone", "herringbone", "chevron", "fern", "feather", "cretan",
                          "blanket", "overlock", "e_stitch", "arrow_chain", "lattice", "couching",
                          "cross_stitch", "woven_wheel", "spider_web", "lazy_daisy", "bullion", "tulip"]},
}
DECORATIVE = {
    "chain": 2, "couching": 2, "overlock": 2,
    "fern": 3, "woven_wheel": 3, "blanket": 3, "e_stitch": 3, "arrow_chain": 3,
    "chevron": 4, "fishbone": 4, "lattice": 4,
    "feather": 5, "cretan": 5, "herringbone": 5, "spider_web": 5, "lazy_daisy": 5, "cross_stitch": 5,
}
# Le point classique qui raconte la même forme, quand le cran demandé refuse le point décoratif.
PLAINER = {
    "chain": "split",
    "couching": "stem",
    "overlock": "back",
    "back": "split",
    "e_stitch": "blanket",
    "arrow_chain": "chain",
    "lattice": "herringbone",
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
    """Ramène un point au cran de créativité demandé (1 sobre … 5 décoratif).

    Utilise la carte CREATIVITY_STITCH_MAP : si le point n'est pas dans la liste autorisée
    pour le cran de créativité, on retombe vers le point sobre de base (running),
    sauf pour french_knot qui reste autorisé à tous les crans.
    """
    level = max(1, min(5, int(level if level is not None else 3)))
    allowed = CREATIVITY_STITCH_MAP[level]["allowed_kinds"]
    if kind in allowed:
        return kind
    # Le point décoratif refusé cède la place à son point classique, s'il est permis à ce cran
    plain = PLAINER.get(kind)
    if plain in allowed:
        return plain
    # Sinon, retombe sur le point sobre de base selon la forme
    if kind == "french_knot":
        return "french_knot"
    # Un remplissage de grande surface devient un point couché quand le cran le permet
    if kind == "darning" and "tatami" in allowed:
        return "tatami"
    if kind == "darning" and "laid" in allowed:
        return "laid"
    # Par défaut, point avant sobre
    return "running"


def _classify(width_mm, length_mm, *, roundness=0.0, encloses=False, area_mm2=0.0, style="auto",
              taper=0.0, straight=0.0, big_ring=False, filled=None):
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
    if style == "bandes":   # bande large : un filet ; bande étroite : son bord ourlé
        if width >= 3.2:
            return "lattice"
        return "arrow_chain" if width >= 1.6 and ratio >= 8.0 and straight >= 0.9 else "e_stitch"
    # --- automatique : le cas le plus général d'abord, puis les formes étroites ---
    if 0 < area_mm2 <= 1.2 and ratio < 1.9:   # un vrai point rond, sans axe médian
        return "french_knot"
    if roundness >= 0.72 and 3.0 < area_mm2 <= WOVEN_MAX_AREA:
        return "woven_wheel"
    if roundness >= 0.72 and ratio < 2.0 and area_mm2 > WOVEN_MAX_AREA:
        return "seed"      # grande surface ronde et courte : un semis, pas un satin en éventail
    if filled is True and ratio < 2.0 and area_mm2 > WOVEN_MAX_AREA:
        return "tatami"    # grande surface pleine et ramassée : petits points irréguliers en rangs
    # --- la forme avant la largeur : feuille, bordure, bande ---
    if filled is not False and taper >= 0.35 and 1.4 <= width <= 7.5 and ratio >= 2.0 and length_mm >= 5.0:
        return "fern" if width <= 3.2 else "fishbone"   # feuille : nervure et bras, ou arête
    if big_ring and encloses and 1.2 <= width <= 3.4 and length_mm >= 8.0:
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
                  taper=0.0, straight=0.0, big_ring=False, creativity=3, filled=None):
    """Point de main le plus adapté à une forme, à un cran de créativité donné.

    `creativity` (1 à 5) ne change jamais la géométrie : il fixe seulement jusqu'où le moteur a le
    droit de s'écarter des points classiques. À 1, feston, fougère et roue enroulée sont remplacés
    par point arrière, satin et nœud ; à 5, tout est permis. Une famille imposée (`style`) n'est
    jamais tempérée, puisque l'utilisateur l'a demandée.
    """
    kind = _classify(width_mm, length_mm, roundness=roundness, encloses=encloses, area_mm2=area_mm2,
                     style=style, taper=taper, straight=straight, big_ring=big_ring, filled=filled)
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
    inner = [i for i in range(1, count) if i not in border and int((labels == i).sum()) >= 4]
    encloses = bool(inner)
    # Distance aux vides intérieurs : un contour de feuille les longe, une tige traverse le fond.
    hole_dist = (cv2.distanceTransform(np.isin(labels, inner).astype(np.uint8), cv2.DIST_L2, 5)
                 if encloses else None)
    outlines = [(contour.reshape(-1, 2).astype(float) + np.array([x0, y0])) / ppm for contour in contours]
    return {"paths": paths, "outlines": outlines, "half": half, "dist": dist, "hole_dist": hole_dist,
            "origin": (x0, y0), "area": area, "roundness": roundness, "length": length,
            "encloses": encloses, "center": ((x0 + x1) / 2 / ppm, (y0 + y1) / 2 / ppm)}


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
    elif kind in ("satin", "fishbone", "herringbone", "chevron", "fern", "feather", "cretan", "blanket",
                  "overlock", "e_stitch", "arrow_chain", "lattice"):
        getattr(hs, kind)(pen, pts, half_fn)
    elif kind == "running":
        hs.running(pen, pts)
    else:
        hs.back(pen, pts)


MULTI_STRAND_KIND = "running"   # deux brins parallèles : un point sobre, sauf cran supérieur (voir _draw_path)


def _draw_path(pen, pts, half_fn, half, kind):
    """Trace une branche ; un trait plus fin qu'un fil est brodé en deux brins parallèles.

    Un dessin au trait fait souvent 0,3 mm : piqué d'un seul fil, il reste plus maigre que
    l'original. Deux passages décalés de part et d'autre donnent l'épaisseur d'une aiguillée.
    """
    if kind in ("satin", "fishbone", "herringbone", "chevron", "fern", "feather",
                "cretan", "blanket", "cross_stitch", "overlock", "e_stitch", "arrow_chain", "lattice"):
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
                  kind if strands == 1 else MULTI_STRAND_KIND)


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
    elif kind == "laid":
        hs.laid(pen, comp, ppm, spacing=1.2, length=3.0)
    elif kind == "tatami":
        hs.tatami(pen, comp, ppm)
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


def _runs_along_hole(info, path, ppm):
    """La branche longe-t-elle un vide intérieur (contour de feuille, de pétale) ?

    C'est le meilleur indice de rôle : un contour borde l'intérieur vide de la forme, une tige
    traverse le fond. Deux traits de même largeur n'ont donc pas le même point selon qu'ils cernent
    une forme ou courent en ligne ouverte.
    """
    dist = info.get("hole_dist")
    if dist is None or len(path) < 2:
        return False
    step = max(1, len(path) // 40)
    height, width = dist.shape
    tolerance = max(2.0, 1.1 * 2 * info["half"] * ppm)
    hits = 0
    for x, y in path[::step]:
        ix = int(np.clip(round(x) - info["origin"][0], 0, width - 1))
        iy = int(np.clip(round(y) - info["origin"][1], 0, height - 1))
        if dist[iy, ix] <= tolerance:
            hits += 1
    return hits * 2 >= len(path[::step])


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
             "straight": chord / arc if arc > 1e-6 else 0.0,
             "outline": _runs_along_hole(info, path, ppm)}
    return arc, half, half_fn, shape


def _alike(a, b):
    """Deux traits se ressemblent-ils assez pour recevoir le même point ?

    Largeur en rapport (pas en millimètres : 1,5 et 2 mm sont le même trait d'un dessin à l'autre),
    puis effilement, rectitude et rapport longueur / largeur. Les drapeaux de forme (contour d'une
    grande forme, bordure) doivent être identiques : un feston ne se compare pas à un trait libre.
    """
    if a["encloses"] != b["encloses"] or a["big_ring"] != b["big_ring"]:
        return False
    if a.get("filled") != b.get("filled") or a.get("outline") != b.get("outline"):
        return False
    if abs(math.log(max(a["width"], 0.2) / max(b["width"], 0.2))) > 0.35:
        return False
    if abs(a["taper"] - b["taper"]) > 0.25 or abs(a["straight"] - b["straight"]) > 0.22:
        return False
    # La longueur ne sépare pas les familles : une feuille courte et une feuille longue du même
    # dessin sont le même motif, et doivent recevoir le même point.
    return True


def _assign_families(units, style, creativity):
    """Un point par famille de traits semblables, pas un point par trait.

    Trois niveaux de cohérence, dans cet ordre :
    1. le **poids du trait** du dessin (20e centile des largeurs) sépare un tracé au stylo d'une
       forme réellement plus grasse. Une branche mesure 1,0 à 1,6 mm selon sa longueur : sans ce
       poids, la même tige changeait de point d'un segment à l'autre ;
    2. tous les traits fins du motif reçoivent **le même** point de ligne (arrière si le dessin
       cerne des formes, tige s'il court en lignes ouvertes) : c'est ce qui rend une tige continue
       et des contours de feuilles homogènes ;
    3. les formes plus grasses sont regroupées par mesures voisines et partagent la moyenne du
       groupe (une feuille courte et une feuille longue, le même point).
    """
    if not units:
        return
    if style != "auto":
        for unit in units:
            unit["kind"] = choose_stitch(unit["width"], unit["length"], roundness=0.0,
                                         encloses=unit["encloses"], area_mm2=0.0, style=style,
                                         taper=unit["taper"], straight=unit["straight"],
                                         big_ring=unit["big_ring"], creativity=creativity)
        return
    weight = float(np.percentile([unit["width"] for unit in units], 20))
    for unit in units:
        unit["filled"] = unit["width"] >= 1.45 * weight and unit["taper"] >= 0.45
    for unit in units:
        if unit["width"] >= 1.45 * weight:
            continue
        # Un trait de même largeur ne raconte pas la même chose s'il cerne une forme (arrière,
        # ligne nette) ou s'il court en ligne ouverte (tige, torsade).
        # Au cran 5, le contour prend l'arrière (ligne nette) ; plus bas, un point fendu.
        outline_kind = "back" if creativity >= 5 else "split"
        unit["kind"] = _temper(outline_kind if unit.get("outline") else "stem", creativity)
    groups = []
    for unit in units:
        if unit["width"] < 1.45 * weight:
            continue
        for group in groups:
            if _alike(group["mean"], unit):
                count = group["count"] = group["count"] + 1
                for key in ("width", "length", "taper", "straight"):
                    group["mean"][key] += (unit[key] - group["mean"][key]) / count
                group["members"].append(unit)
                break
        else:
            groups.append({"count": 1, "members": [unit],
                           "mean": {key: unit[key] for key in ("width", "length", "taper", "straight",
                                                               "encloses", "big_ring", "filled", "outline")}})
    for group in groups:
        mean = group["mean"]
        # area_mm2=0 : une branche est un trait, jamais un point. Avec une surface estimée,
        # un petit trait de 2 mm passait sous le seuil du nœud et devenait une pastille.
        kind = choose_stitch(mean["width"], mean["length"], roundness=0.0,
                             encloses=mean["encloses"], area_mm2=0.0, style="auto",
                             taper=mean["taper"], straight=mean["straight"],
                             big_ring=mean["big_ring"], filled=mean["filled"], creativity=creativity)
        for unit in group["members"]:
            unit["kind"] = kind


def _component_units(comp, ppm, style, seed, creativity=3):
    """Mesure une forme connexe sans décider seul du point de ses traits.

    Renvoie `(units, strokes, kind)` : les traits sont rendus sans point, pour une décision à
    l'échelle de tout le motif ; les pastilles, semis et reprises sont dessinés tout de suite, car
    leur point ne se compare à rien. La largeur se mesure **branche par branche** : sur un dessin
    au trait, toutes les lignes forment une seule forme de plusieurs millimètres de large, et une
    mesure globale ferait broder un remplissage à la place du tracé.
    """
    info = _shape(comp, ppm)
    pen = hs.Pen(seed=seed)
    # Une pastille ronde et compacte est un nœud colonial, pas un trait : à cette taille, l'axe
    # médian la réduit à un point ou à une roue minuscule. C'est LE point classique de la main.
    # Un anneau (œil, boucle) est exclu : il doit rester un contour, pas devenir une boule.
    if (info["roundness"] >= 0.72 and info["length"] <= 4.0 and info["area"] <= 9.0
            and not info["encloses"]):
        hs.french_knot(pen, info["center"], max(0.6, min(1.5, info["length"] / 2)))
        return [], pen.strokes, "french_knot"
    if info["paths"]:
        units = []
        for path in info["paths"]:
            arc, half, half_fn, shape = _branch_measures(info, path, ppm)
            if arc < 0.8:
                continue
            units.append({"mode": "stroke", "path": path, "half": half, "half_fn": half_fn,
                          "length": arc, "width": 2 * half if half > 0.05 else 0.6,
                          "taper": shape["taper"], "straight": shape["straight"],
                          "outline": shape["outline"],
                          "encloses": info["encloses"], "big_ring": info["area"] >= 40})
        if units:
            return units, [], None
    width = 2 * info["half"] if info["half"] > 0.05 else info["length"]
    kind = choose_stitch(width, info["length"], roundness=info["roundness"], encloses=info["encloses"],
                         area_mm2=info["area"], style=style, creativity=creativity)
    if kind in ("seed", "darning", "laid", "tatami"):
        _draw_area(pen, comp, info, ppm, kind)
    elif not info["paths"] and kind not in ("french_knot", "woven_wheel", "spider_web", "lazy_daisy"):
        if info["length"] >= 1.6 * max(2 * info["half"], 0.2):
            _draw_short_stroke(pen, comp, ppm)      # petite marque allongée : un point, pas une pastille
            # Apply creativity tempering: at low creativity, back becomes running
            kind = _temper(kind, creativity)
        else:
            hs.french_knot(pen, info["center"], max(0.6, min(1.4, info["half"])))
            kind = "french_knot"
    else:
        _draw(pen, info, ppm, kind)
    return [], pen.strokes, kind


def _draw_units(pen, units, ppm, overrides=None):
    """Dessine les traits d'une forme avec les familles décidées à l'échelle du motif.

    `overrides` remplace le point d'un tracé précis (clé = identifiant du tracé) : c'est le choix
    fait à la main dans l'éditeur, qui ne doit pas être tempéré.
    """
    for unit in units:
        kind = (overrides or {}).get(unit["id"], unit["kind"])
        pts = unit["path"].astype(float) / ppm
        if kind == "french_knot":
            # Un nœud ne se pose pas sur une ligne : il se fait au centre du tracé, comme un motif rond.
            center = pts[len(pts) // 2]
            hs.french_knot(pen, (float(center[0]), float(center[1])), max(0.5, min(1.4, unit["half"])))
        else:
            _draw_path(pen, pts, unit["half_fn"], unit["half"], kind)
    return Counter((overrides or {}).get(unit["id"], unit["kind"]) for unit in units).most_common(1)[0][0]


def hand_pattern(source, background="outside", *, n_colors=None, exclude=(), style="auto", seed=0,
                 creativity=3, target_mm=None, color_overrides=None, stroke_overrides=None):
    """Image -> motif « broderie à la main » (EmbPattern pyembroidery).

    `stroke_overrides` : {identifiant de tracé: point} choisi à la main ; les identifiants sont
    listés dans `pattern.img2stitch_hand["strokes"]`.

    `target_mm` est le rectangle (largeur, hauteur) que la broderie ne doit pas dépasser : le motif
    est mis à l'échelle pour y tenir au plus juste, en gardant ses proportions. Sans lui, le motif
    garde sa taille naturelle, seulement ramenée sous MAX_END_MM.
    """
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
        pattern.img2stitch_hand = {"stitches": {}, "shapes": 0, "strokes": []}
        pattern.img2stitch_frame = {"w": width_px, "h": height, "ppm": float(WORK_PX_PER_MM),
                                    "cx": 0.0, "cy": 0.0, "scale": 1.0}
        return pattern

    # Les grandes surfaces d'abord, les détails ensuite : c'est l'ordre du travail à la main.
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    measured, pending, shapes = [], [], 0
    for color_index in order:
        rgb, mask = palette[color_index], masks[color_index]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        entries = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] / WORK_PX_PER_MM ** 2 < MIN_AREA_MM2:
                continue
            component_seed = seed + 7919 * (index + 1)
            units, ready, kind = _component_units(labels == index, WORK_PX_PER_MM, style,
                                                  component_seed, creativity=creativity)
            for number, unit in enumerate(units):
                unit["id"] = f"{color_index}-{index}-{number}"
            entries.append({"units": units, "strokes": ready, "kind": kind, "seed": component_seed,
                            "id": f"{color_index}-{index}", "mask_stats": [int(v) for v in stats[index, :4]]})
            pending.extend(units)
            shapes += 1
        measured.append((rgb, entries))

    # Une seule décision par famille de formes semblables, toutes couleurs confondues : c'est ce
    # qui rend une tige continue et des feuilles répétées cohérentes.
    _assign_families(pending, style, creativity)

    zones, segments, tally = [], [], {}
    stroke_list, stroke_paths = [], []
    first = True
    for rgb, entries in measured:
        strokes, chosen = [], []
        for entry in entries:
            if entry["units"]:
                pen = hs.Pen(seed=entry["seed"])
                chosen.append(_draw_units(pen, entry["units"], WORK_PX_PER_MM, stroke_overrides))
                for unit in entry["units"]:
                    stroke_list.append({"id": unit["id"], "color": "#%02x%02x%02x" % tuple(int(v) for v in rgb),
                                        "kind": (stroke_overrides or {}).get(unit["id"], unit["kind"]),
                                        "length": round(float(unit["length"]), 2)})
                    stroke_paths.append(unit["path"].astype(float) / WORK_PX_PER_MM)
                strokes.extend(pen.strokes)
            else:
                chosen.append(entry["kind"])
                strokes.extend(entry["strokes"])
        if not strokes:
            continue
        for kind in chosen:
            tally[kind] = tally.get(kind, 0) + 1
        color = tuple(int(v) for v in rgb)
        # Surcharge de couleur : seule la teinte du fil change, pas les points.
        source_hex = "#%02x%02x%02x" % color
        if color_overrides and source_hex in color_overrides:
            color = tuple(int(color_overrides[source_hex].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
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

    # Cadrage : le motif est mis à l'échelle pour tenir dans la taille demandée ; sans taille, on ne
    # réduit qu'au-delà de 11 cm, où les points perdraient l'échelle de la main. Le repère (ppm) suit
    # l'échelle, sinon l'aperçu ne superposerait plus les points et l'image source.
    xs = [s[0] for s in pattern.stitches]
    ys = [s[1] for s in pattern.stitches]
    ppm = float(WORK_PX_PER_MM)
    factor = 1.0
    if xs:
        width_mm = (max(xs) - min(xs)) / 10
        height_mm = (max(ys) - min(ys)) / 10
        if target_mm:
            target_w, target_h = (max(1.0, float(v)) for v in target_mm)
            factor = min(target_w / max(width_mm, 1e-6), target_h / max(height_mm, 1e-6))
        else:
            longest = max(width_mm, height_mm)
            factor = MAX_END_MM / longest if longest > MAX_END_MM else 1.0
        if abs(factor - 1.0) > 1e-6:
            for stitch in pattern.stitches:
                stitch[0] *= factor
                stitch[1] *= factor
            ppm /= factor
    pattern.end()
    cx, cy = _center_on_origin(pattern)
    # Tracés dans le repère final des points (dixièmes de mm), pour la carte de sélection.
    pattern.img2stitch_stroke_paths = [p * 10 * factor - np.array([cx, cy]) for p in stroke_paths]
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_hand = {"stitches": tally, "shapes": shapes,
                               "strokes": stroke_list}
    pattern.img2stitch_frame = {"w": width_px, "h": height, "ppm": ppm,
                                "cx": cx, "cy": cy, "scale": 1.0}
    return pattern


def hand_stitch_sample_pattern(kind):
    """Construit un petit exemple avec les mêmes primitives de points que la broderie à la main."""
    if kind not in hs.STITCH_TYPES:
        raise ValueError("Type de point inconnu.")

    pen = hs.Pen(seed=7)
    path = np.array([(3, 5), (38, 5)], dtype=float)
    half_width = 2.0
    half_fn = lambda _s: half_width

    if kind == "running":
        hs.running(pen, path, length=2.6, gap=0.7, spread=0)
    elif kind == "back":
        hs.back(pen, path)
    elif kind in ("stem", "split", "chain", "couching", "cross_stitch",
                  "satin", "fishbone", "herringbone", "chevron", "fern", "feather",
                  "cretan", "blanket", "overlock", "e_stitch", "arrow_chain", "lattice"):
        _draw_one(pen, path, half_fn, half_width, kind)
    elif kind == "darning":
        hs.darning(pen, np.ones((15, 52), np.uint8), 1.5)
    elif kind == "tatami":
        hs.tatami(pen, np.ones((15, 52), np.uint8), 1.5)
    elif kind == "seed":
        hs.seed(pen, np.ones((15, 52), np.uint8), 1.5, density=0.5)
    elif kind == "french_knot":
        hs.french_knot(pen, (20, 6), radius=2.2)
    elif kind == "bullion":
        hs.bullion(pen, (7, 6), (33, 6), radius=1.7)
    elif kind == "lazy_daisy":
        hs.lazy_daisy(pen, (7, 6), (33, 6), bulge=0.45)
    elif kind == "woven_wheel":
        hs.woven_wheel(pen, (20, 6), radius=5)
    elif kind == "spider_web":
        hs.spider_web(pen, (20, 6), radius=5)
    elif kind == "tulip":
        hs.tulip(pen, (20, 12), (20, 0), width=1.8)

    pattern = pe.EmbPattern()
    pattern.add_thread({"color": 0xB5233E})
    for stroke in pen.strokes:
        pattern.add_stitch_absolute(pe.JUMP, stroke[0][0] * 10, stroke[0][1] * 10)
        for x, y in stroke:
            pattern.add_stitch_absolute(pe.STITCH, x * 10, y * 10)
    pattern.end()
    return pattern
