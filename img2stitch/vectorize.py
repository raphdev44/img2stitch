"""Numérisation par vectorisation : l'image devient une géométrie, puis un parcours de broderie.

Le pipeline standard (``img2stitch.convert``) découpe les couleurs en masques de pixels et remplit
ces masques rangée par rangée. Ici, on change d'approche : chaque région de couleur est convertie en
polygones lissés (contours sous-échantillonnés puis simplifiés), ces polygones sont **dilatés** pour
compenser la tension du fil (*pull compensation*), puis :

1. une **sous-couche** (contour intérieur, *edge walk*) est piquée pour stabiliser le tissu ;
2. le **remplissage** est calculé par intersection vectorielle : les rangées sont coupées exactement
   par le bord des formes, sans escalier de pixels ;
3. le **parcours** est ordonné au plus court (fusion des tracés alignés puis tri par proximité), ce
   que font ``linemerge`` et ``linesort`` dans vpype.

vpype et vtracer ne sont pas utilisés ici : vpype épingle ``Shapely==1.8.0`` (plus de wheel à partir
de Python 3.11) et les wheels vtracer cp314 plantent à l'appel. Les étapes utiles du document de
spécification sont donc reproduites avec OpenCV + Shapely, en restant compatible avec le modèle de
motif de img2stitch (zones, cadrage, aperçus, éditeur).
"""
import math

import cv2
import numpy as np
import pyembroidery as pe

from .fill import zone_fill
from .outline import overlap_masks
from .planner import Sewer, plan_fill, plan_outline
from .preprocess import load_and_quantize

ENGINE = "vector"
SUPERSAMPLE = 2          # contours extraits à 20 px/mm : bords plus fins qu'un masque à 10 px/mm
SIMPLIFY_MM = 0.12       # simplification Douglas-Peucker des contours (anti-escalier)
PULL_MM = 0.25           # compensation de tension : les formes débordent sous le fil voisin
UNDERLAY_INSET_MM = 0.6  # la sous-couche reste à l'intérieur du remplissage
UNDERLAY_MIN_MM2 = 12.0  # en dessous, une sous-couche coûte plus de points qu'elle n'apporte
UNDERLAY_STEP_MM = 2.5   # longueur des points de la sous-couche
MIN_RUN_MM = 0.8         # une rangée plus courte n'apporte rien : elle sera recouverte


class VectorizerUnavailable(RuntimeError):
    """Dépendances de vectorisation absentes (Shapely n'est pas installé)."""


def available():
    try:
        import shapely  # noqa: F401
    except ImportError:
        return False
    return True


# --- Géométrie --------------------------------------------------------------------------------
def polygons_of(mask, px_per_mm, pull_mm=0.0, min_area_mm2=0.3):
    """Masque binaire -> polygones Shapely lissés (trous compris), dilatés de `pull_mm`.

    Le masque est suréchantillonné puis adouci avant extraction : les contours suivent mieux les
    courbes qu'un simple relevé de pixels, et la simplification retire l'escalier sans arrondir les
    angles vifs.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise VectorizerUnavailable(
            "La numérisation par vectorisation nécessite le paquet « shapely » "
            "(pip install shapely).") from exc

    src = (mask.astype(np.uint8) * 255)
    if SUPERSAMPLE > 1:
        src = cv2.resize(src, None, fx=SUPERSAMPLE, fy=SUPERSAMPLE, interpolation=cv2.INTER_CUBIC)
        src = cv2.GaussianBlur(src, (0, 0), 0.8)
        src = (src > 110).astype(np.uint8)
    else:
        src = (src > 127).astype(np.uint8)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    scale = px_per_mm * SUPERSAMPLE
    eps = max(0.6, SIMPLIFY_MM * scale)

    def ring(contour, area=None):
        # Même seuil que le moteur standard, appliqué AVANT la dilatation : sinon une tache de
        # 0,8 mm² deviendrait une « zone » de 1,7 mm² et créerait un fil de plus que demandé.
        if area is not None and abs(area) / (scale * scale) < min_area_mm2:
            return None     # tache de bruit : inutile de la vectoriser
        pts = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2) / scale
        return [tuple(float(v) for v in p) for p in pts] if len(pts) >= 3 else None

    shells, holes = {}, {}
    for index, (contour, info) in enumerate(zip(contours, hierarchy)):
        points = ring(contour, cv2.contourArea(contour))
        if points is None:
            continue
        if info[3] == -1:
            shells[index] = points
        else:
            holes.setdefault(info[3], []).append(points)

    geoms = []
    for index, points in shells.items():
        poly = Polygon(points, holes.get(index, []))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area > 0:
            geoms.append(poly)
    if not geoms:
        return []
    merged = unary_union(geoms)
    if pull_mm:
        merged = unary_union([part.buffer(pull_mm, join_style=1, quad_segs=4)
                              for part in explode(merged)])
    return [part.simplify(0.01) for part in explode(merged) if part.area >= min_area_mm2 * 0.5]


def explode(geom):
    """Géométrie Shapely -> liste de polygones (aplatit MultiPolygon et collections)."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [item for part in geom.geoms for item in explode(part)]
    return []


def loops_of(polygons):
    """Contours extérieurs et trous, en boucles de points mm (pour les contours brodés)."""
    loops = []
    for poly in polygons:
        loops.append([tuple(p) for p in poly.exterior.coords[:-1]])
        for hole in poly.interiors:
            loops.append([tuple(p) for p in hole.coords[:-1]])
    return loops


def hatch_rows(polygons, spacing_mm, angle_deg):
    """Rangées parallèles exactes : chaque rangée est coupée par le bord des polygones."""
    from shapely.geometry import LineString

    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    rows = []
    row = 0
    for comp, poly in enumerate(polygons):
        minx, miny, maxx, maxy = poly.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        if diag <= 0:
            continue
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        count = int(diag / spacing_mm) + 2
        for k in range(-count, count + 1):
            off = k * spacing_mm
            ox, oy = cx + nx * off, cy + ny * off
            line = LineString([(ox - dx * diag, oy - dy * diag),
                               (ox + dx * diag, oy + dy * diag)])
            for part in lines_of(poly.intersection(line)):
                p0, p1 = tuple(part.coords[0]), tuple(part.coords[-1])
                if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < MIN_RUN_MM:
                    continue
                rows.append({"p0": p0, "p1": p1, "comp": comp, "row": row, "line": k})
                row += 1
    return rows


def lines_of(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        return [item for part in geom.geoms for item in lines_of(part)]
    return []


def underlay_loops(polygons, inset_mm=UNDERLAY_INSET_MM, step_mm=UNDERLAY_STEP_MM):
    """Sous-couche « edge walk » : un contour intérieur qui fixe le tissu avant le remplissage."""
    loops = []
    for poly in polygons:
        if poly.area < UNDERLAY_MIN_MM2:
            continue
        inner = poly.buffer(-inset_mm, join_style=1, quad_segs=4)
        for part in explode(inner):
            if part.area < UNDERLAY_MIN_MM2 * 0.5:
                continue
            ring = resample([tuple(p) for p in part.exterior.coords[:-1]], step_mm)
            if len(ring) >= 3:
                loops.append(ring)
    return loops


def resample(ring, step_mm):
    """Ré-échantillonne une boucle à pas constant (points de sous-couche réguliers)."""
    out, acc = [], 0.0
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 0:
            continue
        t = 0.0 if not out else step_mm - acc
        while t < seg:
            out.append((a[0] + (b[0] - a[0]) * t / seg, a[1] + (b[1] - a[1]) * t / seg))
            t += step_mm
        acc = seg - (t - step_mm)
    return out


# --- Parcours ---------------------------------------------------------------------------------
def merge_runs(rows, tolerance_mm=0.15):
    """Fusionne les rangées alignées qui se touchent (équivalent de ``linemerge``).

    La fusion se fait à l'intérieur d'une même forme : deux rangées de formes différentes ne peuvent
    pas être reliées sans traverser du tissu nu.
    """
    by_comp = {}
    for row in rows:
        # Deux morceaux d'une même rangée (coupée par un trou) ne se touchent pas : les comparer
        # entre rangées ne servirait à rien et coûterait O(n²) sur les images riches en détails.
        by_comp.setdefault((row["comp"], row.get("line")), []).append(
            dict(row, a=row["p0"], b=row["p1"]))
    out = []
    for (comp, _), runs in by_comp.items():
        changed = True
        while changed:
            changed = False
            for i in range(len(runs)):
                if runs[i] is None:
                    continue
                for j in range(i + 1, len(runs)):
                    if runs[j] is None:
                        continue
                    merged = _join(runs[i], runs[j], tolerance_mm)
                    if merged is not None:
                        runs[i] = merged
                        runs[j] = None
                        changed = True
                        break
        out.extend({"p0": r["a"], "p1": r["b"], "comp": comp, "row": r["row"]}
                   for r in runs if r is not None)
    return out


def _join(first, second, tol):
    """Deux rangées bout à bout et dans la même direction -> une seule, sinon None."""
    fa, fb, sa, sb = first["a"], first["b"], second["a"], second["b"]
    for (a0, a1), (b0, b1) in (((fa, fb), (sa, sb)), ((fa, fb), (sb, sa)),
                               ((fb, fa), (sa, sb)), ((fb, fa), (sb, sa))):
        if math.hypot(a1[0] - b0[0], a1[1] - b0[1]) > tol:
            continue
        la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
        lb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
        if la > 0 and lb > 0:
            cross = (a1[0] - a0[0]) * (b1[1] - b0[1]) - (a1[1] - a0[1]) * (b1[0] - b0[0])
            if abs(cross) / (la * lb) > 0.08:   # pas la même direction : on ne colle pas
                continue
        return {"a": a0, "b": b1, "comp": first["comp"], "row": min(first["row"], second["row"])}
    return None


def _hatch_angle(kind, settings):
    if kind == "tatami_h":
        return 0.0
    if kind == "tatami_v":
        return 90.0
    return settings.fill_angle_deg


SPECIALIZED = ("satin", "satin_flow", "wave", "zigzag", "contour")


def is_specialized(kind):
    """Motifs dont le remplissage reste confié au moteur raster (satin, relief, vagues, contour)."""
    return kind in SPECIALIZED or kind.startswith("relief_")


def _zone_passes(kind, polygons, fmask, px_per_mm, settings):
    """Passes de remplissage d'une zone : vectorielles, ou moteur raster pour les motifs spécialisés."""
    if kind == "none":
        return []
    if kind == "contour":
        inner = [p.buffer(-max(0.4, settings.row_spacing_mm * 0.6), join_style=1, quad_segs=4)
                 for p in polygons]
        return [("loops", loops_of([part for g in inner for part in explode(g)]))]
    if is_specialized(kind) or not polygons:
        # Satin tournant, relief, vagues : le moteur raster les traite déjà bien.
        return zone_fill(kind, fmask, px_per_mm, settings)
    spacing = max(0.2, settings.row_spacing_mm)
    if kind == "cross":
        spacing *= 1.75
        return [("runs", merge_runs(hatch_rows(polygons, spacing, _hatch_angle(kind, settings)))),
                ("runs", merge_runs(hatch_rows(polygons, spacing, _hatch_angle(kind, settings) + 90.0)))]
    rows = merge_runs(hatch_rows(polygons, spacing, _hatch_angle(kind, settings)))
    if not rows:    # forme trop fine pour une rangée : le moteur raster sait encore la remplir
        return zone_fill(kind, fmask, px_per_mm, settings)
    return [("runs", rows)]


def plan_shapes(sw, groups):
    """Visite chaque forme une seule fois : sous-couche puis remplissage.

    Le planificateur standard enchaîne les passes ; en vectorisation la sous-couche ferait alors un
    deuxième tour de toutes les formes, avec un saut et une coupe à chaque déplacement entre formes.
    """
    left = list(range(len(groups)))
    starts = [np.array([p for p in groups[i]["entry"]], float) for i in range(len(groups))]
    while left:
        if sw.cur is None:
            index = left[0]
        else:
            # Les k formes dont une entrée est la plus proche : évaluer le vrai coût (qui simule le
            # déplacement) sur toutes les formes serait quadratique sur les images très découpées.
            cur = np.array(sw.cur, float)
            dist = [float(np.min(np.hypot(starts[i][:, 0] - cur[0], starts[i][:, 1] - cur[1]))) for i in left]
            k = min(8, len(left))
            candidates = [left[j] for j in np.argsort(dist)[:k]]
            best = None
            for index in candidates:
                for point in groups[index]["entry"]:
                    cost = sw.cost(sw.cur, point)
                    if best is None or cost < best[0]:
                        best = (cost, index)
            index = best[1]
        left.remove(index)
        group = groups[index]
        if group["underlay"]:
            plan_outline(sw, group["underlay"], 1)
        if group["rows"]:
            plan_fill(sw, group["rows"])


def _groups(polygons, rows):
    """Une entrée par forme : ses rangées et sa sous-couche, visitées d'un seul tenant."""
    by_comp = {}
    for row in rows:
        by_comp.setdefault(row["comp"], []).append(row)
    groups = []
    for index, poly in enumerate(polygons):
        underlay = underlay_loops([poly])
        shape_rows = by_comp.get(index, [])
        entry = []
        if underlay:
            entry.append(underlay[0][0])
        if shape_rows:
            entry.extend([shape_rows[0]["p0"], shape_rows[0]["p1"]])
        if entry:
            groups.append({"rows": shape_rows, "underlay": underlay, "entry": entry})
    return groups


def build_vector_pattern(path, settings):
    """Image -> motif, par vectorisation. Même contrat que ``convert._build_pattern``."""
    # Import tardif : convert importe ce module, on évite le cycle à l'import du paquet.
    from .convert import _center_on_origin, _merge_order, _region_look, _split_regions, _zone_override

    px_per_mm = 10
    palette, masks = load_and_quantize(
        path, round(settings.width_mm * px_per_mm), settings.n_colors, settings.background,
        settings.background_tolerance, settings.background_scope, settings.exclude_colors,
        thicken_lines=settings.thicken_lines)
    pattern = pe.EmbPattern()
    if not masks:
        pattern.end()
        return pattern

    frame = {"w": masks[0].shape[1], "h": masks[0].shape[0], "ppm": px_per_mm,
             "cx": 0.0, "cy": 0.0, "scale": 1.0}
    palette, masks, regions = _split_regions(palette, masks, settings.region_overrides, px_per_mm)
    looks = [_zone_override(rgb, settings.zone_overrides) if regions[i] is None
             else _region_look(settings.region_overrides[regions[i]], rgb)
             for i, rgb in enumerate(palette)]
    if settings.merge_same_colors:
        order = _merge_order([c for c, _, _ in looks], masks, px_per_mm, settings.overlap_mm)
        palette, masks, regions, looks = ([lst[k] for k in order] for lst in (palette, masks, regions, looks))
    fill_masks = overlap_masks(masks, px_per_mm, settings.overlap_mm)

    later = [None] * len(masks)
    acc = np.zeros_like(masks[0])
    for i in range(len(masks) - 1, -1, -1):
        later[i] = acc.copy()
        acc |= masks[i]

    sw = Sewer(pattern, settings, px_per_mm)
    first, thread = True, None
    zones, segments = [], []
    # Le remplissage déborde de PULL_MM (compensation de tension) : un déplacement qui reste dedans
    # est couvert par le fil, donc acceptable. Sans cette marge, la navette ferait un saut à chaque
    # changement de rangée près du bord.
    reach = 2 * int(round(PULL_MM * px_per_mm)) + 1
    tol = np.ones((reach, reach), np.uint8)
    # Même seuil que le moteur standard : une tache plus petite que `min_area_mm2` est du bruit, pas
    # une zone à broder (sinon elle créerait un fil et des changements de couleur inutiles).
    min_area = max(0.5, settings.min_area_mm2)
    for i, (rgb, mask, fmask) in enumerate(zip(palette, masks, fill_masks)):
        color, kind, outline = looks[i]
        zone_index = len(zones)
        zones.append((tuple(rgb), regions[i]))
        shape_polys = polygons_of(mask, px_per_mm, 0.0, min_area)
        fill_polys = polygons_of(fmask, px_per_mm, PULL_MM, min_area)
        passes = _zone_passes(kind, fill_polys, fmask, px_per_mm, settings)
        want_outline = settings.outline if outline is None else outline
        loops = loops_of(shape_polys) if want_outline else []
        if not any(data for _, data in passes) and not loops:
            continue

        same = color == thread
        if not same:
            if not first:
                pattern.add_command(pe.COLOR_CHANGE)
            pattern.add_thread({"color": (color[0] << 16) | (color[1] << 8) | color[2]})
        first, thread = False, color
        segments.append((len(pattern.stitches), zone_index))

        hidden = cv2.dilate((fmask | later[i]).astype(np.uint8), tol).astype(bool)
        sw.new_color(hidden, later[i])
        if same:
            sw.fresh = False
        if kind == "none" or is_specialized(kind):
            # Moteur raster : on garde l'ordre de passes historique.
            underlay = underlay_loops(fill_polys) if want_outline else []
            if underlay:
                plan_outline(sw, underlay, 1)
            for what, data in passes:
                if what == "runs":
                    plan_fill(sw, data)
                elif data:
                    plan_outline(sw, data, 1)
            if loops:
                plan_outline(sw, loops, settings.outline_passes)
        else:
            rows = next((data for what, data in passes if what == "runs"), [])
            plan_shapes(sw, _groups(fill_polys, rows))
            if loops:
                plan_outline(sw, loops, settings.outline_passes)

    pattern.end()
    pattern.img2stitch_zones = zones
    pattern.img2stitch_segments = segments
    pattern.img2stitch_engine = ENGINE
    if settings.center_on_origin:
        frame["cx"], frame["cy"] = _center_on_origin(pattern)
    pattern.img2stitch_frame = frame
    return pattern
