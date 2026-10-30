"""Optimisation radicale d'un motif de broderie, pensée pour l'usage en machine.

Contrairement aux passes prudentes de :mod:`img2stitch.optimize`, cette optimisation travaille
sur une **copie** du motif et accepte de le réorganiser profondément pour réduire le nombre de
points piqués et le nombre de coupes de fil :

1. nettoyage agressif des passages : doublons, micro-points, points quasi alignés fusionnés
   (un point peut alors atteindre ``stitch_mm``, 4,5 mm par défaut, la valeur conseillée pour
   un remplissage tatami) ;
2. **chasse aux points recouverts** : les points d'une couleur entièrement cachés par une couche
   brodée plus tard sont retirés ;
3. **réordonnancement** : les blocs d'un même fil sont rapprochés, l'ordre des couleurs est choisi
   pour raccourcir les déplacements, et les passages d'un même bloc sont enchaînés au plus proche
   voisin (avec inversion du sens de piquage si c'est plus court) ;
4. **suppression des coupes** : les déplacements courts, ou entièrement cachés sous une zone
   brodée plus tard (ou sous le même fil), sont piqués au lieu d'être coupés ;
5. contrôle final de l'apparence : le rendu couleur du motif optimisé est comparé à l'original ;
   si la réorganisation change trop l'image, elle est annulée et seules les passes de points
   sont conservées.

L'ordre de superposition est préservé : deux blocs qui se recouvrent gardent leur ordre relatif
(sauf cycle impossible, signalé dans le rapport). Les commandes non prises en charge (arrêt
opérateur, appliqué, sequins...) et les changements de fil consécutifs font abandonner
l'optimisation plutôt que de les perdre.

Coordonnées pyembroidery : dixièmes de millimètre.
"""
import numpy as np
import cv2
import pyembroidery as pe
from PIL import Image, ImageDraw

from .optimize import (THREAD_MM, UNITS_PER_MM, _Grid, _copy_pattern, _distance, _iou, _parse,
                       _sep_kind, _segment_distance, _simplify, _split_long)

MASK = pe.COMMAND_MASK
SUPPORTED = {pe.STITCH, pe.JUMP, pe.TRIM, pe.COLOR_CHANGE, pe.END}
TRAVEL_PIECE_MM = 2.0      # longueur des morceaux d'un déplacement piqué
MIN_VISIBLE_IOU = 0.99     # seuil de contrôle d'un passage après suppression de points cachés
MAX_COLOR_MASKS = 48       # au-delà, on ne précalcule plus la couverture d'une couleur
MAX_TWO_OPT_RUNS = 600     # au-delà, le 2-opt coûterait plus qu'il ne rapporte

DEFAULTS = {
    "min_stitch_mm": 0.4,           # en dessous, un point est un micro-point
    "simplify_tol_mm": 0.2,         # écart autorisé en fusionnant des points alignés
    "stitch_mm": 4.5,               # longueur maximale en fusionnant (tatami : 4 mm conseillés)
    "max_stitch_mm": 12.0,          # au-delà, le point est subdivisé (limite des formats)
    "color_change_cost_s": 30.0,    # temps perdu par changement de fil (aiguille unique)
    "trim_cost_s": 5.0,             # temps perdu par coupe de fil
    "jump_cost_s": 0.4,             # temps perdu par saut sans coupe
    "hidden_travel_mm": 60.0,       # longueur maximale d'un déplacement caché piqué
    "hidden_ratio": 0.90,           # part minimale du trajet qui doit être recouverte
    "max_gap_mm": 1.2,              # plus longue trouée visible tolérée dans un trajet caché
    "hidden_margin_mm": 0.15,       # marge de sécurité autour d'un point caché
    "appearance_tolerance": 0.02,   # part de l'image brodée qui peut changer
}


class _Run:
    """Suite de points d'un passage ; ``origin`` = index du premier point d'origine."""

    __slots__ = ("points", "origin")

    def __init__(self, points, origin=None):
        self.points = points
        self.origin = points[0][3] if origin is None else origin

    def flipped(self):
        return _Run(self.points[::-1], self.origin)


class _Block:
    """Suite de passages d'un même fil, dans l'ordre du motif d'origine."""

    __slots__ = ("runs", "index", "thread", "color", "key", "label", "surface")

    def __init__(self, runs, index, thread):
        self.runs = runs
        self.index = index
        self.thread = thread
        self.color = None    # couleur RGB du fil, None si inconnue (fichier sans couleurs)
        self.key = None      # regroupement : couleur du fil, ou couple unique si inconnue
        self.label = None    # entier utilisé pour peindre l'aperçu de contrôle
        self.surface = 0


def optimize_pattern_radical(pattern, **overrides):
    """Retourne ``(motif_optimisé, rapport)`` sans modifier ``pattern``.

    Les passes sont tentées du plus agressif au plus prudent ; le premier résultat dont
    l'apparence reste fidèle est retenu. En cas d'échec total, une copie inchangée est renvoyée
    avec un avertissement dans le rapport.
    """
    prm = dict(DEFAULTS)
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in prm:
            raise TypeError("paramètre inconnu : %s" % key)
        prm[key] = value

    optimized = _copy_pattern(pattern)
    report = {
        "mode": "radical", "level": "radical",
        "removed_duplicates": 0, "removed_micro_stitches": 0, "simplified_points": 0,
        "removed_hidden_stitches": 0, "merged_trims": 0, "hidden_travels_merged": 0,
        "split_long_stitches": 0, "reordered": False, "color_changes_saved": 0,
        "appearance_change_pct": 0.0, "coverage_iou": 1.0, "warnings": [],
    }
    optimizer = _Radical(pattern, prm, report)
    report["before"] = optimizer.before_stats
    if optimizer.error:
        report["warnings"].append(optimizer.error)
        report["after"] = dict(optimizer.before_stats)
        optimized.stitches = [list(stitch[:3]) for stitch in pattern.stitches]
        return optimized, report

    attempts = [
        dict(reorder=True, hidden_travel=True, remove_hidden=True),
        dict(reorder=False, hidden_travel=True, remove_hidden=True),
        dict(reorder=False, hidden_travel=False, remove_hidden=False),
    ]
    keep = _score(report["before"])
    candidates = []
    for number, attempt in enumerate(attempts):
        info = optimizer.build(**attempt)
        faithful = (info["appearance"] <= prm["appearance_tolerance"]
                    and info["coverage_iou"] >= 0.95)
        if not faithful:
            if number < len(attempts) - 1:
                report["warnings"].append(
                    "Réorganisation trop visible (%.1f %% de l'image modifiée) : repli sur une "
                    "passe plus prudente." % (100 * info["appearance"]))
            continue
        candidates.append(info)
        if _score(info["after"]) <= keep:
            break
    candidates = [info for info in candidates if _score(info["after"]) <= keep]
    if not candidates:
        report["warnings"].append(
            "Aucune passe n'améliore le motif sans en changer l'aspect : motif laissé en l'état.")
        report["after"] = dict(report["before"])
        optimized.stitches = [list(stitch[:3]) for stitch in pattern.stitches]
        return optimized, report

    result = min(candidates, key=lambda info: _score(info["after"]))
    optimized.stitches = result["stitches"]
    if result["threads"] is not None:
        optimized.threadlist = result["threads"]
    optimized.img2stitch_segments = result["segments"] or []

    report.update({
        "removed_hidden_stitches": result["removed_hidden_stitches"],
        "merged_trims": result["merged_trims"],
        "hidden_travels_merged": result["hidden_travels_merged"],
        "split_long_stitches": result["split_long_stitches"],
        "reordered": result["reordered"],
        "blocks_before": result["blocks_before"],
        "blocks_after": result["blocks_after"],
        "color_changes_saved": max(0, report["before"]["color_changes"]
                                   - result["after"]["color_changes"]),
        "appearance_change_pct": round(100 * result["appearance"], 2),
        "coverage_iou": round(float(result["coverage_iou"]), 4),
        "after": result["after"],
    })
    return optimized, report


class _Radical:
    """État partagé des passes : blocs, passages nettoyés, rasters et contraintes d'ordre."""

    def __init__(self, pattern, prm, report):
        self.prm = prm
        self.report = report
        self.error = None
        self.before_stats = _stats(pattern.stitches)
        self.segments = [tuple(item) for item in getattr(pattern, "img2stitch_segments", None)
                         or []]
        self.relaxed = 0
        self.reorder_allowed = False

        parsed = _parse_pattern(pattern)
        if isinstance(parsed, str):
            self.error = parsed
            return
        self.blocks, self.tail, self.trim_threshold = parsed
        if not self.blocks:
            self.error = "Motif vide : rien à optimiser."
            return
        self.reorder_allowed = all(block.color is not None for block in self.blocks)
        if not self.reorder_allowed:
            report["warnings"].append(
                "Couleurs de fil inconnues dans le fichier : l'ordre des couleurs est conservé, "
                "seuls les passages de chaque couleur sont optimisés.")
        self.before_stats["blocks"] = len({block.key for block in self.blocks})

        items = [list(stitch) + [index] for index, stitch in enumerate(pattern.stitches)]
        self.grid = _Grid(items)
        self._reduce_runs()
        self.before_paint = self._paint(
            [(block.label, [point for run in block.runs for point in run.points])
             for block in self.blocks])
        self.before_covered = max(1, int(np.count_nonzero(self.before_paint >= 0)))
        self.before_mask = self.before_paint >= 0
        self.block_masks = [self._mask(block.runs) for block in self.blocks]
        for block, (box, mask) in zip(self.blocks, self.block_masks):
            block.surface = int(np.count_nonzero(mask))
        self.preds = self._precedence()

    # --- Préparation ----------------------------------------------------------------------
    def _reduce_runs(self):
        """Doublons, micro-points et points alignés : la passe la plus agressive du moteur."""
        protected = {int(start) for start, _ in self.segments}
        counts = dict(duplicates=0, micro=0, simplified=0)
        for block in self.blocks:
            runs = []
            for run in block.runs:
                points = _reduce(run.points, self.prm, protected, counts)
                if points:
                    runs.append(_Run(points, run.origin))
            block.runs = runs
        self.report["removed_duplicates"] = counts["duplicates"]
        self.report["removed_micro_stitches"] = counts["micro"]
        self.report["simplified_points"] = counts["simplified"]

    def _mask(self, runs):
        return _mask_of(self.grid, runs, THREAD_MM)

    def _precedence(self):
        """Un bloc reste avant ceux qu'il précède déjà s'ils se recouvrent (superposition)."""
        count = len(self.blocks)
        preds = [set() for _ in range(count)]
        for j in range(count):
            for i in range(j):
                if self.blocks[i].key == self.blocks[j].key:
                    continue        # même fil : l'ordre entre ces blocs ne change rien
                if _overlap(self.block_masks[i], self.block_masks[j]):
                    preds[j].add(i)
        return preds

    # --- Séquencement ---------------------------------------------------------------------
    def _sequence(self):
        count = len(self.blocks)
        if count < 2:
            return list(range(count))
        roots = [index for index in range(count) if not self.preds[index]] or list(range(count))
        roots.sort(key=lambda index: (-self.blocks[index].surface, index))
        best, best_score = list(range(count)), None
        for start in roots[:6]:
            order, score = self._greedy(start)
            if best_score is None or score < best_score:
                best, best_score = order, score
        if self.relaxed and not getattr(self, "_warned", False):
            self._warned = True
            self.report["warnings"].append(
                "Superpositions croisées entre blocs : l'ordre de certaines couches a été relâché "
                "pour pouvoir réorganiser le parcours (vérifiez l'aperçu).")
        return best

    def _greedy(self, start):
        count = len(self.blocks)
        order, done = [start], {start}
        remaining = set(range(count)) - done
        position = None
        previous = self.blocks[start].key
        score = 0.0
        while remaining:
            available = [index for index in remaining if self.preds[index] <= done]
            if not available:       # cycle de superpositions : on relâche le moins contraint
                self.relaxed += 1
                available = sorted(remaining, key=lambda i: len(self.preds[i] - done))[:3]
            costs = []
            for index in available:
                entry, distance = (None, 0.0) if position is None else \
                    _nearest_endpoint(position, self.blocks[index].runs)
                penalty = 0.0 if self.blocks[index].key == previous else \
                    self.prm["color_change_cost_s"]
                costs.append((self._travel_cost(distance) + penalty, index, entry))
            cost, pick, entry = min(costs, key=lambda item: (item[0], item[1]))
            score += cost
            done.add(pick)
            remaining.discard(pick)
            previous = self.blocks[pick].key
            position = _exit_hint(self.blocks[pick].runs, entry)
            order.append(pick)
        return order, score

    def _travel_cost(self, distance):
        """Coût machine d'un déplacement, en secondes (coupe ou saut)."""
        length = distance / UNITS_PER_MM
        if length <= self.trim_threshold:
            return self.prm["jump_cost_s"] + 0.01 * length
        return self.prm["trim_cost_s"] + self.prm["jump_cost_s"] + 0.01 * length

    # --- Construction ---------------------------------------------------------------------
    def build(self, *, reorder, hidden_travel, remove_hidden):
        self.report["split_long_stitches"] = 0
        count = len(self.blocks)
        order = self._sequence() if reorder and self.reorder_allowed else list(range(count))
        blocks = [self.blocks[index] for index in order]
        moved = order != list(range(count))

        # Ordre des passages de chaque bloc, en suivant l'aiguille depuis la position atteinte
        ordered, position = [], None
        for block in blocks:
            runs = _order_runs(block.runs, position, self.prm, self.trim_threshold)
            ordered.append(runs)
            if runs:
                position = runs[-1].points[-1]

        pruned = [None] * count
        links = [None] * count
        after_hidden = [False] * count
        hidden_removed = 0
        same_cover = self._same_color_covers()
        acc = np.zeros((self.grid.h, self.grid.w), bool)
        visible_before = np.zeros((self.grid.h, self.grid.w), bool)
        visible_after = np.zeros((self.grid.h, self.grid.w), bool)
        for index in reversed(range(count)):
            block = blocks[index]
            runs = ordered[index]
            if remove_hidden:
                runs, removed = self._prune(runs, acc)
                hidden_removed += removed
            pruned[index] = runs
            own = _solid(self.grid.coverage([run.points for run in runs], THREAD_MM))
            # Couverture réellement visible : ce qui n'est pas recouvert par une couche suivante.
            # C'est elle qui doit rester identique, pas la couverture brute (des points cachés
            # sont retirés volontairement).
            visible_before |= _solid(self.grid.coverage(
                [run.points for run in ordered[index]], THREAD_MM)) & ~acc
            visible_after |= own & ~acc
            same = same_cover.get(block.key)
            mask = acc | (own if same is None else same)
            links[index] = [bool(hidden_travel) and self._hidden(mask, first.points[-1],
                                                                 second.points[0])
                            for first, second in zip(runs, runs[1:])]
            if hidden_travel and index + 1 < count:
                after_hidden[index] = self._hidden(mask, runs[-1].points[-1],
                                                   pruned[index + 1][0].points[0])
            acc = acc | own

        stitches, painted, new_start = [], [], {}
        travel, hidden_count, merged = 0.0, 0, 0
        previous_key, position = None, None
        for index, block in enumerate(blocks):
            runs = pruned[index]
            start = runs[0].points[0]
            if previous_key is None:
                stitches.append([start[0], start[1], pe.JUMP, start[3]])
            elif block.key != previous_key:
                stitches.append([position[0], position[1], pe.COLOR_CHANGE, position[3]])
                stitches.append([start[0], start[1], pe.JUMP, start[3]])
            else:
                travel += _distance(position, start) / UNITS_PER_MM
                commands, trace, kind = self._connector(position, start, after_hidden[index - 1])
                stitches.extend(commands)
                if kind == "hidden":
                    hidden_count += 1
                elif kind == "jump":
                    merged += 1
                if trace:
                    painted.append((block.label, trace))
            for number, run in enumerate(runs):
                if number:
                    first = runs[number - 1].points[-1]
                    travel += _distance(first, run.points[0]) / UNITS_PER_MM
                    commands, trace, kind = self._connector(first, run.points[0],
                                                            links[index][number - 1])
                    stitches.extend(commands)
                    if kind == "hidden":
                        hidden_count += 1
                    elif kind == "jump":
                        merged += 1
                    if trace:
                        painted.append((block.label, trace))
                points = _split_long(run.points, self.prm["max_stitch_mm"] * UNITS_PER_MM,
                                     self.report)
                new_start[run.origin] = len(stitches)
                stitches.extend(points)
                painted.append((block.label, points))
                position = points[-1]
            previous_key = block.key
        stitches.append([position[0], position[1], pe.END, position[3]])

        after_paint = self._paint(painted)
        appearance = self._appearance_change(self.before_paint, after_paint)
        coverage = float(_iou(visible_before, visible_after))
        after_stats = _stats(stitches)
        after_stats["travel_mm"] = round(travel, 1)
        after_stats["blocks"] = len({block.key for block in blocks})
        after_stats["runs"] = sum(len(runs) for runs in pruned)

        self.report.update({"merged_trims": merged, "hidden_travels_merged": hidden_count,
                            "reordered": bool(moved)})
        return {
            "stitches": [list(item[:3]) for item in stitches],
            "threads": self._threads(blocks) if self.reorder_allowed else None,
            "segments": self._remap_segments(new_start),
            "split_long_stitches": self.report["split_long_stitches"],
            "appearance": appearance,
            "coverage_iou": coverage,
            "removed_hidden_stitches": hidden_removed,
            "merged_trims": merged,
            "hidden_travels_merged": hidden_count,
            "reordered": bool(moved),
            "blocks_before": len(self.blocks),
            "blocks_after": len(blocks),
            "after": after_stats,
        }

    def _connector(self, position, start, hidden):
        """Commandes pour relier deux passages, et tracé visible éventuel.

        Renvoie ``(commandes, tracé, type)`` ; le type vaut ``hidden`` (piqué sous une couche),
        ``jump`` (saut que la machine ne coupe pas) ou ``trim`` (coupe puis saut). Un déplacement
        visible court reste un saut : le piquer ajouterait des points sans éviter de coupe.
        """
        if hidden:
            pieces = _travel(start, position, TRAVEL_PIECE_MM * UNITS_PER_MM)
            return pieces, [position] + pieces + [start], "hidden"
        if _distance(position, start) <= self.trim_threshold * UNITS_PER_MM:
            return [[start[0], start[1], pe.JUMP, start[3]]], None, "jump"
        return ([[position[0], position[1], pe.TRIM, position[3]],
                 [start[0], start[1], pe.JUMP, start[3]]], None, "trim")

    def _appearance_change(self, before, after):
        """Part de l'image brodée où une couleur apparaît là où elle n'était pas.

        Une tolérance de quelques dixièmes de millimètre évite de compter les micro-décalages
        de tracé dus à la fusion de points alignés.
        """
        covered = np.count_nonzero(before >= 0)
        if not covered:
            return 0.0
        size = max(1, int(round(0.35 / self.grid.px_mm)))
        kernel = np.ones((2 * size + 1, 2 * size + 1), np.uint8)
        labels = np.unique(np.concatenate((before[before >= 0].ravel(),
                                           after[after >= 0].ravel())))
        mismatch = 0
        for label in labels:
            old = cv2.dilate((before == label).astype(np.uint8), kernel) > 0
            mismatch += int(np.count_nonzero((after == label) & ~old))
        return mismatch / covered

    def _same_color_covers(self):
        """Couverture de tous les blocs d'un même fil (trajet invisible dans cette couleur)."""
        groups = {}
        for block in self.blocks:
            groups.setdefault(block.key, []).append(block)
        covers = {}
        if len(groups) > MAX_COLOR_MASKS:
            return covers
        for key, blocks in groups.items():
            if len(blocks) > 1:
                covers[key] = _solid(self.grid.coverage(
                    [run.points for block in blocks for run in block.runs], THREAD_MM))
        return covers

    def _hidden(self, mask, start, end):
        prm = self.prm
        distance = _distance(start, end)
        if distance <= 0:
            return True
        if distance > prm["hidden_travel_mm"] * UNITS_PER_MM:
            return False
        grid = self.grid
        steps = max(2, int(distance * grid.scale) + 1)
        t = np.linspace(0.0, 1.0, steps)
        xs = start[0] + (end[0] - start[0]) * t
        ys = start[1] + (end[1] - start[1]) * t
        ix = np.clip(np.rint((xs - grid.x0) * grid.scale).astype(int), 0, grid.w - 1)
        iy = np.clip(np.rint((ys - grid.y0) * grid.scale).astype(int), 0, grid.h - 1)
        covered = mask[iy, ix]
        if not covered.all() and covered.mean() < prm["hidden_ratio"]:
            return False
        return _longest_gap(covered) * (distance / UNITS_PER_MM / (steps - 1)) \
            <= prm["max_gap_mm"]

    def _prune(self, runs, covered):
        """Retire les points d'un passage entièrement recouverts par les couches suivantes."""
        if not covered.any():
            return runs, 0
        dist = cv2.distanceTransform(covered.astype(np.uint8), cv2.DIST_L2, 3)
        need = max(1.0, (THREAD_MM / 2 + self.prm["hidden_margin_mm"]) / self.grid.px_mm)
        out, removed = [], 0
        for run in runs:
            points = run.points
            if len(points) < 3:
                out.append(run)
                continue
            xy = np.array([self.grid.px(point) for point in points])
            ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, self.grid.w - 1)
            iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, self.grid.h - 1)
            safe = dist[iy, ix] >= need
            if safe.all() or not safe.any():
                out.append(run)
                continue
            kept = _hidden_indices(points, safe,
                                   self.prm["simplify_tol_mm"] * UNITS_PER_MM)
            if len(kept) == len(points):
                out.append(run)
                continue
            candidate = _Run([points[index] for index in kept], run.origin)
            if not _visible_preserved(self.grid, points, candidate.points, covered):
                out.append(run)
                continue
            out.append(candidate)
            removed += len(points) - len(kept)
        return out, removed

    def _paint(self, strokes):
        grid = self.grid
        image = Image.new("I", (grid.w, grid.h), -1)
        draw = ImageDraw.Draw(image)
        width = grid._width(THREAD_MM)
        for label, points in strokes:
            pixel = [grid.px(point) for point in points]
            for first, second in zip(pixel, pixel[1:]):
                draw.line([first, second], fill=int(label), width=width)
            if len(pixel) == 1:      # point isolé : une piqûre, pas un tracé
                x, y = pixel[0]
                draw.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2],
                             fill=int(label))
        return np.asarray(image)

    def _threads(self, blocks):
        """Un fil par changement de couleur émis (même couleur possible à deux endroits)."""
        threads, previous = [], None
        for block in blocks:
            if block.key == previous:
                continue        # même fil que le bloc précédent : pas de changement de couleur
            previous = block.key
            threads.append(pe.EmbThread({"color": block.color}))
        return threads

    def _remap_segments(self, new_start):
        if not self.segments:
            return None
        ranges = sorted((run.origin, block.index, number) for block in self.blocks
                        for number, run in enumerate(block.runs))
        if not ranges:
            return None
        starts = [item[0] for item in ranges]
        mapped = {}
        for stitch_index, zone in self.segments:
            position = _bisect(starts, int(stitch_index))
            if position < 0:
                continue
            origin = starts[position]
            if origin in mapped or origin not in new_start:
                continue
            mapped[origin] = (new_start[origin], zone)
        return [mapped[key] for key in sorted(mapped, key=lambda k: mapped[k][0])]


# ----------------------------------------------------------------------------------------
# Découpage du motif
# ----------------------------------------------------------------------------------------
def _separator_kind(separator):
    commands = {item[2] & MASK for item in separator}
    if not commands:
        return "jump"
    if pe.COLOR_CHANGE in commands:
        return "color"
    if commands <= {pe.JUMP, pe.TRIM}:
        return "trim"
    return "other"


def _trim_threshold(runs, kinds, seps):
    """Distance à partir de laquelle le motif d'origine coupe le fil (3 mm par défaut).

    Un motif qui ne coupe pas (ou presque jamais) laisse la machine gérer les sauts : on n'ajoute
    alors aucune coupe, sinon l'optimisation dégraderait le fichier d'origine.
    """
    cut = [_distance(run[-1], nxt[0]) / UNITS_PER_MM
           for run, nxt, kind, separator in zip(runs, runs[1:], kinds, seps)
           if kind == "trim" and any(item[2] & MASK == pe.TRIM for item in separator)]
    if len(cut) < max(3.0, 0.1 * max(1, len(kinds))):
        return float("inf")
    return min(12.0, max(3.0, min(cut)))


def _parse_pattern(pattern):
    """Retourne ``(blocs, queue, seuil_de_coupe)`` ou un message d'erreur (chaîne)."""
    commands = [(stitch[2] & MASK) for stitch in pattern.stitches]
    if any(command not in SUPPORTED for command in commands):
        return ("Commandes non prises en charge (arrêt opérateur, appliqué, sequins...) : "
                "optimisation radicale ignorée.")
    items = [list(stitch) + [index] for index, stitch in enumerate(pattern.stitches)]
    head, runs, seps, tail = _parse(items)
    if not runs:
        return [], tail
    kinds = [_separator_kind(separator) for separator in seps]
    if "other" in kinds:
        return "Séparateurs inconnus dans le motif : optimisation radicale ignorée."
    changes = [sum(1 for item in group if item[2] & MASK == pe.COLOR_CHANGE)
               for group in [head] + seps + [tail]]
    if any(count > 1 for count in changes) or sum(changes[1:-1]) != kinds.count("color"):
        return ("Changements de fil consécutifs (arrêt opérateur ?) : optimisation radicale "
                "ignorée pour ne pas les perdre.")
    threshold = _trim_threshold(runs, kinds, seps)

    threads = pattern.threadlist
    thread = sum(1 for item in head if item[2] & MASK == pe.COLOR_CHANGE)
    blocks, current, thread_at = [], [_Run(runs[0])], [thread]
    for run, kind, separator in zip(runs[1:], kinds, seps):
        if kind == "color":
            thread += sum(1 for item in separator if item[2] & MASK == pe.COLOR_CHANGE)
            blocks.append(_Block(current, len(blocks), thread_at[-1]))
            current, thread_at = [_Run(run)], thread_at + [thread]
        else:
            current.append(_Run(run))
    blocks.append(_Block(current, len(blocks), thread_at[-1]))

    for block in blocks:
        thread_object = threads[block.thread] if block.thread < len(threads) else None
        block.color = None if thread_object is None else thread_object.color & 0xFFFFFF
        block.key = block.color if block.color is not None else ("bloc", block.index)
        block.label = block.color if block.color is not None else 0x1000000 + block.index
    # Une seule couleur pour plusieurs blocs : couleurs de remplissage du lecteur (DST, XXX...),
    # pas une information fiable. On ne regroupe alors rien et on garde l'ordre du fichier.
    if len(blocks) > 1 and len({block.color for block in blocks}) < 2:
        for block in blocks:
            block.key = ("bloc", block.index)
            block.label = 0x1000000 + block.index
            block.color = None
    return blocks, tail, threshold


def _reduce(points, prm, protected, counts):
    """Doublons, micro-points et points alignés : la passe la plus agressive du moteur."""
    cleaned = []
    for item in points:
        if cleaned and _distance(cleaned[-1], item) <= 0.5:
            counts["duplicates"] += 1
            continue
        cleaned.append(item)
    minimum = prm["min_stitch_mm"] * UNITS_PER_MM
    if minimum > 0 and len(cleaned) > 2:
        kept = [cleaned[0]]
        for item in cleaned[1:-1]:
            if item[3] in protected or _distance(kept[-1], item) >= minimum:
                kept.append(item)
            else:
                counts["micro"] += 1
        kept.append(cleaned[-1])
        cleaned = kept
    tolerance = prm["simplify_tol_mm"] * UNITS_PER_MM
    target = prm["stitch_mm"] * UNITS_PER_MM
    if tolerance > 0 and len(cleaned) > 2:
        # 1. tracé fidèle à ``tolerance`` près, sans limite de longueur de corde
        line = _simplify(cleaned, tolerance, target * 4, protected)
        counts["simplified"] += len(cleaned) - len(line)
        # 2. points de piquage replacés à intervalles réguliers le long de ce tracé —
        #    uniquement si le motif est plus fin que la cible : un motif déjà plus espacé
        #    (grandes satins, points de recouvrement) ne doit pas gagner de points.
        length = sum(_distance(a, b) for a, b in zip(cleaned, cleaned[1:]))
        average = length / max(1, len(cleaned) - 1)
        if average < target:
            respaced = _respace(line, target, protected)
            counts["simplified"] += len(line) - len(respaced)
            line = respaced
        cleaned = line
    return cleaned


def _respace(points, target, protected):
    """Replace les points de piquage à ``target`` le long du tracé, en conservant ses coins.

    Les points ajoutés sont interpolés sur le segment : le tracé piqué, lui, ne bouge pas ; seuls
    les points de piquage sont plus espacés (moins de piqûres pour la même couverture).
    """
    if len(points) < 2 or target <= 0:
        return points
    out = [points[0]]
    for first, second in zip(points, points[1:]):
        length = _distance(first, second)
        if length <= 0:
            continue
        if protected and second[3] in protected:
            out.append(second)
            continue
        position = 0.0
        while length - position >= target:
            ratio = (position + target) / length
            point = [round(first[0] + (second[0] - first[0]) * ratio),
                     round(first[1] + (second[1] - first[1]) * ratio), second[2], second[3]]
            if _distance(out[-1], point) > target * 0.3:
                out.append(point)
            position += target
        out.append(second)
    return out


def _score(stats):
    """Coût machine approché : une coupe vaut une cinquantaine de piqûres."""
    return stats["trims"] * 50 + stats["stitches"]


def _stats(items):
    stitches = sum(1 for item in items if item[2] & MASK == pe.STITCH)
    trims = sum(1 for item in items if item[2] & MASK == pe.TRIM)
    jumps = sum(1 for item in items if item[2] & MASK == pe.JUMP)
    colors = sum(1 for item in items if item[2] & MASK == pe.COLOR_CHANGE)
    return {
        "stitches": stitches, "trims": trims, "jumps": jumps, "color_changes": colors,
        "travel_mm": 0.0, "runs": 0, "blocks": colors + 1 if items else 0,
        # estimation indicative : 600 points/min, 5 s par coupe, 30 s par changement de fil
        "est_time_min": round(stitches / 600.0 + trims * 5 / 60.0 + colors * 0.5, 1),
    }


# ----------------------------------------------------------------------------------------
# Outils géométriques
# ----------------------------------------------------------------------------------------
def _solid(mask):
    """Comble les trous d'un pixel des rasters de couverture (rangées de remplissage jointives)."""
    if not mask.any():
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def _mask_of(grid, runs, width_mm):
    pixel = [grid.px(point) for run in runs for point in run.points]
    if not pixel:
        return (0, 0, 0, 0), np.zeros((0, 0), bool)
    pad = grid._width(width_mm) + 1
    left = max(0, int(min(p[0] for p in pixel)) - pad)
    top = max(0, int(min(p[1] for p in pixel)) - pad)
    right = min(grid.w, int(max(p[0] for p in pixel)) + pad + 1)
    bottom = min(grid.h, int(max(p[1] for p in pixel)) + pad + 1)
    image = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
    draw = ImageDraw.Draw(image)
    width = grid._width(width_mm)
    for run in runs:
        points = [grid.px(point) for point in run.points]
        for first, second in zip(points, points[1:]):
            draw.line([(first[0] - left, first[1] - top), (second[0] - left, second[1] - top)],
                      fill=255, width=width)
    return (left, top, right, bottom), np.asarray(image) > 0


def _overlap(first, second):
    (left_a, top_a, right_a, bottom_a), mask_a = first
    (left_b, top_b, right_b, bottom_b), mask_b = second
    left, top = max(left_a, left_b), max(top_a, top_b)
    right, bottom = min(right_a, right_b), min(bottom_a, bottom_b)
    if left >= right or top >= bottom:
        return False
    return bool(np.any(mask_a[top - top_a:bottom - top_a, left - left_a:right - left_a]
                       & mask_b[top - top_b:bottom - top_b, left - left_b:right - left_b]))


def _nearest_endpoint(position, runs):
    """Extrémité de passage la plus proche, et sa distance."""
    best, point = None, None
    for run in runs:
        for candidate in (run.points[0], run.points[-1]):
            distance = _distance(position, candidate)
            if best is None or distance < best:
                best, point = distance, candidate
    return point, 0.0 if best is None else best


def _exit_hint(runs, entry=None):
    """Position approximative en fin de bloc : extrémité la plus éloignée de l'entrée."""
    far, best = None, -1.0
    for run in runs:
        for point in (run.points[0], run.points[-1]):
            distance = 0.0 if entry is None else _distance(entry, point)
            if far is None or distance > best:
                far, best = point, distance
    return far


def _order_runs(runs, position, prm, threshold):
    """Ordre de piquage des passages d'un même fil : plus proche voisin puis 2-opt.

    Chaque passage peut être piqué dans les deux sens. Le coût n'est pas la distance mais le
    temps machine : une coupe coûte cher, un saut court presque rien, un raccord piqué quelques
    piqûres.
    """
    count = len(runs)
    if count < 2:
        return list(runs)
    starts = np.array([run.points[0][:2] for run in runs], float)
    ends = np.array([run.points[-1][:2] for run in runs], float)
    cost = _pair_costs(starts, ends, prm, threshold)

    order, current = [], None if position is None else np.array(position[:2], float)
    remaining = np.ones(count, bool)
    while remaining.any():
        indexes = np.nonzero(remaining)[0]
        if current is None:
            pick = int(indexes[0])
        else:
            distance = np.minimum(
                np.hypot(starts[indexes, 0] - current[0], starts[indexes, 1] - current[1]),
                np.hypot(ends[indexes, 0] - current[0], ends[indexes, 1] - current[1]))
            pick = int(indexes[int(np.argmin(distance))])
        order.append(pick)
        remaining[pick] = False
        current = _closest_end(current, starts[pick], ends[pick])

    if 2 < count <= MAX_TWO_OPT_RUNS:
        order = _two_opt(order, cost)

    out, current = [], position
    for index in order:
        run = runs[index]
        flipped = False
        if current is not None:
            forward = _distance(current, run.points[0])
            backward = _distance(current, run.points[-1])
            flipped = backward < forward
        chosen = run.flipped() if flipped else run
        out.append(chosen)
        current = chosen.points[-1]
    return out


def _pair_costs(starts, ends, prm, threshold):
    """Coût machine minimal entre deux passages, quelle que soit leur orientation."""
    def spread(first, second):
        return np.hypot(first[:, None, 0] - second[None, :, 0],
                        first[:, None, 1] - second[None, :, 1])
    distance = np.minimum(np.minimum(spread(starts, starts), spread(starts, ends)),
                          np.minimum(spread(ends, starts), spread(ends, ends)))
    return _costs(distance, prm, threshold)


def _costs(distance, prm, threshold):
    """Distance (unités) -> coût machine en secondes."""
    length = distance / UNITS_PER_MM
    return np.where(length <= threshold, prm["jump_cost_s"] + 0.01 * length,
                    prm["trim_cost_s"] + prm["jump_cost_s"] + 0.01 * length)


def _closest_end(current, start, end):
    if current is None:
        return start
    return start if _distance(current, start) <= _distance(current, end) else end


def _two_opt(order, costs):
    """Amélioration 2-opt : inverse un tronçon quand les deux raccords raccourcissent."""
    order = list(order)
    count = len(order)
    for _ in range(4):
        improved = False
        for i in range(1, count - 1):
            previous, first = order[i - 1], order[i]
            rest = np.array(order[i + 1:count - 1], int)
            following = np.array(order[i + 2:], int)
            if not rest.size:
                break
            delta = (costs[previous, rest] + costs[first, following]
                     - costs[previous, first] - costs[rest, following])
            pick = int(np.argmin(delta))
            if delta[pick] < -1e-6:
                end = i + 1 + pick
                order[i:end + 1] = order[i:end + 1][::-1]
                improved = True
        if not improved:
            break
    return order


def _travel(target, start, piece_units):
    """Points intermédiaires d'un déplacement piqué de ``start`` vers ``target``."""
    distance = _distance(start, target)
    pieces = max(1, int(distance / piece_units) + 1)
    return [[round(start[0] + (target[0] - start[0]) * step / pieces),
             round(start[1] + (target[1] - start[1]) * step / pieces),
             pe.STITCH, target[3]] for step in range(1, pieces)]


def _longest_gap(flags):
    """Plus longue suite de False dans un tableau booléen."""
    if flags.all():
        return 0
    missing = np.flatnonzero(~flags)
    breaks = np.flatnonzero(np.diff(missing) > 1)
    sizes = np.diff(np.concatenate(([0], breaks + 1, [missing.size])))
    return int(sizes.max()) if sizes.size else 0


def _hidden_indices(points, safe, tolerance):
    """Points à conserver dans un passage dont certains points sont recouverts.

    Un point recouvert peut partir ; le tracé le remplaçant doit rester à ``tolerance`` du
    tracé d'origine, sinon le point est un coin à conserver. Les extrémités recouvertes sont
    retirées : le passage commence et finit alors sous la couche qui le cache.
    """
    high = len(points) - 1
    low = 0
    while low + 2 < high and safe[low]:
        low += 1
    while high - 2 > low and safe[high]:
        high -= 1
    kept = [low]
    for index in range(low + 1, high):
        if not safe[index] or _segment_distance(points[index], points[kept[-1]],
                                                points[index + 1]) > tolerance:
            kept.append(index)
    if high != kept[-1]:
        kept.append(high)
    return kept


def _visible_preserved(grid, before, after, covered):
    """Vérifie que les points retirés étaient bien invisibles (couverture visible identique)."""
    pixel = [grid.px(point) for point in before] + [grid.px(point) for point in after]
    pad = grid._width(THREAD_MM) + 1
    left = max(0, int(min(p[0] for p in pixel)) - pad)
    top = max(0, int(min(p[1] for p in pixel)) - pad)
    right = min(grid.w, int(max(p[0] for p in pixel)) + pad + 1)
    bottom = min(grid.h, int(max(p[1] for p in pixel)) + pad + 1)
    if right <= left or bottom <= top:
        return True
    hidden = covered[top:bottom, left:right]
    width = grid._width(THREAD_MM)
    masks = []
    for points in (before, after):
        image = Image.new("L", (right - left, bottom - top), 0)
        draw = ImageDraw.Draw(image)
        for first, second in zip(points, points[1:]):
            draw.line([(grid.px(first)[0] - left, grid.px(first)[1] - top),
                       (grid.px(second)[0] - left, grid.px(second)[1] - top)],
                      fill=255, width=width)
        masks.append((np.asarray(image) > 0) & ~hidden)
    return _iou(masks[0], masks[1]) >= MIN_VISIBLE_IOU


def _bisect(values, target):
    """Index du plus grand élément <= target (recherche binaire)."""
    if not values or target < values[0]:
        return -1
    low, high = 0, len(values) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if values[middle] <= target:
            low = middle
        else:
            high = middle - 1
    return low
