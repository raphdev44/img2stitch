"""Stockage des motifs sur disque et génération en tâche de fond."""
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyembroidery as pe
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from img2stitch import (Settings, color_palettes, estimate_colors, image_to_pattern, natural_size_mm,
                        render_photo_preview, render_raw_preview)
from img2stitch.algorithm_standard import (SIZE_MARGIN_MM, _region_look, _zone_override, pattern_stats,
                                           pattern_to_image, thread_lengths_mm)
from img2stitch.cross_stitch import MAX_COLORS as CROSS_STITCH_MAX_COLORS, cross_stitch_palettes, cross_stitch_pattern
from img2stitch.fill import FILL_TYPES
from img2stitch.hand import CREATIVITY as HAND_CREATIVITY
from img2stitch.hand import STYLES as HAND_STYLES, hand_pattern
from img2stitch.optimize import optimize_pattern
from img2stitch.radical import optimize_pattern_radical
from img2stitch.algorithm_vectorize import VectorizerUnavailable
from img2stitch.relief import LEGACY as RELIEF_LEGACY
from img2stitch.smart import thin_band_regions
from img2stitch.photo import FABRICS, stitch_bbox_mm

log = logging.getLogger(__name__)

DEFAULT_FABRIC = "creme"

# Algorithmes de numérisation proposés à la création : libellé affiché dans l'interface
SCAN_ENGINES = {
    "standard": "Standard",
    "vector": "Par vectorisation (vpype)",
    "industrial": "Industriel",
}

# Optimisations proposées sur un motif prêt : libellé affiché dans l'interface
OPTIMIZATIONS = {
    "simple": "Optimisation simple",
    "radical": "Optimisation radicale",
}

FORMATS = {
    "pes": "Brother · Babylock",
    "dst": "Tajima",
    "jef": "Janome",
    "exp": "Melco",
    "vp3": "Pfaff · Husqvarna",
    "xxx": "Singer",
    "u01": "Barudan",
    "pec": "Brother (PEC)",
}
SOURCE, RAW, PHOTO, THUMB, META = "source.png", "preview_raw.png", "preview_photo.png", "thumb.jpg", "meta.json"
PATTERN_DATA = "pattern.json"  # points et couleurs exacts, pour refaire un aperçu sans régénérer le motif
ZONES = "zones.png"  # carte des zones (index du bloc de couleur + 1), même cadrage que l'aperçu réaliste
PUBLIC_FILES = {SOURCE, RAW, PHOTO, THUMB, ZONES}
MIN_COLORS, MAX_COLORS = 1, 12
MAX_NAME = 80
MIN_SIZE_MM, MAX_SIZE_MM, DEFAULT_SIZE_MM = 10, 400, 100
# Les aperçus (réaliste en 22 px/mm) pèsent plusieurs Mo : sous cette marge on ne tente pas de les
# écrire, un disque plein ferait échouer la génération après les fichiers de broderie.
MIN_FREE_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 40_000_000
ACCEPTED_FORMATS = {"PNG", "JPEG", "WEBP", "AVIF"}
# mode web -> (Settings.background, Settings.background_scope)
BACKGROUND_MODES = {
    "outside": ("auto", "outside"),
    "all": ("auto", "all"),
    "keep": ("none", "all"),
}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class InvalidImage(ValueError):
    pass


class DesignStore:
    def __init__(self, root, workers=2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="img2stitch")
        self._lock = threading.Lock()
        self._modify_lock = threading.Lock()  # lecture-modification-écriture de meta.json hors génération initiale
        self._recover_interrupted()

    # --- Accès ---
    def dir(self, design_id):
        if not isinstance(design_id, str) or not _ID_RE.match(design_id):
            return None
        d = self.root / design_id
        return d if (d / META).is_file() else None

    def get(self, design_id):
        d = self.dir(design_id)
        if d is None:
            return None
        meta = self._read(d)
        meta.setdefault("background", "all" if meta.pop("remove_background", True) else "keep")
        meta.setdefault("fabric", DEFAULT_FABRIC)
        meta.setdefault("photo_version", 0)
        meta["has_raw"] = (d / RAW).is_file()
        meta["has_photo"] = (d / PHOTO).is_file()
        meta["has_thumb"] = (d / THUMB).is_file()
        return meta

    def thread_lengths(self, design_id):
        """Longueur de fil estimée (m) par fil, dans l'ordre de d.threads.

        Les motifs générés avant ce calcul n'ont pas la valeur en mémoire : on la déduit de
        pattern.json, sans relancer la conversion.
        """
        meta = self.get(design_id)
        if meta is None:
            return []
        cached = meta.get("thread_lengths")
        if cached is not None:
            return cached
        d = self.dir(design_id)
        if meta.get("status") != "ready" or not (d / PATTERN_DATA).is_file():
            return []
        return [round(v / 1000, 2) for v in thread_lengths_mm(_load_pattern(d))]

    def list(self):
        items = [self.get(d.name) for d in self.root.iterdir() if d.is_dir()]
        return sorted((m for m in items if m), key=lambda m: m["created_at"], reverse=True)

    def file_path(self, design_id, name):
        d = self.dir(design_id)
        if d is None or name not in PUBLIC_FILES or not (d / name).is_file():
            return None
        return d / name

    def pattern_path(self, design_id, fmt):
        d = self.dir(design_id)
        if d is None or fmt not in FORMATS or not (d / f"pattern.{fmt}").is_file():
            return None
        return d / f"pattern.{fmt}"

    def delete(self, design_id):
        with self._modify_lock:
            d = self.dir(design_id)
            if d is None:
                return False
            meta = self._read(d)
            if meta.get("status") == "processing" or meta.get("photo_rendering"):
                raise InvalidImage("Impossible de supprimer un motif pendant son traitement.")
            shutil.rmtree(d)
            return True

    # --- Nom du motif ---
    def rename(self, design_id, name):
        d = self.dir(design_id)
        with self._modify_lock:
            meta = self._read(d)
            if meta.get("status") == "processing":  # la génération réécrit meta.json en entier
                raise InvalidImage("Le motif est en cours de génération, réessayez dans un instant.")
            meta["name"] = name
            self._write(d, meta)

    # --- Fusion des fils de même couleur ---
    def recolor(self, design_id, old, new):
        """Change la couleur d'un fil dans toute la broderie : toutes les zones de ce fil prennent `new`."""
        if new == old:
            return
        if self.get(design_id).get("mode") == "hand":
            raise InvalidImage("Une broderie à la main ne se modifie pas : recréez un motif depuis l'image.")
        if self.get(design_id).get("mode") == "cross_stitch":
            self._change_cross_stitch_palette(design_id, recolor=(old, new))
            return
        self.apply_edits(design_id, [{"color": new if z["color"] == old else z["color"], "fill": z["fill"],
                                      "outline": z["outline"]} for z in self.zones(design_id)])

    def set_cross_stitch_colors(self, design_id, n_colors):
        if n_colors is not None and (isinstance(n_colors, bool) or not isinstance(n_colors, int)
                                     or not 1 <= n_colors <= CROSS_STITCH_MAX_COLORS):
            raise InvalidImage("Le nombre de couleurs doit être compris entre 1 et 256.")
        self._change_cross_stitch_palette(design_id, n_colors=n_colors)

    def _change_cross_stitch_palette(self, design_id, *, n_colors=None, recolor=None):
        d = self.dir(design_id)
        with self._modify_lock:
            meta = self._read(d)
            if meta.get("mode") != "cross_stitch":
                raise InvalidImage("Ce réglage est réservé aux motifs en point de croix.")
            if meta.get("status") != "ready" or meta.get("photo_rendering"):
                raise InvalidImage("Le motif est en cours de traitement, réessayez dans un instant.")
            previous = {key: meta.get(key) for key in ("cross_stitch_color_limit", "cross_stitch_colors")}
            if recolor is not None:
                old, new = recolor
                colors = dict(meta.get("cross_stitch_colors") or {})
                sources = meta.get("cross_stitch_source_palette") or [z["source"] for z in meta.get("zones") or []]
                for source in sources:
                    if colors.get(source, source) == old:
                        colors[source] = new
                meta["cross_stitch_colors"] = colors
            else:
                meta.update(cross_stitch_color_limit=n_colors, cross_stitch_colors={})
            meta.update(cross_stitch_previous=previous, editing=True, zones_previous=meta.get("zones"),
                        regions_previous=meta.get("regions"), status="processing", error=None, edit_error=None,
                        step="Mise à jour des couleurs du point de croix", progress=5)
            self._write(d, meta)
        self._pool.submit(self._generate, d)

    def merge_colors(self, design_id, enabled=True):
        d = self.dir(design_id)
        with self._modify_lock:
            meta = self._read(d)
            if meta.get("mode") == "cross_stitch":
                raise InvalidImage("Les fils identiques du point de croix sont déjà regroupés automatiquement.")
            if meta.get("mode") == "hand":
                raise InvalidImage("Les fils d'une broderie à la main ne sont pas regroupés.")
            if meta.get("status") != "ready" or meta.get("photo_rendering"):
                raise InvalidImage("Le motif est en cours de traitement, réessayez dans un instant.")
            # "editing" : en cas d'échec le motif reste utilisable, avec ses zones d'avant
            meta.update(merge_colors=enabled, editing=True, zones_previous=meta.get("zones"),
                        regions_previous=meta.get("regions"), status="processing",
                        step="Fusion des fils de même couleur", progress=5, error=None, edit_error=None)
            self._write(d, meta)
        self._pool.submit(self._generate, d)

    def optimize(self, design_id, mode="simple"):
        """Optimise une **copie** du motif ; l'original reste intact.

        Le dossier du motif est dupliqué (nouvel identifiant, nouveau nom « … - optimisé » ou
        « … - optimisé+ ») puis la copie est optimisée en tâche de fond. Retourne l'identifiant
        du nouveau motif.
        """
        if mode not in OPTIMIZATIONS:
            raise InvalidImage("Optimisation inconnue : choisissez l'optimisation simple ou radicale.")
        source = self.dir(design_id)
        if source is None:
            raise InvalidImage("Motif introuvable.")
        with self._modify_lock:
            meta = self._read(source)
            if meta.get("mode") in ("cross_stitch", "hand"):
                raise InvalidImage("L'optimisation est réservée aux motifs de broderie classiques.")
            if meta.get("status") != "ready" or meta.get("photo_rendering"):
                raise InvalidImage("Le motif est en cours de traitement, réessayez dans un instant.")
            copy_id = uuid.uuid4().hex
            target = self.root / copy_id
            try:
                shutil.copytree(source, target)
            except OSError:
                log.exception("Copie du motif %s impossible", design_id)
                shutil.rmtree(target, ignore_errors=True)
                raise InvalidImage("La copie du motif a échoué, réessayez dans un instant.")
            copy_meta = dict(meta)
            copy_meta.update(
                id=copy_id,
                name=_optimized_name(meta.get("name"), mode),
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="processing",
                step="Réorganisation du parcours" if mode == "radical"
                else "Nettoyage des points et déplacements",
                progress=5,
                editing=True,           # en cas d'échec la copie reste utilisable telle quelle
                zones_previous=meta.get("zones"),
                regions_previous=meta.get("regions"),
                optimization_mode=mode,
                optimization_report=None,
                optimized_from=design_id,
                optimized_from_name=meta.get("name"),
                photo_rendering=False,
                photo_request=0,
                photo_error=None,
                error=None,
                edit_error=None,
            )
            self._write(target, copy_meta)
        self._pool.submit(self._optimize, target)
        return copy_id

    def _optimize(self, d):
        meta = self._read(d)
        mode = meta.get("optimization_mode", "simple")
        staged = []
        try:
            pattern = _load_pattern(d)
            if mode == "radical":
                try:
                    pattern, report = optimize_pattern_radical(pattern)
                except Exception:
                    log.exception("Optimisation radicale impossible pour %s, repli sur l'optimisation "
                                  "simple", d.name)
                    pattern, report = optimize_pattern(pattern)
                    report["warnings"].append(
                        "Optimisation radicale indisponible sur ce motif : optimisation simple appliquée.")
            else:
                pattern, report = optimize_pattern(pattern)
            self._update(d, meta, step="Préparation des fichiers machine", progress=35)
            stats = pattern_stats(pattern)
            x0, y0, x1, y1 = stitch_bbox_mm(pattern)
            zones_path = d / ".optimized.tmp.png"
            photo_path = d / ".optimized-photo.tmp.png"
            raw_path = d / ".optimized-raw.tmp.png"
            thumb_path = d / ".optimized-thumb.tmp.jpg"
            data_path = d / ".optimized-data.tmp.json"
            staged.extend((zones_path, photo_path, raw_path, thumb_path, data_path))
            _save_pattern(pattern, str(data_path))
            _render_zone_map(pattern, str(zones_path))
            render_raw_preview(pattern, str(raw_path))
            photo_error = None
            if _has_room_for_preview(d):
                try:
                    render_photo_preview(pattern, str(photo_path),
                                         fabric=meta.get("fabric", DEFAULT_FABRIC))
                    _make_thumb_from(photo_path, thumb_path)
                except OSError:
                    log.warning("Rendu réaliste impossible pour %s", d.name, exc_info=True)
                    photo_error = _write_error(d)
            else:
                log.warning("Aperçu réaliste ignoré pour %s : espace disque insuffisant", d.name)
                photo_error = _write_error(d)
            for index, fmt in enumerate(FORMATS, 1):
                path = d / f".optimized.tmp.{fmt}"
                staged.append(path)
                pe.write(pattern, str(path), {"tie_on": True, "tie_off": True})
                self._update(d, meta, progress=35 + round(40 * index / len(FORMATS)))

            self._update(d, meta, step="Mise à jour des aperçus", progress=82)
            replacements = [
                (data_path, d / PATTERN_DATA),
                (zones_path, d / ZONES),
                (raw_path, d / RAW),
            ] + [(d / f".optimized.tmp.{fmt}", d / f"pattern.{fmt}") for fmt in FORMATS]
            if photo_error is None:     # sinon l'aperçu précédent est conservé tel quel
                replacements += [(photo_path, d / PHOTO), (thumb_path, d / THUMB)]
            for source, target in replacements:
                os.replace(source, target)

            report["mode"] = mode
            report["after"]["colors"] = len(pattern.threadlist)
            report["after"]["stats"] = stats
            report["before"]["stats"] = meta.get("stats")
            self._update(
                d, meta,
                status="ready",
                step=None,
                progress=100,
                editing=False,
                zones_previous=None,
                regions_previous=None,
                stats=stats,
                threads=["#%06x" % (t.color & 0xFFFFFF) for t in pattern.threadlist],
                thread_lengths=[round(v / 1000, 2) for v in thread_lengths_mm(pattern)],
                size_mm=[round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
                photo_version=meta.get("photo_version", 0) + 1,
                optimization_report=report,
                optimization_mode=mode,
                photo_error=photo_error,
                edit_error=None,
            )
        except OSError:
            log.exception("Échec d'écriture pendant l'optimisation de %s", d.name)
            self._fail(d, meta, _write_error(d))
        except Exception:
            log.exception("Échec de l'optimisation pour %s", d.name)
            self._fail(d, meta, "L'optimisation du motif a échoué.")
        finally:
            for path in staged:
                path.unlink(missing_ok=True)

    # --- Édition des zones ---
    def zones(self, design_id):
        """Blocs de couleur du motif, dans l'ordre de piqué :
        [{"source", "color", "fill", "outline", "region"}, ...].
        "outline" : point de contour activé pour la zone ; "region" est l'index de la retouche de
        zone fermée (meta["regions"]) qui a créé le bloc, sinon None."""
        meta = self.get(design_id)
        if meta.get("zones"):
            return [dict(z, region=z.get("region"), outline=z.get("outline", True)) for z in meta["zones"]]
        return [{"source": c, "color": c, "fill": "tatami", "outline": True, "region": None}
                for c in meta.get("threads") or []]

    def zone_map_path(self, design_id):
        d = self.dir(design_id)
        if d is None:
            return None
        if not (d / ZONES).is_file():  # motifs créés avant l'éditeur
            _write_atomic(d / ZONES, lambda p: _render_zone_map(_load_pattern(d), p))
        return d / ZONES

    def apply_edits(self, design_id, edits, new_regions=()):
        """Enregistre la couleur et le motif de chaque bloc, ajoute les retouches de zones fermées
        (point cliqué en fraction de l'aperçu), puis régénère le motif."""
        d = self.dir(design_id)
        current = self.zones(design_id)
        if len(edits) != len(current):
            raise InvalidImage("Les zones ne correspondent plus au motif : rechargez la page.")
        meta = self.get(design_id)
        if meta.get("mode") == "cross_stitch":
            raise InvalidImage("Un pixel art en point de croix ne peut pas être modifié dans cet éditeur.")
        if meta.get("mode") == "hand":
            raise InvalidImage("Une broderie à la main ne se modifie pas : recréez un motif depuis l'image.")
        regions = [dict(r, outline=r.get("outline", True)) for r in meta.get("regions") or []]
        zones = []
        for z, e in zip(current, edits):
            outline = e.get("outline", z.get("outline", True))
            if z["region"] is not None and z["region"] < len(regions):
                regions[z["region"]].update(color=e["color"], fill=e["fill"], outline=outline)
            zones.append({"source": z["source"], "color": e["color"], "fill": e["fill"],
                          "outline": outline, "region": z["region"]})
        new_regions = [r for r in new_regions if current[r["block"]]["region"] is None]
        if new_regions:
            frame, (x0, y0, x1, y1) = self._frame_and_bbox(d, meta)
            for r in new_regions:
                # même cadrage que l'aperçu réaliste et la carte des zones : boîte des points + 6 mm de marge
                x = (x0 - 6 + r["fx"] * (x1 - x0 + 12)) * 10
                y = (y0 - 6 + r["fy"] * (y1 - y0 + 12)) * 10
                u, v = pattern_to_image(frame, x, y)
                regions.append({"source": current[r["block"]]["source"], "u": float(u), "v": float(v),
                                "color": r["color"], "fill": r["fill"], "outline": r.get("outline", True)})
        # Une zone revenue aux réglages de sa couleur n'a plus besoin d'être isolée
        whole = {z["source"]: z for z in zones if z["region"] is None}
        regions = [r for r in regions if not (r["source"] in whole and whole[r["source"]]["color"] == r["color"]
                                              and whole[r["source"]]["fill"] == r["fill"]
                                              and whole[r["source"]].get("outline", True) == r.get("outline", True))]
        with self._modify_lock:
            meta = self._read(d)
            if meta.get("status") != "ready" or meta.get("photo_rendering"):
                raise InvalidImage("Le motif est en cours de traitement, réessayez dans un instant.")
            meta.update(zones_previous=meta.get("zones"), regions_previous=meta.get("regions"),
                        zones=[z for z in zones if z["region"] is None], regions=regions, editing=True,
                        status="processing", step="Application des modifications", progress=5,
                        error=None, edit_error=None)
            self._write(d, meta)
        self._pool.submit(self._generate, d)

    def _frame_and_bbox(self, d, meta):
        """Repère motif -> image et boîte des points (mm) du motif affiché."""
        if meta.get("frame"):
            return meta["frame"], stitch_bbox_mm(_load_pattern(d))
        # Motif antérieur à l'enregistrement du repère : on le recalcule (même image, mêmes réglages)
        pattern = image_to_pattern(str(d / SOURCE), _settings(meta))
        return pattern.img2stitch_frame, stitch_bbox_mm(pattern)

    # --- Création ---
    def create(self, stream, filename, n_colors, background, target_mm, excluded=(), thicken_lines=False,
               *, smart_fill=False, mode="classic", engine="standard", hand_style="auto", hand_creativity=3,
               cross_stitch_double=False):
        if engine not in SCAN_ENGINES:
            raise InvalidImage("Algorithme de numérisation inconnu.")
        img = _load_image(stream)
        design_id = uuid.uuid4().hex
        d = self.root / design_id
        d.mkdir()
        img.save(d / SOURCE, "PNG")
        stem = secure_filename(Path(filename or "").stem)[:60] or "motif"
        meta = {
            "id": design_id,
            "name": stem,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "source_format": img.info.get("source_format"),
            "n_colors": n_colors,
            "background": background,
            "target_mm": list(target_mm) if target_mm else None,
            "excluded": list(excluded),
            "thicken_lines": bool(thicken_lines),
            "smart_fill": bool(smart_fill),
            "engine": engine,
            "hand_style": hand_style if mode == "hand" else None,
            "hand_creativity": int(hand_creativity) if mode == "hand" else None,
            "fabric": DEFAULT_FABRIC,
            "status": "processing",
            "step": "Analyse de l'image",
            "progress": 5,
            "error": None,
        }
        if smart_fill and mode == "classic":
            meta["regions"] = _smart_regions(d, n_colors, background, target_mm, excluded, thicken_lines)
        if mode == "cross_stitch":
            meta.update(cross_stitch_color_limit=n_colors or None, cross_stitch_colors={},
                        cross_stitch_double=bool(cross_stitch_double))
        self._write(d, meta)
        self._pool.submit(self._generate, d)
        return design_id

    def _generate(self, d):
        meta = self._read(d)
        try:
            cross_stitch = meta.get("mode") == "cross_stitch"
            hand = meta.get("mode") == "hand"
            s = Settings() if (cross_stitch or hand) else _settings(meta)
            if cross_stitch:
                try:
                    pattern = cross_stitch_pattern(d / SOURCE, meta["background"],
                                                   source_format=meta.get("source_format"),
                                                   n_colors=meta.get("cross_stitch_color_limit"),
                                                   exclude=_exclude_setting(meta.get("excluded", [])),
                                                   double=bool(meta.get("cross_stitch_double")),
                                                   color_overrides=[(_rgb(src), _rgb(dst)) for src, dst in
                                                                    (meta.get("cross_stitch_colors") or {}).items()])
                except ValueError as exc:
                    raise InvalidImage(str(exc)) from exc
            elif hand:
                pattern = hand_pattern(d / SOURCE, meta["background"], n_colors=meta.get("n_colors") or None,
                                       exclude=_exclude_setting(meta.get("excluded", [])),
                                       style=meta.get("hand_style") or "auto",
                                       creativity=meta.get("hand_creativity") or 3,
                                       target_mm=meta.get("target_mm") or None)
            else:
                pattern = image_to_pattern(str(d / SOURCE), s)
            self._update(d, meta, step="Préparation des points", progress=35)
            stats = pattern_stats(pattern)
            if stats["stitches"] == 0:
                off = [z for z in meta.get("zones") or [] if z["fill"] == "none" and not z.get("outline", True)]
                if off and len(off) == len(meta["zones"]):
                    raise InvalidImage("Toutes les zones sont éteintes : rallumez au moins un remplissage "
                                       "ou un point de contour.")
                raise InvalidImage("Aucun point à broder : l'image semble vide une fois le fond supprimé.")

            self._update(d, meta, step="Écriture des fichiers de broderie", progress=50)
            opts = {"tie_on": True, "tie_off": True} if s.lock_stitches else {}
            for index, fmt in enumerate(FORMATS, 1):
                pe.write(pattern, str(d / f"pattern.{fmt}"), opts)
                self._update(d, meta, progress=50 + round(25 * index / len(FORMATS)))
            _write_atomic(d / PATTERN_DATA, lambda p: _save_pattern(pattern, p))
            _write_atomic(d / ZONES, lambda p: _render_zone_map(pattern, p))

            x0, y0, x1, y1 = stitch_bbox_mm(pattern)
            _write_atomic(d / RAW, lambda p: render_raw_preview(pattern, p))
            if hand:
                # Une broderie à la main n'a pas de zones : chaque forme a reçu son point à la création.
                zones = [{"source": _hex(src), "color": _hex(src), "fill": "hand", "outline": False,
                          "region": None} for src, _ in getattr(pattern, "img2stitch_zones", [])]
                regions = []
                meta.update(n_colors=len(pattern.threadlist),
                            hand_mix=getattr(pattern, "img2stitch_hand", None))
            else:
                zones, regions = _zones_of(pattern, s, meta.get("regions") or [])
                if cross_stitch:
                    for zone in zones:
                        zone["fill"] = "cross_stitch"
                    meta.update(n_colors=len(pattern.threadlist), grid=pattern.img2stitch_grid,
                                cross_stitch_source_palette=[_hex(rgb) for rgb in pattern.img2stitch_source_palette],
                                cross_stitch_previous=None)
            frame = {k: float(v) for k, v in pattern.img2stitch_frame.items()}
            self._update(
                d, meta,
                step="Rendu réaliste",
                progress=78,
                stats=stats,
                threads=["#%06x" % (t.color & 0xFFFFFF) for t in pattern.threadlist],
                thread_lengths=[round(v / 1000, 2) for v in thread_lengths_mm(pattern)],
                zones=zones,
                regions=regions,
                frame=frame,
                zones_previous=zones,  # fichiers de broderie écrits : la retouche est acquise
                regions_previous=regions,
                size_mm=[round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
                optimization_report=None if meta.get("editing") else meta.get("optimization_report"),
            )

            fabric = meta.get("fabric", DEFAULT_FABRIC)
            photo_error = None
            if _has_room_for_preview(d):
                try:
                    _write_atomic(d / PHOTO, lambda p: render_photo_preview(pattern, p, fabric=fabric))
                    _make_thumb(d)
                except OSError:
                    # L'aperçu réaliste est décoratif : sans lui le motif reste brodable et téléchargeable.
                    log.warning("Rendu réaliste impossible pour %s", d.name, exc_info=True)
                    photo_error = _write_error(d)
            else:
                log.warning("Aperçu réaliste ignoré pour %s : espace disque insuffisant", d.name)
                photo_error = _write_error(d)
            self._update(d, meta, status="ready", step=None, editing=False, zones_previous=None,
                         regions_previous=None, progress=100, photo_error=photo_error,
                         error=None,  # une génération qui aboutit efface l'échec précédent
                         photo_version=meta.get("photo_version", 0) + 1)
        except InvalidImage as e:
            self._fail(d, meta, str(e))
        except VectorizerUnavailable as e:
            self._fail(d, meta, str(e))
        except OSError:
            log.exception("Échec d'écriture pour %s", d.name)
            self._fail(d, meta, _write_error(d))
        except Exception:
            log.exception("Échec de génération pour %s", d.name)
            self._fail(d, meta, "La génération du motif a échoué.")

    def _fail(self, d, meta, message):
        if meta.get("editing"):  # une retouche qui échoue ne doit pas rendre le motif inutilisable
            meta.update(meta.pop("cross_stitch_previous", None) or {})
            self._update(d, meta, status="ready", step=None, editing=False, zones=meta.get("zones_previous"),
                         regions=meta.get("regions_previous"), zones_previous=None, regions_previous=None,
                         progress=None, edit_error=message)
        else:
            self._update(d, meta, status="error", step=None, progress=None, error=message)

    # --- Tissu de l'aperçu réaliste ---
    def set_fabric(self, design_id, fabric):
        d = self.dir(design_id)
        with self._modify_lock:
            meta = self._read(d)
            if meta.get("status") != "ready":
                raise InvalidImage("Le motif n'est pas encore prêt.")
            req = meta.get("photo_request", 0) + 1
            meta.update(fabric=fabric, photo_request=req, photo_rendering=True, photo_progress=10,
                        photo_error=None)
            self._write(d, meta)
        self._pool.submit(self._render_photo, d, fabric, req)

    def _render_photo(self, d, fabric, req):
        tmp = d / f"preview_photo.{req}.tmp.png"
        try:
            render_photo_preview(_load_pattern(d), str(tmp), fabric=fabric)
            with self._modify_lock:
                meta = self._read(d)
                if meta.get("photo_request") != req:  # un autre tissu a été choisi entre-temps
                    return
                meta["photo_progress"] = 75
                self._write(d, meta)
                os.replace(tmp, d / PHOTO)
                _make_thumb(d)
                meta.update(photo_rendering=False, photo_progress=100,
                            photo_version=meta.get("photo_version", 0) + 1)
                self._write(d, meta)
        except Exception as exc:
            log.exception("Échec du rendu réaliste pour %s", d.name)
            message = _write_error(d) if isinstance(exc, OSError) else "Le nouveau rendu a échoué."
            with self._modify_lock:
                meta = self._read(d)
                if meta.get("photo_request") == req:
                    meta.update(photo_rendering=False, photo_progress=0, photo_error=message)
                    self._write(d, meta)
        finally:
            tmp.unlink(missing_ok=True)

    # --- Persistance ---
    def _recover_interrupted(self):
        for d in self.root.iterdir():
            if (d / META).is_file():
                meta = self._read(d)
                if meta.get("status") == "processing" and meta.get("editing"):
                    self._fail(d, meta, "Modification interrompue (redémarrage du serveur).")
                elif meta.get("status") == "processing":
                    self._update(d, meta, status="error", step=None, error="Génération interrompue (redémarrage du serveur).")
                elif meta.get("photo_rendering"):
                    self._update(d, meta, photo_rendering=False)

    def _update(self, d, meta, **changes):
        meta.update(changes)
        self._write(d, meta)

    def _read(self, d):
        with self._lock, open(d / META, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, d, meta):
        tmp = d / (META + ".tmp")
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            os.replace(tmp, d / META)


_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _optimized_name(name, mode):
    """Nom de la copie optimisée : suffixe explicite, longueur maximale respectée."""
    suffix = " - optimisé+" if mode == "radical" else " - optimisé"
    base = " ".join(str(name or "Motif").split())[:MAX_NAME - len(suffix)].strip()
    return (base or "Motif") + suffix


def clean_name(raw):
    """Nom saisi par l'utilisateur : espaces normalisés, caractères de contrôle retirés, 1 à 80 caractères."""
    name = " ".join("".join(ch for ch in str(raw or "") if unicodedata.category(ch)[0] != "C").split())
    if not name or len(name) > MAX_NAME:
        raise InvalidImage(f"Le nom doit contenir entre 1 et {MAX_NAME} caractères.")
    return name


def export_filename(name, ext):
    """Nom de fichier d'export sûr pour toutes les brodeuses : ASCII (accents retirés), lettres, chiffres, - et _."""
    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_).strip("_-")[:60].rstrip("_-") or "motif"
    return f"{stem}.{ext}"


def parse_exclusions(raw):
    """JSON envoyé par le formulaire -> [{"color": "#rrggbb", "others": ["#rrggbb", ...]}, ...] validé."""
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        raise InvalidImage("Couleurs supprimées invalides.")
    ok = (isinstance(items, list) and len(items) <= MAX_COLORS and all(
        isinstance(it, dict) and isinstance(it.get("color"), str) and _HEX_RE.match(it["color"])
        and isinstance(it.get("others"), list) and len(it["others"]) <= MAX_COLORS
        and all(isinstance(o, str) and _HEX_RE.match(o) for o in it["others"]) for it in items))
    if not ok:
        raise InvalidImage("Couleurs supprimées invalides.")
    return [{"color": it["color"].lower(), "others": [o.lower() for o in it["others"]]} for it in items]


def _exclude_setting(excluded):
    return tuple((_rgb(e["color"]), tuple(_rgb(o) for o in e["others"])) for e in excluded)


def analyse_cross_stitch_image(stream, background, excluded=()):
    image = _load_image(stream)
    try:
        analysis = cross_stitch_palettes(image, background, source_format=image.info.get("source_format"),
                                         max_colors=MAX_COLORS, exclude=_exclude_setting(excluded))
    except ValueError as exc:
        raise InvalidImage(str(exc)) from exc
    return {"colors": analysis["colors"],
            "palettes": {count: [_hex(color) for color in colors] for count, colors in analysis["palettes"].items()}}


def analyse_image(stream, background, excluded=()):
    """Valeurs conseillées pour l'image envoyée : nombre de couleurs et taille optimale calculée par le moteur."""
    bg, scope = BACKGROUND_MODES[background]
    ex = _exclude_setting(excluded)
    img = _load_image(stream)
    colors = max(MIN_COLORS, estimate_colors(img, background=bg, bg_scope=scope, max_colors=MAX_COLORS, exclude=ex))
    w, h = natural_size_mm(img, Settings(n_colors=colors, background=bg, background_scope=scope, exclude_colors=ex))
    if not w or not h:
        w = h = DEFAULT_SIZE_MM
    # Arrondi au 1/10 mm supérieur + marge : garder ces valeurs redonne exactement cette taille
    fit = lambda v: min(MAX_SIZE_MM, max(MIN_SIZE_MM, math.ceil((v + SIZE_MARGIN_MM) * 10) / 10))
    palettes = {n: ["#%02x%02x%02x" % c for c in pal]
                for n, pal in color_palettes(img, background=bg, bg_scope=scope, max_colors=MAX_COLORS,
                                             exclude=ex).items()}
    return {"colors": colors, "width_mm": fit(w), "height_mm": fit(h), "palettes": palettes}


def _hex(c):
    return "#%02x%02x%02x" % tuple(c)


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _settings(meta):
    bg, scope = BACKGROUND_MODES[meta["background"]]
    max_w, max_h = meta.get("target_mm") or (None, None)  # motifs antérieurs au cadre en mm
    # Une régénération repart du même cadre que la génération précédente : sans ça, éteindre le
    # remplissage d'une zone réduirait la surface piquée, et l'ajustement au cadre agrandirait
    # tout le reste (le motif changerait d'échelle à chaque retouche).
    frame_w = (meta.get("frame") or {}).get("w")
    frame_ppm = (meta.get("frame") or {}).get("ppm")
    same_frame = {"width_mm": frame_w / frame_ppm} if frame_w and frame_ppm else {}
    if same_frame:
        max_w = max_h = None
    return Settings(n_colors=meta["n_colors"], background=bg, background_scope=scope,
                    max_width_mm=max_w, max_height_mm=max_h, **same_frame,
                    exclude_colors=_exclude_setting(meta.get("excluded", [])),
                    engine=meta.get("engine", "standard"),
                    thicken_lines=bool(meta.get("thicken_lines")),
                    zone_overrides=_zone_setting(meta.get("zones") or []),
                    # Une zone restée de la couleur de sa forme suit la palette réellement générée :
                    # la couleur mémorisée à la création (quantification à une autre résolution) peut
                    # différer de quelques unités et créer un fil en trop.
                    region_overrides=tuple(((r["u"], r["v"]), _rgb(r["source"]),
                                            None if r["color"].lower() == r["source"].lower() else _rgb(r["color"]),
                                            r["fill"], r.get("outline", True)) for r in meta.get("regions") or []),
                    merge_same_colors=bool(meta.get("merge_colors")))


def _smart_regions(d, n_colors, background, target_mm, excluded, thicken_lines):
    """Zones fermées des bandes fines à broder en satin bombé (« Remplissage intelligent »).

    Détectées une fois, à la création : elles deviennent des zones comme les autres, modifiables
    dans l'éditeur, et survivent aux retouches qui régénèrent le motif.
    """
    bg, scope = BACKGROUND_MODES[background]
    s = Settings(n_colors=n_colors, background=bg, background_scope=scope,
                 exclude_colors=_exclude_setting(excluded), thicken_lines=thicken_lines,
                 max_width_mm=target_mm[0] if target_mm else None,
                 max_height_mm=target_mm[1] if target_mm else None)
    try:
        shapes = thin_band_regions(d / SOURCE, s)
    except Exception:
        log.exception("Remplissage intelligent : détection impossible pour %s", d.name)
        return []
    log.info("Remplissage intelligent : %d bande(s) fine(s) sur %s", len(shapes), d.name)
    return [{"source": _hex(rgb), "u": round(u, 4), "v": round(v, 4), "color": _hex(rgb),
             "fill": "satin_relief", "outline": False} for u, v, rgb in shapes]


def _zone_setting(zones):
    """Blocs de meta.json -> Settings.zone_overrides (couleurs entières qui diffèrent du défaut)."""
    return tuple((_rgb(z["source"]), _rgb(z["color"]), z["fill"], z.get("outline", True)) for z in zones
                 if z.get("region") is None and (z["color"] != z["source"] or z["fill"] != "tatami"
                                                 or z.get("outline", True) is False))


def _zones_of(pattern, s, regions):
    """Blocs du motif généré et retouches de zones réellement appliquées (réindexées)."""
    used = sorted({ri for _, ri in getattr(pattern, "img2stitch_zones", []) if ri is not None})
    new_index = {ri: k for k, ri in enumerate(used)}
    out = []
    for src, ri in getattr(pattern, "img2stitch_zones", []):
        if ri is None:
            color, kind, outline = _zone_override(src, s.zone_overrides)
            out.append({"source": _hex(src), "color": _hex(color), "fill": kind,
                        "outline": True if outline is None else outline, "region": None})
        else:
            r = regions[ri]
            # couleur réellement piquée : la forme suit sa palette si elle n'a pas été recolorée
            color = _hex(_region_look(s.region_overrides[ri], _rgb(r["source"]))[0])
            out.append({"source": _hex(src), "color": color, "fill": r["fill"],
                        "outline": r.get("outline", True), "region": new_index[ri]})
    return out, [regions[ri] for ri in used]


def parse_edits(payload, n_zones):
    """Retouches envoyées par l'éditeur -> (blocs [{"color", "fill"}], zones fermées [{"block", "fx", "fy", "color", "fill"}])."""
    payload = payload if isinstance(payload, dict) else {}
    items, regions = payload.get("zones"), payload.get("regions") or []
    # RELIEF_LEGACY : noms d'avant la refonte des reliefs, encore présents dans les projets
    # existants ; relief.zone_fill les traduit vers le motif équivalent.
    valid = lambda it: (isinstance(it, dict) and isinstance(it.get("color"), str) and _HEX_RE.match(it["color"])
                        and (it.get("fill") in FILL_TYPES or it.get("fill") in RELIEF_LEGACY)
                        and isinstance(it.get("outline", True), bool))
    ok = (isinstance(items, list) and len(items) == n_zones and all(valid(it) for it in items)
          and isinstance(regions, list) and len(regions) <= 200 and all(
              valid(r) and isinstance(r.get("block"), int) and 0 <= r["block"] < n_zones
              and all(isinstance(r.get(k), (int, float)) and 0 <= r[k] <= 1 for k in ("fx", "fy")) for r in regions))
    if not ok:
        raise InvalidImage("Modifications invalides.")
    return ([{"color": it["color"].lower(), "fill": it["fill"], "outline": it.get("outline", True)} for it in items],
            [{"block": r["block"], "fx": float(r["fx"]), "fy": float(r["fy"]), "color": r["color"].lower(),
              "fill": r["fill"], "outline": r.get("outline", True)} for r in regions])


def _render_zone_map(pattern, path, ppm=5, margin_mm=6):
    """Carte des zones : chaque bloc de couleur dessiné avec la valeur (index + 1), cadrage identique à
    render_photo_preview pour retrouver la zone cliquée sur l'aperçu par simple proportion."""
    x0, y0, x1, y1 = stitch_bbox_mm(pattern)
    W, H = int((x1 - x0 + 2 * margin_mm) * ppm), int((y1 - y0 + 2 * margin_mm) * ppm)
    zmap = np.zeros((H, W), np.uint8)
    px = lambda x, y: (round((x / 10 - x0 + margin_mm) * ppm), round((y / 10 - y0 + margin_mm) * ppm))
    ci, last = 0, None
    width = max(2, round(0.7 * ppm))
    # Zones fusionnées sur un même fil : on suit les segments (début de chaque zone), sinon les changements de fil
    starts = dict(getattr(pattern, "img2stitch_segments", None) or [])
    for n, (x, y, cmd) in enumerate(pattern.stitches):
        c = cmd & pe.COMMAND_MASK
        cur = px(x, y)
        if starts and n in starts:
            ci = starts[n]
        if c == pe.STITCH and last is not None:
            cv2.line(zmap, last, cur, ci + 1, width)
        elif c == pe.COLOR_CHANGE and not starts:
            ci += 1
        last = cur
    Image.fromarray(zmap, "L").save(path, "PNG")


def _save_pattern(pattern, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stitches": [[float(x), float(y), int(c)] for x, y, c in pattern.stitches],
                   "threads": [t.color & 0xFFFFFF for t in pattern.threadlist],
                   "segments": getattr(pattern, "img2stitch_segments", None),
                   "relief": getattr(pattern, "img2stitch_relief", None)}, f)


def _load_pattern(d):
    if not (d / PATTERN_DATA).is_file():
        # Motifs créés avant pattern.json : le VP3 change les sauts en points (fils de déplacement visibles),
        # on prend donc le parcours du DST (sauts et coupes conservés) et les couleurs exactes du VP3.
        pattern = pe.read(str(d / "pattern.dst"))
        pattern.threadlist = pe.read(str(d / "pattern.vp3")).threadlist
        return pattern
    with open(d / PATTERN_DATA, encoding="utf-8") as f:
        data = json.load(f)
    pattern = pe.EmbPattern()
    pattern.stitches = [list(s) for s in data["stitches"]]
    for c in data["threads"]:
        pattern.add_thread({"color": c})
    if data.get("segments"):
        pattern.img2stitch_segments = [tuple(s) for s in data["segments"]]
    if data.get("relief") and isinstance(data["relief"], dict):
        pattern.img2stitch_relief = {layer: [tuple(r) for r in ranges]
                                     for layer, ranges in data["relief"].items()}
    return pattern


def _make_thumb(d):
    with Image.open(d / PHOTO) as ph:
        ph = ph.convert("RGB")
        ph.thumbnail((640, 640), Image.LANCZOS)
        _write_atomic(d / THUMB, lambda p: ph.save(p, "JPEG", quality=86, optimize=True))


def _make_thumb_from(source, target):
    with Image.open(source) as ph:
        ph = ph.convert("RGB")
        ph.thumbnail((640, 640), Image.LANCZOS)
        ph.save(target, "JPEG", quality=86, optimize=True)


def _free_bytes(path):
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def _has_room_for_preview(path):
    free = _free_bytes(path)
    return free is None or free >= MIN_FREE_BYTES


def _write_error(path):
    """Message utilisateur pour un échec d'écriture, avec la cause probable quand le disque est plein."""
    free = _free_bytes(path)
    if free is not None and free < MIN_FREE_BYTES:
        return ("Espace disque insuffisant (%d Mo libres) : libérez de l'espace puis relancez la "
                "génération." % max(0, free // (1024 * 1024)))
    return "Les fichiers du motif n'ont pas pu être écrits sur le disque."


def _write_atomic(path, write):
    """Écrit via un fichier temporaire puis renomme : le fichier final n'existe jamais à moitié écrit."""
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        write(str(tmp))
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows : fichier en cours de lecture (navigateur, téléchargement)
                if attempt == 19:
                    raise
                time.sleep(0.15)
    except OSError:
        tmp.unlink(missing_ok=True)     # un temporaire à moitié écrit ne doit pas traîner
        raise


def _load_image(stream):
    """Valide le fichier envoyé et le normalise (orientation EXIF, mode RGB/RGBA)."""
    try:
        with Image.open(stream) as probe:
            if probe.format not in ACCEPTED_FORMATS:
                raise InvalidImage("Format non pris en charge : envoyez une image PNG, JPG, WebP ou AVIF.")
            probe.verify()
        stream.seek(0)
        img = Image.open(stream)
        source_format = img.format
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImage("Image trop grande.")
        img = ImageOps.exif_transpose(img)
        has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        normalized = img.convert("RGBA" if has_alpha else "RGB")
        normalized.info["source_format"] = source_format
        return normalized
    except InvalidImage:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise InvalidImage("Fichier image illisible ou corrompu.")
