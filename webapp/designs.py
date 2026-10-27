"""Stockage des motifs sur disque et génération en tâche de fond."""
import json
import logging
import math
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyembroidery as pe
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from img2stitch import (Settings, color_palettes, estimate_colors, image_to_pattern, natural_size_mm,
                        render_photo_preview, render_raw_preview)
from img2stitch.convert import SIZE_MARGIN_MM, pattern_stats
from img2stitch.photo import stitch_bbox_mm

log = logging.getLogger(__name__)

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
PUBLIC_FILES = {SOURCE, RAW, PHOTO, THUMB}
MIN_COLORS, MAX_COLORS = 1, 12
MIN_SIZE_MM, MAX_SIZE_MM, DEFAULT_SIZE_MM = 10, 400, 100
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
        meta["has_raw"] = (d / RAW).is_file()
        meta["has_photo"] = (d / PHOTO).is_file()
        meta["has_thumb"] = (d / THUMB).is_file()
        return meta

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

    # --- Création ---
    def create(self, stream, filename, n_colors, background, target_mm, excluded=()):
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
            "n_colors": n_colors,
            "background": background,
            "target_mm": list(target_mm),
            "excluded": list(excluded),
            "status": "processing",
            "step": "Analyse de l'image",
            "error": None,
        }
        self._write(d, meta)
        self._pool.submit(self._generate, d)
        return design_id

    def _generate(self, d):
        meta = self._read(d)
        try:
            bg, scope = BACKGROUND_MODES[meta["background"]]
            max_w, max_h = meta["target_mm"]
            s = Settings(n_colors=meta["n_colors"], background=bg, background_scope=scope,
                         max_width_mm=max_w, max_height_mm=max_h,
                         exclude_colors=_exclude_setting(meta.get("excluded", [])))
            pattern = image_to_pattern(str(d / SOURCE), s)
            stats = pattern_stats(pattern)
            if stats["stitches"] == 0:
                raise InvalidImage("Aucun point à broder : l'image semble vide une fois le fond supprimé.")

            self._update(d, meta, step="Écriture des fichiers de broderie")
            opts = {"tie_on": True, "tie_off": True} if s.lock_stitches else {}
            for fmt in FORMATS:
                pe.write(pattern, str(d / f"pattern.{fmt}"), opts)

            x0, y0, x1, y1 = stitch_bbox_mm(pattern)
            _write_atomic(d / RAW, lambda p: render_raw_preview(pattern, p))
            self._update(
                d, meta,
                step="Rendu réaliste",
                stats=stats,
                threads=["#%06x" % (t.color & 0xFFFFFF) for t in pattern.threadlist],
                size_mm=[round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
            )

            _write_atomic(d / PHOTO, lambda p: render_photo_preview(pattern, p))
            with Image.open(d / PHOTO) as ph:
                ph = ph.convert("RGB")
                ph.thumbnail((640, 640), Image.LANCZOS)
                _write_atomic(d / THUMB, lambda p: ph.save(p, "JPEG", quality=86, optimize=True))
            self._update(d, meta, status="ready", step=None)
        except InvalidImage as e:
            self._update(d, meta, status="error", step=None, error=str(e))
        except Exception:
            log.exception("Échec de génération pour %s", d.name)
            self._update(d, meta, status="error", step=None, error="La génération du motif a échoué.")

    # --- Persistance ---
    def _recover_interrupted(self):
        for d in self.root.iterdir():
            if (d / META).is_file():
                meta = self._read(d)
                if meta.get("status") == "processing":
                    self._update(d, meta, status="error", step=None, error="Génération interrompue (redémarrage du serveur).")

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
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    return tuple((rgb(e["color"]), tuple(rgb(o) for o in e["others"])) for e in excluded)


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


def _write_atomic(path, write):
    """Écrit via un fichier temporaire puis renomme : le fichier final n'existe jamais à moitié écrit."""
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    write(str(tmp))
    os.replace(tmp, path)


def _load_image(stream):
    """Valide le fichier envoyé et le normalise (orientation EXIF, mode RGB/RGBA)."""
    try:
        with Image.open(stream) as probe:
            if probe.format not in ACCEPTED_FORMATS:
                raise InvalidImage("Format non pris en charge : envoyez une image PNG, JPG, WebP ou AVIF.")
            probe.verify()
        stream.seek(0)
        img = Image.open(stream)
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImage("Image trop grande.")
        img = ImageOps.exif_transpose(img)
        has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        return img.convert("RGBA" if has_alpha else "RGB")
    except InvalidImage:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise InvalidImage("Fichier image illisible ou corrompu.")
