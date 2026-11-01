from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from img2stitch.algorithm_vectorize import available as geometry_available
from img2stitch.hand import hand_stitch_sample_pattern
from img2stitch.handstitch import STITCH_TYPES
from img2stitch.photo import render_photo_preview

from .designs import (BACKGROUND_MODES, CROSS_STITCH_MAX_COLORS, DEFAULT_SIZE_MM, FABRICS, FILL_TYPES, FORMATS, MAX_COLORS, MAX_SIZE_MM,
                      MIN_COLORS, MIN_SIZE_MM, SCAN_ENGINES, ZONES, InvalidImage, analyse_cross_stitch_image, analyse_image,
                      clean_name, export_filename, parse_edits, parse_exclusions)
from .designs import _HEX_RE as HEX_RE
from .designs import HAND_CREATIVITY, HAND_STYLES

bp = Blueprint("web", __name__)

BG_LABELS = {"outside": "fond extérieur supprimé", "all": "fond et zones identiques supprimés", "keep": "fond conservé"}
FABRIC_LABELS = {"blanc": "Blanc", "creme": "Crème", "gris": "Gris", "noir": "Noir", "marine": "Marine",
                 "denim": "Denim", "rouge": "Rouge", "emeraude": "Vert émeraude"}


def store():
    return current_app.extensions["designs"]


def _file_url(meta, name):
    # ?v= : l'aperçu réaliste change quand on change de tissu, le cache navigateur ne doit pas resservir l'ancien
    return url_for("web.design_file", design_id=meta["id"], name=name, v=meta.get("photo_version", 0))


def _public(meta):
    out = {k: meta.get(k) for k in ("id", "name", "created_at", "n_colors", "background", "target_mm",
                                     "status", "step", "error", "stats", "threads", "size_mm", "fabric",
                                     "photo_error", "photo_progress", "edit_error", "mode", "grid", "engine",
                                     "cross_stitch_color_limit", "progress", "optimization_report",
                                     "optimization_mode", "optimized_from", "optimized_from_name",
                                     "hand_style", "hand_creativity", "hand_mix", "cross_stitch_double",
                                     "streamline", "streamline_info")}
    out["photo_rendering"] = bool(meta.get("photo_rendering"))
    out["urls"] = {
        "page": url_for("web.design", design_id=meta["id"]),
        "raw": _file_url(meta, "preview_raw.png") if meta["has_raw"] else None,
        "photo": _file_url(meta, "preview_photo.png") if meta["has_photo"] else None,
    }
    return out


@bp.app_template_filter("datefr")
def datefr(value):
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d/%m/%Y · %H:%M")
    except (TypeError, ValueError):
        return ""


@bp.get("/")
def home():
    return render_template("home.html", count=len(store().list()))


@bp.get("/numerisation")
def create():
    return render_template("create.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS,
                           min_size=MIN_SIZE_MM, max_size=MAX_SIZE_MM, default_size=DEFAULT_SIZE_MM,
                           engines=SCAN_ENGINES, geometry_ready=geometry_available())


@bp.get("/galerie")
def gallery():
    return render_template("gallery.html", designs=store().list())


@bp.get("/pixel-art")
def create_cross_stitch():
    return render_template("cross_stitch.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS)


@bp.get("/broderie-main")
def create_hand():
    return render_template("hand_embroidery.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS,
                           min_size=MIN_SIZE_MM, max_size=MAX_SIZE_MM, default_size=DEFAULT_SIZE_MM,
                           styles=HAND_STYLES, creativity=HAND_CREATIVITY, stitch_types=STITCH_TYPES)


@bp.get("/streamlines")
def create_streamline():
    return render_template("streamline.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS,
                           min_size=MIN_SIZE_MM, max_size=MAX_SIZE_MM, default_size=DEFAULT_SIZE_MM)


@bp.get("/api/broderie-main/exemples/<stitch>.png")
def hand_stitch_sample(stitch):
    if stitch not in STITCH_TYPES:
        abort(404)
    pattern = hand_stitch_sample_pattern(stitch)
    with TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "stitch-sample.png"
        render_photo_preview(pattern, str(image_path), fabric="creme")
        image = image_path.read_bytes()
    return send_file(BytesIO(image), mimetype="image/png", max_age=86400)


@bp.post("/api/broderie-main")
def api_create_hand():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify(error="Aucune image envoyée."), 400
    background = request.form.get("background", "outside")
    if background not in BACKGROUND_MODES:
        return jsonify(error="Option d'arrière-plan invalide."), 400
    try:
        n_colors = int(request.form.get("colors", ""))
    except ValueError:
        return jsonify(error="Nombre de couleurs invalide."), 400
    if not MIN_COLORS <= n_colors <= MAX_COLORS:
        return jsonify(error=f"Le nombre de couleurs doit être compris entre {MIN_COLORS} et {MAX_COLORS}."), 400
    style = request.form.get("hand_style", "auto")
    if style not in HAND_STYLES:
        return jsonify(error="Famille de points inconnue."), 400
    try:
        creativity = int(request.form.get("creativity", "3"))
    except ValueError:
        creativity = 0
    if creativity not in HAND_CREATIVITY:
        return jsonify(error="Niveau de créativité invalide."), 400
    raw = [request.form.get(k) for k in ("width_mm", "height_mm")]
    if any(not v for v in raw):  # ancien client sans champs de taille : on garde la taille par défaut
        target = (DEFAULT_SIZE_MM, DEFAULT_SIZE_MM)
    else:
        try:
            target = tuple(float(v.replace(",", ".")) for v in raw)
        except ValueError:
            return jsonify(error="Dimensions invalides."), 400
    if not all(MIN_SIZE_MM <= v <= MAX_SIZE_MM for v in target):
        return jsonify(error=f"La largeur et la hauteur doivent être comprises entre {MIN_SIZE_MM} "
                             f"et {MAX_SIZE_MM} mm."), 400
    try:
        design_id = store().create(image.stream, image.filename, n_colors, background, target,
                                   excluded=parse_exclusions(request.form.get("exclude")),
                                   mode="hand", hand_style=style, hand_creativity=creativity)
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/streamlines")
def api_create_streamline():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify(error="Aucune image envoyée."), 400
    # Les streamlines privilégient une interprétation de la photo entière : fond conservé par défaut.
    background = request.form.get("background", "keep")
    if background not in BACKGROUND_MODES:
        return jsonify(error="Option d'arrière-plan invalide."), 400
    try:
        n_colors = int(request.form.get("colors", ""))
    except ValueError:
        return jsonify(error="Nombre de couleurs invalide."), 400
    if not MIN_COLORS <= n_colors <= MAX_COLORS:
        return jsonify(error=f"Le nombre de couleurs doit être compris entre {MIN_COLORS} et {MAX_COLORS}."), 400
    raw = [request.form.get(k) for k in ("width_mm", "height_mm")]
    if any(not v for v in raw):  # champs absents : la taille du dépôt (99 mm) s'applique
        target = None
    else:
        try:
            target = tuple(float(v.replace(",", ".")) for v in raw)
        except ValueError:
            return jsonify(error="Dimensions invalides."), 400
        if not all(MIN_SIZE_MM <= v <= MAX_SIZE_MM for v in target):
            return jsonify(error=f"La largeur et la hauteur doivent être comprises entre {MIN_SIZE_MM} "
                                 f"et {MAX_SIZE_MM} mm."), 400
    try:
        line_width = float(request.form.get("line_width_mm", "0.4").replace(",", "."))
        relax = int(request.form.get("relax_iterations", "14"))
    except ValueError:
        return jsonify(error="Réglages streamlines invalides."), 400
    if not 0.15 <= line_width <= 2.0:
        return jsonify(error="La largeur du fil doit être comprise entre 0,15 et 2 mm."), 400
    if not 4 <= relax <= 40:
        return jsonify(error="Le lissage doit être compris entre 4 et 40 passes."), 400
    try:
        design_id = store().create(image.stream, image.filename, n_colors, background, target,
                                   excluded=parse_exclusions(request.form.get("exclude")),
                                   mode="streamline",
                                   streamline_options={"line_width_mm": line_width,
                                                       "background_fill": request.form.get("background_fill") == "true",
                                                       "relax_iterations": relax})
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/pixel-art/analyse")
def api_analyse_cross_stitch():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify(error="Aucune image envoyée."), 400
    background = request.form.get("background", "outside")
    if background not in BACKGROUND_MODES:
        return jsonify(error="Option d'arrière-plan invalide."), 400
    try:
        return jsonify(analyse_cross_stitch_image(image.stream, background,
                                                  parse_exclusions(request.form.get("exclude"))))
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 400


def _cross_stitch_colors(raw):
    if isinstance(raw, bool) or not isinstance(raw, (str, int, type(None))):
        raise InvalidImage("Nombre de couleurs invalide.")
    try:
        count = int(raw) if raw not in (None, "") else 0
    except ValueError:
        raise InvalidImage("Nombre de couleurs invalide.")
    if count == 0:
        return None
    if not 1 <= count <= CROSS_STITCH_MAX_COLORS:
        raise InvalidImage("Le nombre de couleurs doit être compris entre 1 et 256.")
    return count


@bp.post("/api/pixel-art")
def api_create_cross_stitch():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify(error="Aucune image envoyée."), 400
    background = request.form.get("background", "outside")
    if background not in BACKGROUND_MODES:
        return jsonify(error="Option d'arrière-plan invalide."), 400
    stitch = request.form.get("stitch", "simple")
    if stitch not in ("simple", "double"):
        return jsonify(error="Type de point de croix inconnu."), 400
    try:
        colors = _cross_stitch_colors(request.form.get("colors"))
        excluded = parse_exclusions(request.form.get("exclude"))
        design_id = store().create(image.stream, image.filename, colors or 0, background, None,
                       excluded=excluded, mode="cross_stitch", cross_stitch_double=stitch == "double")
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/motifs/<design_id>/couleurs")
def api_cross_stitch_colors(design_id):
    if store().get(design_id) is None:
        abort(404)
    body = request.get_json(silent=True)
    try:
        if not isinstance(body, dict) or "colors" not in body:
            raise InvalidImage("Nombre de couleurs invalide.")
        colors = _cross_stitch_colors(body["colors"])
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 400
    try:
        store().set_cross_stitch_colors(design_id, colors)
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/motifs/<design_id>/supprimer")
def api_delete(design_id):
    try:
        if not store().delete(design_id):
            abort(404)
    except InvalidImage as e:
        return jsonify(error=str(e)), 409
    return "", 204


@bp.post("/api/motifs/<design_id>/optimiser")
def api_optimize(design_id):
    if store().get(design_id) is None:
        abort(404)
    body = request.get_json(silent=True) or {}
    # Sans précision (ancien client, formulaire vide) : l'optimisation prudente d'origine
    mode = body.get("mode", "simple") if isinstance(body, dict) else "simple"
    try:
        optimized_id = store().optimize(design_id, mode)
    except InvalidImage as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(_public(store().get(optimized_id))), 202


@bp.get("/motifs/<design_id>")
def design(design_id):
    meta = store().get(design_id) or abort(404)
    return render_template("design.html", d=meta, formats=FORMATS, bg_labels=BG_LABELS,
                           thread_lengths=store().thread_lengths(design_id),
                           hand_labels={key: value[0] for key, value in STITCH_TYPES.items()},
                           fabrics={k: (FABRIC_LABELS.get(k, k.capitalize()), v) for k, v in FABRICS.items()})


@bp.get("/motifs/<design_id>/edition")
def edit(design_id):
    meta = store().get(design_id) or abort(404)
    if meta["status"] != "ready" or meta.get("mode") in ("cross_stitch", "hand", "streamline"):
        return redirect(url_for("web.design", design_id=design_id))
    return render_template("edit.html", d=meta, zones=store().zones(design_id), fills=FILL_TYPES)


@bp.get("/motifs/<design_id>/fichiers/<name>")
def design_file(design_id, name):
    if name == ZONES and store().get(design_id):
        store().zone_map_path(design_id)
    path = store().file_path(design_id, name) or abort(404)
    return send_file(path, max_age=3600)


@bp.get("/motifs/<design_id>/telecharger/<fmt>")
def download(design_id, fmt):
    path = store().pattern_path(design_id, fmt) or abort(404)
    meta = store().get(design_id)
    return send_file(path, as_attachment=True, download_name=export_filename(meta["name"], fmt),
                     mimetype="application/octet-stream")


@bp.post("/api/motifs")
def api_create():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="Aucune image envoyée."), 400
    try:
        n_colors = int(request.form.get("colors", ""))
    except ValueError:
        return jsonify(error="Nombre de couleurs invalide."), 400
    if not MIN_COLORS <= n_colors <= MAX_COLORS:
        return jsonify(error=f"Le nombre de couleurs doit être compris entre {MIN_COLORS} et {MAX_COLORS}."), 400
    background = request.form.get("background", "outside")
    if background not in BACKGROUND_MODES:
        return jsonify(error="Option d'arrière-plan invalide."), 400
    try:
        target = tuple(float(request.form.get(k, "").replace(",", ".")) for k in ("width_mm", "height_mm"))
    except ValueError:
        return jsonify(error="Dimensions invalides."), 400
    if not all(MIN_SIZE_MM <= v <= MAX_SIZE_MM for v in target):
        return jsonify(error=f"La largeur et la hauteur doivent être comprises entre {MIN_SIZE_MM} et {MAX_SIZE_MM} mm."), 400
    engine = request.form.get("engine", "standard")
    if engine not in SCAN_ENGINES:
        return jsonify(error="Algorithme de numérisation inconnu."), 400
    try:
        excluded = parse_exclusions(request.form.get("exclude"))
        thicken_lines = request.form.get("thicken_lines") == "true"
        smart_fill = request.form.get("smart_fill") == "true"
        design_id = store().create(f.stream, f.filename, n_colors, background, target, excluded, thicken_lines,
                                   smart_fill=smart_fill, engine=engine)
    except InvalidImage as e:
        return jsonify(error=str(e)), 400
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/analyse")
def api_analyse():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="Aucune image envoyée."), 400
    background = request.form.get("background", "outside")
    if background not in BACKGROUND_MODES:
        return jsonify(error="Option d'arrière-plan invalide."), 400
    try:
        return jsonify(analyse_image(f.stream, background, parse_exclusions(request.form.get("exclude"))))
    except InvalidImage as e:
        return jsonify(error=str(e)), 400


@bp.post("/api/motifs/<design_id>/edition")
def api_edit(design_id):
    if store().get(design_id) is None:
        abort(404)
    try:
        edits, regions = parse_edits(request.get_json(silent=True), len(store().zones(design_id)))
    except InvalidImage as e:
        return jsonify(error=str(e)), 400
    try:
        store().apply_edits(design_id, edits, regions)
    except InvalidImage as e:
        return jsonify(error=str(e)), 409
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/motifs/<design_id>/nom")
def api_rename(design_id):
    if store().get(design_id) is None:
        abort(404)
    try:
        name = clean_name((request.get_json(silent=True) or {}).get("name"))
    except InvalidImage as e:
        return jsonify(error=str(e)), 400
    try:
        store().rename(design_id, name)
    except InvalidImage as e:
        return jsonify(error=str(e)), 409
    return jsonify(name=name, filename=export_filename(name, "pes"))


@bp.post("/api/motifs/<design_id>/fil")
def api_recolor(design_id):
    meta = store().get(design_id) or abort(404)
    body = request.get_json(silent=True) or {}
    old, new = str(body.get("from", "")).lower(), str(body.get("to", "")).lower()
    if old not in (meta.get("threads") or []) or not HEX_RE.match(new):
        return jsonify(error="Couleur invalide."), 400
    try:
        store().recolor(design_id, old, new)
    except InvalidImage as e:
        return jsonify(error=str(e)), 409
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/motifs/<design_id>/fusion")
def api_merge(design_id):
    if store().get(design_id) is None:
        abort(404)
    try:
        store().merge_colors(design_id, bool((request.get_json(silent=True) or {}).get("enabled", True)))
    except InvalidImage as e:
        return jsonify(error=str(e)), 409
    return jsonify(_public(store().get(design_id))), 202


@bp.post("/api/motifs/<design_id>/tissu")
def api_fabric(design_id):
    if store().get(design_id) is None:
        abort(404)
    fabric = (request.get_json(silent=True) or {}).get("fabric")
    if fabric not in FABRICS:
        return jsonify(error="Tissu inconnu."), 400
    try:
        store().set_fabric(design_id, fabric)
    except InvalidImage as e:
        return jsonify(error=str(e)), 409
    return jsonify(_public(store().get(design_id))), 202


@bp.get("/api/motifs/<design_id>")
def api_get(design_id):
    meta = store().get(design_id) or abort(404)
    return jsonify(_public(meta))


@bp.app_errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error="Fichier trop volumineux (20 Mo maximum)."), 413
