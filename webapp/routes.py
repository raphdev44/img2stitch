from datetime import datetime

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from img2stitch.vectorize import available as vectorize_available

from .designs import (BACKGROUND_MODES, CROSS_STITCH_MAX_COLORS, DEFAULT_SIZE_MM, FABRICS, FILL_TYPES, FORMATS, MAX_COLORS, MAX_SIZE_MM,
                      MIN_COLORS, MIN_SIZE_MM, SCAN_ENGINES, ZONES, InvalidImage, analyse_cross_stitch_image, analyse_image,
                      clean_name, export_filename, parse_edits, parse_exclusions)
from .designs import _HEX_RE as HEX_RE

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
                                     "optimization_mode", "optimized_from", "optimized_from_name")}
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
def create():
    return render_template("create.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS,
                           min_size=MIN_SIZE_MM, max_size=MAX_SIZE_MM, default_size=DEFAULT_SIZE_MM,
                           engines=SCAN_ENGINES, vector_ready=vectorize_available())


@bp.get("/galerie")
def gallery():
    return render_template("gallery.html", designs=store().list())


@bp.get("/pixel-art")
def create_cross_stitch():
    return render_template("cross_stitch.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS)


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
    try:
        colors = _cross_stitch_colors(request.form.get("colors"))
        excluded = parse_exclusions(request.form.get("exclude"))
        design_id = store().create(image.stream, image.filename, colors or 0, background, None,
                       excluded=excluded, mode="cross_stitch")
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
                           fabrics={k: (FABRIC_LABELS.get(k, k.capitalize()), v) for k, v in FABRICS.items()})


@bp.get("/motifs/<design_id>/edition")
def edit(design_id):
    meta = store().get(design_id) or abort(404)
    if meta["status"] != "ready" or meta.get("mode") == "cross_stitch":
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
