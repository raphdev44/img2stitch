from datetime import datetime

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from .designs import (BACKGROUND_MODES, DEFAULT_SIZE_MM, FORMATS, MAX_COLORS, MAX_SIZE_MM, MIN_COLORS, MIN_SIZE_MM,
                      InvalidImage, analyse_image, parse_exclusions)

bp = Blueprint("web", __name__)

BG_LABELS = {"outside": "fond extérieur supprimé", "all": "fond et zones identiques supprimés", "keep": "fond conservé"}


def store():
    return current_app.extensions["designs"]


def _public(meta):
    out = {k: meta.get(k) for k in ("id", "name", "created_at", "n_colors", "background", "target_mm",
                                     "status", "step", "error", "stats", "threads", "size_mm")}
    out["urls"] = {
        "page": url_for("web.design", design_id=meta["id"]),
        "raw": url_for("web.design_file", design_id=meta["id"], name="preview_raw.png") if meta["has_raw"] else None,
        "photo": url_for("web.design_file", design_id=meta["id"], name="preview_photo.png") if meta["has_photo"] else None,
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
                           min_size=MIN_SIZE_MM, max_size=MAX_SIZE_MM, default_size=DEFAULT_SIZE_MM)


@bp.get("/galerie")
def gallery():
    return render_template("gallery.html", designs=store().list())


@bp.get("/motifs/<design_id>")
def design(design_id):
    meta = store().get(design_id) or abort(404)
    return render_template("design.html", d=meta, formats=FORMATS, bg_labels=BG_LABELS)


@bp.get("/motifs/<design_id>/fichiers/<name>")
def design_file(design_id, name):
    path = store().file_path(design_id, name) or abort(404)
    return send_file(path, max_age=3600)


@bp.get("/motifs/<design_id>/telecharger/<fmt>")
def download(design_id, fmt):
    path = store().pattern_path(design_id, fmt) or abort(404)
    meta = store().get(design_id)
    return send_file(path, as_attachment=True, download_name=f"{meta['name']}.{fmt}",
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
    try:
        excluded = parse_exclusions(request.form.get("exclude"))
        design_id = store().create(f.stream, f.filename, n_colors, background, target, excluded)
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


@bp.get("/api/motifs/<design_id>")
def api_get(design_id):
    meta = store().get(design_id) or abort(404)
    return jsonify(_public(meta))


@bp.app_errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error="Fichier trop volumineux (20 Mo maximum)."), 413
