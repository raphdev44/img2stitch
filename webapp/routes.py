from datetime import datetime

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from .designs import FORMATS, MAX_COLORS, MIN_COLORS, InvalidImage

bp = Blueprint("web", __name__)


def store():
    return current_app.extensions["designs"]


def _public(meta):
    out = {k: meta.get(k) for k in ("id", "name", "created_at", "n_colors", "remove_background",
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
    return render_template("create.html", min_colors=MIN_COLORS, max_colors=MAX_COLORS)


@bp.get("/galerie")
def gallery():
    return render_template("gallery.html", designs=store().list())


@bp.get("/motifs/<design_id>")
def design(design_id):
    meta = store().get(design_id) or abort(404)
    return render_template("design.html", d=meta, formats=FORMATS)


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
    remove_bg = request.form.get("remove_background") in ("1", "true", "on")
    try:
        design_id = store().create(f.stream, f.filename, n_colors, remove_bg)
    except InvalidImage as e:
        return jsonify(error=str(e)), 400
    return jsonify(_public(store().get(design_id))), 202


@bp.get("/api/motifs/<design_id>")
def api_get(design_id):
    meta = store().get(design_id) or abort(404)
    return jsonify(_public(meta))


@bp.app_errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error="Fichier trop volumineux (20 Mo maximum)."), 413
