"""Rendu de contrôle avec les vraies couleurs de fil (le DST ne les stocke pas)."""
from PIL import Image, ImageDraw
import pyembroidery as pe
from .photo import stitch_bbox_mm


def render_raw_preview(pattern, path, px_per_mm=8, margin_mm=3, line_px=2):
    x0, y0, x1, y1 = [v * 10 for v in stitch_bbox_mm(pattern)]
    W = int((x1 - x0) / 10 * px_per_mm + 2 * margin_mm * px_per_mm)
    H = int((y1 - y0) / 10 * px_per_mm + 2 * margin_mm * px_per_mm)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = lambda x, y: ((x - x0) / 10 * px_per_mm + margin_mm * px_per_mm,
                      (y - y0) / 10 * px_per_mm + margin_mm * px_per_mm)
    ci, last = 0, None
    th = pattern.threadlist
    color = lambda: "#%06x" % (th[min(ci, len(th) - 1)].color & 0xFFFFFF)
    for x, y, cmd in pattern.stitches:
        c = cmd & pe.COMMAND_MASK
        if c == pe.STITCH and last is not None:
            d.line([f(*last), f(x, y)], fill=color(), width=line_px)
        elif c == pe.COLOR_CHANGE:
            ci += 1
        last = (x, y)
    img.save(path)


render_preview = render_raw_preview  # ancien nom, conservé


def render_previews(pattern, stem, fabric="creme", **photo_kw):
    """Écrit DEUX aperçus : <stem>_brut.png (tracé des points, vérification technique)
    et <stem>_photo.png (simulation réaliste de la broderie terminée sur un tissu). Retourne les 2 chemins."""
    from .photo import render_photo_preview
    stem = stem[:-4] if stem.lower().endswith(".png") else stem
    raw, photo = f"{stem}_brut.png", f"{stem}_photo.png"
    render_raw_preview(pattern, raw)
    render_photo_preview(pattern, photo, fabric=fabric, **photo_kw)
    return raw, photo
