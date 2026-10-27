import argparse
from .config import Settings
from .convert import convert, pattern_stats
from .preview import render_previews


def main():
    p = argparse.ArgumentParser(description="Image -> fichier de broderie machine")
    p.add_argument("input")
    p.add_argument("output", help="extension = format : .dst .pes .jef .exp .vp3 ...")
    d = Settings()
    p.add_argument("--width", type=float, default=d.width_mm, help="largeur en mm")
    p.add_argument("--colors", type=int, default=d.n_colors)
    p.add_argument("--spacing", type=float, default=d.row_spacing_mm, help="écart des rangées en mm")
    p.add_argument("--angle", type=float, default=d.fill_angle_deg)
    p.add_argument("--background", default=d.background, help="auto | none | #RRGGBB")
    p.add_argument("--overlap", type=float, default=d.overlap_mm, help="chevauchement entre couleurs en mm (0 = aucun)")
    p.add_argument("--no-outline", action="store_true", help="désactive les contours")
    p.add_argument("--trim-threshold", type=float, default=d.trim_threshold_mm, help="coupe le fil pour les sauts plus longs (mm)")
    p.add_argument("--max-visible-travel", type=float, default=d.max_visible_travel_mm,
                   help="déplacement piqué visible toléré avant de préférer un saut (mm) ; plus grand = moins de sauts")
    p.add_argument("--no-lock", action="store_true", help="désactive les points d'arrêt")
    p.add_argument("--preview", metavar="PREFIXE", help="écrit PREFIXE_brut.png et PREFIXE_photo.png")
    p.add_argument("--fabric", default="creme", help="tissu du rendu photo : blanc creme gris noir marine denim rouge ou #rrggbb")
    a = p.parse_args()
    s = Settings(width_mm=a.width, n_colors=a.colors, row_spacing_mm=a.spacing, fill_angle_deg=a.angle,
                 background=a.background, overlap_mm=a.overlap, outline=not a.no_outline,
                 trim_threshold_mm=a.trim_threshold, max_visible_travel_mm=a.max_visible_travel,
                 lock_stitches=not a.no_lock)
    pat = convert(a.input, a.output, s)
    st = pattern_stats(pat)
    print(f"{st['stitches']} points, {len(pat.threadlist)} fils, {st['jumps']} sauts, "
          f"{st['trims']} coupes, {st['color_changes']} changements de couleur -> {a.output}")
    if a.preview:
        for f in render_previews(pat, a.preview, fabric=a.fabric):
            print("aperçu :", f)
