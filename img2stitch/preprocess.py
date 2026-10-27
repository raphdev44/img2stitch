"""Étape 1 : image -> N couleurs + masques binaires (un par fil)."""
import numpy as np
from PIL import Image, ImageFilter


def _parse_hex(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)])


def _kmeans(pixels, k, iters=15):
    """k-means avec initialisation 'farthest point' : garde les petites zones de couleur distinctes."""
    pixels = pixels.astype(np.float32)
    centers = [pixels.mean(axis=0)]
    d = np.linalg.norm(pixels - centers[0], axis=1)
    for _ in range(k - 1):
        centers.append(pixels[int(d.argmax())])
        d = np.minimum(d, np.linalg.norm(pixels - centers[-1], axis=1))
    centers = np.array(centers)
    for _ in range(iters):
        lab = np.linalg.norm(pixels[:, None] - centers[None], axis=2).argmin(axis=1)
        for j in range(k):
            sel = pixels[lab == j]
            if len(sel):
                centers[j] = sel.mean(axis=0)
    return centers


def load_and_quantize(path, width_px, n_colors, background="auto", bg_tol=40):
    """Retourne (palette, masks) ; palette = liste de (r,g,b), masks = liste de tableaux bool."""
    img = Image.open(path).convert("RGBA")
    h = max(1, round(img.height * width_px / img.width))
    img = img.resize((width_px, h), Image.LANCZOS)
    arr = np.array(img)
    rgb = arr[..., :3].astype(int)

    # 1) Quels pixels sont du fond ?
    keep = arr[..., 3] > 128
    bg_rgb = None
    if background == "auto":
        if keep.all():
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
            bg_rgb = np.median(border, axis=0)
    elif background not in ("none", None):
        bg_rgb = _parse_hex(background)
    if bg_rgb is not None:
        keep &= np.linalg.norm(rgb - bg_rgb, axis=-1) > bg_tol
    if not keep.any():
        return [], []

    # 2) K-means sur les seuls pixels à broder (aucune couleur gaspillée sur le fond)
    pix = rgb[keep]
    if len(pix) > 50000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 50000, replace=False)]
    centers = _kmeans(pix, n_colors)

    labels = np.linalg.norm(rgb[:, :, None] - centers[None, None], axis=3).argmin(axis=2).astype(np.uint8)
    labels[~keep] = 255
    # 3) Lissage : supprime les pixels isolés et les franges d'anti-aliasing
    labels = np.array(Image.fromarray(labels).filter(ImageFilter.ModeFilter(5)))

    palette, masks = [], []
    for j in range(n_colors):
        m = labels == j
        if m.any():
            palette.append(tuple(int(c) for c in centers[j]))
            masks.append(m)
    # Clair d'abord : on pose les grandes zones claires avant les détails sombres
    order = sorted(range(len(palette)), key=lambda i: -sum(palette[i]))
    return [palette[i] for i in order], [masks[i] for i in order]
