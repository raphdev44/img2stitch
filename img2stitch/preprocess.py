"""Étape 1 : image -> N couleurs + masques binaires (un par fil)."""
import cv2
import numpy as np
from PIL import Image, ImageFilter


def _parse_hex(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)])


def _to_lab(rgb):
    """(N,3) RGB 0-255 -> (N,3) CIELAB : espace où la distance suit l'écart de couleur perçu."""
    rgb = np.ascontiguousarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _nearest(points, centers, chunk=200_000):
    """Indice du centre le plus proche de chaque point (par blocs pour limiter la mémoire)."""
    out = np.empty(len(points), np.intp)
    c2 = (centers ** 2).sum(axis=1)
    for i in range(0, len(points), chunk):
        out[i:i + chunk] = (c2 - 2 * points[i:i + chunk] @ centers.T).argmin(axis=1)
    return out


def _reduce_colors(pix, n_colors, n_fine=48, min_share=0.0005, stop_delta=None):
    """Réduit les couleurs en fusionnant toujours les deux groupes les plus proches (distance Lab).
    Si `stop_delta` est donné, s'arrête dès que toutes les couleurs restantes en sont plus éloignées.

    Retourne (centres_fins_lab, groupe_de_chaque_centre_fin, palette_rgb)."""
    lab = _to_lab(pix)
    uniq, inv = np.unique(pix, axis=0, return_inverse=True)
    if len(uniq) <= n_fine:
        centers, labels = _to_lab(uniq), inv.ravel()
    else:
        # Sur-segmentation fine, déterministe ; les fusions ci-dessous font la vraie réduction
        cv2.setRNGSeed(0)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv2.kmeans(lab, n_fine, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
    k = len(centers)
    n = np.bincount(labels, minlength=k).astype(np.float64)
    g_lab = centers.astype(np.float64)
    g_rgb = np.zeros((k, 3))
    np.add.at(g_rgb, labels, pix)
    group = np.arange(k)
    active = n > 0

    def merge(a, b):
        g_lab[a] = (g_lab[a] * n[a] + g_lab[b] * n[b]) / (n[a] + n[b])
        g_rgb[a] += g_rgb[b]
        n[a] += n[b]
        active[b] = False
        group[group == b] = a

    # 1) Poussières (bruit, franges d'anti-crénelage) : rattachées à leur couleur la plus proche
    while active.sum() > n_colors:
        idx = np.flatnonzero(active)
        small = idx[n[idx] < min_share * n.sum()]
        if not len(small):
            break
        s = small[n[small].argmin()]
        others = idx[idx != s]
        merge(others[np.linalg.norm(g_lab[others] - g_lab[s], axis=1).argmin()], s)

    # 2) Fusion itérative des deux couleurs les plus proches
    while active.sum() > n_colors:
        idx = np.flatnonzero(active)
        d = np.linalg.norm(g_lab[idx, None] - g_lab[None, idx], axis=2)
        np.fill_diagonal(d, np.inf)
        if stop_delta is not None and d.min() >= stop_delta:
            break
        i, j = np.unravel_index(d.argmin(), d.shape)
        a, b = (idx[i], idx[j]) if n[idx[i]] >= n[idx[j]] else (idx[j], idx[i])
        merge(a, b)

    final = np.flatnonzero(active)
    remap = np.full(k, -1)
    remap[final] = np.arange(len(final))
    palette = [tuple(int(round(c)) for c in g_rgb[f] / n[f]) for f in final]
    return centers.astype(np.float32), remap[group], palette


def _touching_border(mask):
    """Ne garde que les zones de `mask` reliées au bord de l'image."""
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    edge_mask = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return np.isin(labels, np.unique(edge[edge_mask]))


def _embroidered_pixels(src, width_px, background="auto", bg_tol=40, bg_scope="all"):
    """Image redimensionnée -> (rgb int (H,W,3), masque des pixels à broder). `src` : chemin, fichier ou image PIL."""
    img = (src if isinstance(src, Image.Image) else Image.open(src)).convert("RGBA")
    h = max(1, round(img.height * width_px / img.width))
    img = img.resize((width_px, h), Image.LANCZOS)
    arr = np.array(img)
    rgb = arr[..., :3].astype(int)

    keep = arr[..., 3] > 128
    bg_rgb = None
    if background == "auto":
        if keep.all():
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
            bg_rgb = np.median(border, axis=0)
    elif background not in ("none", None):
        bg_rgb = _parse_hex(background)
    if bg_rgb is not None:
        is_bg = np.linalg.norm(rgb - bg_rgb, axis=-1) <= bg_tol
        if bg_scope == "outside":
            is_bg = _touching_border(is_bg)
        keep &= ~is_bg
    return rgb, keep


def estimate_colors(src, background="auto", bg_tol=40, bg_scope="all", max_colors=12,
                    min_delta_e=15.0, min_share=0.005, width_px=400):
    """Nombre de couleurs réellement distinctes à l'œil dans la zone à broder (fond exclu).

    Deux couleurs à moins de `min_delta_e` (CIELAB) sont confondues ; les transitions (flou,
    anti-crénelage) et les zones de moins de `min_share` de la surface sont ignorées."""
    rgb, keep = _embroidered_pixels(src, width_px, background, bg_tol, bg_scope)
    if not keep.any():
        return 0
    lab = _to_lab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    # Écart de couleur par pixel (ΔE/px) : fort sur les bords flous ou anti-crénelés, nul dans les aplats
    grad = np.sqrt(sum(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
                       for c in range(3))) / 8
    flat = keep & (grad < 4)
    if flat.sum() < 0.2 * keep.sum():  # image sans aplats (photo, dégradés) : on garde la moitié la plus stable
        flat = keep & (grad <= np.percentile(grad[keep], 50))
    pix = rgb[flat]
    if len(pix) > 50000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 50000, replace=False)]
    _, _, palette = _reduce_colors(pix, 1, min_share=min_share, stop_delta=min_delta_e)
    return int(np.clip(len(palette), 1, max_colors))


def load_and_quantize(path, width_px, n_colors, background="auto", bg_tol=40, bg_scope="all"):
    """Retourne (palette, masks) ; palette = liste de (r,g,b), masks = liste de tableaux bool.
    bg_scope : "all" retire tous les pixels de la couleur du fond, "outside" seulement ceux reliés aux bords."""
    rgb, keep = _embroidered_pixels(path, width_px, background, bg_tol, bg_scope)
    if not keep.any():
        return [], []

    # 2) Réduction des couleurs sur les seuls pixels à broder (aucune couleur gaspillée sur le fond)
    pix = rgb[keep]
    if len(pix) > 50000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 50000, replace=False)]
    centers, group, colors = _reduce_colors(pix, n_colors)

    labels = np.full(keep.shape, 255, np.uint8)
    labels[keep] = group[_nearest(_to_lab(rgb[keep]), centers)]
    # 3) Lissage : supprime les pixels isolés et les franges d'anti-aliasing
    labels = np.array(Image.fromarray(labels).filter(ImageFilter.ModeFilter(5)))

    palette, masks = [], []
    for j, rgb_j in enumerate(colors):
        m = labels == j
        if m.any():
            palette.append(rgb_j)
            masks.append(m)
    # Clair d'abord : on pose les grandes zones claires avant les détails sombres
    order = sorted(range(len(palette)), key=lambda i: -sum(palette[i]))
    return [palette[i] for i in order], [masks[i] for i in order]
