"""Rendu "photo" : simule l'aspect réel de la broderie (fils en relief, brillance, ombres, tissu tissé).

Principe : chaque point est un petit cylindre de fil posé sur le tissu. On construit une carte de hauteur,
une carte de couleur et une carte de direction du fil, puis on éclaire le tout (diffus + reflet anisotrope
type soie/rayonne + occlusion ambiante + ombres portées sur le tissu).
"""
import math
import cv2
import numpy as np
import pyembroidery as pe

FABRICS = {
    "blanc": "#f1eee6", "creme": "#e6dcc4", "gris": "#8d8f92",
    "noir": "#1c1c1e", "marine": "#22304f", "denim": "#46628a", "rouge": "#8c2a2a",
}
_LIGHT = np.array([-0.45, -0.55, 0.70], np.float32)
_LIGHT /= np.linalg.norm(_LIGHT)


def stitch_bbox_mm(pattern):
    """Boîte englobante (x0, y0, x1, y1) en mm des points réellement piqués.
    (pattern.extents() inclut toujours l'origine, ce qui décentre les aperçus.)"""
    pts = np.array([(x, y) for x, y, c in pattern.stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)]) / 10
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def _rgb(c):
    c = FABRICS.get(str(c).lower(), c).lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _lin(x):
    return np.power(x, 2.2)


def _srgb(x):
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _normals(h, k=1.0):
    gy, gx = np.gradient(h)
    nx, ny = -gx * k, -gy * k
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    return nx * inv, ny * inv, inv


def _fabric(H, W, ppm, base_rgb, rng):
    """Toile à armure toile (plain weave) : fils de chaîne et de trame qui passent dessus/dessous."""
    period = 0.5 * ppm
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / period, yy / period
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    over = ((iu + iv) % 2) == 0
    warp = np.cos(np.pi * (fu - 0.5)) * (0.55 + 0.45 * np.sin(np.pi * fv))
    weft = np.cos(np.pi * (fv - 0.5)) * (0.55 + 0.45 * np.sin(np.pi * fu))
    h = np.where(over, warp, weft).astype(np.float32)
    # irrégularités propres à chaque fil de fil (épaisseur/teinte) + duvet de fibres
    nu = rng.normal(0, 1, int(W / period) + 3).astype(np.float32)
    nv = rng.normal(0, 1, int(H / period) + 3).astype(np.float32)
    yarn = np.where(over, nu[iu.astype(int)], nv[iv.astype(int)]) * 0.04
    fiber = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7) * 0.10
    mott = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), ppm * 2.5) * 0.9
    tone = 0.78 + 0.30 * h + yarn + fiber + 0.05 * mott
    col = _lin(base_rgb)[None, None, :] * tone[..., None]
    return col.astype(np.float32), (h * 0.20 * period + fiber * 0.15 * period).astype(np.float32)


def _draw_stitch(Ht, Ct, Tt, cov, a, b, col, r, thick, twist_px, rng):
    """Dessine un point (capsule) dans les cartes de hauteur / couleur / direction / couverture."""
    H, W = Ht.shape
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay)
    if L < 0.3:
        return
    ux, uy = (bx - ax) / L, (by - ay) / L
    pad = r + 2
    x0, x1 = max(int(min(ax, bx) - pad), 0), min(int(max(ax, bx) + pad) + 1, W)
    y0, y1 = max(int(min(ay, by) - pad), 0), min(int(max(ay, by) + pad) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    px, py = xs - ax, ys - ay
    t = px * ux + py * uy
    s = -px * uy + py * ux
    tc = np.clip(t, 0, L)
    dist = np.hypot(px - tc * ux, py - tc * uy)
    alpha = np.clip(r + 0.5 - dist, 0, 1)
    if not alpha.any():
        return
    prof = np.sqrt(np.clip(1 - (dist / r) ** 2, 0, 1))             # section ronde du fil
    endd = np.minimum(t, L - t)                                       # distance à l'extrémité la plus proche
    dip = 0.30 + 0.70 * _smooth((endd + 0.5 * r) / (1.5 * r))        # le fil plonge dans le tissu aux extrémités
    sl = (slice(y0, y1), slice(x0, x1))
    under = Ht[sl]
    hnew = np.minimum(thick * prof * dip + 0.5 * under, 1.5 * thick)
    Ht[sl] = under * (1 - alpha) + hnew * alpha
    tw = 1 + 0.07 * np.sin(2 * np.pi * (t / twist_px + 0.4 * s / r))  # torsion du fil
    pix = col[None, None, :] * (tw * (1 + rng.normal(0, 0.03)))[..., None]
    a3 = alpha[..., None]
    Ct[sl] = Ct[sl] * (1 - a3) + pix * a3
    m = alpha > 0.5
    Tt[sl][..., 0] = np.where(m, ux, Tt[sl][..., 0])
    Tt[sl][..., 1] = np.where(m, uy, Tt[sl][..., 1])
    cov[sl] = np.maximum(cov[sl], alpha)


def render_photo_preview(pattern, path, px_per_mm=22, thread_mm=0.42, fabric="creme",
                         margin_mm=6, seed=7, shine=0.30):
    """Écrit un PNG simulant la broderie terminée sur `fabric` (nom de FABRICS ou '#rrggbb')."""
    if len(pattern.stitches) < 2:
        raise ValueError("motif vide")
    rng = np.random.default_rng(seed)
    ppm = px_per_mm
    x0, y0, x1, y1 = stitch_bbox_mm(pattern)
    W = int((x1 - x0 + 2 * margin_mm) * ppm)
    H = int((y1 - y0 + 2 * margin_mm) * ppm)
    to_px = lambda x, y: ((x / 10 - x0 + margin_mm) * ppm, (y / 10 - y0 + margin_mm) * ppm)

    fab_col, fab_h = _fabric(H, W, ppm, _rgb(fabric), rng)

    r = thread_mm * ppm / 2
    thick = r * 0.9
    Ht = np.zeros((H, W), np.float32)
    Ct = np.zeros((H, W, 3), np.float32)
    Tt = np.zeros((H, W, 2), np.float32)
    cov = np.zeros((H, W), np.float32)

    # -- 1) poser les points dans l'ordre où la machine les pique ---------------------------------
    threads = [_lin(np.array([(t.color >> 16) & 255, (t.color >> 8) & 255, t.color & 255], np.float32) / 255.0)
               for t in pattern.threadlist] or [_lin(np.array([.8, .1, .1], np.float32))]
    ci, last = 0, None
    pen = np.zeros((H, W), np.float32)  # piqûres de l'aiguille
    for x, y, cmd in pattern.stitches:
        c = cmd & pe.COMMAND_MASK
        cur = to_px(x, y)
        if c == pe.STITCH and last is not None:
            _draw_stitch(Ht, Ct, Tt, cov, last, cur, threads[min(ci, len(threads) - 1)], r, thick,
                         twist_px=0.9 * ppm, rng=rng)
            ix, iy = int(cur[0]), int(cur[1])
            if 0 <= ix < W and 0 <= iy < H:
                pen[iy, ix] += 1
        elif c == pe.COLOR_CHANGE:
            ci += 1
        last = cur

    # Le fil est tiré dans le tissu à chaque piqûre : des piqûres alignées d'une rangée à l'autre creusent
    # un sillon visible (c'est ce qui dessine les remplissages en relief) ; dispersées, un simple grain.
    sig = 0.2 * ppm
    groove = cv2.GaussianBlur(pen, (0, 0), sig) * (2 * math.pi * sig * sig)
    Ht *= 1 - 0.62 * np.clip(groove, 0, 1.6) / 1.6

    # -- 2) éclairage des fils --------------------------------------------------------------------
    Hs = cv2.GaussianBlur(Ht, (0, 0), 0.6)
    nx, ny, nz = _normals(Hs, 1.0)
    Lx, Ly, Lz = _LIGHT
    diffuse = 0.28 + 0.95 * np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1)
    tl = Tt[..., 0] * Lx + Tt[..., 1] * Ly                           # angle fil <-> lumière (Kajiya-Kay)
    sheen = np.power(np.sqrt(np.clip(1 - tl * tl, 0, 1)), 24) * nz ** 10
    ao = np.clip(1 - 1.3 * np.maximum(cv2.GaussianBlur(Ht, (0, 0), 0.5 * ppm) - Ht, 0) / thick, 0.40, 1)
    spec_tint = 0.70 * Ct + 0.30
    thread = Ct * (diffuse * ao)[..., None] + shine * (sheen * (0.35 + 0.65 * ao))[..., None] * spec_tint

    # -- 3) éclairage du tissu + ombres portées des fils ------------------------------------------
    fx, fy, fz = _normals(fab_h, 1.0)
    fab = fab_col * (0.50 + 0.62 * np.clip(fx * Lx + fy * Ly + fz * Lz, 0, 1))[..., None]
    occ = np.clip(Ht / thick, 0, 1)
    M = np.float32([[1, 0, 0.16 * ppm], [0, 1, 0.22 * ppm]])
    soft = cv2.GaussianBlur(cv2.warpAffine(occ, M, (W, H)), (0, 0), 0.20 * ppm)
    contact = cv2.GaussianBlur(occ, (0, 0), 0.07 * ppm)
    shadow = np.clip(0.60 * soft + 0.40 * contact, 0, 0.88)
    fab = fab * (1 - shadow * (1 - cov))[..., None]

    # -- 4) composition + "effet appareil photo" ---------------------------------------------------
    out = fab * (1 - cov[..., None]) + thread * cov[..., None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2
    out *= (1 - 0.22 * rr)[..., None]                                 # léger vignettage
    out = _srgb(out * 1.08)
    out += rng.normal(0, 0.008, out.shape).astype(np.float32)        # grain
    out = cv2.GaussianBlur(out, (0, 0), 0.5)
    img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path
