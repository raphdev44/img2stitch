"""Vérifie que les rangées de remplissage tombent bien DANS la forme, quel que soit l'angle."""
import cv2
import numpy as np
from img2stitch.fill import fill_runs


def test_fill_points_inside_mask():
    mask = np.zeros((300, 500), bool)
    mask[40:120, 300:460] = True          # rectangle excentré en haut à droite
    tol = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    for angle in (0, 30, 45, 90, 135):
        runs = fill_runs(mask, 10, 0.4, angle, 0.1)
        assert runs, angle
        pts = np.array([r[k] for r in runs for k in ("p0", "p1")]) * 10
        xs, ys = pts[:, 0].round().astype(int), pts[:, 1].round().astype(int)
        inside = tol[np.clip(ys, 0, 299), np.clip(xs, 0, 499)]
        # tolérance de 2 px : les extrémités des rangées sont sur le bord de la forme
        assert inside.mean() > 0.95, (angle, inside.mean())
