"""Chevauchement entre couleurs et extraction des contours."""
import cv2
import numpy as np


def overlap_masks(masks, px_per_mm, overlap_mm):
    """
    Les couleurs sont piquées dans l'ordre de la liste : chacune recouvre les précédentes.
    Pour éviter les jours de tissu, chaque masque est élargi, mais SEULEMENT là où une
    couleur posée plus tard viendra le recouvrir. Le bord extérieur du motif ne grossit pas.
    """
    r = round(overlap_mm * px_per_mm)
    if r <= 0:
        return [m.copy() for m in masks]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    later = np.zeros_like(masks[0])
    out = [None] * len(masks)
    for i in range(len(masks) - 1, -1, -1):  # on remonte : 'later' = union des masques suivants
        grown = cv2.dilate(masks[i].astype(np.uint8), kernel).astype(bool)
        out[i] = masks[i] | (grown & later)
        later = later | masks[i]
    return out


def contours_mm(mask, px_per_mm, simplify_mm, min_area_mm2):
    """Contours fermés (extérieurs ET trous) du masque, en liste de polylignes [(x_mm, y_mm), ...]."""
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    eps = max(0.5, simplify_mm * px_per_mm)
    result = []
    for c in found:
        if cv2.contourArea(c) / px_per_mm ** 2 < min_area_mm2 or len(c) < 4:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        # +0.5 : on vise le bord du pixel, pas son centre
        result.append([((x + 0.5) / px_per_mm, (y + 0.5) / px_per_mm) for x, y in c])
    return result
