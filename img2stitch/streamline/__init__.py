"""Digitalisation de photos par streamlines.

Méthode de Zhenyuan Desmond Liu, Michal Piovarči, Christian Hafner, Raphaël Charrondière et Bernd
Bickel, « Directionality-Aware Design of Embroidery Patterns » (Computer Graphics Forum, 2023),
reprise de https://github.com/desmondlzy/embroidery-streamlines.

Le principe : un champ de direction tiré du dessin et un champ de densité tiré des couleurs
orientent un réseau de lignes de courant — les *streamlines*. Elles suivent les contours de la
photo, s'espacent selon la densité voulue, puis sont relâchées par un système de ressorts pour
former un remplissage lisse et régulier, impossible à obtenir avec des rangées parallèles.

Découpage :

- `fields`   : région -> champs de densité et de direction, bord et trous ;
- `tracing`  : sources, puits, tracé Runge-Kutta, coupe et rééchantillonnage ;
- `relax`    : ressorts d'alignement, d'espacement et de régularité (leur énergie, relaxée) ;
- `pattern`  : passages fond / motif, points machine et centrage.

Le dépôt d'origine s'appuie sur `triangle`, `scipy.sparse`, `networkx`, `numba`, `scikit-image`
et des annotations Labelme ; ici tout est réécrit avec numpy et OpenCV, les seules dépendances
déjà nécessaires à l'application, et la segmentation vient de notre quantification de couleurs.
"""
from .fields import GRID, WORK_PX, Patch, load_patches
from .pattern import DEFAULT_SIZE_MM, LINE_WIDTH_MM, STITCH_MM, streamline_pattern

__all__ = ["GRID", "WORK_PX", "Patch", "load_patches", "streamline_pattern",
           "DEFAULT_SIZE_MM", "LINE_WIDTH_MM", "STITCH_MM"]
