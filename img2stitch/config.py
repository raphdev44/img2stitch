from dataclasses import dataclass


@dataclass
class Settings:
    """Paramètres de conversion. Toutes les longueurs sont en millimètres."""

    width_mm: float = 80.0          # largeur finale de l'IMAGE (ignorée si max_width_mm / max_height_mm est donné)
    max_width_mm: float = None      # le motif brodé tient strictement dans max_width_mm x max_height_mm,
    max_height_mm: float = None     # à la plus grande taille possible (l'un des deux peut rester None)
    n_colors: int = 4               # nombre de couleurs de fil, fond exclu
    row_spacing_mm: float = 0.4     # écart entre deux rangées de remplissage (densité)
    max_stitch_mm: float = 3.0      # longueur max d'un point (au-delà, on subdivise)
    fill_angle_deg: float = 45.0    # angle des rangées de remplissage
    min_area_mm2: float = 1.0       # ignore les taches plus petites (bruit)
    background: str = "auto"        # "auto", "none" ou "#RRGGBB"
    background_scope: str = "all"   # "all" : toute zone de la couleur du fond ; "outside" : seulement le fond relié aux bords
    background_tolerance: int = 40  # distance RGB pour considérer un pixel comme fond
    exclude_colors: tuple = ()      # couleurs à ne pas broder : ((r,g,b), ((r,g,b) des autres couleurs de la palette)), ...
    center_on_origin: bool = True   # centre le motif sur (0,0) = centre du cerceau, comme l'attendent les machines
    stagger: bool = True            # décale les points d'une rangée à l'autre (aspect "brique", plus régulier)

    # --- Chevauchement et contours ---
    overlap_mm: float = 0.4         # une couleur déborde de ce montant SOUS les couleurs posées après elle
    outline: bool = True            # piquer un contour par-dessus chaque zone
    outline_stitch_mm: float = 1.5  # longueur des points du contour (plus court = plus lisse dans les courbes)
    outline_passes: int = 2         # 1 = simple passage, 2 = aller-retour (trait plus épais et plus net)
    outline_simplify_mm: float = 0.15  # tolérance de simplification de la bordure (anti-dentelure)

    # --- Sauts, coupes de fil et points d'arrêt ---
    trim_threshold_mm: float = 3.0  # un saut plus long que ça est précédé d'une coupe de fil (TRIM)
    short_travel_mm: float = 1.0    # un déplacement plus court est toujours piqué (pas de saut)
    max_visible_travel_mm: float = 2.0  # un déplacement piqué qui resterait visible sur plus de ça devient un saut
    jump_penalty_mm: float = 8.0    # "coût" d'un saut dans l'optimisation : on préfère piquer jusqu'à 8 mm plus loin
    lock_stitches: bool = True      # points d'arrêt (3 micro-points) avant/après chaque coupe et changement de couleur
