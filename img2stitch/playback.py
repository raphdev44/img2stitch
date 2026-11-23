"""Parcours compact d'un motif, pour l'aperçu animé (le « film » du piquage).

L'aperçu brut montre le résultat ; il ne dit rien de l'ordre. Ces données décrivent la suite
exacte des gestes de la machine — piqûres, sauts, coupes, changements de fil — dans un format
léger que le navigateur rejoue image par image : on voit l'aiguille avancer, sauter et couper.
Aucune simulation réaliste : chaque point vaut une unité de temps.
"""
import pyembroidery as pe

# Opérations du film (le navigateur applique le même codage)
SEW, JUMP, TRIM, COLOR_CHANGE, STOP = 0, 1, 2, 3, 4


def playback_data(pattern):
    """Parcours d'un motif : {"bbox", "threads", "ops", "counts"}.

    `bbox` vaut [x0, y0, largeur, hauteur] en 1/10 mm : les coordonnées des opérations sont
    relatives à son coin haut-gauche, ce qui garde des entiers courts.
    Chaque opération est [type, index de fil, x, y, x, y, ...] : une polyligne pour une piqûre
    ou une suite de sauts, aucun point pour une coupe ou un changement de fil.
    """
    stitches = pattern.stitches
    points = [(x, y) for x, y, c in stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)]
    threads = ["#%06x" % (t.color & 0xFFFFFF) for t in pattern.threadlist] or ["#333333"]
    counts = {"stitches": 0, "jumps": 0, "trims": 0, "color_changes": 0, "stops": 0}
    if not points:
        return {"bbox": [0, 0, 0, 0], "threads": threads, "ops": [], "counts": counts}

    x0 = int(round(min(x for x, _ in points)))
    y0 = int(round(min(y for _, y in points)))
    w = int(round(max(x for x, _ in points))) - x0
    h = int(round(max(y for _, y in points))) - y0

    ops, color, cur = [], 0, None
    for x, y, command in stitches:
        kind = command & pe.COMMAND_MASK
        if kind not in (pe.STITCH, pe.JUMP):
            if kind == pe.TRIM:
                counts["trims"] += 1
                ops.append([TRIM, color])
            elif kind == pe.STOP:
                counts["stops"] += 1
                ops.append([STOP, color])
            elif kind == pe.COLOR_CHANGE:
                color += 1
                counts["color_changes"] += 1
                ops.append([COLOR_CHANGE, color])
            continue

        pt = (int(round(x)) - x0, int(round(y)) - y0)
        if kind == pe.STITCH:
            counts["stitches"] += 1
            if ops and ops[-1][0] == SEW and ops[-1][1] == color:
                ops[-1].extend(pt)
            elif cur is None:  # tout premier point du motif : pas encore d'aiguille posée
                ops.append([SEW, color, *pt])
            else:
                ops.append([SEW, color, *cur, *pt])
            cur = pt
        else:
            counts["jumps"] += 1
            if ops and ops[-1][0] == JUMP and ops[-1][1] == color:
                ops[-1].extend(pt)
            else:
                ops.append([JUMP, color, *pt])
            cur = pt

    return {"bbox": [x0, y0, w, h], "threads": threads, "ops": ops, "counts": counts}
