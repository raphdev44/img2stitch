"""Compare le moteur standard et la numérisation par vectorisation (aspect + parcours)."""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".freebuff", "bench"))
from img2stitch import Settings                       # noqa: E402
from img2stitch.convert import image_to_pattern, pattern_stats  # noqa: E402
from img2stitch.photo import stitch_bbox_mm           # noqa: E402
from evaluate import appearance, compare, grid_of     # noqa: E402

ARGS = dict(n_colors=4, background="auto", background_scope="outside",
            max_width_mm=80, max_height_mm=80)


def coverage(img):
    return np.count_nonzero(img >= 0) / img.size


def build(path, engine):
    start = time.perf_counter()
    pattern = image_to_pattern(path, Settings(engine=engine, **ARGS))
    return pattern, time.perf_counter() - start


def main(paths):
    total = {"standard": 0, "vector": 0}
    for path in paths:
        try:
            standard, t1 = build(path, "standard")
            vector, t2 = build(path, "vector")
        except Exception as exc:  # noqa: BLE001
            print("ECHEC", os.path.basename(path), type(exc).__name__, exc)
            continue
        grid = grid_of([standard, vector])
        a_std, a_vec = appearance(standard, grid), appearance(vector, grid)
        s1, s2 = pattern_stats(standard), pattern_stats(vector)
        print("\n== %s" % os.path.basename(path))
        print("   standard : %5d points, %3d coupes, %3d sauts, %5.1f%% couverture, %.2fs" % (
            s1["stitches"], s1["trims"], s1["jumps"], 100 * coverage(a_std), t1))
        print("   vector   : %5d points, %3d coupes, %3d sauts, %5.1f%% couverture, %.2fs" % (
            s2["stitches"], s2["trims"], s2["jumps"], 100 * coverage(a_vec), t2))
        print("   écart standard->vector : %.2f%% | vector->standard : %.2f%%" % (
            100 * compare(a_std, a_vec), 100 * compare(a_vec, a_std)))
        print("   bbox std %s / vec %s" % ([round(float(v), 1) for v in stitch_bbox_mm(standard)],
                                           [round(float(v), 1) for v in stitch_bbox_mm(vector)]))
        total["standard"] += s1["stitches"]
        total["vector"] += s2["stitches"]
        if os.environ.get("SAVE"):
            side = Image.new("RGB", (a_std.shape[1] * 2 + 20, a_std.shape[0]), "white")
            for i, arr in enumerate((a_std, a_vec)):
                rgb = np.zeros(arr.shape + (3,), np.uint8)
                mask = arr >= 0
                rgb[mask] = np.stack([(arr >> 16) & 255, (arr >> 8) & 255, arr & 255], -1)[mask]
                rgb[~mask] = 255
                side.paste(Image.fromarray(rgb), (i * (a_std.shape[1] + 20), 0))
            name = os.path.splitext(os.path.basename(path))[0]
            side.save(os.path.join(".freebuff", "bench", "vec_%s.png" % name))
    print("\nTOTAL points standard %d / vector %d (%+.1f%%)" % (
        total["standard"], total["vector"],
        100 * (total["vector"] - total["standard"]) / max(1, total["standard"])))


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("examples/*.png"))[:6]
    main(files)
