"""Banc d'essai du moteur de vectorisation sur un corpus d'images réelles et synthétiques."""
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".freebuff", "bench"))
from img2stitch import Settings                        # noqa: E402
from img2stitch.convert import image_to_pattern, pattern_stats  # noqa: E402
from evaluate import appearance, compare, grid_of      # noqa: E402

ARGS = dict(n_colors=4, background="auto", background_scope="outside",
            max_width_mm=80, max_height_mm=80)
OUT = os.path.join(".freebuff", "bench", "corpus_images")


def synthetic():
    os.makedirs(OUT, exist_ok=True)
    path = lambda name: os.path.join(OUT, name)

    image = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(image)
    d.ellipse((40, 40, 360, 360), fill=(210, 40, 40))
    d.ellipse((150, 150, 250, 250), fill="white")          # trou
    image.save(path("anneau.png"))

    image = Image.new("RGB", (500, 300), "white")
    d = ImageDraw.Draw(image)
    d.text((20, 100), "BRODER", fill=(20, 20, 20))          # traits fins (police par défaut)
    image.save(path("texte.png"))

    image = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(image)
    for i in range(4):
        d.rectangle((60 + i * 70, 80, 90 + i * 70, 320), fill=(30, 60, 200))
    image.save(path("bandes.png"))

    image = Image.new("RGB", (300, 300), (240, 240, 240))
    d = ImageDraw.Draw(image)
    d.polygon([(150, 30), (280, 270), (20, 270)], fill=(20, 140, 80))
    d.ellipse((120, 180, 180, 240), fill=(250, 200, 20))
    image.save(path("triangle.png"))

    image = Image.new("RGB", (320, 320), "white")
    d = ImageDraw.Draw(image)
    d.ellipse((20, 20, 300, 300), fill=(120, 60, 160))
    image = image.filter(ImageFilter.GaussianBlur(6)).resize((320, 320))  # bords flous
    image.save(path("flou.png"))

    gradient = np.zeros((300, 300, 3), np.uint8)
    for y in range(300):
        gradient[y, :] = (y % 256, 120, max(0, 255 - y // 2))
    Image.fromarray(gradient).save(path("degrade.png"))
    return sorted(glob.glob(os.path.join(OUT, "*.png")))


def inputs():
    return sorted(glob.glob("instance/designs/*/source.png")) + ["examples/exemple.png"] + synthetic()


def run(path):
    out = {"path": path}
    try:
        t = time.perf_counter()
        std = image_to_pattern(path, Settings(engine="standard", **ARGS))
        out["standard"] = dict(pattern_stats(std), seconds=round(time.perf_counter() - t, 2))
        t = time.perf_counter()
        vec = image_to_pattern(path, Settings(engine="vector", **ARGS))
        out["vector"] = dict(pattern_stats(vec), seconds=round(time.perf_counter() - t, 2))
    except Exception as exc:  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    grid = grid_of([std, vec])
    a_std, a_vec = appearance(std, grid), appearance(vec, grid)
    out["appearance"] = dict(to_vector=round(100 * compare(a_std, a_vec), 2),
                             to_standard=round(100 * compare(a_vec, a_std), 2),
                             coverage_standard=round(100 * np.count_nonzero(a_std >= 0) / a_std.size, 1),
                             coverage_vector=round(100 * np.count_nonzero(a_vec >= 0) / a_vec.size, 1))
    out["threads"] = [len(std.threadlist), len(vec.threadlist)]
    return out


def main():
    rows = [run(path) for path in inputs()]
    for row in rows:
        name = os.path.basename(os.path.dirname(row["path"]))[:8] + "/" + os.path.basename(row["path"])
        if "error" in row:
            print("ECHEC %-28s %s" % (name, row["error"]))
            continue
        s, v, a = row["standard"], row["vector"], row["appearance"]
        print("%-30s std %5d pts %2d coupes %5.1fs | vec %5d pts %2d coupes %4.1fs | "
              "aspect %4.1f%%/%4.1f%% | couv %4.1f%%->%4.1f%% | fils %s" % (
                  name, s["stitches"], s["trims"], s["seconds"],
                  v["stitches"], v["trims"], v["seconds"],
                  a["to_standard"], a["to_vector"], a["coverage_standard"], a["coverage_vector"],
                  row["threads"]))
    ok = [r for r in rows if "error" not in r]
    if ok:
        print("\n%d images, %d échecs" % (len(rows), len(rows) - len(ok)))
        print("points : standard %d, vector %d (%+.1f%%)" % (
            sum(r["standard"]["stitches"] for r in ok), sum(r["vector"]["stitches"] for r in ok),
            100 * (sum(r["vector"]["stitches"] for r in ok) - sum(r["standard"]["stitches"] for r in ok))
            / max(1, sum(r["standard"]["stitches"] for r in ok))))
        print("temps  : standard %.1fs, vector %.1fs" % (
            sum(r["standard"]["seconds"] for r in ok), sum(r["vector"]["seconds"] for r in ok)))
        print("aspect : max %.1f%%" % max(max(r["appearance"]["to_vector"], r["appearance"]["to_standard"])
                                          for r in ok))
        with open(os.path.join(".freebuff", "bench", "corpus_images.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
