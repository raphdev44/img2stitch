"""Bout en bout : création d'un motif, optimisation simple puis radicale, exports."""
import json
import os
import shutil
import sys
import time

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from webapp.designs import FORMATS, DesignStore  # noqa: E402

ROOT = os.path.join(".freebuff", "bench", "store")
shutil.rmtree(ROOT, ignore_errors=True)
store = DesignStore(ROOT, workers=1)


def wait(design_id, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        meta = store.get(design_id)
        if meta["status"] in ("ready", "error"):
            return meta
        time.sleep(0.2)
    raise RuntimeError("délai dépassé")


def summary(meta):
    stats = meta.get("stats") or {}
    return {
        "status": meta["status"], "points": stats.get("stitches"), "trims": stats.get("trims"),
        "colors": stats.get("color_changes"), "threads": len(meta.get("threads") or []),
        "lengths": (meta.get("thread_lengths") or [])[:4],
        "report": {key: (meta.get("optimization_report") or {}).get(key) for key in
                   ("mode", "reordered", "removed_hidden_stitches", "hidden_travels_merged",
                    "appearance_change_pct", "coverage_iou")},
        "warnings": (meta.get("optimization_report") or {}).get("warnings"),
        "edit_error": meta.get("edit_error"),
    }


with open("examples/exemple.png", "rb") as stream:
    design_id = store.create(stream, "exemple.png", 3, "outside", (60, 60))
meta = wait(design_id)
print("création :", json.dumps(summary(meta), ensure_ascii=False))

store.optimize(design_id, "simple")
meta = wait(design_id)
print("simple   :", json.dumps(summary(meta), ensure_ascii=False))
print("  fichiers :", sorted(os.listdir(store.dir(design_id)))[:12])

store.optimize(design_id, "radical")
meta = wait(design_id)
print("radicale :", json.dumps(summary(meta), ensure_ascii=False))

directory = store.dir(design_id)
for fmt in FORMATS:
    path = directory / f"pattern.{fmt}"
    pattern = pe.read(str(path))
    stitches = sum(1 for s in pattern.stitches if s[2] & pe.COMMAND_MASK == pe.STITCH)
    print("  %-4s %6d points, %2d fils" % (fmt, stitches, len(pattern.threadlist)))

with open(directory / "pattern.json", encoding="utf-8") as handle:
    data = json.load(handle)
print("  pattern.json : %d points, %d segments" % (len(data["stitches"]),
                                                   len(data.get("segments") or [])))
print("  aperçus :", {name: (directory / name).stat().st_size for name in
                      ("preview_raw.png", "preview_photo.png", "thumb.jpg", "zones.png")})
print("  fil total (m) :", sum(meta.get("thread_lengths") or []))
