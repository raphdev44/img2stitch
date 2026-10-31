"""Bout en bout : l'optimisation crée une copie, l'original reste intact."""
import hashlib
import json
import os
import shutil
import sys
import time

import pyembroidery as pe

sys.path.insert(0, os.getcwd())
from webapp.designs import FORMATS, DesignStore  # noqa: E402

ROOT = os.path.join(".freebuff", "bench", "store3")
shutil.rmtree(ROOT, ignore_errors=True)
store = DesignStore(ROOT, workers=1)


def wait(design_id, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        meta = store.get(design_id)
        if meta["status"] in ("ready", "error"):
            return meta
        time.sleep(0.2)
    raise RuntimeError("délai dépassé")


def digest(design_id):
    d = store.dir(design_id)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def summary(meta):
    stats = meta.get("stats") or {}
    report = meta.get("optimization_report") or {}
    return {"name": meta["name"], "status": meta["status"], "points": stats.get("stitches"),
            "trims": stats.get("trims"), "colors": stats.get("color_changes"),
            "mode": report.get("mode"), "from": meta.get("optimized_from"),
            "appearance_pct": report.get("appearance_change_pct"),
            "iou": report.get("coverage_iou"), "warnings": report.get("warnings")}


with open("examples/exemple.png", "rb") as stream:
    original = store.create(stream, "exemple.png", 3, "outside", (60, 60))
meta = wait(original)
print("origine  :", json.dumps(summary(meta), ensure_ascii=False))
before = digest(original)

simple = store.optimize(original, "simple")
meta = wait(simple)
print("simple   :", json.dumps(summary(meta), ensure_ascii=False))
print("  original inchangé :", digest(original) == before)
print("  copie identique à l'origine au départ ?", sorted(before) == sorted(digest(simple)))
d = store.dir(simple)
for fmt in FORMATS:
    pattern = pe.read(str(d / f"pattern.{fmt}"))
    stitches = sum(1 for s in pattern.stitches if s[2] & pe.COMMAND_MASK == pe.STITCH)
    assert stitches > 0, fmt
print("  exports relus :", {fmt: (d / f'pattern.{fmt}').stat().st_size for fmt in FORMATS})

radical = store.optimize(original, "radical")
meta = wait(radical)
print("radicale :", json.dumps(summary(meta), ensure_ascii=False))
print("  original inchangé :", digest(original) == before)
print("  noms galerie :", [(m["name"], len(m["threads"] or [])) for m in store.list()])
print("  tuiles :", len(store.list()))
