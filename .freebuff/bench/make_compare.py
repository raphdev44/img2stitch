import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, ".freebuff/bench")
from evaluate import load  # noqa: E402
from img2stitch import render_photo_preview, render_raw_preview  # noqa: E402
from img2stitch.radical import optimize_pattern_radical  # noqa: E402

cases = [
    ("50d894507a784a87939efbd0e0c75a7e", "instance/designs/50d894507a784a87939efbd0e0c75a7e/pattern.json"),
    ("exemple", "examples/exemple.pes"),
]
out = ".freebuff/bench/compare"
os.makedirs(out, exist_ok=True)
for name, path in cases:
    pattern = load(path)
    optimized, report = optimize_pattern_radical(pattern)
    render_raw_preview(pattern, f"{out}/{name}_avant.png")
    render_raw_preview(optimized, f"{out}/{name}_apres.png")
    render_photo_preview(pattern, f"{out}/{name}_photo_avant.png")
    render_photo_preview(optimized, f"{out}/{name}_photo_apres.png")
    print(name, report["before"]["stitches"], "->", report["after"]["stitches"], "points,",
          report["before"]["trims"], "->", report["after"]["trims"], "coupes, aspect",
          report["appearance_change_pct"], "%")
