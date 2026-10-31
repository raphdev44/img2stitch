import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from tests.test_optimize_radical import _pattern, _rows, RED, GREEN  # noqa: E402
from img2stitch import radical  # noqa: E402

pattern = _pattern([_rows(0, 0, 120, 120), _rows(200, 0, 320, 120),
                    _rows(0, 200, 120, 320)], [RED, GREEN, RED])
report = {"split_long_stitches": 0, "removed_hidden_stitches": 0, "warnings": []}
engine = radical._Radical(pattern, dict(radical.DEFAULTS), report)
print("blocs", [(block.label, len(block.runs), block.surface) for block in engine.blocks])
info = engine.build(reorder=True, hidden_travel=True, remove_hidden=True)
print("info", {k: v for k, v in info.items() if k not in ("stitches", "threads", "segments", "after")})
print("après", info["after"])
print("before paint pixels", int(np.count_nonzero(engine.before_paint >= 0)))
print("labels avant", {int(label): int(np.count_nonzero(engine.before_paint == label))
                       for label in np.unique(engine.before_paint[engine.before_paint >= 0])})
