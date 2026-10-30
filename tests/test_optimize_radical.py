import json

import pyembroidery as pe

from img2stitch.radical import optimize_pattern_radical
from webapp import create_app

MASK = pe.COMMAND_MASK
RED, GREEN = 0xcc2222, 0x22aa44


def _rows(x0, y0, x1, y1, spacing=4, step=30):
    """Rangées horizontales en boustrophédon (coordonnées en 0,1 mm)."""
    rows, y, forward = [], y0, True
    while y <= y1:
        xs = list(range(x0, x1 + 1, step))
        if xs[-1] != x1:
            xs.append(x1)
        if not forward:
            xs.reverse()
        rows.append([(x, y) for x in xs])
        forward = not forward
        y += spacing
    return rows


def _pattern(blocks, colors=()):
    """Motif : une suite de rangées par bloc, un changement de fil entre deux blocs."""
    pattern = pe.EmbPattern()
    stitches = []
    for index, rows in enumerate(blocks):
        if index:
            first = rows[0][0]
            stitches.append([first[0], first[1], pe.COLOR_CHANGE])
        for number, row in enumerate(rows):
            if number:
                last = stitches[-1]
                stitches.append([last[0], last[1], pe.TRIM])
                stitches.append([row[0][0], row[0][1], pe.JUMP])
            stitches.extend([[x, y, pe.STITCH] for x, y in row])
    pattern.stitches = stitches
    for color in colors:
        pattern.add_thread({"color": color})
    return pattern


def _count(pattern, command):
    return sum(1 for stitch in pattern.stitches if stitch[2] & MASK == command)


def _stitches(pattern):
    return [stitch for stitch in pattern.stitches if stitch[2] & MASK == pe.STITCH]


def test_does_not_mutate_input_and_returns_a_copy():
    pattern = _pattern([_rows(0, 0, 300, 120)], [RED])
    before = [list(stitch) for stitch in pattern.stitches]

    optimized, report = optimize_pattern_radical(pattern)

    assert pattern.stitches == before
    assert optimized is not pattern
    assert report["mode"] == "radical"
    assert report["before"]["stitches"] == sum(1 for stitch in before
                                               if stitch[2] & MASK == pe.STITCH)


def test_reduces_stitch_count_without_changing_appearance():
    pattern = _pattern([_rows(0, 0, 400, 400)], [RED])

    optimized, report = optimize_pattern_radical(pattern)

    assert len(_stitches(optimized)) < len(_stitches(pattern))
    assert report["simplified_points"] > 0
    assert report["appearance_change_pct"] <= 2
    assert report["coverage_iou"] >= 0.97
    longest = max(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                  for stitch in _split(optimized) for a, b in zip(stitch, stitch[1:]))
    assert longest <= 120


def _split(pattern):
    runs, current = [], []
    for stitch in pattern.stitches:
        if stitch[2] & MASK == pe.STITCH:
            current.append(stitch)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def test_merges_blocks_of_the_same_thread():
    # rouge, vert, rouge : les deux blocs rouges ne se recouvrent pas, ils peuvent être réunis
    pattern = _pattern([_rows(0, 0, 120, 120), _rows(200, 0, 320, 120),
                        _rows(0, 200, 120, 320)], [RED, GREEN, RED])

    optimized, report = optimize_pattern_radical(pattern)

    assert _count(optimized, pe.COLOR_CHANGE) == 1
    assert [thread.color & 0xFFFFFF for thread in optimized.threadlist] == [RED, GREEN] \
        or [thread.color & 0xFFFFFF for thread in optimized.threadlist] == [GREEN, RED]
    assert report["color_changes_saved"] == 1
    assert report["reordered"] is True


def test_keeps_overlapping_blocks_in_their_original_order():
    # le vert est brodé sur le rouge : il doit rester au-dessus
    pattern = _pattern([_rows(0, 0, 400, 400), _rows(150, 150, 250, 250)], [RED, GREEN])

    optimized, report = optimize_pattern_radical(pattern)

    assert [thread.color & 0xFFFFFF for thread in optimized.threadlist] == [RED, GREEN]
    assert report["appearance_change_pct"] <= 1


def test_removes_stitches_hidden_under_a_later_color():
    red = _rows(0, 0, 400, 300)
    green = _rows(150, 100, 250, 200)     # 10 x 10 mm posés sur le rouge
    pattern = _pattern([red, green], [RED, GREEN])

    def inside(points):
        return [point for point in points if 170 < point[0] < 230 and 120 < point[1] < 180]

    before = len(inside(_stitches(pattern)))
    optimized, report = optimize_pattern_radical(pattern)
    after = len(inside(_stitches(optimized)))

    assert report["removed_hidden_stitches"] > 0
    assert before > 0
    assert after < before * 0.5
    assert report["appearance_change_pct"] <= 1


def test_keeps_color_changes_when_thread_colors_are_unknown():
    pattern = _pattern([_rows(0, 0, 200, 100), _rows(300, 0, 500, 100),
                        _rows(600, 0, 800, 100)], [])

    optimized, report = optimize_pattern_radical(pattern)

    assert len(pattern.threadlist) == 0
    assert _count(optimized, pe.COLOR_CHANGE) == 2
    assert report["color_changes_saved"] == 0
    assert report["warnings"]


def test_gives_up_on_operator_stops_without_touching_the_pattern():
    pattern = _pattern([_rows(0, 0, 200, 100)], [RED, GREEN])
    # deux changements de fil consécutifs = arrêt opérateur : on ne doit pas les fusionner
    pattern.stitches[1:1] = [[0, 0, pe.COLOR_CHANGE], [0, 0, pe.COLOR_CHANGE]]
    before = [list(stitch) for stitch in pattern.stitches]

    optimized, report = optimize_pattern_radical(pattern)

    assert report["warnings"]
    assert optimized.stitches == before


def test_remaps_zone_starts_after_reordering():
    pattern = _pattern([_rows(0, 0, 300, 100), _rows(400, 0, 700, 100)], [RED, GREEN])
    second = next(index for index, stitch in enumerate(pattern.stitches)
                  if stitch[2] & MASK == pe.COLOR_CHANGE)
    pattern.img2stitch_segments = [(0, 0), (second, 1)]

    optimized, report = optimize_pattern_radical(pattern)

    segments = optimized.img2stitch_segments
    assert [zone for _, zone in segments] == [0, 1]
    for index, _ in segments:
        assert 0 <= index < len(optimized.stitches)
        assert optimized.stitches[index][2] & MASK == pe.STITCH


def test_web_option_rejects_unknown_optimization(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    design_id = "b" * 32
    design_dir = tmp_path / design_id
    design_dir.mkdir()
    (design_dir / "meta.json").write_text(json.dumps({
        "id": design_id, "name": "Motif de test", "created_at": "2026-10-29T12:00:00+00:00",
        "status": "ready", "mode": "classic",
    }), encoding="utf-8")

    response = app.test_client().post(f"/api/motifs/{design_id}/optimiser",
                                      json={"mode": "turbo"})

    assert response.status_code == 409
    assert "optimisation" in response.json["error"].lower()
