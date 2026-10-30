"""Numérisation par vectorisation : géométrie, remplissage, intégration web."""
import io
import math
import time

import numpy as np
import pyembroidery as pe
import pytest
from PIL import Image, ImageDraw

import img2stitch.vectorize as vectorize
from img2stitch import Settings
from img2stitch.convert import image_to_pattern, pattern_stats, pattern_to_image
from webapp import create_app

PPM = 10
MASK = pe.COMMAND_MASK


def _ring_mask(size=120, outer=50, inner=25):
    yy, xx = np.mgrid[0:size, 0:size]
    d2 = (xx - size / 2) ** 2 + (yy - size / 2) ** 2
    return (d2 < outer ** 2) & (d2 > inner ** 2)


def _pattern_image(tmp_path, draw, size=(160, 160), name="motif.png"):
    image = Image.new("RGB", size, "white")
    draw(ImageDraw.Draw(image))
    path = tmp_path / name
    image.save(path)
    return str(path)


# --- Géométrie ---------------------------------------------------------------------------------
def test_vectorized_ring_keeps_its_hole():
    polygons = vectorize.polygons_of(_ring_mask(), PPM, 0.0, 0.3)

    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1
    area = polygons[0].area
    expected = math.pi * (50 ** 2 - 25 ** 2) / PPM ** 2
    assert 0.9 * expected < area < 1.1 * expected


def test_pull_compensation_expands_the_shape():
    mask = _ring_mask()
    plain = vectorize.polygons_of(mask, PPM, 0.0, 0.3)[0]
    grown = vectorize.polygons_of(mask, PPM, 0.25, 0.3)[0]

    assert grown.area > plain.area
    for side_plain, side_grown in zip(plain.bounds, grown.bounds):
        assert abs(side_grown - side_plain) == pytest.approx(0.25, abs=0.06)


def test_hatch_rows_avoid_the_hole():
    polygons = vectorize.polygons_of(_ring_mask(), PPM, 0.25, 0.3)
    rows = vectorize.hatch_rows(polygons, 0.4, 45.0)
    center = (60 / PPM, 60 / PPM)

    assert len(rows) > 25
    assert max(math.dist(row["p0"], row["p1"]) for row in rows) > 8.0   # rangées qui manquent le trou
    for row in rows:
        (x0, y0), (x1, y1) = row["p0"], row["p1"]
        length = math.hypot(x1 - x0, y1 - y0)
        assert length >= vectorize.MIN_RUN_MM
        # aucun morceau de rangée ne traverse le trou central
        for t in np.linspace(0, 1, 7):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            assert math.hypot(x - center[0], y - center[1]) > 2.5 - 0.35


def test_underlay_runs_inside_the_shape():
    polygons = vectorize.polygons_of(_ring_mask(), PPM, 0.25, 0.3)
    loops = vectorize.underlay_loops(polygons)

    assert loops
    for loop in loops:
        xs, ys = zip(*loop)
        assert min(xs) > min(y for y in polygons[0].bounds) + 0.3


def test_aligned_rows_are_merged():
    rows = [{"p0": (0.0, 0.0), "p1": (2.0, 0.0), "comp": 0, "row": 0, "line": 0},
            {"p0": (2.05, 0.0), "p1": (4.0, 0.0), "comp": 0, "row": 1, "line": 0},
            {"p0": (0.0, 1.0), "p1": (2.0, 1.0), "comp": 0, "row": 2, "line": 1}]

    merged = vectorize.merge_runs(rows)

    assert len(merged) == 2
    joined = max(merged, key=lambda r: math.dist(r["p0"], r["p1"]))
    assert math.dist(joined["p0"], joined["p1"]) == pytest.approx(4.0)


def test_crossing_rows_are_not_merged():
    rows = [{"p0": (0.0, 0.0), "p1": (2.0, 0.0), "comp": 0, "row": 0, "line": 0},
            {"p0": (2.0, 0.0), "p1": (2.0, 2.0), "comp": 0, "row": 1, "line": 1}]

    assert len(vectorize.merge_runs(rows)) == 2


# --- Moteur complet ----------------------------------------------------------------------------
def _disk(tmp_path):
    return _pattern_image(tmp_path, lambda d: d.ellipse((20, 20, 140, 140), fill=(220, 30, 30)))


def test_vector_engine_builds_a_complete_pattern(tmp_path):
    pattern = image_to_pattern(_disk(tmp_path), Settings(
        engine="vector", n_colors=1, background="auto", background_scope="outside", width_mm=40))

    stats = pattern_stats(pattern)
    assert stats["stitches"] > 100
    assert len(pattern.threadlist) == 1
    assert len(pattern.img2stitch_segments) == len(pattern.img2stitch_zones) == 1
    assert pattern.img2stitch_engine == "vector"
    assert pattern.img2stitch_frame["ppm"] == PPM


def test_vector_engine_does_not_stitch_the_hole(tmp_path):
    # anneau rouge : le centre blanc est du fond, il ne doit pas être brodé
    path = _pattern_image(tmp_path, lambda d: (d.ellipse((20, 20, 140, 140), fill=(220, 30, 30)),
                                               d.ellipse((60, 60, 100, 100), fill="white")))
    pattern = image_to_pattern(path, Settings(
        engine="vector", n_colors=1, background="auto", background_scope="all", width_mm=40))
    frame = pattern.img2stitch_frame
    width, height = 160, 160           # image de test : trou de 40 px de diamètre, centré
    stitches = [(x, y) for x, y, command in pattern.stitches if command & MASK == pe.STITCH]
    assert stitches

    for x, y in stitches:
        u, v = pattern_to_image(frame, x, y)
        assert math.hypot(u * width - 80, v * height - 80) > 17   # jamais dans le trou (rayon 20 px)


def test_standard_engine_is_unchanged(tmp_path):
    pattern = image_to_pattern(_disk(tmp_path), Settings(engine="standard", n_colors=1,
                                                         background="auto", width_mm=40))

    assert pattern_stats(pattern)["stitches"] > 100


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
        Settings(engine="turbo")


# --- Formulaire et API -------------------------------------------------------------------------
def test_creation_form_offers_both_engines(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})

    html = app.test_client().get("/").get_data(as_text=True)

    assert '<select id="scan-engine" name="engine">' in html
    assert 'value="standard" selected' in html
    assert 'value="vector"' in html
    assert "Par vectorisation (vpype)" in html


def test_vector_option_is_disabled_without_shapely(tmp_path, monkeypatch):
    from webapp import routes

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    monkeypatch.setattr(routes, "vectorize_available", lambda: False)

    html = app.test_client().get("/").get_data(as_text=True)

    assert 'value="vector"' in html
    assert "disabled" in html.split('value="vector"')[1].split(">")[0]
    assert "shapely" in html


def _wait(store, design_id, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        meta = store.get(design_id)
        if meta["status"] != "processing":
            return meta
        time.sleep(0.1)
    raise AssertionError("le motif n'a pas fini d'être généré")


def test_api_creates_a_vector_design(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    image = Image.new("RGB", (160, 160), "white")
    ImageDraw.Draw(image).ellipse((20, 20, 140, 140), fill=(220, 30, 30))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)

    response = app.test_client().post("/api/motifs", data={
        "image": (stream, "rond.png"), "background": "outside", "colors": "1",
        "width_mm": "40", "height_mm": "40", "engine": "vector",
    }, content_type="multipart/form-data")

    assert response.status_code == 202, response.get_json()
    assert response.json["engine"] == "vector"
    meta = _wait(app.extensions["designs"], response.json["id"])
    assert meta["status"] == "ready", meta.get("error")
    assert meta["engine"] == "vector"
    assert meta["stats"]["stitches"] > 100


def test_api_rejects_an_unknown_engine(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    stream = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(stream, "PNG")
    stream.seek(0)

    response = app.test_client().post("/api/motifs", data={
        "image": (stream, "x.png"), "background": "outside", "colors": "1",
        "width_mm": "40", "height_mm": "40", "engine": "turbo",
    }, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "numérisation" in response.json["error"]
