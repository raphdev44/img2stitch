import io
import json

import numpy as np
import pyembroidery as pe
import pytest
from PIL import Image, ImageDraw

from img2stitch.cross_stitch import (_grid_lines, _limit_cell_colors, _prepare_cells, cross_stitch_palettes,
                                    cross_stitch_pattern, read_cells)
from webapp import create_app


def _source(image):
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)
    return stream


def test_native_pixels_are_crosses():
    image = Image.new("RGB", (3, 3), "white")
    image.putpixel((1, 1), (255, 0, 0))
    pattern = cross_stitch_pattern(_source(image))
    stitches = [st for st in pattern.stitches if st[2] & pe.COMMAND_MASK == pe.STITCH]
    assert len(stitches) == 2
    assert pattern.img2stitch_grid["crosses"] == 1
    assert [(st[0], st[1]) for st in stitches] == [(10, 10), (-10, 10)]
    assert len(pattern.threadlist) == 1


def test_printed_grid_is_sampled_without_grid_lines():
    image = Image.new("RGB", (121, 121), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((41, 41, 59, 59), fill="red")
    for position in range(0, 121, 20):
        draw.line((position, 0, position, 120), fill=(160, 160, 160))
        draw.line((0, position, 120, position), fill=(160, 160, 160))
    cells = read_cells(_source(image))
    assert cells.shape == (6, 6, 4)
    pattern = cross_stitch_pattern(_source(image))
    assert pattern.img2stitch_grid["crosses"] == 1
    assert pattern.threadlist[0].color & 0xFFFFFF == 0xFF0000


@pytest.fixture
def block_art():
    native = Image.new("RGB", (32, 32), "white")
    draw = ImageDraw.Draw(native)
    draw.polygon([(3, 12), (5, 12), (5, 8), (9, 8), (9, 5), (23, 5),
                  (23, 8), (27, 8), (27, 12), (29, 12), (29, 24),
                  (25, 24), (25, 27), (7, 27), (7, 24), (3, 24)], fill="black")
    draw.rectangle((7, 9, 24, 13), fill=(225, 160, 65))
    draw.rectangle((5, 15, 26, 16), fill=(135, 195, 55))
    draw.rectangle((7, 18, 24, 20), fill=(100, 55, 55))
    draw.rectangle((9, 22, 22, 25), fill=(225, 160, 65))
    draw.point((11, 10), fill=(255, 240, 180))
    draw.point((19, 11), fill=(255, 240, 180))
    return native


@pytest.mark.parametrize("side, image_format", [(96, "PNG"), (448, "PNG"), (600, "PNG"), (600, "JPEG")])
def test_enlarged_pixel_art_without_grid(block_art, side, image_format):
    image = block_art.resize((side, side), Image.Resampling.NEAREST)
    stream = io.BytesIO()
    image.save(stream, image_format)
    stream.seek(0)
    cells = read_cells(stream)
    assert cells.shape == (32, 32, 4)
    assert np.allclose(cells[:, :, :3], np.array(block_art), atol=0 if image_format == "PNG" else 8)
    stream.seek(0)
    assert cross_stitch_pattern(stream).img2stitch_grid == cross_stitch_pattern(block_art).img2stitch_grid


def test_enlarged_blocks_with_unaligned_margins(block_art):
    image = Image.new("RGB", (468, 468), "white")
    image.paste(block_art.resize((448, 448), Image.Resampling.NEAREST), (7, 9))
    expected = Image.new("RGB", (34, 34), "white")
    expected.paste(block_art, (1, 1))
    cells = read_cells(image)
    assert cells.shape == (34, 34, 4)
    np.testing.assert_array_equal(cells[:, :, :3], np.array(expected))
    assert cross_stitch_pattern(image).img2stitch_grid["crosses"] == cross_stitch_pattern(block_art).img2stitch_grid["crosses"]


def test_enlarged_transparent_blocks(block_art):
    pixels = np.array(block_art.convert("RGBA"))
    pixels[np.all(pixels[:, :, :3] == 255, axis=2)] = (0, 0, 0, 0)
    image = Image.fromarray(pixels).resize((600, 600), Image.Resampling.NEAREST)
    np.testing.assert_array_equal(read_cells(image), pixels)
    assert cross_stitch_pattern(image, "keep").img2stitch_grid["crosses"] == cross_stitch_pattern(block_art).img2stitch_grid["crosses"]


def test_non_square_blocks_are_rejected(block_art):
    image = block_art.resize((320, 640), Image.Resampling.NEAREST)
    with pytest.raises(ValueError, match="carrées"):
        read_cells(image)


def test_unstructured_large_image_is_not_treated_as_blocks():
    pixels = np.random.default_rng(42).integers(0, 256, (600, 600, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Quadrillage"):
        read_cells(Image.fromarray(pixels))


@pytest.mark.parametrize("side, image_format", [(768, "PNG"), (768, "JPEG"), (1080, "JPEG")])
def test_dense_grid_with_dark_borders_and_footer(side, image_format):
    native = Image.new("RGB", (32, 32), "white")
    draw = ImageDraw.Draw(native)
    draw.ellipse((4, 3, 27, 29), fill=(20, 20, 20))
    draw.ellipse((5, 4, 26, 28), fill=(225, 35, 25))
    draw.ellipse((5, 11, 18, 24), fill=(20, 20, 20))
    draw.ellipse((6, 12, 17, 23), fill=(55, 155, 210))
    draw.rectangle((8, 13, 9, 14), fill="white")
    image = native.resize((side, side), Image.Resampling.NEAREST)
    pixels = np.array(image)
    boundaries = np.minimum(side - 1, np.rint(np.linspace(0, side, 33)).astype(int))
    pixels[boundaries] = (pixels[boundaries].astype(float) * 0.65).astype(np.uint8)
    pixels[:, boundaries] = (pixels[:, boundaries].astype(float) * 0.65).astype(np.uint8)
    image = Image.fromarray(pixels)
    ImageDraw.Draw(image).text((side - 200, side - 24), "pixel art", fill=(40, 40, 40))
    stream = io.BytesIO()
    image.save(stream, image_format)
    stream.seek(0)
    cells = read_cells(stream)
    assert cells.shape == (32, 32, 4)
    assert np.allclose(cells[:, :, :3], np.array(native), atol=0 if image_format == "PNG" else 8)


def test_grid_reconstructs_lines_hidden_by_dark_regions():
    pixels = np.full((321, 321, 3), 255, dtype=np.uint8)
    pixels[:, ::20] = 160
    pixels[:, 100:221] = 25
    pixels[:, 73] = 30
    result = _grid_lines(pixels, 0)
    assert result is not None
    lines, pitch = result
    assert pitch == pytest.approx(20)
    assert lines == pytest.approx(np.arange(0, 321, 20))


def test_grid_detects_lines_over_colored_pixels():
    pixels = np.full((321, 321, 3), (80, 160, 220), dtype=np.uint8)
    pixels[:, ::20] = (pixels[:, ::20].astype(float) * 0.9).astype(np.uint8)
    pixels[::20, :] = (pixels[::20, :].astype(float) * 0.9).astype(np.uint8)

    result = _grid_lines(pixels, 0)

    assert result is not None
    lines, pitch = result
    assert pitch == pytest.approx(20)
    assert lines == pytest.approx(np.arange(0, 321, 20))


def test_background_modes_preserve_enclosed_white():
    pixels = np.full((5, 5, 3), 255, dtype=np.uint8)
    pixels[1:4, 1:4] = (255, 0, 0)
    pixels[2, 2] = 255
    image = Image.fromarray(pixels)
    for mode, crosses in [("outside", 9), ("all", 8), ("keep", 25)]:
        assert cross_stitch_pattern(_source(image), mode).img2stitch_grid["crosses"] == crosses


def test_transparent_pixels_are_never_stitched():
    image = Image.new("RGBA", (2, 2))
    image.putpixel((0, 0), (0, 80, 255, 255))
    pattern = cross_stitch_pattern(_source(image), "keep")
    assert pattern.img2stitch_grid["crosses"] == 1
    pattern = cross_stitch_pattern(_source(image), "outside")
    assert pattern.img2stitch_grid["crosses"] == 1


def test_native_png_preserves_similar_colors():
    image = Image.new("RGB", (2, 1), (255, 0, 0))
    image.putpixel((1, 0), (245, 0, 0))
    pattern = cross_stitch_pattern(_source(image), "keep")
    assert len(pattern.threadlist) == 2


@pytest.mark.parametrize("background", ["outside", "all", "keep"])
def test_excluded_colors_use_nearest_remaining_color(background):
    image = Image.new("RGBA", (6, 4), "white")
    colors = [(250, 20, 20), (200, 20, 20), (20, 20, 240), (20, 20, 200)]
    for column, color in enumerate(colors, 1):
        image.putpixel((column, 1), (*color, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    excluded = [(colors[index], tuple(color for color in colors if color != colors[index]))
                for index in (0, 2)]
    _, palette, labels = _prepare_cells(image, background, exclude=excluded)
    assert [palette[labels[1, column]] for column in range(1, 5)] == [
        colors[1], colors[1], colors[3], colors[3]]
    assert labels[0, 0] == -1
    assert (labels[0, 1] == -1) == (background != "keep")
    assert cross_stitch_pattern(image, background, exclude=excluded).img2stitch_grid == (
        cross_stitch_pattern(image, background).img2stitch_grid)


def test_excluded_color_cannot_be_replaced_by_removed_background():
    image = Image.new("RGB", (5, 5), "white")
    image.putpixel((1, 1), (240, 240, 240))
    image.putpixel((2, 1), (0, 0, 0))
    excluded = [((240, 240, 240), ((0, 0, 0),))]
    _, palette, labels = _prepare_cells(image, "outside", exclude=excluded)
    assert palette[labels[1, 1]] == (0, 0, 0)
    assert labels[0, 0] == -1


def test_excluding_all_colors_is_rejected():
    image = Image.new("RGB", (2, 1), "red")
    excluded = [((255, 0, 0), ())]
    for function in (cross_stitch_palettes, cross_stitch_pattern):
        with pytest.raises(ValueError, match="au moins une couleur"):
            function(image, "keep", exclude=excluded)


def test_color_reduction_preserves_crosses_and_size():
    image = Image.new("RGB", (5, 5), "white")
    image.putpixel((1, 1), (255, 0, 0))
    image.putpixel((2, 1), (240, 20, 0))
    image.putpixel((3, 1), (0, 0, 255))
    original = cross_stitch_pattern(_source(image))
    reduced = cross_stitch_pattern(_source(image), n_colors=2)
    assert len(original.threadlist) == 3
    assert len(reduced.threadlist) == 2
    assert original.img2stitch_grid == reduced.img2stitch_grid
    assert original.bounds() == reduced.bounds()
    assert sum(st[2] == pe.STITCH for st in reduced.stitches) == 6


def test_recolor_merges_threads_without_changing_crosses():
    image = Image.new("RGB", (2, 1), "red")
    image.putpixel((1, 0), (0, 0, 255))
    pattern = cross_stitch_pattern(_source(image), "keep", color_overrides=[((255, 0, 0), (0, 0, 255))])
    assert len(pattern.threadlist) == 1
    assert pattern.threadlist[0].color & 0xFFFFFF == 0x0000FF
    assert pattern.img2stitch_grid["crosses"] == 2
    assert set(pattern.img2stitch_source_palette) == {(255, 0, 0), (0, 0, 255)}


def test_suggested_palettes_match_generated_threads():
    image = Image.new("RGB", (5, 5), "white")
    image.putpixel((1, 1), (255, 0, 0))
    image.putpixel((2, 1), (0, 0, 255))
    image.putpixel((3, 1), (255, 255, 0))
    analysis = cross_stitch_palettes(_source(image))
    assert analysis["colors"] == 3
    assert set(analysis["palettes"]) == set(range(1, 13))
    for count, palette in analysis["palettes"].items():
        pattern = cross_stitch_pattern(_source(image), n_colors=count)
        actual = [((thread.color >> 16) & 255, (thread.color >> 8) & 255, thread.color & 255)
                  for thread in pattern.threadlist]
        assert actual == palette


def test_excluded_palette_color_is_replaced_without_losing_crosses():
    image = Image.new("RGB", (5, 5), "white")
    image.putpixel((1, 1), (255, 0, 0))
    image.putpixel((2, 1), (0, 0, 255))
    excluded = [((255, 0, 0), ((0, 0, 255),))]
    analysis = cross_stitch_palettes(_source(image), exclude=excluded)
    assert analysis["colors"] == 1
    assert analysis["palettes"][2] == [(0, 0, 255)]
    pattern = cross_stitch_pattern(_source(image), n_colors=2, exclude=excluded)
    original = cross_stitch_pattern(image)
    assert pattern.img2stitch_grid == original.img2stitch_grid
    assert sorted(tuple(stitch[:2]) for stitch in pattern.stitches if stitch[2] == pe.STITCH) == sorted(
        tuple(stitch[:2]) for stitch in original.stitches if stitch[2] == pe.STITCH)
    assert len(pattern.threadlist) == 1
    assert pattern.threadlist[0].color & 0xFFFFFF == 0x0000FF


@pytest.mark.parametrize("count", [0, -1, 257, 1.5, True])
def test_invalid_color_limits_are_rejected(count):
    with pytest.raises(ValueError, match="nombre de couleurs"):
        cross_stitch_pattern(_source(Image.new("RGB", (2, 1), "red")), "keep", n_colors=count)


def test_only_long_cross_travel_is_trimmed():
    adjacent = cross_stitch_pattern(_source(Image.new("RGB", (3, 1), "red")), "keep")
    assert not any(st[2] == pe.TRIM for st in adjacent.stitches)
    image = Image.new("RGBA", (5, 1))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((4, 0), (255, 0, 0, 255))
    separated = cross_stitch_pattern(_source(image), "keep")
    assert sum(st[2] == pe.TRIM for st in separated.stitches) == 1


def test_oversized_and_non_pixel_art_images_are_rejected():
    with pytest.raises(ValueError, match="200 cases"):
        read_cells(_source(Image.new("RGB", (201, 1), "red")))
    with pytest.raises(ValueError, match="Quadrillage"):
        read_cells(_source(Image.new("RGB", (768, 768), "red")))
    pixels = np.arange(300, dtype=np.uint32)
    rgb = np.stack((pixels % 256, pixels // 256, pixels * 0), axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb.reshape(15, 20, 3))
    pattern = cross_stitch_pattern(_source(image), "keep")
    assert len(pattern.img2stitch_source_palette) <= 20
    assert len(pattern.threadlist) <= 20
    assert pattern.img2stitch_grid["crosses"] == 300


def test_color_reduction_merges_nearest_shades_first():
    anchors = [
        (135, 7, 80), (42, 253, 169), (26, 4, 162), (125, 145, 138), (208, 187, 75),
        (43, 57, 93), (235, 27, 49), (237, 142, 167), (68, 188, 213), (186, 12, 247),
        (138, 5, 201), (35, 214, 16), (90, 157, 116), (143, 212, 208), (203, 27, 96),
        (108, 232, 15), (128, 122, 56), (213, 134, 95), (201, 173, 242),
    ]
    colors = [(130, 110, 94), (130, 111, 94), *anchors]
    cells = np.array([[(*color, 255) for color in colors]], dtype=np.uint8)

    reduced = _limit_cell_colors(cells)

    assert tuple(reduced[0, 0, :3]) == tuple(reduced[0, 1, :3])
    assert len(np.unique(reduced[0, :, :3], axis=0)) == 20


def test_lossy_upload_preserves_original_format(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, "JPEG")
    stream.seek(0)
    store = app.extensions["designs"]
    try:
        response = app.test_client().post("/api/pixel-art", data={
            "image": (stream, "pixel.jpg"), "background": "keep",
        })
        assert response.status_code == 202
        store._pool.shutdown(wait=True)
        meta = store.get(response.json["id"])
        assert meta["status"] == "ready"
        assert meta["source_format"] == "JPEG"
        assert meta["n_colors"] == 1
    finally:
        store._pool.shutdown(wait=True)


def test_web_cross_stitch_workflow(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    client = app.test_client()
    store = app.extensions["designs"]
    try:
        response = client.post("/api/pixel-art", data={
            "image": (_source(Image.new("RGB", (3, 2), "red")), "pixel.png"),
            "background": "keep", "colors": "1", "width_mm": "400", "height_mm": "400",
        })
        assert response.status_code == 202
        design_id = response.json["id"]
        store._pool.shutdown(wait=True)
        meta = store.get(design_id)
        assert meta["status"] == "ready", meta.get("error")
        assert meta["mode"] == "cross_stitch"
        assert meta["grid"]["crosses"] == 6
        assert meta["n_colors"] == 1
        assert meta["size_mm"] == [6.0, 4.0]
        assert meta["target_mm"] is None
        assert meta["has_raw"] and meta["has_photo"] and meta["has_thumb"]
        assert client.get("/galerie").status_code == 200
        page = client.get(f"/motifs/{design_id}").get_data(as_text=True)
        assert "6</dd>" in page
        assert "Modifier le motif" not in page
        assert page.count('type="color"') == 1
        assert 'id="cross-color-form"' not in page
        for fmt in ("pes", "dst", "jef", "exp", "vp3", "xxx", "u01", "pec"):
            assert client.get(f"/motifs/{design_id}/telecharger/{fmt}").status_code == 200
        assert client.get(f"/motifs/{design_id}/edition").status_code == 302
        assert client.post(f"/api/motifs/{design_id}/fusion", json={"enabled": True}).status_code == 409
        assert client.post(f"/api/motifs/{design_id}/edition", json={
            "zones": [{"color": "#00ff00", "fill": "tatami"}],
        }).status_code == 409
    finally:
        store._pool.shutdown(wait=True)


def test_web_enlarged_blocks_workflow(block_art, tmp_path, monkeypatch):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    client = app.test_client()
    image = block_art.resize((600, 600), Image.Resampling.NEAREST)
    try:
        analysis = client.post("/api/pixel-art/analyse", data={
            "image": (_source(image), "blocks.png"), "background": "outside",
        })
        assert analysis.status_code == 200, analysis.json
        response = client.post("/api/pixel-art", data={
            "image": (_source(image), "blocks.png"), "background": "outside",
        })
        assert response.status_code == 202
        meta = store.get(response.json["id"])
        assert meta["status"] == "ready", meta.get("error")
        assert meta["grid"] == cross_stitch_pattern(block_art).img2stitch_grid
        assert meta["size_mm"] == [54.0, 46.0]
        assert client.get(f"/motifs/{meta['id']}/telecharger/pes").status_code == 200
    finally:
        store._pool.shutdown(wait=True)


def test_cross_stitch_palette_api(tmp_path, monkeypatch):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    client = app.test_client()
    image = Image.new("RGB", (3, 1), "red")
    image.putpixel((1, 0), (0, 0, 255))
    image.putpixel((2, 0), (255, 255, 0))
    try:
        response = client.post("/api/pixel-art", data={
            "image": (_source(image), "pixel.png"), "background": "keep", "colors": "2",
        })
        assert response.status_code == 202
        design_id = response.json["id"]
        meta = store.get(design_id)
        assert meta["status"] == "ready", meta.get("error")
        assert meta["n_colors"] == 2
        grid, size = meta["grid"], meta["size_mm"]
        for count in (1, 3):
            response = client.post(f"/api/motifs/{design_id}/couleurs", json={"colors": count})
            assert response.status_code == 202
            assert response.json["n_colors"] == count
            assert response.json["grid"] == grid
            assert response.json["size_mm"] == size
        for old in ("#ff0000", "#0000ff"):
            response = client.post(f"/api/motifs/{design_id}/fil", json={"from": old, "to": "#00ff00"})
            assert response.status_code == 202
        assert store.get(design_id)["n_colors"] == 2
        response = client.post(f"/api/motifs/{design_id}/fil", json={"from": "#00ff00", "to": "#a000f0"})
        assert response.status_code == 202
        assert set(response.json["threads"]) == {"#a000f0", "#ffff00"}
        assert response.json["grid"] == grid
        assert response.json["size_mm"] == size
        saved = pe.read(str(store.pattern_path(design_id, "pes")))
        assert len(saved.threadlist) == 2
        data = json.loads((store.dir(design_id) / "pattern.json").read_text(encoding="utf-8"))
        assert set(data["threads"]) == {0xA000F0, 0xFFFF00}
        assert client.post(f"/api/motifs/{design_id}/tissu", json={"fabric": "emeraude"}).status_code == 202
        assert set(store.get(design_id)["threads"]) == {"#a000f0", "#ffff00"}
        assert client.post(f"/api/motifs/{design_id}/couleurs", json={"colors": None}).status_code == 202
        assert store.get(design_id)["n_colors"] == 3
        for invalid in (-1, 257, 1.5, True, "invalid", []):
            assert client.post(f"/api/motifs/{design_id}/couleurs", json={"colors": invalid}).status_code == 400
            assert client.post("/api/pixel-art", data={
                "image": (_source(image), "pixel.png"), "colors": str(invalid),
            }).status_code == 400
        assert client.post(f"/api/motifs/{design_id}/couleurs", json={}).status_code == 400
        assert client.post(f"/api/motifs/{'f' * 32}/couleurs", json={"colors": 2}).status_code == 404
    finally:
        store._pool.shutdown(wait=True)


def test_failed_palette_change_restores_settings(tmp_path, monkeypatch):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    client = app.test_client()
    try:
        response = client.post("/api/pixel-art", data={
            "image": (_source(Image.new("RGB", (2, 1), "red")), "pixel.png"), "background": "keep",
        })
        design_id = response.json["id"]
        client.post(f"/api/motifs/{design_id}/fil", json={"from": "#ff0000", "to": "#00ff00"})
        before = store.get(design_id)
        def fail(*args, **kwargs):
            raise ValueError("Erreur de contrôle")
        monkeypatch.setattr("webapp.designs.cross_stitch_pattern", fail)
        response = client.post(f"/api/motifs/{design_id}/couleurs", json={"colors": 2})
        assert response.status_code == 202
        after = store.get(design_id)
        assert after["status"] == "ready"
        assert after["edit_error"] == "Erreur de contrôle"
        for key in ("threads", "zones", "cross_stitch_colors", "cross_stitch_color_limit"):
            assert after[key] == before[key]
    finally:
        store._pool.shutdown(wait=True)


def test_legacy_cross_stitch_recolor_and_busy_guard(tmp_path, monkeypatch):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    client = app.test_client()
    try:
        response = client.post("/api/pixel-art", data={
            "image": (_source(Image.new("RGB", (2, 1), "red")), "pixel.png"), "background": "keep",
        })
        design_id = response.json["id"]
        directory = store.dir(design_id)
        meta = store.get(design_id)
        for key in ("cross_stitch_colors", "cross_stitch_color_limit", "cross_stitch_source_palette"):
            meta.pop(key)
        store._write(directory, meta)
        response = client.post(f"/api/motifs/{design_id}/fil", json={"from": "#ff0000", "to": "#00ff00"})
        assert response.status_code == 202
        assert response.json["threads"] == ["#00ff00"]
        for changes in ({"status": "processing"}, {"status": "ready", "photo_rendering": True}):
            store._update(directory, store.get(design_id), **changes)
            assert client.post(f"/api/motifs/{design_id}/couleurs", json={"colors": 2}).status_code == 409
            assert client.post(f"/api/motifs/{design_id}/fil", json={"from": "#00ff00", "to": "#0000ff"}).status_code == 409
    finally:
        store._pool.shutdown(wait=True)


def test_cross_stitch_upload_validation(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    client = app.test_client()
    try:
        assert client.post("/api/pixel-art").status_code == 400
        assert client.post("/api/pixel-art", data={"image": (io.BytesIO(b"invalid"), "image.png")}).status_code == 400
        assert client.post("/api/pixel-art", data={
            "image": (_source(Image.new("RGB", (2, 2))), "image.png"), "background": "invalid",
        }).status_code == 400
    finally:
        app.extensions["designs"]._pool.shutdown(wait=True)


def test_cross_stitch_analysis_and_exclusions_api(tmp_path, monkeypatch):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    client = app.test_client()
    image = Image.new("RGB", (5, 5), "white")
    image.putpixel((1, 1), (255, 0, 0))
    image.putpixel((2, 1), (0, 0, 255))
    image.putpixel((3, 1), (255, 255, 0))
    try:
        response = client.post("/api/pixel-art/analyse", data={"image": (_source(image), "pixel.png")})
        assert response.status_code == 200
        assert response.json["colors"] == 3
        assert set(response.json["palettes"]) == {str(count) for count in range(1, 13)}
        assert set(response.json["palettes"]["3"]) == {"#ff0000", "#0000ff", "#ffff00"}
        excluded = json.dumps([{"color": "#0000ff", "others": ["#ff0000", "#ffff00"]}])
        response = client.post("/api/pixel-art/analyse", data={
            "image": (_source(image), "pixel.png"), "exclude": excluded,
        })
        assert response.status_code == 200
        assert response.json["colors"] == 2
        palette = response.json["palettes"]["2"]
        assert set(palette) == {"#ff0000", "#ffff00"}
        response = client.post("/api/pixel-art", data={
            "image": (_source(image), "pixel.png"), "colors": "2", "exclude": excluded,
        })
        assert response.status_code == 202
        assert response.json["grid"]["crosses"] == 3
        assert response.json["threads"] == palette
        meta = store.get(response.json["id"])
        assert meta["excluded"] == json.loads(excluded)
        response = client.post(f"/api/motifs/{meta['id']}/fil", json={"from": "#ff0000", "to": "#00ff00"})
        assert response.status_code == 202
        assert response.json["grid"]["crosses"] == 3
        assert set(response.json["threads"]) == {"#00ff00", "#ffff00"}
        assert client.post("/api/pixel-art/analyse").status_code == 400
        for endpoint in ("/api/pixel-art/analyse", "/api/pixel-art"):
            for extra in ({"exclude": "invalid"}, {"background": "invalid"}):
                assert client.post(endpoint, data={"image": (_source(image), "pixel.png"), **extra}).status_code == 400
        assert client.post("/api/pixel-art/analyse", data={
            "image": (io.BytesIO(b"invalid"), "pixel.png"),
        }).status_code == 400
    finally:
        store._pool.shutdown(wait=True)


def test_cross_stitch_creation_has_color_options_without_size_controls(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    try:
        client = app.test_client()
        assert "/pixel-art" in client.get("/galerie").get_data(as_text=True)
        response = client.get("/pixel-art")
        assert response.status_code == 200
        page = response.get_data(as_text=True)
        assert 'name="image"' in page
        assert 'name="colors"' in page
        colors_markup = app.jinja_env.get_template("_colors.html").render(min_colors=1, max_colors=12)
        assert colors_markup in page
        assert colors_markup in client.get("/").get_data(as_text=True)
        assert 'id="cross-auto-colors"' not in page
        assert 'type="number"' not in page
        assert 'data-analyse="/api/pixel-art/analyse"' in page
        assert page.count('name="background"') == 3
        for field in ("width_mm", "height_mm", "thicken_lines"):
            assert f'name="{field}"' not in page
    finally:
        app.extensions["designs"]._pool.shutdown(wait=True)