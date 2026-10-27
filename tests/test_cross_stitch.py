import io

import numpy as np
import pyembroidery as pe
import pytest
from PIL import Image, ImageDraw

from img2stitch.cross_stitch import _grid_lines, cross_stitch_pattern, read_cells
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
    with pytest.raises(ValueError, match="trop de nuances"):
        cross_stitch_pattern(_source(image), "keep")


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
        assert 'type="color"' not in page
        for fmt in ("pes", "dst", "jef", "exp", "vp3", "xxx", "u01", "pec"):
            assert client.get(f"/motifs/{design_id}/telecharger/{fmt}").status_code == 200
        assert client.get(f"/motifs/{design_id}/edition").status_code == 302
        assert client.post(f"/api/motifs/{design_id}/fil", json={"from": "#ff0000", "to": "#00ff00"}).status_code == 409
        assert client.post(f"/api/motifs/{design_id}/fusion", json={"enabled": True}).status_code == 409
        assert client.post(f"/api/motifs/{design_id}/edition", json={
            "zones": [{"color": "#00ff00", "fill": "tatami"}],
        }).status_code == 409
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


def test_cross_stitch_creation_has_only_background_options(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    try:
        client = app.test_client()
        assert "/pixel-art" in client.get("/galerie").get_data(as_text=True)
        response = client.get("/pixel-art")
        assert response.status_code == 200
        page = response.get_data(as_text=True)
        assert 'name="image"' in page
        assert page.count('name="background"') == 3
        for field in ("colors", "width_mm", "height_mm", "thicken_lines"):
            assert f'name="{field}"' not in page
    finally:
        app.extensions["designs"]._pool.shutdown(wait=True)