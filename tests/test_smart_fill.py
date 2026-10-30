"""Remplissage intelligent : bandes fines détectées et brodées en satin bombé, sans contour."""
import io
import time

from PIL import Image, ImageDraw

from img2stitch import Settings
from img2stitch.smart import thin_band_regions

RED, BLUE = (220, 30, 30), (30, 60, 200)


def _shapes(size=(300, 300)):
    """Une bande fine (ruban) et un disque épais sur fond blanc."""
    im = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle((20, size[1] // 2, size[0] - 20, size[1] // 2 + 12), fill=RED)
    draw.ellipse((40, 40, 140, 140), fill=BLUE)
    return im


def test_detects_the_band_but_not_the_disc(tmp_path):
    path = tmp_path / "shapes.png"
    _shapes().save(path)
    regions = thin_band_regions(path, Settings(n_colors=2, background="auto", background_scope="outside",
                                               max_width_mm=60, max_height_mm=60), probe_px=300)
    assert len(regions) == 1
    u, v, rgb = regions[0]
    assert 0.4 < v < 0.6                       # le point est dans le ruban horizontal…
    assert u < 0.95                            # …pas sur le disque épais
    assert rgb[0] > 150 and max(rgb[1:]) < 80  # fil rouge, pas le fond blanc


def test_a_thin_outline_is_a_band_even_when_it_encloses_areas(tmp_path):
    # Le liseré d'un dessin est un réseau : il enferme des surfaces, donc il a des trous. Un trou
    # ne dit rien de la finesse d'un brin — c'est la largeur lue sur l'axe médian qui compte.
    im = Image.new("RGB", (300, 300), "white")
    dr = ImageDraw.Draw(im)
    dr.ellipse((40, 40, 200, 200), fill=(230, 200, 40))            # surface claire, large
    dr.ellipse((40, 40, 200, 200), outline=(20, 20, 20), width=6)  # liseré fin, fermé
    path = tmp_path / "lisere.png"
    im.save(path)
    regions = thin_band_regions(path, Settings(n_colors=2, background="auto",
                                               background_scope="outside",
                                               max_width_mm=60, max_height_mm=60), probe_px=300)
    assert len(regions) == 1
    _, _, rgb = regions[0]
    assert max(rgb) < 90                       # le liseré sombre, pas la surface qu'il enferme


def _wait(store, design_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        meta = store.get(design_id)
        if meta["status"] != "processing":
            return meta
        time.sleep(0.05)
    raise AssertionError("le motif n'a pas fini d'être généré")


def test_smart_fill_creates_satin_regions_without_outline(tmp_path, monkeypatch):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    buf = io.BytesIO()
    _shapes().save(buf, "PNG")
    buf.seek(0)
    response = app.test_client().post("/api/motifs", data={
        "image": (buf, "formes.png"), "colors": "2", "background": "outside",
        "width_mm": "60", "height_mm": "60", "smart_fill": "true", "exclude": "[]",
    })
    assert response.status_code == 202, response.get_json()
    meta = _wait(store, response.json["id"])
    assert meta["status"] == "ready", meta.get("error")
    assert meta["smart_fill"] is True
    assert meta["regions"], "la bande fine aurait dû être isolée"
    assert all(r["fill"] == "satin_relief" and r["outline"] is False for r in meta["regions"])
    satin = [z for z in meta["zones"] if z["fill"] == "satin_relief"]
    assert satin and all(z["outline"] is False for z in satin)
    # Deux fils : le satin reprend le fil de sa couleur, sans arrêt supplémentaire.
    assert len(meta["threads"]) == 2
    assert meta["stats"]["color_changes"] == 1


def test_smart_fill_off_keeps_plain_fill(tmp_path, monkeypatch):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    monkeypatch.setattr(store._pool, "submit", lambda function, *args: function(*args))
    buf = io.BytesIO()
    _shapes().save(buf, "PNG")
    buf.seek(0)
    response = app.test_client().post("/api/motifs", data={
        "image": (buf, "formes.png"), "colors": "2", "background": "outside",
        "width_mm": "60", "height_mm": "60", "exclude": "[]",
    })
    assert response.status_code == 202
    meta = _wait(store, response.json["id"])
    assert meta["status"] == "ready", meta.get("error")
    assert meta.get("smart_fill") is False
    assert not meta.get("regions")
    assert all(z["fill"] == "tatami" for z in meta["zones"])


def test_create_page_offers_the_option_unchecked(tmp_path):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    html = app.test_client().get("/numerisation").get_data(as_text=True)
    assert 'id="smart-fill"' in html
    assert 'id="smart-fill" checked' not in html
