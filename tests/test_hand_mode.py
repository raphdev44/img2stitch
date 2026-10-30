"""Canal « broderie à la main » : création, formulaire, motif non modifiable."""
import io
import time

from PIL import Image, ImageDraw

YELLOW, DARK, RED = (240, 200, 40), (23, 22, 20), (200, 41, 40)


def _picture(size=(360, 360)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    d.ellipse((40, 40, size[0] - 40, size[1] - 40), fill=YELLOW)
    d.line((70, size[1] // 2, size[0] - 70, size[1] // 2), fill=DARK, width=8)
    d.ellipse((180, 220, 215, 255), fill=RED)
    return im


def _wait(store, design_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        meta = store.get(design_id)
        if meta["status"] != "processing":
            return meta
        time.sleep(0.05)
    raise AssertionError("le motif n'a pas fini d'être généré")


def _create(app):
    store = app.extensions["designs"]
    buf = io.BytesIO()
    _picture().save(buf, "PNG")
    buf.seek(0)
    response = app.test_client().post("/api/broderie-main", data={
        "image": (buf, "fleur.png"), "colors": "3", "background": "outside",
        "hand_style": "auto", "exclude": "[]",
    })
    assert response.status_code == 202, response.get_json()
    return _wait(store, response.json["id"])


def test_creates_a_hand_stitched_design(tmp_path, monkeypatch):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    monkeypatch.setattr(app.extensions["designs"]._pool, "submit", lambda function, *args: function(*args))
    meta = _create(app)
    assert meta["status"] == "ready", meta.get("error")
    assert meta["mode"] == "hand"
    assert meta["hand_style"] == "auto"
    assert meta["hand_creativity"] == 3
    assert meta["stats"]["stitches"] > 100
    assert meta["hand_mix"]["shapes"] >= 2
    assert meta["hand_mix"]["stitches"]
    assert meta["size_mm"][0] > 10 and meta["size_mm"][1] > 10


def test_a_hand_design_cannot_be_modified(tmp_path, monkeypatch):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    monkeypatch.setattr(app.extensions["designs"]._pool, "submit", lambda function, *args: function(*args))
    meta = _create(app)
    client = app.test_client()
    design_id = meta["id"]
    assert client.get(f"/motifs/{design_id}/edition").status_code == 302
    assert client.post(f"/api/motifs/{design_id}/optimiser", json={"mode": "simple"}).status_code == 409
    assert client.post(f"/api/motifs/{design_id}/fusion", json={"enabled": True}).status_code == 409
    assert client.post(f"/api/motifs/{design_id}/fil",
                       json={"from": meta["threads"][0], "to": "#112233"}).status_code == 409


def test_the_page_offers_the_hand_form(tmp_path):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    html = app.test_client().get("/broderie-main").get_data(as_text=True)
    assert 'id="hand-form"' in html
    assert 'name="hand_style"' in html
    assert 'name="creativity"' in html
    assert 'id="size-field"' in html
    assert 'name="width_mm"' in html
    assert 'name="height_mm"' in html
    assert "Très décoratif" in html
    assert "Automatique" in html


def test_a_sober_motif_keeps_the_classic_stitches(tmp_path, monkeypatch):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    monkeypatch.setattr(app.extensions["designs"]._pool, "submit", lambda function, *args: function(*args))
    store = app.extensions["designs"]
    buf = io.BytesIO()
    _picture().save(buf, "PNG")
    buf.seek(0)
    response = app.test_client().post("/api/broderie-main", data={
        "image": (buf, "fleur.png"), "colors": "3", "background": "outside",
        "hand_style": "auto", "creativity": "1", "exclude": "[]",
    })
    assert response.status_code == 202, response.get_json()
    meta = _wait(store, response.json["id"])
    assert meta["status"] == "ready", meta.get("error")
    assert meta["hand_creativity"] == 1
    # au premier cran, aucun point décoratif ne doit sortir
    assert not {"fern", "fishbone", "blanket", "chevron", "feather", "cretan", "herringbone",
                "spider_web", "lazy_daisy", "woven_wheel"} & set(meta["hand_mix"]["stitches"])


def test_an_unknown_creativity_level_is_refused(tmp_path):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    buf = io.BytesIO()
    _picture().save(buf, "PNG")
    buf.seek(0)
    response = app.test_client().post("/api/broderie-main", data={
        "image": (buf, "fleur.png"), "colors": "3", "background": "outside",
        "hand_style": "auto", "creativity": "9", "exclude": "[]",
    })
    assert response.status_code == 400
    assert "créativité" in response.get_json()["error"]


def test_the_target_size_frames_the_hand_embroidery(tmp_path, monkeypatch):
    from webapp import create_app

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    monkeypatch.setattr(app.extensions["designs"]._pool, "submit", lambda function, *args: function(*args))
    store = app.extensions["designs"]

    def create(width, height):
        buf = io.BytesIO()
        _picture().save(buf, "PNG")
        buf.seek(0)
        return app.test_client().post("/api/broderie-main", data={
            "image": (buf, "fleur.png"), "colors": "3", "background": "outside",
            "hand_style": "auto", "creativity": "3", "exclude": "[]",
            "width_mm": str(width), "height_mm": str(height),
        })

    response = create(60, 80)
    assert response.status_code == 202, response.get_json()
    meta = _wait(store, response.json["id"])
    assert meta["status"] == "ready", meta.get("error")
    assert meta["target_mm"] == [60.0, 80.0]
    # Le motif tient dans le rectangle demandé et en occupe un côté au plus juste
    assert meta["size_mm"][0] <= 60.2 and meta["size_mm"][1] <= 80.2
    assert max(meta["size_mm"][0] / 60, meta["size_mm"][1] / 80) > 0.97
    assert create(5, 80).status_code == 400
    assert create(500, 80).status_code == 400
