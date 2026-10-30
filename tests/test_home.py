"""Page d'accueil : les quatre canaux sont accessibles d'un clic."""
from webapp import create_app


def _app(tmp_path):
    return create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})


def test_home_offers_the_four_channels(tmp_path):
    client = _app(tmp_path).test_client()
    html = client.get("/").get_data(as_text=True)
    for label in ("Galerie de motifs", "Numérisation classique",
                  "Pixel art en point de croix", "Broderie à la main"):
        assert label in html
    for url in ("/galerie", "/numerisation", "/pixel-art", "/broderie-main"):
        assert f'href="{url}"' in html
    for illustration in ("icone_gallerie.png", "icone_numeriser.png",
                         "icone_pixel.png", "icone_main.png"):
        assert f"/static/img/{illustration}" in html


def test_the_classic_form_moved_to_its_own_page(tmp_path):
    client = _app(tmp_path).test_client()
    assert client.get("/numerisation").status_code == 200
    html = client.get("/numerisation").get_data(as_text=True)
    assert 'id="create-form"' in html
    assert 'id="create-form"' not in client.get("/").get_data(as_text=True)


def test_each_tool_page_sets_its_color_theme(tmp_path):
    client = _app(tmp_path).test_client()
    for path, theme in (("/galerie", "gallery"), ("/numerisation", "create"),
                        ("/pixel-art", "cross-stitch"), ("/broderie-main", "hand-embroidery")):
        response = client.get(path)
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert f'data-page="{theme}"' in html
        assert f'data-theme="{theme}"' in html


def test_design_and_edit_pages_keep_javascript_page_identifiers(tmp_path):
    import io
    import time

    from PIL import Image, ImageDraw

    app = _app(tmp_path)
    store = app.extensions["designs"]
    client = app.test_client()
    image = Image.new("RGB", (80, 80), "white")
    ImageDraw.Draw(image).ellipse((10, 10, 70, 70), fill=(220, 30, 30))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)

    try:
        response = client.post("/api/motifs", data={
            "image": (stream, "rond.png"), "background": "outside", "colors": "1",
            "width_mm": "40", "height_mm": "40",
        })
        assert response.status_code == 202
        design_id = response.json["id"]
        deadline = time.monotonic() + 30
        while store.get(design_id)["status"] == "processing" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert store.get(design_id)["status"] == "ready"

        design_html = client.get(f"/motifs/{design_id}").get_data(as_text=True)
        edit_html = client.get(f"/motifs/{design_id}/edition").get_data(as_text=True)
        assert 'data-page="design" data-theme="create"' in design_html
        assert 'data-page="edit" data-theme="create"' in edit_html
    finally:
        store._pool.shutdown(wait=True)
