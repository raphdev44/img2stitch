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


def test_the_classic_form_moved_to_its_own_page(tmp_path):
    client = _app(tmp_path).test_client()
    assert client.get("/numerisation").status_code == 200
    html = client.get("/numerisation").get_data(as_text=True)
    assert 'id="create-form"' in html
    assert 'id="create-form"' not in client.get("/").get_data(as_text=True)
