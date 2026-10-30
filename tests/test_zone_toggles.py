"""Interrupteurs par zone : remplissage et point de contour."""
import pytest
from PIL import Image, ImageDraw

from img2stitch import Settings
from img2stitch.algorithm_standard import _region_look, _split_regions, _zone_override, image_to_pattern, pattern_stats
from webapp.designs import InvalidImage, _zone_setting, parse_edits

RED = (220, 30, 30)


def _img(tmp_path):
    im = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(im).ellipse((60, 60, 340, 340), fill=RED)
    f = tmp_path / "t.png"
    im.save(f)
    return str(f)


# --- modèle : contour par zone ---------------------------------------------------------------
def test_zone_override_carries_the_outline_flag():
    assert _zone_override(RED, ((RED, (0, 0, 255), "satin", False),)) == ((0, 0, 255), "satin", False)


def test_zone_override_accepts_old_three_element_tuples():
    # projets créés avant l'interrupteur : le contour suit le réglage global (None)
    assert _zone_override(RED, ((RED, (0, 0, 255), "satin"),)) == ((0, 0, 255), "satin", None)
    assert _zone_override(RED, ()) == (RED, "tatami", None)


def test_region_look_carries_the_outline_flag():
    override = ((0.5, 0.5), RED, (0, 0, 255), "satin", False)
    assert _region_look(override, RED) == ((0, 0, 255), "satin", False)
    assert _region_look(((0.5, 0.5), RED, None, None), RED) == (RED, "tatami", None)


def test_split_regions_accepts_five_element_overrides():
    import numpy as np

    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True
    regions = (((0.5, 0.5), RED, RED, "tatami", False),)
    palette, masks, indexes = _split_regions([RED], [mask], regions, 10)
    assert indexes == [0]           # la forme cliquée porte bien la retouche n° 0
    assert masks[0].sum() == 40 * 40  # et couvre toute la forme : rien ne reste dans le bloc entier


def test_zone_setting_only_stores_what_differs_from_default():
    base = {"source": "#dc1e1e", "color": "#dc1e1e", "fill": "tatami", "region": None}
    assert _zone_setting([dict(base, outline=True)]) == ()
    assert _zone_setting([dict(base, outline=False)]) == ((RED, RED, "tatami", False),)
    # ancien projet : pas de clé "outline" -> contour actif, aucune retouche
    assert _zone_setting([base]) == ()


# --- moteur : le contour d'une zone est bien retiré ------------------------------------------
def test_disabling_outline_removes_stitches_but_keeps_the_fill(tmp_path):
    path = _img(tmp_path)
    on = image_to_pattern(path, Settings(width_mm=40, n_colors=1, zone_overrides=((RED, RED, "tatami", True),)))
    off = image_to_pattern(path, Settings(width_mm=40, n_colors=1, zone_overrides=((RED, RED, "tatami", False),)))
    st_on, st_off = pattern_stats(on), pattern_stats(off)

    assert st_off["stitches"] < st_on["stitches"]          # le contour a disparu
    assert st_off["stitches"] > st_on["stitches"] * 0.5    # le remplissage est toujours là


def test_global_outline_setting_still_applies(tmp_path):
    path = _img(tmp_path)
    off = image_to_pattern(path, Settings(width_mm=40, n_colors=1, outline=False))
    zone_off = image_to_pattern(path, Settings(width_mm=40, n_colors=1,
                                               zone_overrides=((RED, RED, "tatami", False),)))
    assert pattern_stats(off)["stitches"] == pattern_stats(zone_off)["stitches"]


# --- API : l'éditeur transmet les deux interrupteurs -----------------------------------------
def test_parse_edits_round_trips_both_toggles():
    zones, regions = parse_edits({
        "zones": [{"color": "#dc1e1e", "fill": "none", "outline": False}],
        "regions": [{"block": 0, "fx": 0.5, "fy": 0.5, "color": "#dc1e1e", "fill": "tatami", "outline": True}],
    }, 1)
    assert zones == [{"color": "#dc1e1e", "fill": "none", "outline": False}]
    assert regions[0]["outline"] is True and regions[0]["fill"] == "tatami"


def test_parse_edits_defaults_outline_for_old_payloads():
    zones, _ = parse_edits({"zones": [{"color": "#dc1e1e", "fill": "tatami"}]}, 1)
    assert zones[0]["outline"] is True


def test_parse_edits_rejects_a_non_boolean_outline():
    with pytest.raises(InvalidImage):
        parse_edits({"zones": [{"color": "#dc1e1e", "fill": "tatami", "outline": "non"}]}, 1)


# --- éditeur complet : les deux interrupteurs vont jusqu'au fichier de broderie -----------------
def _wait(store, design_id, timeout=120):
    import time

    end = time.time() + timeout
    while time.time() < end:
        meta = store.get(design_id)
        if meta["status"] != "processing":
            return meta
        time.sleep(0.1)
    raise AssertionError("le motif n'a pas fini d'être généré")


def _edit(client, design_id, zones, index, **changes):
    payload = [{"color": z["color"], "fill": z["fill"], "outline": z["outline"]} for z in zones]
    payload[index].update(changes)
    response = client.post(f"/api/motifs/{design_id}/edition", json={"zones": payload})
    assert response.status_code == 202, response.get_json()


def test_web_turning_everything_off_explains_instead_of_failing_silently(tmp_path):
    import io

    from PIL import Image, ImageDraw

    from webapp import create_app

    image = Image.new("RGB", (120, 120), "white")
    ImageDraw.Draw(image).ellipse((10, 10, 110, 110), fill=RED)
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    client = app.test_client()
    try:
        design_id = client.post("/api/motifs", data={
            "image": (stream, "rond.png"), "background": "outside", "colors": "1",
            "width_mm": "40", "height_mm": "40",
        }).json["id"]
        _wait(store, design_id)
        _edit(client, design_id, store.zones(design_id), 0, fill="none", outline=False)
        meta = _wait(store, design_id)
        assert meta["status"] == "ready"  # le motif reste utilisable
        assert "éteintes" in (meta.get("edit_error") or "")
        # la retouche refusée n'a pas été enregistrée : la zone a retrouvé son remplissage
        assert store.zones(design_id)[0]["fill"] == "tatami"
        assert meta["stats"]["stitches"] > 0
    finally:
        store._pool.shutdown(wait=True)


def test_web_toggles_reach_the_stitch_file_and_survive_regeneration(tmp_path):
    import io

    from PIL import Image, ImageDraw

    from webapp import create_app

    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 150, 150), fill=RED)
    draw.rectangle((150, 150, 185, 185), fill=(30, 30, 220))
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)

    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    store = app.extensions["designs"]
    client = app.test_client()
    try:
        created = client.post("/api/motifs", data={
            "image": (stream, "rond.png"), "background": "outside", "colors": "2",
            "width_mm": "40", "height_mm": "40",
        })
        assert created.status_code == 202, created.get_json()
        design_id = created.json["id"]
        meta = _wait(store, design_id)
        assert meta["status"] == "ready", meta.get("error")
        zones = store.zones(design_id)
        assert len(zones) == 2
        before = meta["stats"]["stitches"]
        assert zones[0]["outline"] is True

        # contour coupé : le remplissage reste, les points du bord disparaissent
        _edit(client, design_id, zones, 0, outline=False)
        meta = _wait(store, design_id)
        assert meta["status"] == "ready", meta.get("error")
        assert store.zones(design_id)[0]["outline"] is False
        assert meta["stats"]["stitches"] < before

        # remplissage ET contour coupés : la zone reste au catalogue, sans aucun point à elle
        _edit(client, design_id, store.zones(design_id), 0, fill="none", outline=False)
        meta = _wait(store, design_id)
        assert meta["status"] == "ready", meta.get("error", meta.get("edit_error"))
        assert meta.get("edit_error") is None
        assert meta["stats"]["stitches"] < before
        kept = store.zones(design_id)
        assert len(kept) == 2  # la zone éteinte n'a pas disparu du catalogue
        assert kept[0]["fill"] == "none" and kept[0]["outline"] is False
        assert kept[1]["fill"] == "tatami" and kept[1]["outline"] is True

        # on rallume : la zone doit retrouver exactement ses points
        _edit(client, design_id, kept, 0, fill="tatami", outline=True)
        meta = _wait(store, design_id)
        assert meta["status"] == "ready", meta.get("error")
        assert meta["stats"]["stitches"] == before
        page = client.get(f"/motifs/{design_id}/edition").get_data(as_text=True)
        assert 'id="zm-toggle-fill"' in page and 'id="zm-toggle-outline"' in page
    finally:
        store._pool.shutdown(wait=True)
