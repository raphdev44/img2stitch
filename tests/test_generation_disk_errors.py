"""Écritures disque : un aperçu réaliste qui échoue ne doit pas rendre le motif inutilisable."""
import io
import time

import pytest
from PIL import Image, ImageDraw

from webapp import create_app
from webapp import designs as designs_module
from webapp.designs import _write_atomic, _write_error

RED = (220, 30, 30)


def _app(tmp_path):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    return app, app.extensions["designs"]


def _create(client, store, wait=True):
    image = Image.new("RGB", (160, 160), "white")
    ImageDraw.Draw(image).ellipse((20, 20, 140, 140), fill=RED)
    stream = io.BytesIO()
    image.save(stream, "PNG")
    stream.seek(0)
    design_id = client.post("/api/motifs", data={
        "image": (stream, "rond.png"), "background": "outside", "colors": "1",
        "width_mm": "40", "height_mm": "40",
    }, content_type="multipart/form-data").json["id"]
    return _wait(store, design_id) if wait else design_id


def _wait(store, design_id, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        meta = store.get(design_id)
        if meta["status"] != "processing":
            return meta
        time.sleep(0.1)
    raise AssertionError("le motif n'a pas fini d'être généré")


def test_preview_failure_keeps_the_pattern_usable(tmp_path, monkeypatch):
    app, store = _app(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("écriture de l'aperçu réaliste impossible")

    monkeypatch.setattr(designs_module, "render_photo_preview", boom)
    meta = _create(app.test_client(), store)

    assert meta["status"] == "ready"          # les fichiers de broderie restent disponibles
    assert meta["photo_error"]
    directory = store.dir(meta["id"])
    assert (directory / "pattern.dst").is_file()
    assert (directory / "pattern.json").is_file()
    assert not (directory / "preview_photo.png").exists()
    assert not list(directory.glob("*.tmp.png"))


def test_preview_is_skipped_when_the_disk_is_full(tmp_path, monkeypatch):
    app, store = _app(tmp_path)
    monkeypatch.setattr(designs_module, "_free_bytes", lambda path: 1024)

    meta = _create(app.test_client(), store)

    assert meta["status"] == "ready"
    assert "Espace disque insuffisant" in meta["photo_error"]


def test_write_error_names_the_full_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(designs_module, "_free_bytes", lambda path: 3 * 1024 * 1024)

    assert "Espace disque insuffisant" in _write_error(tmp_path)
    assert "3 Mo libres" in _write_error(tmp_path)


def test_write_error_without_a_full_disk_stays_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(designs_module, "_free_bytes", lambda path: None)

    assert "écrits sur le disque" in _write_error(tmp_path)


def test_write_atomic_cleans_up_after_a_failed_write(tmp_path):
    target = tmp_path / "preview.png"

    def boom(path):
        raise OSError("disque plein")

    with pytest.raises(OSError):
        _write_atomic(target, boom)

    assert not (tmp_path / "preview.tmp.png").exists()
    assert not target.exists()


def test_write_atomic_cleans_up_when_the_temp_file_never_appears(tmp_path):
    target = tmp_path / "preview.png"

    with pytest.raises(OSError):
        _write_atomic(target, lambda path: None)   # n'écrit rien : os.replace échoue

    assert not target.exists()
