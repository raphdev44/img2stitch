import json

from webapp import create_app
from webapp.designs import MAX_NAME, _optimized_name

ORIGINAL = "c" * 32


def _app(tmp_path, monkeypatch, name="Motif de test"):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    design_dir = tmp_path / ORIGINAL
    design_dir.mkdir()
    (design_dir / "meta.json").write_text(json.dumps({
        "id": ORIGINAL, "name": name, "created_at": "2026-10-29T12:00:00+00:00",
        "status": "ready", "mode": "classic", "zones": [], "regions": [],
        "stats": {"stitches": 10},
    }), encoding="utf-8")
    (design_dir / "source.png").write_bytes(b"image")
    (design_dir / "pattern.dst").write_bytes(b"dst")
    # L'optimisation tourne en tâche de fond : on l'empêche de démarrer pour tester la copie seule.
    monkeypatch.setattr(app.extensions["designs"]._pool, "submit", lambda *a, **k: None)
    return app, design_dir


def _meta(path):
    return json.loads((path / "meta.json").read_text(encoding="utf-8"))


def test_optimization_duplicates_the_design(tmp_path, monkeypatch):
    app, design_dir = _app(tmp_path, monkeypatch)

    new_id = app.extensions["designs"].optimize(ORIGINAL, "simple")

    copy = tmp_path / new_id
    assert new_id != ORIGINAL and len(new_id) == 32
    assert (copy / "source.png").read_bytes() == b"image"
    assert (copy / "pattern.dst").read_bytes() == b"dst"
    meta = _meta(copy)
    assert meta["id"] == new_id
    assert meta["name"] == "Motif de test - optimisé"
    assert meta["status"] == "processing"
    assert meta["optimized_from"] == ORIGINAL
    assert meta["optimized_from_name"] == "Motif de test"
    assert meta["optimization_mode"] == "simple"


def test_radical_copy_is_named_optimise_plus(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)

    new_id = app.extensions["designs"].optimize(ORIGINAL, "radical")

    meta = _meta(tmp_path / new_id)
    assert meta["name"] == "Motif de test - optimisé+"
    assert meta["optimization_mode"] == "radical"


def test_original_design_is_left_untouched(tmp_path, monkeypatch):
    app, design_dir = _app(tmp_path, monkeypatch)
    before = _meta(design_dir)

    app.extensions["designs"].optimize(ORIGINAL, "radical")

    after = _meta(design_dir)
    assert after == before
    assert after["name"] == "Motif de test"
    assert after["status"] == "ready"
    assert "optimized_from" not in after


def test_optimization_api_returns_the_copy(tmp_path, monkeypatch):
    app, design_dir = _app(tmp_path, monkeypatch)
    before = _meta(design_dir)

    response = app.test_client().post(f"/api/motifs/{ORIGINAL}/optimiser", json={"mode": "radical"})

    assert response.status_code == 202
    payload = response.json
    assert payload["id"] != ORIGINAL
    assert payload["name"] == "Motif de test - optimisé+"
    assert payload["optimized_from"] == ORIGINAL
    assert payload["urls"]["page"].endswith(payload["id"])
    assert _meta(design_dir) == before


def test_copy_name_stays_within_the_limit():
    long_name = "A" * MAX_NAME

    name = _optimized_name(long_name, "radical")

    assert name.endswith(" - optimisé+")
    assert len(name) <= MAX_NAME
