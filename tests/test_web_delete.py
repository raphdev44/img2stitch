import json

from webapp import create_app


def _app_with_design(tmp_path, status="ready", photo_rendering=False):
    app = create_app({"TESTING": True, "DESIGNS_DIR": str(tmp_path)})
    design_id = "a" * 32
    design_dir = tmp_path / design_id
    design_dir.mkdir()
    (design_dir / "meta.json").write_text(json.dumps({
        "id": design_id,
        "name": "Motif de test",
        "created_at": "2026-10-27T12:00:00+00:00",
        "status": status,
        "photo_rendering": photo_rendering,
    }), encoding="utf-8")
    return app, design_id, design_dir


def test_delete_ready_design(tmp_path):
    app, design_id, design_dir = _app_with_design(tmp_path)

    response = app.test_client().post(f"/api/motifs/{design_id}/supprimer")

    assert response.status_code == 204
    assert not design_dir.exists()


def test_delete_rejects_design_being_processed(tmp_path):
    app, design_id, design_dir = _app_with_design(tmp_path, status="processing")

    response = app.test_client().post(f"/api/motifs/{design_id}/supprimer")

    assert response.status_code == 409
    assert response.json["error"]
    assert design_dir.exists()


def test_delete_rejects_photo_rendering(tmp_path):
    app, design_id, design_dir = _app_with_design(tmp_path, photo_rendering=True)

    response = app.test_client().post(f"/api/motifs/{design_id}/supprimer")

    assert response.status_code == 409
    assert design_dir.exists()