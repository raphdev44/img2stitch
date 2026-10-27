"""Application web Flask pour img2stitch.

Lancement :  flask --app webapp run
"""
import os
from pathlib import Path

from flask import Flask

from .designs import DesignStore


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.update(
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        DESIGNS_DIR=os.environ.get("IMG2STITCH_DATA", str(Path(app.instance_path) / "designs")),
    )
    if config:
        app.config.update(config)

    app.extensions["designs"] = DesignStore(app.config["DESIGNS_DIR"])

    from .routes import bp
    app.register_blueprint(bp)
    return app
