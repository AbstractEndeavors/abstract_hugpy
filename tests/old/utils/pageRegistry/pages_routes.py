# pages_routes.py
import os
from flask import render_template, abort
from abstract_flask import get_bp
from .pages_registry import get_page, pages_by_category

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

pages_bp, logger = get_bp("pages_bp", __name__, template_folder=TEMPLATES_DIR)

@pages_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", grouped=pages_by_category())

@pages_bp.route("/pages/<path:key>", methods=["GET"])
def page(key: str):
    try:
        spec = get_page(key)
    except KeyError:
        abort(404)
    return render_template("page.html", spec=spec)
