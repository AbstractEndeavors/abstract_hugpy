from abstract_flask import *
from .hugpy_proxyvideo_flask import hugpy_proxyvideo_bp

bp_list = [
    hugpy_proxyvideo_bp
]
URL_PREFIX = "hugpy/proxyvideo"

def hugpy_proxyvideo_app(debug=True):
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "https://clownworld.biz",
        "https://www.clownworld.biz",
        "https://www.thedailydialectics.com",
        "https://www.typicallyoutliers.com"
    ]
    app = get_Flask_app(
        name=__name__,
        bp_list=bp_list,
        allowed_origins=ALLOWED_ORIGINS,
        url_prefix=URL_PREFIX
    )

    @app.route(f"/{URL_PREFIX}/endpoints", methods=["GET"])
    def list_endpoints():
        """Return all available endpoints with methods."""
        output = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint != "static":
                methods = list(rule.methods - {"HEAD", "OPTIONS"})
                output.append({
                    "endpoint": rule.endpoint,
                    "url": str(rule),
                    "methods": methods
                })
        return jsonify(sorted(output, key=lambda x: x["url"]))

    return app
