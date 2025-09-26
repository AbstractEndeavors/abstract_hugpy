import os
from abstract_utilities import get_caller_dir

abs_dir = get_caller_dir()
HUGPY_PRE = 'hugpy'
FLASK_POST = 'flask'
APP_POST = 'app'
WSGI_PRE = 'wsgi'

# where the flask apps live
FLASK_DIR_BASE = os.path.join(abs_dir, "hugpy_console", "hugpy_flasks")

# where the wsgi entrypoints should be placed
WSGI_DIR_BASE = "/home/computron/Documents/pythonTools/modules/abstract_hugpy/src/abstract_hugpy"


# -------------------
# Naming Helpers
# -------------------

def get_flask_dir_name(flask_name: str) -> str:
    return f"{HUGPY_PRE}_{flask_name}_{FLASK_POST}_{APP_POST}"

def get_flask_dir_path(flask_name: str) -> str:
    return os.path.join(FLASK_DIR_BASE, get_flask_dir_name(flask_name))

def get_module_name(flask_name: str) -> str:
    return f"{HUGPY_PRE}_{flask_name}_{FLASK_POST}"

def get_init_path(flask_name: str) -> str:
    return os.path.join(get_flask_dir_path(flask_name), "__init__.py")

def get_wsgi_basename(flask_name: str) -> str:
    return f"{HUGPY_PRE}_{flask_name}_{WSGI_PRE}.py"

def get_wsgi_path(flask_name: str) -> str:
    return os.path.join(WSGI_DIR_BASE, get_wsgi_basename(flask_name))


# -------------------
# File Creators
# -------------------

def create_flask_dir(flask_name: str) -> str:
    flask_dir = get_flask_dir_path(flask_name)
    os.makedirs(flask_dir, exist_ok=True)
    return flask_dir

def create_init_file(flask_name: str):
    """Generate __init__.py for each Flask app package."""
    init_path = get_init_path(flask_name)
    module_name = get_module_name(flask_name)
    app_func_name = f"{HUGPY_PRE}_{flask_name}_{APP_POST}"

    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write(f'''from abstract_flask import *
from .{module_name} import {module_name}_bp

bp_list = [
    {module_name}_bp
]
URL_PREFIX = "hugpy/{flask_name}"

def {app_func_name}(debug=True):
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

    @app.route(f"/{{URL_PREFIX}}/endpoints", methods=["GET"])
    def list_endpoints():
        """Return all available endpoints with methods."""
        output = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint != "static":
                methods = list(rule.methods - {{"HEAD", "OPTIONS"}})
                output.append({{
                    "endpoint": rule.endpoint,
                    "url": str(rule),
                    "methods": methods
                }})
        return jsonify(sorted(output, key=lambda x: x["url"]))

    return app
''')
    return init_path


def create_wsgi_file(flask_name: str):
    """Generate WSGI entrypoint under abstract_hugpy/"""
    wsgi_path = get_wsgi_path(flask_name)
    app_func_name = f"{HUGPY_PRE}_{flask_name}_{APP_POST}"
    package_name = get_flask_dir_name(flask_name)

    if not os.path.exists(wsgi_path):
        with open(wsgi_path, "w") as f:
            f.write(f'''from hugpy_console.hugpy_flasks.{package_name} import {app_func_name}

app = {app_func_name}()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
''')
    return wsgi_path


# -------------------
# Orchestrator
# -------------------

def scaffold_flask(flask_name: str):
    create_flask_dir(flask_name)
    init_file = create_init_file(flask_name)
    wsgi_file = create_wsgi_file(flask_name)
    print(f"✅ Scaffolded Flask app: {flask_name}")
    print(f"   Init file → {init_file}")
    print(f"   WSGI file → {wsgi_file}")


if __name__ == "__main__":
    flask_names = ["deepcoder", "proxyvideo", "video"]
    for name in flask_names:
        scaffold_flask(name)
