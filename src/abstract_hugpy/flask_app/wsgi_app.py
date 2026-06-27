from .app import *
from .app import routes as routes
def get_hugpy_flask(name=None,allowed_origins=None,debug=False):
    name = name or "hugpy_flask"
    return get_Flask_app(
        name=name,
        routes=routes,
        allowed_origins=allowed_origins,
        debug=debug
    )
