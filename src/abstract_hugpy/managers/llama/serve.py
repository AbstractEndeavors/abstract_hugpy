"""llama serving — re-export of the canonical :mod:`abstract_hugpy.managers.serve.serve`.

This used to be a byte-for-byte duplicate of ``managers/serve/serve.py``; the two
drifting apart is exactly how a flag could end up fixed in one place and not the
other. There is now a single source of truth, and the llama HTTP runner
(``ccp_runner``) imports the same ``serve_endpoint`` / ``serve_model_name`` the
serve CLI uses.
"""
from ..serve.serve import *  # noqa: F401,F403
from ..serve.serve import serve_endpoint, serve_model_name  # explicit for clarity
