# /mnt/24T/hugging_face/new_hugs/hugging_face_app/modules/keybertManager/src/client.py
import requests
import logging
from abstract_utilities import SingletonMeta

logger = logging.getLogger(__name__)

def get_args_kwargs_json(*args, **kwargs):
    kwargs["args"] = args
    return kwargs

class KeybertClient(metaclass=SingletonMeta):
    """
    Strict KeyBERT client: always calls the running Flask/Gunicorn service.
    If the service is unreachable, raises an exception instead of falling back.
    """

    def __init__(self, service_url: str, timeout: int = 30):
        """
        Args:
            service_url (str): Base URL of the Flask service,
                               e.g. "http://localhost:6081/hugpy".
            timeout (int): HTTP timeout in seconds.
        """
        if not hasattr(self, "initialized"):
            if not service_url:
                raise ValueError("service_url must be provided for KeybertClient")
            self.initialized = True
            self.service_url = service_url.rstrip("/")
            self.timeout = timeout
            logger.info('hi im rstyarting again')
    def _post(self, endpoint: str, *args, **kwargs):
        url = f"{self.service_url}{endpoint}"
        try:
            resp = requests.post(
                url,
                json=get_args_kwargs_json(*args, **kwargs),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"HTTP KeyBERT {endpoint} failed: {e.__class__.__name__}: {e}")
            raise   # <-- hard fail (no local fallback)

    # ------------------------
    # Keywords
    # ------------------------
    def keywords(self, *args, **kwargs):
        return self._post("/keywords", *args, **kwargs).get("keywords", [])

    # ------------------------
    # Refine
    # ------------------------
    def refine(self, *args, **kwargs):
        return self._post("/refine", *args, **kwargs)

    # ------------------------
    # Density
    # ------------------------
    def density(self, *args, **kwargs):
        return self._post("/density", *args, **kwargs)
