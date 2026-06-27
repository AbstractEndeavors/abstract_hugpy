from .src import *

# ---------------------------------------------------------------------------
# Process-local singleton cache for the heavy GGUF runners.
# Keyed by model_key (str). The adapter wrappers in chat_runner share these.
# ---------------------------------------------------------------------------

_LLAMA_INSTANCES: Dict[str, "LlamaCppBaseRunner"] = {}
_LLAMA_LOCK = threading.Lock()


def get_llama_runner(model_key: str) -> "LlamaCppBaseRunner":
    """Get-or-build the singleton runner for a model_key.

    HTTP runner first (cheap probe); falls back to in-process Python.
    """
    if not isinstance(model_key, str):
        raise TypeError(
            f"get_llama_runner expects model_key: str, got {type(model_key).__name__}"
        )

    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.get(model_key)
        if runner is None:
            runner = _build_runner(model_key)
            _LLAMA_INSTANCES[model_key] = runner
        return runner


def _build_runner(model_key: str) -> "LlamaCppBaseRunner":
    try:
        candidate = LlamaCppRunner(model_key)  # HTTP runner
        # quick probe — if the server isn't up this will throw
        with httpx.Client(timeout=2.0) as client:
            client.get(f"{candidate.base_url}/health").raise_for_status()
        logger.info("get_llama_runner: using HTTP runner for %s", model_key)
        return candidate
    except Exception:
        logger.info(
            "get_llama_runner: HTTP unavailable, falling back to in-process for %s",
            model_key,
        )
        return LlamaCppPythonRunner(model_key)
