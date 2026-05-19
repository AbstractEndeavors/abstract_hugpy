from .src import *
# ---------------------------------------------------------------------------
# Process-local registry (kept as-is; survives across HTTP requests)
# ---------------------------------------------------------------------------

_LLAMA_INSTANCES: Dict[str, "LlamaCppPythonRunner"] = {}
_LLAMA_LOCK = threading.Lock()


def get_llama_runner(model_key: str) -> LlamaCppBaseRunner:
    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.get(model_key)
        if runner is None:
            runner = _build_runner(model_key)
            _LLAMA_INSTANCES[model_key] = runner
        return runner


def _build_runner(model_key: str) -> LlamaCppBaseRunner:
    try:
        candidate = LlamaCppRunner(model_key)  # HTTP runner
        # quick probe — if the server isn't up this will throw
        with httpx.Client(timeout=2.0) as client:
            client.get(f"{candidate.base_url}/health").raise_for_status()
        logger.info("get_llama_runner: using HTTP runner for %s", model_key)
        return candidate
    except Exception:
        logger.info("get_llama_runner: HTTP unavailable, falling back to in-process for %s", model_key)
        return LlamaCppPythonRunner(model_key)
