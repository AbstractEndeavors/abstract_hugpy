from .init_imports import *
# ---------------------------------------------------------------------------
# Defaults — single source of truth for "what does the runner do when the
# request omits a value." Override per-runner via constructor or per-request
# via ChatRequest.
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 2048      # was 512 / 256 / scattered; 2048 is the floor for useful coder outputs
DEFAULT_N_CTX = 16384          # was 4096; small ctx silently truncated long outputs
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_HTTP_TIMEOUT = 120.0   # non-streaming HTTP only; streaming uses None


# ---------------------------------------------------------------------------
# Env / port wiring (host:port discovery for the HTTP runner)
# ---------------------------------------------------------------------------

LLAMA_HOST_DEFAULT = "http://127.0.0.1"

LLAMA_MODEL_PORTS: Dict[str, int] = {
    "Qwen2.5-Coder-1.5B-GGUF":6008,
    "Qwen3-Coder-Next-Q4_K_M":6009,
    "DAN-L3-R1-8B-i1-GGUF":6090,
    "Qwen2.5-Coder-3B-GGUF":6091,
    "flux":6092,

}

# llama.cpp says 'length' / 'stop'; schema says 'max_tokens' / 'stop'.
FINISH_REASON_MAP = {
    "length": "max_tokens",
    "stop": "stop",
    None: "stop",
}
