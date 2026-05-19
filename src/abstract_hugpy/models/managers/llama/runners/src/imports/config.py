from .constants import *
def _load_llama_config(env_path: Optional[str] = None) -> Dict[str, str | int]:
    """Resolve host + per-model ports from env, with defaults as fallback.

    Env keys are the uppercased model_key:
        LLAMA_HOST=http://127.0.0.1
        Qwen2.5-Coder-1.5B-GGUF=6008
        Qwen3-Coder-Next-Q4_K_M=6009
        DAN-L3-R1-8B-i1-GGUF=6090
        Qwen2.5-Coder-3B-GGUF=6091

    """
    cfg: Dict[str, str | int] = {}
    cfg["LLAMA_HOST"] = get_env_value("LLAMA_HOST", path=env_path) or LLAMA_HOST_DEFAULT

    for model_key, default_port in LLAMA_MODEL_PORTS.items():
        raw = get_env_value(model_key.upper(), path=env_path)
        cfg[model_key] = int(raw) if raw else default_port

    return cfg
