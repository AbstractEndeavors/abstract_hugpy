from .imports import *
from .models import *



def resolve_hf_model_dir(base_dir: str) -> str:

    if config_exists(base_dir):
        return base_dir

    snapshots = join_path(base,"snapshots")
    if is_dir(snapshots):
        candidates = [
            p for p in itter_dir(snapshots)
            if is_dir(p) and config_exists(p)
        ]

        if candidates:
            return max(candidates, key=lambda p: st_mtime(p))

    raise FileNotFoundError(f"No usable Hugging Face model dir found under: {base}")

# ---------------------------------------------------------------------
# Registry utilities
# ---------------------------------------------------------------------

def list_models():
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_key: str=None,dict_return=False,return_dict=False,key: str=None) -> ModelConfig or dict:
    model_key = model_key or key
    model_registry = get_model_registry(dict_return=dict_return,return_dict=return_dict)
   
    if model_key not in model_registry:
        raise KeyError(f"Unknown model: {model_key}")
    return model_registry[model_key]


def list_model_options():
    return {
        key: {
            "name": cfg.name,
            "hub_id": cfg.hub_id,
            "folder": cfg.folder,
            "tasks": cfg.tasks,
            "framework": cfg.framework,
            "filename": cfg.filename,
            "max_new_tokens": cfg.max_new_tokens,
            "port": cfg.port,
        
        }
        for key, cfg in MODEL_REGISTRY.items()
    }

# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

def get_model_path(key: str):
    env_override = get_env_value(f"MODEL_{key.upper()}")
    if env_override:
        return env_override
    cfg = get_model_config(key)
    path = os.path.join(MODELS_HOME,cfg.folder)
    return path


def get_gguf_file(path: str, cfg: ModelConfig) -> Optional[str]:
    if cfg.filename:
        candidate = os.path.join(path,cfg.filename)
        if exists(candidate):
            return candidate

    ggufs = get_glob(path,"*.gguf")
    if ggufs:
        return ggufs[0]

    recursive_ggufs = get_glob(path,"*.gguf")
    if recursive_ggufs:
        return recursive_ggufs[0]

    return None


def model_looks_downloaded(path: str, cfg: Optional[ModelConfig] = None) -> bool:
    """
    Lightweight check to avoid treating partial Hugging Face / Git-LFS
    pointer directories as usable model directories.

    Supports both:
      - transformers model dirs
      - GGUF model dirs for llama.cpp
    """
    if not exists(path) or not is_dir(path):
        return False

    if cfg and cfg.framework == "llama_cpp":
        gguf = get_gguf_file(path, cfg)
        return bool(gguf and exists(gguf) and st_size(gguf) > 1024 * 1024)

    if not config_exists(path):
        return False

    safetensor_files = list(get_glob(path,"*.safetensors"))

    if safetensor_files:
        for file_path in safetensor_files:
            if st_size(file_path) < 1024 * 1024:
                return False
        return True

    expected_any = [
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]

    return any(exists(join_path(path,name)) for name in expected_any)

# ---------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------




def resolve_model_source(key: str) -> str:
    cfg = get_model_config(key)
    local = get_model_path(key)
    env_override = get_env_value(f"MODEL_{key.upper()}")

    if env_override and not exists(local):
        raise FileNotFoundError(
            f"MODEL_{key.upper()}={env_override} was set but path does not exist"
        )

    if cfg.framework == "llama_cpp":
        if not model_looks_downloaded(local, cfg):
            return cfg.hub_id

        gguf = get_gguf_file(local, cfg)
        if not gguf:
            raise FileNotFoundError(f"No GGUF file found in {local}")

        return str(gguf)

    if model_looks_downloaded(local, cfg):
        return str(local)

    return cfg.hub_id


# ---------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------

class _LazyModelPaths:
    """
    Dict-like that resolves on access, not import.

    Managers that cache DEFAULT_PATHS["foo"] in __init__ get the
    correct value at construction time — even if the model was
    downloaded or deleted after the module was first imported.
    """

    def __getitem__(self, key: str) -> str:
        return resolve_model_source(key)

    def get(self, key: str, default=None) -> str:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in MODEL_REGISTRY


DEFAULT_PATHS: _LazyModelPaths = _LazyModelPaths()
