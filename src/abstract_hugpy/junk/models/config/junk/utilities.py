from .constants import *
# ---------------------------------------------------------------------
# Registry utilities
# ---------------------------------------------------------------------

def list_models():
    return list(MODEL_REGISTRY.keys())+list(names_js.keys())


def get_model_config(key: str) -> ModelConfig:
    key = get_legacy(key)
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {key}")
    return MODEL_REGISTRY[key]


def list_model_options():
    return {
        key: {
            "name": cfg.name,
            "hub_id": cfg.hub_id,
            "folder": cfg.folder,
            "task": cfg.task,
            "framework": cfg.framework,
            "filename": cfg.filename,
        }
        for key, cfg in MODEL_REGISTRY.items()
    }

# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

def get_model_path(key: str) -> Path:
    key = get_legacy(key)
    env_override = os.environ.get(f"MODEL_{key.upper()}")

    if env_override:
        return Path(env_override)

    cfg = get_model_config(key)
    return MODEL_HOME / cfg.folder


def get_gguf_file(path: Path, cfg: ModelConfig) -> Optional[Path]:
    if cfg.filename:
        candidate = path / cfg.filename
        if candidate.exists():
            return candidate

    ggufs = sorted(path.glob("*.gguf"))
    if ggufs:
        return ggufs[0]

    recursive_ggufs = sorted(path.rglob("*.gguf"))
    if recursive_ggufs:
        return recursive_ggufs[0]

    return None


def model_looks_downloaded(path: Path, cfg: Optional[ModelConfig] = None) -> bool:
    """
    Lightweight check to avoid treating partial Hugging Face / Git-LFS
    pointer directories as usable model directories.

    Supports both:
      - transformers model dirs
      - GGUF model dirs for llama.cpp
    """
    if not path.exists() or not path.is_dir():
        return False

    if cfg and cfg.framework == "llama_cpp":
        gguf = get_gguf_file(path, cfg)
        return bool(gguf and gguf.exists() and gguf.stat().st_size > 1024 * 1024)

    if not (path / "config.json").exists():
        return False

    safetensor_files = list(path.glob("*.safetensors"))

    if safetensor_files:
        for file_path in safetensor_files:
            if file_path.stat().st_size < 1024 * 1024:
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

    return any((path / name).exists() for name in expected_any)

# ---------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------

def ensure_model(key: str) -> Path:
    key = get_legacy(key)
    cfg = get_model_config(key)
    path = get_model_path(key)

    if model_looks_downloaded(path, cfg):
        return path

    path.mkdir(parents=True, exist_ok=True)

    download_kwargs = {
        "repo_id": cfg.hub_id,
        "local_dir": path,
        "local_dir_use_symlinks": False,
    }

    if cfg.include:
        download_kwargs["allow_patterns"] = cfg.include

    snapshot_download(**download_kwargs)

    return path


def resolve_model_source(key: str) -> str:
    key = get_legacy(key)
    cfg = get_model_config(key)
    local = get_model_path(key)
    env_override = os.environ.get(f"MODEL_{key.upper()}")

    if env_override and not local.exists():
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
        key = get_legacy(key)
        return resolve_model_source(key)

    def get(self, key: str, default=None) -> str:
        key = get_legacy(key)
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        key = get_legacy(key)
        return key in MODEL_REGISTRY


DEFAULT_PATHS: _LazyModelPaths = _LazyModelPaths()
