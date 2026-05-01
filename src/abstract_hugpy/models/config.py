import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download
from abstract_security import get_env_value


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hub_id: str
    folder: str
    task: str
    framework: str = "transformers"


# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------

MODEL_HOME = Path(
    get_env_value("MODEL_HOME")
    or os.path.expanduser("~/.cache/abstract_models")
)


# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "whisper": ModelConfig(
        name="openai_whisper",
        hub_id="openai/whisper-base",
        folder="whisper_base",
        task="speech-to-text",
    ),

    "keybert": ModelConfig(
        name="keybert",
        hub_id="sentence-transformers/all-MiniLM-L6-v2",
        folder="all_minilm_l6_v2",
        task="embeddings",
    ),

    "summarizer": ModelConfig(
        name="summarizer",
        hub_id="Falconsai/text_summarization",
        folder="text_summarization",
        task="summarization",
    ),

    "flan": ModelConfig(
        name="flan",
        hub_id="google/flan-t5-xl",
        folder="flan_t5_xl",
        task="text-generation",
    ),

    "bigbird": ModelConfig(
        name="bigbird",
        hub_id="allenai/led-large-16384",
        folder="led_large_16384",
        task="long-summarization",
    ),

    "deepcoder": ModelConfig(
        name="deepcoder",
        hub_id="agentica-org/DeepCoder-14B-Preview",
        folder="DeepCoder-14B",
        task="code-generation",
    ),

    # Qwen2.5-VL local image analysis model
    "qwen_vl": ModelConfig(
        name="qwen_vl",
        hub_id="Qwen/Qwen2.5-VL-7B-Instruct",
        folder="Qwen2.5-VL-7B-Instruct",
        task="vision-language",
    ),

    "huggingface": ModelConfig(
        name="huggingface",
        hub_id="huggingface/hub",
        folder="hugging_face_models",
        task="hub-utils",
    ),

    "zerosearch": ModelConfig(
        name="zerosearch",
        hub_id="ZeroSearch/dataset",
        folder="ZeroSearch_dataset",
        task="dataset",
    ),
}


# ---------------------------------------------------------------------
# Registry utilities
# ---------------------------------------------------------------------

def list_models():
    return list(MODEL_REGISTRY.keys())


def get_model_config(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {key}")
    return MODEL_REGISTRY[key]


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

def get_model_path(key: str) -> Path:
    env_override = os.environ.get(f"MODEL_{key.upper()}")

    if env_override:
        return Path(env_override)

    cfg = get_model_config(key)

    return MODEL_HOME / cfg.folder


def model_looks_downloaded(path: Path) -> bool:
    """
    Lightweight check to avoid treating partial Hugging Face / Git-LFS
    pointer directories as usable model directories.
    """
    if not path.exists() or not path.is_dir():
        return False

    if not (path / "config.json").exists():
        return False

    safetensor_files = list(path.glob("*.safetensors"))

    if safetensor_files:
        # Real shards should not be tiny Git-LFS pointer files.
        for file_path in safetensor_files:
            if file_path.stat().st_size < 1024 * 1024:
                return False
        return True

    # Some small models may not use safetensors, so allow tokenizer/config-only
    # models to pass if they have expected Hugging Face files.
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
    cfg = get_model_config(key)
    path = get_model_path(key)

    if not model_looks_downloaded(path):
        path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=cfg.hub_id,
            local_dir=path,
            local_dir_use_symlinks=False,
        )

    return path


def resolve_model_source(key: str) -> str:
    local = get_model_path(key)
    env_override = os.environ.get(f"MODEL_{key.upper()}")

    if env_override and not local.exists():
        raise FileNotFoundError(
            f"MODEL_{key.upper()}={env_override} was set but path doesn't exist"
        )

    if model_looks_downloaded(local):
        return str(local)

    return get_model_config(key).hub_id


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
