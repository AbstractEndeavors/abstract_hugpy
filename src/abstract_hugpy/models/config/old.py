import os
from dataclasses import dataclass
from pathlib import Path
from typing import *

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
    filename: Optional[str] = None
    include: Optional[str] = None

@dataclass(frozen=True)
class DeepCoderRuntime:
    model_dir: str
    device: str
    torch_dtype: Any
    use_quantization: bool = False
    use_flash_attention: bool = False
    local_files_only: bool = True
    max_new_tokens_cap: int = 512
    max_concurrent_generations: int = 1
 
    def cache_key(self) -> tuple:
        return (
            self.model_dir,
            self.device,
            str(self.torch_dtype),
            self.use_quantization,
            self.use_flash_attention,
            self.local_files_only,
            self.max_new_tokens_cap,
            self.max_concurrent_generations,
        )
 

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
    "dan_qwen3_1_7b": ModelConfig(
        name="dan_qwen3_1_7b",
        hub_id="UnfilteredAI/DAN-Qwen3-1.7B",  # for reference/metadata only
        folder="UnfilteredAI/DAN-Qwen3-1.7B",  # logical folder name
        task="text-generation"
    ),
    # Qwen2.5-VL local image analysis model
    "qwen_vl": ModelConfig(
        name="qwen_vl",
        hub_id="Qwen/Qwen2.5-VL-7B-Instruct",
        folder="Qwen2.5-VL-7B-Instruct",
        task="vision-language",
    ),
    "qwen25_coder_1_5b_gguf": ModelConfig(
        name="qwen25_coder_1_5b_gguf",
        hub_id="bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        folder="Qwen/Qwen2.5-Coder-1.5B-GGUF",
        task="code-generation",
        framework="llama_cpp",
        filename="Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        include="*Q4_K_M.gguf",
    ),
    "qwen36_35b_a3b": ModelConfig(
        name="qwen36_35b_a3b",
        hub_id="Qwen/Qwen3.6-35B-A3B",
        folder="Qwen/Qwen3.6-35B-A3B",
        task="vision-language",
        framework="transformers",
        filename=None,
        include="*.safetensors",
    ),
    "qwen25_coder_3b_gguf": ModelConfig(
        name="qwen25_coder_3b_gguf",
        hub_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        folder="Qwen/Qwen2.5-Coder-3B-GGUF",
        task="code-generation",
        framework="llama_cpp",
        filename=None,
        include="*Q4_K_M.gguf",
    ),

    "qwen3_coder_next_gguf": ModelConfig(
        name="qwen3_coder_next_gguf",
        hub_id="Qwen/Qwen3-Coder-Next-GGUF",
        folder="Qwen/Qwen3-Coder-Next-GGUF",
        task="code-generation",
        framework="llama_cpp",
        filename="Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
        include="Qwen3-Coder-Next-Q4_K_M/*.gguf",
    ),
    "dan_l3_r1_8b_i1_gguf": ModelConfig(
        name="dan_l3_r1_8b_i1_gguf",
        hub_id="mradermacher/DAN-L3-R1-8B-i1-GGUF",
        folder="mradermacher/DAN-L3-R1-8B-i1-GGUF",
        task="text-generation",
        framework="llama_cpp",
        filename="DAN-L3-R1-8B.i1-Q4_K_M.gguf",
        include="*Q4_K_M.gguf",
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

def get_legacy(name):
    names_js = {"bigbird":'led_large_16384',"summarizer":'text_summarization',"flan":'flan_t5_xl',"deepcoder":'DeepCoder-14B',"keybert":'all_minilm_l6_v2',"zerosearch":'ZeroSearch_model',"whisper":'whisper-large-v3'}
    return names_js.get(key) or key
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
