import os
from abstract_security import get_env_value
from typing import Literal, Optional
from .init_imports import Path
# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------
HUGGINGFACE_DOMAIN = "https://huggingface.co"

MODEL_HOME = Path(
    get_env_value("MODEL_HOME")
    or os.path.expanduser("~/.cache/abstract_models")
)
MODELS_DICT_PATH = Path(
    get_env_value("MODELS_DICT_PATH")
    or os.path.join(MODEL_HOME,'variables.json')
)

EXCLUDE_DIR_NAMES = frozenset({
    ".cache", ".git", ".locks",
    "snapshots", "blobs", "refs",     # HF cache internals
    "1_Pooling", "2_Normalize",       # sentence-transformers submodules
    "onnx",                            # alt-runtime subfolder, not a model
})
EXCLUDE_DIR_PREFIXES = ("models--",)  # HF cache root naming

DEFAULT_MAX_TOKENS=32768
DEFAULT_CHAT_MODEL = DEFAULT_MODEL"Qwen2.5-Coder-3B-Instruct-GGUF"
DEFAULT_VISION_MODEL = "Qwen2.5-VL-7B-Instruct"  # whatever key resolve_qwen_vl_path expects
SOURCEKIND = Literal["text", "url", "file", "image"]
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
FINISH_REASONS = Literal["stop", "max_tokens", "cancelled", "error"]
ROLES = Literal["system", "user", "assistant"]
