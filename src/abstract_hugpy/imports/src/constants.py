import os
from abstract_security import get_env_value
from typing import Literal, Optional

# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------
HUGGINGFACE_DOMAIN = "https://huggingface.co"

MODELS_HOME =  get_env_value("MODELS_HOME") or os.path.expanduser("~/.cache/abstract_models")

MODELS_DICT_PATH = get_env_value("MODELS_DICT_PATH") or os.path.join(MODELS_HOME,'variables.json')

LLAMA_HOST= get_env_value("LLAMA_HOST") or "http://127.0.0.1"
VISION_HOST= get_env_value("VISION_HOST") or "http://127.0.0.1"

EXCLUDE_DIR_NAMES = frozenset({
    ".cache", ".git", ".locks",
    "snapshots", "blobs", "refs",     # HF cache internals
    "1_Pooling", "2_Normalize",       # sentence-transformers submodules
    "onnx",                            # alt-runtime subfolder, not a model
})
EXCLUDE_DIR_PREFIXES = ("models--",)  # HF cache root naming

DEFAULT_TIMEOUT=3600.0
DEFAULT_MAX_TOKENS=32768
MIN_INPUT_WORDS_DEFAULT = 10
  # whatever key resolve_qwen_vl_path expects
SOURCEKIND = Literal["text", "url", "file", "image"]
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
FINISH_REASONS = Literal["stop", "max_tokens", "cancelled", "error"]
ROLES = Literal["system", "user", "assistant"]


DEFAULT_CHAT_MODEL = "Qwen2.5-Coder-3B-Instruct-GGUF"
DEFAULT_VISION_MODEL = "Qwen2.5-VL-7B-Instruct"
DEFAULT_WHISPER_MODEL = "whisper-large-v3"
DEFAULT_SUMMARIZE_MODEL = "text_summarization"
