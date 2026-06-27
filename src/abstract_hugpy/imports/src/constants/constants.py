import os
from abstract_security import get_env_value
from typing import Literal, Optional
from .imports import make_list,HfApi,re
# Tokenizers set this as a sentinel for "no enforced limit". It's never a real window.

# ---------------------------------------------------------------------
# Model storage root
# ---------------------------------------------------------------------
HUGGINGFACE_DOMAIN = "https://huggingface.co"

HF_TOKEN = get_env_value("HF_TOKEN") or False

hfApi = HfApi(token=HF_TOKEN)

DEFAULT_ROOT =  get_env_value("DEFAULT_ROOT") or "/mnt/llm_storage"

MODELS_HOME = MODELS_DIR =  get_env_value("MODELS_HOME") or os.path.join(DEFAULT_ROOT,"models")

UPLOADS_HOME = CHAT_UPLOAD_DIR =  get_env_value("UPLOADS_HOME") or os.path.join(DEFAULT_ROOT,"uploads")

PROJECTS_HOME = PROJECTS_DIR =  get_env_value("PROJECTS_HOME") or os.path.join(DEFAULT_ROOT,"projects")

DATASETS_HOME = DATASETS_DIR =  get_env_value("DATASETS_HOME") or os.path.join(DEFAULT_ROOT,"datasets")

MODELS_DISCOVERY_PATH = get_env_value("MODELS_DISCOVERY_PATH") or os.path.join(PROJECTS_HOME,"model_discovery.json")

MODELS_DICT_PATH = get_env_value("MODELS_DICT_PATH") or os.path.join(PROJECTS_HOME,"model_manifest.json")

HF_CACHE = get_env_value("HF_CACHE") or os.path.join(MODELS_HOME,"cache")

HF_HOME = get_env_value("HF_HOME") or os.path.join(HF_CACHE,"huggingface")

HF_HUB_CACHE = get_env_value("HF_HUB_CACHE") or os.path.join(HF_HOME,"hub")

TORCH_HOME = get_env_value("TORCH_HOME") or os.path.join(HF_CACHE,"torch")

PIP_CACHE_DIR = get_env_value("PIP_CACHE_DIR") or os.path.join(HF_CACHE,"pip")

PATHS = [
    MODELS_DIR,
    DATASETS_DIR,
    HF_HOME,
    HF_HUB_CACHE,
    TORCH_HOME,
    PIP_CACHE_DIR,
]


def _ensure_dirs(paths):
    """Best-effort create the storage dirs.

    Importing abstract_hugpy must never hard-crash just because a storage path
    can't be made — e.g. on a worker box where DEFAULT_ROOT (/mnt/llm_storage)
    is a broken/stale mount (OSError errno 5) or simply not present. Each dir is
    created independently; failures are warned about, not fatal. Set
    DEFAULT_ROOT to a local, writable path on such boxes.
    """
    import logging
    failed = []
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            failed.append((path, exc))
    if failed:
        logging.getLogger("abstract_hugpy").warning(
            "could not create %d storage dir(s); continuing. "
            "Set DEFAULT_ROOT to a writable path to silence this. Details: %s",
            len(failed),
            "; ".join(f"{p} ({e.__class__.__name__}: {e})" for p, e in failed),
        )


_ensure_dirs(PATHS)

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("TORCH_HOME", TORCH_HOME)
os.environ.setdefault("PIP_CACHE_DIR", PIP_CACHE_DIR)

HUGPY_MARKER= get_env_value("HUGPY_MARKER") or "hugpy.json"

LLAMA_HOST= get_env_value("LLAMA_HOST") or "http://127.0.0.1"
VISION_HOST= get_env_value("VISION_HOST") or "http://127.0.0.1"

EXCLUDE_DIR_NAMES = make_list(get_env_value("EXCLUDE_DIR_NAMES") or ".cache,.git,.locks,snapshots,blobs,refs,1_Pooling,2_Normalize,onnx")
EXCLUDE_DIR_NAMES = frozenset(EXCLUDE_DIR_NAMES)

EXCLUDE_DIR_PREFIXES = make_list(get_env_value("EXCLUDE_DIR_PREFIXES") or "models--")
EXCLUDE_DIR_PREFIXES = tuple(EXCLUDE_DIR_PREFIXES)  # HF cache root naming

TOKENIZER_SENTINEL_THRESHOLD = float(get_env_value("TOKENIZER_SENTINEL_THRESHOLD") or 10**9)
DEFAULT_TIMEOUT= float(get_env_value("DEFAULT_TIMEOUT") or 3600.0)
DEFAULT_MAX_TOKENS= int(get_env_value("DEFAULT_MAX_TOKENS") or 32768)
MIN_INPUT_WORDS_DEFAULT = get_env_value("MIN_INPUT_WORDS_DEFAULT") or 10
  # whatever key resolve_qwen_vl_path expects
  
SOURCEKIND = make_list(get_env_value("SOURCEKIND") or "text,url,file,image")
SOURCEKIND =Literal[*SOURCEKIND]

JOBSTATUS = make_list(get_env_value("JOBSTATUS") or "queued,running,completed,failed,cancelled")
JOBSTATUS =Literal[*JOBSTATUS]

DEFAULT_TEMPERATURE = float(get_env_value("DEFAULT_TEMPERATURE") or 0.1)
DEFAULT_TOP_P = float(get_env_value("DEFAULT_TOP_P") or 1)

FINISH_REASONS = make_list(get_env_value("FINISH_REASONS") or "stop,max_tokens,cancelled,error")
FINISH_REASONS =Literal[*FINISH_REASONS]

ROLES = make_list(get_env_value("ROLES") or "system,user,assistant")
ROLES = Literal[*ROLES]

DEFAULT_CHAT_MODEL = get_env_value("DEFAULT_CHAT_MODEL") or "Qwen2.5-Coder-3B-Instruct-GGUF"
DEFAULT_VISION_MODEL = get_env_value("DEFAULT_VISION_MODEL") or "Qwen2.5-VL-7B-Instruct"
DEFAULT_WHISPER_MODEL = get_env_value("DEFAULT_WHISPER_MODEL") or "whisper-large-v3"
DEFAULT_SUMMARIZE_MODEL = get_env_value("DEFAULT_SUMMARIZE_MODEL") or "text_summarization"
DEFAULT_EMBED_MODEL = get_env_value("DEFAULT_EMBED_MODEL") or "all-minilm-l6-v2"

DISK_AUTHORITATIVE = make_list(get_env_value("DISK_AUTHORITATIVE") or "name,folder,framework,filename")
OVERLAY_ALLOWED = set(make_list(get_env_value("OVERLAY_ALLOWED") or "port, host, timeout_s, include"))

GGUF_QUANT = re.compile(r"(Q\d+_[A-Z0-9_]+|F16|BF16|F32)", re.I)

DEFAULT_LOCAL_FILES_ONLY = get_env_value("DEFAULT_LOCAL_FILES_ONLY") or True
