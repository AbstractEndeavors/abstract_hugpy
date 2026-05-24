
from .models_dict import MODELS
from .imports import MODELS_DICT_PATH,ModelConfig,safe_load_from_json,Dict,make_list
# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

# Option A: code is the source of truth, disk can add new models only
# Option B: disk wins, but only on a whitelisted set of fields
_OVERLAY_ALLOWED = {"port", "host", "timeout_s", "include"}  # ops knobs only

def get_models_dict():
    nudict = {}
    overlay = safe_load_from_json(MODELS_DICT_PATH) or {}
    for key, defaults in MODELS.items():
        merged = dict(defaults)
        if key in overlay:
            for k, v in overlay[key].items():
                if k in _OVERLAY_ALLOWED:
                    merged[k] = v
                else:
                    logger.warning(
                        "ignoring overlay field %r for model %s "
                        "(not in _OVERLAY_ALLOWED=%s)", k, key, sorted(_OVERLAY_ALLOWED)
                    )
        merged["model_key"] = key
        nudict[key] = ModelConfig(**merged)
    # disk-only models (not in code) — allow only if you want that
    for key, values in overlay.items():
        if key not in MODELS:
            values["model_key"] = key
            nudict[key] = ModelConfig(**values)
    return nudict
MODEL_REGISTRY: Dict[str, ModelConfig] = get_models_dict()

    
def get_context_tokens():
    context_tokens = {}
    for module_key,values in MODEL_REGISTRY.items():
        context_tokens[module_key] = values.model_max_length
    return context_tokens

DEFAULT_CONTEXT_TOKENS_BY_MODEL: dict[str, int] = get_context_tokens()

def default_context_tokens_for_model(model_key: str) -> int:
    return DEFAULT_CONTEXT_TOKENS_BY_MODEL.get(model_key, 8192)


def get_models_dict_by_tasks(tasks=None):
    tasks = make_list(tasks or [])
    models = {}
    for module_key, values in MODEL_REGISTRY.items():
        # was: if values.task in tasks
        if any(t in tasks for t in values.tasks):
            models[module_key] = values
    return models
def get_models_dict_by_names(names=None):
    names = make_list(names or [])
    models = {}
    for module_key,values in MODEL_REGISTRY.items():
        for name in names:
            if name in values.name:
                models[module_key] = values
                break
    return models

DEFAULT_CHAT_MODEL = "Qwen2.5-Coder-3B-Instruct-GGUF"
CHAT_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["text-generation","code-generation"])
if not CHAT_MODELS_REGISTRY.get(DEFAULT_CHAT_MODEL) and CHAT_MODELS_REGISTRY:
    DEFAULT_CHAT_MODEL = list(CHAT_MODELS_REGISTRY.keys())[0]
DEFAULT_MODEL = DEFAULT_CHAT_MODEL

DEFAULT_VISION_MODEL = "Qwen2.5-VL-7B-Instruct"
VISION_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks="image-text-to-text")
if not VISION_MODELS_REGISTRY.get(DEFAULT_VISION_MODEL) and VISION_MODELS_REGISTRY:
    DEFAULT_VISION_MODEL = list(VISION_MODELS_REGISTRY.keys())[0]

DEFAULT_WHISPER_MODEL = "whisper-large-v3"
WHISPER_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_names(names="whisper")
if not WHISPER_MODELS_REGISTRY.get(DEFAULT_WHISPER_MODEL) and WHISPER_MODELS_REGISTRY:
    DEFAULT_WHISPER_MODEL = list(WHISPER_MODELS_REGISTRY.keys())[0]

DEFAULT_EMBED_MODEL = "all-minilm-l6-v2"
EMBED_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(
    tasks=["feature-extraction", "sentence-similarity"]
)
if not EMBED_MODELS_REGISTRY.get(DEFAULT_EMBED_MODEL) and EMBED_MODELS_REGISTRY:
    DEFAULT_EMBED_MODEL = list(EMBED_MODELS_REGISTRY.keys())[0]

