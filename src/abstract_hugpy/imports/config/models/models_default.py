from .models_config import *

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


CHAT_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["text-generation","text-generation-inference","text2text-generation"])
if not CHAT_MODELS_REGISTRY.get(DEFAULT_CHAT_MODEL) and CHAT_MODELS_REGISTRY:
    DEFAULT_CHAT_MODEL = list(CHAT_MODELS_REGISTRY.keys())[0]
DEFAULT_MODEL = DEFAULT_CHAT_MODEL


VISION_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["image-text-to-text","text-to-image"])
if not VISION_MODELS_REGISTRY.get(DEFAULT_VISION_MODEL) and VISION_MODELS_REGISTRY:
    DEFAULT_VISION_MODEL = list(VISION_MODELS_REGISTRY.keys())[0]


WHISPER_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["automatic-speech-recognition","speech-recognition"])
if not WHISPER_MODELS_REGISTRY.get(DEFAULT_WHISPER_MODEL) and WHISPER_MODELS_REGISTRY:
    DEFAULT_WHISPER_MODEL = list(WHISPER_MODELS_REGISTRY.keys())[0]


EMBED_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks=["feature-extraction", "sentence-similarity","sentence-transformers"])
if not EMBED_MODELS_REGISTRY.get(DEFAULT_EMBED_MODEL) and EMBED_MODELS_REGISTRY:
    DEFAULT_EMBED_MODEL = list(EMBED_MODELS_REGISTRY.keys())[0]
