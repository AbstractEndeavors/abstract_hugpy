
from .models_dict import MODELS
from .imports import MODELS_DICT_PATH,ModelConfig,safe_load_from_json,Dict,make_list
# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

def get_models_dict():
    nudict = {}
    models_dict = safe_load_from_json(MODELS_DICT_PATH) or {}
    MODELS.update(models_dict)
    for key,values in MODELS.items():
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
    for module_key,values in MODEL_REGISTRY.items():
        if values.task in tasks:
            models[module_key] = values
    return models


VISION_MODELS_REGISTRY: Dict[str, ModelConfig] = get_models_dict_by_tasks(tasks="vision-language")
