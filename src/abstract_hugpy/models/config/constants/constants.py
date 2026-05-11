from .classes import *
from .models_dict import MODELS
# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------
def get_models_dict_path():
    return get_env_value('MODELS_DICT_PATH')
def get_models_dict():
    abs_dir = get_caller_dir()
    nudict = {}
    models_json_path = get_models_dict_path()
    models_dict = safe_load_from_json(models_json_path)
    MODELS.update(models_dict)
    for key,values in MODELS.items():
        nudict[key] = ModelConfig(**values)

    return nudict
MODEL_REGISTRY: Dict[str, ModelConfig] = get_models_dict()



