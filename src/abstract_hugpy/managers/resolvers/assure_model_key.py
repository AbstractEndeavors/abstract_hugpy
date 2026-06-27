from .imports import *
from pathlib import PurePosixPath


def _normalize_model_path(value: str) -> str:
    """
    Normalize a registry folder/model path for suffix comparison.
    Works for Unix-style paths and Hugging Face-style repo ids.
    """
    return str(PurePosixPath(str(value).strip().rstrip("/")))


def _path_suffix_matches(folder: str, model_key: str) -> bool:
    """
    Return True when `model_key` matches the trailing path parts of `folder`.

    Examples:
        folder:    /mnt/llm_storage/models/Qwen/Qwen2.5-7B
        model_key: Qwen/Qwen2.5-7B        -> True
        model_key: Qwen2.5-7B             -> True
        model_key: other/Qwen2.5-7B       -> False
    """
    folder_parts = _normalize_model_path(folder).split("/")
    model_parts = _normalize_model_path(model_key).split("/")

    if len(model_parts) > len(folder_parts):
        return False

    return folder_parts[-len(model_parts):] == model_parts


def assure_model_key(model_key):
    """
    Resolve a user-provided model key, repo id, folder name, or folder suffix
    into the canonical key from MODEL_REGISTRY.
    """
    if not model_key:
        return None

    model_key = str(model_key).strip().rstrip("/")

    if model_key in MODEL_REGISTRY:
        return model_key

    for key, values in MODEL_REGISTRY.items():
        folder = getattr(values, "folder", None)

        if not folder:
            continue

        if _path_suffix_matches(folder, model_key):
            return key

    return None
