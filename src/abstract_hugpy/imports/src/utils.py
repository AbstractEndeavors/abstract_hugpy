from __future__ import annotations
import uuid
from pathlib import Path
from typing import *
import glob
import os
import re,os
from .init_imports import get_logFile
logger = get_logFile(__name__)

def get_glob(path,ext):
    return sorted(glob.glob(os.path.join(path, ext)))
def exists(obj):
    try:
        if obj and os.path.exists(str(obj)):
            return True
    except Exception as e:
        print(f"exists: {e}")
        return False
    return False
def is_file(obj):
    try:
        if obj and os.path.isfile(str(obj)):
            return True
    except Exception as e:
        print(f"is_file: {e}")
        return False
    return False
def is_dir(obj):
    try:
        if obj and os.path.isdir(str(obj)):
            return True
    except Exception as e:
        print(f"is_dir: {e}")
        return False
    return False
def get_stat(obj):
    try:
        if obj and isinstance(obj,str):
            obj = Path(obj)
        stat = obj.stat()
        return stat
    except Exception as e:
        print(f"stat: {e}")
        return False
    return False
def st_size(obj):
    try:
        stat = get_stat(obj)
        if stat:
            return stat.st_size
    except Exception as e:
        print(f"st_size: {e}")
        return False
    return False
def st_mtime(obj):
    try:
        stat = get_stat(obj)
        if stat:
            return stat.st_mtime
    except Exception as e:
        print(f"st_size: {e}")
        return False
    return False
def itter_dir(directory):
    itter = []
    if directory and os.path.isdir(directory):
        itter = os.listdir(directory)
    return itter

def get_message(prompt:str=None,role:str=None,content:str=None):
    content = content or prompt
    role = role or "user"
    return {"role": role, "content": content}

def get_messages(prompt:str=None,role:str=None,content:str=None) -> List[dict]:
    message = get_message(prompt=prompt,role=role,content=content)
    return [message]

def config_exists(directory):
    if directory and is_dir(directory):
        json_path = os.path.join(directory,'config.json')
        return is_file(json_path)
    return False
def get_request_id() -> str:
    return str(uuid.uuid1())

def get_parts(obj: str) -> List[str]:
    return [item for item in obj.split("/") if item]

def find_keys_by_type(obj, target_type, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, target_type):
                yield path + (k,)
            yield from find_keys_by_type(v, target_type, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from find_keys_by_type(v, target_type, path + (i,))

def get_unique_part(parts: List[str], comp_parts: List[str]) -> List[str]:
    return [part for part in parts if part not in comp_parts]

def safe_dtype_name(value: Any) -> str:
    """
    Converts torch dtypes or dtype-like objects into stable string values.

    Examples:
        torch.float16  -> "torch.float16"
        torch.bfloat16 -> "torch.bfloat16"
        "auto"         -> "auto"
    """
    if value is None:
        return "None"

    return str(value)


def message_to_dict(message: Any) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()

    if isinstance(message, dict):
        return {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }

    return {
        "role": str(getattr(message, "role", "user")),
        "content": str(getattr(message, "content", "")),
    }


def messages_to_dicts(messages: list[Any]) -> list[dict]:
    return [message_to_dict(message) for message in messages]


def slugify(value: str, fallback: str = "media") -> str:
    value = value.strip()
    value = re.sub(r"[^\w.\- ]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._-")
    return value or fallback


def unique_path(path) -> str:
    if not os.path.exists(path):
        return path
    parent = os.path.dirname(path)
    basename = os.path.basename(path)
    stem,suffix = os.path.splitext(basename)

    for index in range(1, 10_000):
        basename = f"{stem}_{index}{suffix}"
        candidate = os.path.join(parent,basename)
        if not os.path.exists(candidate):
            return candidate

    raise RuntimeError(f"Could not create unique path for: {path}")

