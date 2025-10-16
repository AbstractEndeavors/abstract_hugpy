from ..imports import *
from .manager import *
import os
import json
from abstract_utilities.read_write_utils import read_from_file, write_to_file
from abstract_utilities.list_utils import make_list
from abstract_utilities.path_utils import makeAllDirs
import logging

logger = logging.getLogger(__name__)


def _safe_read_from_json(file_path, remove_if_corrupt=True, default=None):
    """
    Read JSON safely. If the file is corrupt, optionally remove it and return default.
    """
    if not file_path or not os.path.isfile(file_path):
        return default if default is not None else {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Corrupt JSON detected at {file_path}: {e}")
        if remove_if_corrupt:
            try:
                os.remove(file_path)
                logger.warning(f"Removed corrupt JSON file: {file_path}")
            except Exception as rm_err:
                logger.error(f"Failed to remove corrupt JSON {file_path}: {rm_err}")
        return default if default is not None else {}


def _safe_dump_to_file(data, file_path=None, ensure_ascii=False, indent=4):
    """
    Dump JSON safely. If data is not JSON serializable, stringify it.
    """
    if not file_path:
        logger.error("safe_dump_to_file called without file_path")
        return

    if data is None:
        logger.warning(f"safe_dump_to_file got None for {file_path}, defaulting to empty dict")
        data = {}

    # Sanitize data for JSON
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    except (TypeError, ValueError) as e:
        logger.error(f"Non-serializable data for {file_path}: {e}, saving as string")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(data))


def try_get_data(file_path):
    """
    Try to read JSON. If file is missing or corrupt, ensure it's replaced with {}.
    """
    if not file_path:
        return {}

    data = _safe_read_from_json(file_path)
    if not data:
        _safe_dump_to_file({}, file_path)
        return {}
    return data

def try_get_read(file_path):
    try:
        data = _safe_read_from_json(file_path)
        return data
    except:
        return False
    
def try_get_data(file_path):
    data = try_get_read(file_path)
    if not os.path.isfile(file_path) or not data:
        safe_dump_to_json(data={},file_path=file_path)
        return False
    return data
    
def get_whisper_result(
        *args,
        audio_path=None,
        video_url=None,
        video_path=None,
        **kwargs
        ):
    whisper_path = get_whisper_path(
        video_url=video_url,
        video_path=video_path
        )
    data = try_get_data(whisper_path)
    if not data:
        audio_path = get_whisper_audio_path(
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path
            )        
        whisper_result = run_whisper_func(
                whisper_transcribe,
                *args,
                audio_path=audio_path,
                **kwargs
                )
        
        safe_dump_to_file(
            data=whisper_result,
            file_path=whisper_path
            )
    return try_get_data(whisper_path)
def get_whisper_text(
        *args,
        audio_path=None,
        video_url=None,
        video_path=None,
        **kwargs
        ):
    whisper_result = get_whisper_result(
            *args,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path,
            **kwargs
            )
    return whisper_result.get('text')
def get_whisper_segments(
        *args,
        audio_path=None,
        video_url=None,
        video_path=None,
        **kwargs
        ):
    whisper_result = get_whisper_result(
            *args,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path,
            **kwargs
            )
    return whisper_result.get('segments')
