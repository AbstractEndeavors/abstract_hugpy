# /mnt/24T/hugging_face/new_hugs/keybertManager/server.py
from ..imports import *
whisper_bp,logger = get_bp('whisper_bp')
from .manager_utils import (
    get_whisper_result,
    get_whisper_segments,
    get_whisper_text

)


@whisper_bp.route("/whisper_text", methods=["POST"])
def whisperText():
    args,data = get_args_kwargs(request)
    result = get_whisper_text(*args,**data)
    return jsonify(result)

@whisper_bp.route("/whisper_segments", methods=["POST"])
def whisperSegments():
    args,data = get_args_kwargs(request)
    result = get_whisper_segments(*args,**data)
    return jsonify(result)

@whisper_bp.route("/whisper_result", methods=["POST"])
def whisperResult():
    args,data = get_args_kwargs(request)
    result = get_whisper_result(*args,**data)
    return jsonify(result)
