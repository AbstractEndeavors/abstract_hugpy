from abstract_flask import *
from abstract_utilities import *
from ..video_console import *
from .imports import *
proxy_video_url_bp,logger = get_bp('proxy_video_url_bp')

@proxy_video_url_bp.route("/api/download_video", methods=["POST","GET"])
def downloadVideo():
    initialize_call_log()
    try:
        result = get_from_local_host('download_video',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/extract_video_audio", methods=["POST","GET"])
def extractVideoAudio():
    initialize_call_log()
    try:
        result = get_from_local_host('extract_video_audio',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_whisper_result", methods=["POST","GET"])
def getVideoWhisperResult():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_whisper_result',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_whisper_text", methods=["POST","GET"])
def getVideoWhisperText():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_whisper_text',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_whisper_segments", methods=["POST","GET"])
def getVideoWhisperSegments():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_whisper_segments',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_metadata", methods=["POST","GET"])
def getVideoMetadata():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_metadata',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_captions", methods=["POST","GET"])
def getVideoCaptions():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_captions',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_info", methods=["POST","GET"])
def getVideoInfo():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_info',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_directory", methods=["POST","GET"])
def getVideoDirectory():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_directory',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_path", methods=["POST","GET"])
def getVideoPath():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_path',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_audio_path", methods=["POST","GET"])
def getVideoAudioPath():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_audio_path',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_srt_path", methods=["POST","GET"])
def getVideoSrtPath():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_srt_path',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@proxy_video_url_bp.route("/api/get_video_metadata_path", methods=["POST","GET"])
def getVideoMetadataPath():
    initialize_call_log()
    try:
        result = get_from_local_host('get_video_metadata_path',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
