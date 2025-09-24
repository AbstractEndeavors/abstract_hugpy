from abstract_flask import *
from abstract_utilities import *
from ..video_console import *
from .imports import *
video_url_bp,logger = get_bp('video_url_bp')

@video_url_bp.route("/download_video", methods=["POST","GET"])
def downloadVideo():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = download_video(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/extract_video_audio", methods=["POST","GET"])
def extractVideoAudio():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = extract_video_audio(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_whisper_result", methods=["POST","GET"])
def getVideoWhisperResult():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_whisper_result(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_whisper_text", methods=["POST","GET"])
def getVideoWhisperText():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_whisper_text(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_whisper_segments", methods=["POST","GET"])
def getVideoWhisperSegments():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_whisper_segments(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_metadata", methods=["POST","GET"])
def getVideoMetadata():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_metadata(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_captions", methods=["POST","GET"])
def getVideoCaptions():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_captions(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_info", methods=["POST","GET"])
def getVideoInfo():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_info(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_directory", methods=["POST","GET"])
def getVideoDirectory():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_directory(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_path", methods=["POST","GET"])
def getVideoPath():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_audio_path", methods=["POST","GET"])
def getVideoAudioPath():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_audio_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_srt_path", methods=["POST","GET"])
def getVideoSrtPath():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_srt_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)

@video_url_bp.route("/get_video_metadata_path", methods=["POST","GET"])
def getVideoMetadataPath():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_metadata_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_aggregated_data", methods=["POST","GET"])
def getAggregatedData():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_aggregated_data(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_aggregated_data_path", methods=["POST","GET"])
def getAggregatedDataPath():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_aggregated_data_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_aggregated_data_dir", methods=["POST","GET"])
def getAggregatedDataDir():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_aggregated_data_dir(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
