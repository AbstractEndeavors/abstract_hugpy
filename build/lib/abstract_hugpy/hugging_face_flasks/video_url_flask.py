from abstract_flask import *
from abstract_utilities import *
from ..video_utils import *
video_url_bp,logger = get_bp('video_url_bp')
@video_url_bp.route("/download_video", methods=["POST","GET"])
def downloadVideoUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = download_video(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/extract_video_audio", methods=["POST","GET"])
def extractVideoAudioUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = extract_video_audio(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_whisper_result", methods=["POST","GET"])
def getVideoWhisperResultUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_whisper_result(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_whisper_text", methods=["POST","GET"])
def getVideoWhisperTextUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_whisper_text(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_whisper_segments", methods=["POST","GET"])
def getVideoWhisperSegmentsUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_whisper_segments(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_metadata", methods=["POST","GET"])
def getVideoMetadataUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_metadata(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_captions", methods=["POST","GET"])
def getVideoCaptionsUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_captions(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_info", methods=["POST","GET"])
def getVideoInfoUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_info(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_directory", methods=["POST","GET"])
def getVideoDirectoryUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_directory(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_path", methods=["POST","GET"])
def getVideoPathUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_path(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_audio_path", methods=["POST","GET"])
def getVideoAudioPathUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_audio_path(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_srt_path", methods=["POST","GET"])
def getVideoSrtPathUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_srt_path(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)

@video_url_bp.route("/get_video_metadata_path", methods=["POST","GET"])
def getVideoMetadataPathUrl():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        video_url = data.get('url') or data.get('video_url')
        if not video_url:
            return get_json_call_response(value=f"url in {data}",status_code=400)
        result = get_video_metadata_path(video_url)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_call_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)
