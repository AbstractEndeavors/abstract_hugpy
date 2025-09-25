from abstract_flask import *
from abstract_utilities import *
from ..video_console.manager_utils import *
from ..imports.aggregator import aggregate_from_base_dir
from .imports import *
video_url_bp,logger = get_bp('video_url_bp')
@video_url_bp.route("/download_video", methods=["POST","GET"])
def Download_videoDownload_videoDownload_video():
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
def Extract_video_audioExtract_video_audioExtract_video_audio():
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
def Get_video_whisper_resultGet_video_whisper_resultGet_video_whisper_result():
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
def Get_video_whisper_textGet_video_whisper_textGet_video_whisper_text():
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
def Get_video_whisper_segmentsGet_video_whisper_segmentsGet_video_whisper_segments():
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
def Get_video_metadataGet_video_metadataGet_video_metadata():
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
def Get_video_captionsGet_video_captionsGet_video_captions():
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
@video_url_bp.route("/get_video_thumbnails", methods=["POST","GET"])
def Get_video_thumbnailsGet_video_thumbnailsGet_video_thumbnails():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_video_thumbnails(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_video_info", methods=["POST","GET"])
def Get_video_infoGet_video_infoGet_video_info():
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
def Get_video_directoryGet_video_directoryGet_video_directory():
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
def Get_video_pathGet_video_pathGet_video_path():
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
@video_url_bp.route("/get_audio_path", methods=["POST","GET"])
def Get_audio_pathGet_audio_pathGet_audio_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_audio_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_thumbnail_dir", methods=["POST","GET"])
def Get_thumbnail_dirGet_thumbnail_dirGet_thumbnail_dir():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_thumbnail_dir(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_srt_path", methods=["POST","GET"])
def Get_srt_pathGet_srt_pathGet_srt_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_srt_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_metadata_path", methods=["POST","GET"])
def Get_metadata_pathGet_metadata_pathGet_metadata_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_metadata_path(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_all_data", methods=["POST","GET"])
def Get_all_dataGet_all_dataGet_all_data():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_all_data(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@video_url_bp.route("/get_aggregated_data_dir", methods=["POST","GET"])
def Get_aggregated_data_dirGet_aggregated_data_dirGet_aggregated_data_dir():
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
@video_url_bp.route("/get_aggregated_data_path", methods=["POST","GET"])
def Get_aggregated_data_pathGet_aggregated_data_pathGet_aggregated_data_path():
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
@video_url_bp.route("/get_aggregated_data", methods=["POST","GET"])
def Get_aggregated_dataGet_aggregated_dataGet_aggregated_data():
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
