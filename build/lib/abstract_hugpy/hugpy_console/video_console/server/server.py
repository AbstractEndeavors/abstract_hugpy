from ..imports import *
from .src import *
from .src.imports import *
def get_pipeline_data(url,key=None):
    p=VideoPipeline(url);
    pipeline_js = {
        "info": p.get_info,
        "download": p.download_video,
        "download_video": p.download_video,
        "extract_audio": p.ensure_audio,
        "whisper": p.get_whisper,
        "captions": p.get_captions,
        "metadata": p.get_metadata,
        "thumbnails": p.get_thumbnails,
        "seodata": p.get_seodata,
        "get_all": p.get_all}
    if key:
        func = pipeline_js.get(key)
        return func()
    return p.get_all()
def get_spec_value(dict_obj,key):
    values = make_list(get_any_value(result,'directory') or None)
    return values[0]
def get_json_response(value,status_code):
    return jsonify({"result":value}),status_code
hugpy_video_bp,logger = get_bp('hugpy_video_bp')

@hugpy_video_bp.route("/download_video", methods=["POST","GET"])
def Download_videoDownload_videoDownload_video():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='download_video')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/extract_video_audio", methods=["POST","GET"])
def Extract_video_audioExtract_video_audioExtract_video_audio():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='extract_audio')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_whisper_result", methods=["POST","GET"])
def Get_video_whisper_resultGet_video_whisper_resultGet_video_whisper_result():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='whisper')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_whisper_text", methods=["POST","GET"])
def Get_video_whisper_textGet_video_whisper_textGet_video_whisper_text():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='whisper').get('text')

        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_whisper_segments", methods=["POST","GET"])
def Get_video_whisper_segmentsGet_video_whisper_segmentsGet_video_whisper_segments():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='whisper').get('segments')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_metadata", methods=["POST","GET"])
def Get_video_metadata():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='metadata')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_captions", methods=["POST","GET"])
def Get_video_captionsGet_video_captionsGet_video_captions():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='captions')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_thumbnails", methods=["POST","GET"])
def Get_video_thumbnailsGet_video_thumbnailsGet_video_thumbnails():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='thumbnails')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_info", methods=["POST","GET"])
def Get_video_infoGet_video_infoGet_video_info():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_directory", methods=["POST","GET"])
def Get_video_directoryGet_video_directoryGet_video_directory():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'directory')
        
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_video_path", methods=["POST","GET"])
def Get_video_pathGet_video_pathGet_video_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'video_path')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_audio_path", methods=["POST","GET"])
def Get_audio_pathGet_audio_pathGet_audio_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'audio_path')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_thumbnail_dir", methods=["POST","GET"])
def Get_thumbnail_dirGet_thumbnail_dirGet_thumbnail_dir():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'thumbnail_directory')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_srt_path", methods=["POST","GET"])
def Get_srt_pathGet_srt_pathGet_srt_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'captions_path')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_metadata_path", methods=["POST","GET"])
def Get_metadata_pathGet_metadata_pathGet_metadata_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'metadata_path')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_all_data", methods=["POST","GET"])
def Get_all_dataGet_all_dataGet_all_data():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='get_all')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_aggregated_data_dir", methods=["POST","GET"])
def Get_aggregated_data_dirGet_aggregated_data_dirGet_aggregated_data_dir():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'aggregated_data_directory')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_aggregated_data_path", methods=["POST","GET"])
def Get_aggregated_data_pathGet_aggregated_data_pathGet_aggregated_data_path():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='info')
        result =get_spec_value(result,'aggregated_data_path')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
@hugpy_video_bp.route("/get_aggregated_data", methods=["POST","GET"])
def Get_aggregated_dataGet_aggregated_dataGet_aggregated_data():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = get_pipeline_data(url,key='aggregated_data')
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
