from .imports import *
from .main import get_video_mgr
from abstract_utilities import capitalize
def download_video(video_url): return get_video_mgr().download_video(video_url)
def extract_video_audio(video_url): return get_video_mgr().extract_audio(video_url)
def get_video_whisper_result(video_url): return get_video_mgr().get_whisper_result(video_url)
def get_video_whisper_text(video_url): return get_video_mgr().get_whisper_text(video_url)
def get_video_whisper_segments(video_url): return get_video_mgr().get_whisper_segments(video_url)
def get_video_metadata(video_url): return get_video_mgr().get_metadata(video_url)
def get_video_captions(video_url): return get_video_mgr().get_captions(video_url)
def get_video_thumbnails(video_url): return get_video_mgr().get_thumbnails(video_url)
def get_video_info(video_url): return get_video_mgr().get_data(video_url).get('info')
def get_video_directory(video_url): return get_video_mgr().get_data(video_url).get('directory')
def get_video_path(video_url): return get_video_mgr().get_data(video_url).get('video_path')
def get_audio_path(video_url): return get_video_mgr().get_data(video_url).get('audio_path')
def get_thumbnail_dir(video_url): return get_video_mgr().get_data(video_url).get('thumbnail_dir')
def get_srt_path(video_url): return get_video_mgr().get_data(video_url).get('srt_path')
def get_metadata_path(video_url): return get_video_mgr().get_data(video_url).get('metadata_path')
def get_all_data(video_url): return get_video_mgr().get_all_data(video_url).get('metadata_path')
def get_aggregated_data_dir(video_url): return get_video_mgr().get_data(video_url).get('aggregated_dir')
def get_aggregated_data_path(video_url): return get_video_mgr().get_data(video_url).get('total_aggregated_path')
def get_aggregated_data(video_url): return get_video_mgr().get_all_aggregated_data(video_url)


texts="""def download_video(video_url): return get_video_mgr().download_video(video_url)
def extract_video_audio(video_url): return get_video_mgr().extract_audio(video_url)
def get_video_whisper_result(video_url): return get_video_mgr().get_whisper_result(video_url)
def get_video_whisper_text(video_url): return get_video_mgr().get_whisper_text(video_url)
def get_video_whisper_segments(video_url): return get_video_mgr().get_whisper_segments(video_url)
def get_video_metadata(video_url): return get_video_mgr().get_metadata(video_url)
def get_video_captions(video_url): return get_video_mgr().get_captions(video_url)
def get_video_thumbnails(video_url): return get_video_mgr().get_thumbnails(video_url)
def get_video_info(video_url): return get_video_mgr().get_data(video_url).get('info')
def get_video_directory(video_url): return get_video_mgr().get_data(video_url).get('directory')
def get_video_path(video_url): return get_video_mgr().get_data(video_url).get('video_path')
def get_audio_path(video_url): return get_video_mgr().get_data(video_url).get('audio_path')
def get_thumbnail_dir(video_url): return get_video_mgr().get_data(video_url).get('thumbnail_dir')
def get_srt_path(video_url): return get_video_mgr().get_data(video_url).get('srt_path')
def get_metadata_path(video_url): return get_video_mgr().get_data(video_url).get('metadata_path')
def get_all_data(video_url): return get_video_mgr().get_all_data(video_url).get('metadata_path')
def get_aggregated_data_dir(video_url): return get_video_mgr().get_data(video_url).get('aggregated_dir')
def get_aggregated_data_path(video_url): return get_video_mgr().get_data(video_url).get('total_aggregated_path')
def get_aggregated_data(video_url): return get_video_mgr().get_all_aggregated_data(video_url)"""
callFuncs = []
for text in texts.split('def ')[1:]:
    functionName = text.split('(')[0]
    capFunctionName = ''.join([capitalize(functionName) for each in "get_request_data".split('_')])

    callFunc = f"""@video_url_bp.route("/{functionName}", methods=["POST","GET"])
def {capFunctionName}():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {data}",status_code=400)
        result = {functionName}(url)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)"""
    callFuncs.append(callFunc)
print('\n'.join(callFuncs))
