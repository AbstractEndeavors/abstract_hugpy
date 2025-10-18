from .imports import *
from ..main import *
def get_all_text(*args, text=None, url=None, file_path=None,
                 audio_path=None, video_url=None, video_path=None, **kwargs):
    # === 1. Prefer explicit text ===
    if text:
        return text

    # === 2. Audio/video → whisper ===
    if audio_path or video_url or video_path:
        return get_whisper_text(
            *args,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path,
            **kwargs
        )

    # === 3. URL / file ingestion ===
    if url or file_path:
        info = dataRegistry.get_info(url=url, file_path=file_path)
        text = get_text(text=text, url=url, file_path=file_path)

        # persist into registry
        if text and info.get("text_path"):
            write_to_file(info["text_path"], text)
        return text

    return None
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
def get_results(func,req):
    args,data = get_args_kwargs(req)
    return func(*args,**data)
def get_keyword_summary(*args,**data):
    try:
        text = get_all_text(*args,**data)
        text, keywords = get_text_keywords(*args, text=text, **data)
        text_data = {"text":text,"keywords":keywords}
        data.update(text_data)
        result = get_refined_keywords(*args,**data)
        text_data['keywords']=result
        text_data['summary']=get_summary_result(*args,**text_data)
        result = get_keyword_summary(request)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
def get_summary_result(*args,**data):
    try:
        text = get_all_text(*args, **data)
        result = get_summary_result(*args, text=text, **data)
        # persist summary if registry-backed
        if data.get("url") or data.get("file_path"):
            result = dataRegistry.get_info(url=data.get("url"), file_path=data.get("file_path"))
            safe_dump_to_file(result, info["summary_path"])
        return result,True
    except Exception as e:
        message = f"{e}"
        return f"{e}",False
