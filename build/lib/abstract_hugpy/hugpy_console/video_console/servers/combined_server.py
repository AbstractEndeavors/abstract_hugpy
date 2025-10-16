from ..imports import *
from .combined import *
from .jobs import start_job, get_job

combined_bp, logger = get_bp("combined_bp")


def make_pipeline_route(key, extractor=None):
    """Generate a route function for pipeline-based endpoints."""
    def _inner():
        data = get_request_data(request)
        initialize_call_log(data=data)
        try:
            url = data.get("url")
            if not url:
                return get_json_response(value=f"url in {data}", status_code=400)

            result = get_pipeline_data(url, key=key)

            if extractor:
                result = get_spec_value(result, extractor)

            if not result:
                return get_json_response(value=f"no result for {data}", status_code=400)

            return get_json_response(value=result, status_code=200)

        except Exception as e:
            logger.exception(f"Error in {key} route")
            return get_json_response(value=str(e), status_code=500)

    return _inner


# --- Register endpoints dynamically ---
pipeline_routes = {
    "/download_video": "download_video",
    "/extract_video_audio": "extract_audio",
    "/get_whisper_result": "whisper",
    "/get_whisper_text": ("whisper", "text"),
    "/get_whisper_segments": ("whisper", "segments"),
    "/get_metadata": "metadata",
    "/get_captions": "captions",
    "/get_thumbnails": "thumbnails",
    "/get_info": "info",
    "/get_video_directory": ("info", "directory"),
    "/get_video_path": ("info", "video_path"),
    "/get_audio_path": ("info", "audio_path"),
    "/get_thumbnail_dir": ("info", "thumbnail_directory"),
    "/get_srt_path": ("info", "captions_path"),
    "/get_metadata_path": ("info", "metadata_path"),
    "/get_all_data": "get_all",
    "/get_aggregated_data_dir": ("info", "aggregated_data_directory"),
    "/get_aggregated_data_path": ("info", "aggregated_data_path"),
    "/get_aggregated_data": "aggregated_data",
}

for route, key in pipeline_routes.items():
    if isinstance(key, tuple):
        k, extractor = key
        view_func = make_pipeline_route(k, extractor)
        endpoint = f"{k}_{extractor}_ep"
    else:
        view_func = make_pipeline_route(key)
        endpoint = f"{key}_ep"

    combined_bp.add_url_rule(
        route,
        endpoint=endpoint,       # 👈 unique endpoint name
        view_func=view_func,
        methods=["POST", "GET"],
    )

@combined_bp.route("/queue_whisper", methods=["POST", "GET"])
def queueWhisper():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:
        url = data.get("url")
        if not url:
            return get_json_response(value=f"url in {data}", status_code=400)

        # enqueue whisper job
        job_id = start_job(get_pipeline_data, url, "whisper")
        return get_json_response(value={"job_id": job_id, "status": "queued"}, status_code=202)
    except Exception as e:
        return get_json_response(value=str(e), status_code=500)


@combined_bp.route("/job_status/<job_id>", methods=["GET"])
def jobStatus(job_id):
    job = get_job(job_id)
    return get_json_response(value=job, status_code=200)

# --- Special cases (need get_results) ---
@combined_bp.route("/get_text", methods=["POST", "GET"])
def getTexts():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:
        result = get_results(get_all_text, request)
        if not result:
            return get_json_response(value=f"no result for {data}", status_code=400)
        return get_json_response(value=result, status_code=200)
    except Exception as e:
        logger.exception("Error in get_text")
        return get_json_response(value=str(e), status_code=500)


@combined_bp.route("/video_info", methods=["POST", "GET"])
def videoInfo():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:
        result = get_results(get_whisper_video_info, request)
        if not result:
            return get_json_response(value=f"no result for {data}", status_code=400)
        return get_json_response(value=result, status_code=200)
    except Exception as e:
        logger.exception("Error in video_info")
        return get_json_response(value=str(e), status_code=500)


@combined_bp.route("/schema_paths", methods=["POST", "GET"])
def getSchemaPaths():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:
        result = get_results(get_schema_paths, request)
        if not result:
            return get_json_response(value=f"no result for {data}", status_code=400)
        return get_json_response(value=result, status_code=200)
    except Exception as e:
        logger.exception("Error in schema_paths")
        return get_json_response(value=str(e), status_code=500)


@combined_bp.route("/video_path", methods=["POST", "GET"])
def getVideoPaths():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:
        result = get_results(get_whisper_audio_path, request)
        if not result:
            return get_json_response(value=f"no result for {data}", status_code=400)
        return get_json_response(value=result, status_code=200)
    except Exception as e:
        logger.exception("Error in video_path")
        return get_json_response(value=str(e), status_code=500)


@combined_bp.route("/text_keyword_summary", methods=["POST", "GET"])
def getTextKeywordSummary():
    args, data = get_args_kwargs(request)
    initialize_call_log(data=data)
    try:
        result, status = get_keyword_summary(*args, **data)
        if result is None and status:
            return get_json_response(value=f"no result for {data}", status_code=400)
        if not status:
            return get_json_response(value=result, status_code=500)
        return get_json_response(value=result, status_code=200)
    except Exception as e:
        logger.exception("Error in text_keyword_summary")
        return get_json_response(value=str(e), status_code=500)


@combined_bp.route("/summary_result", methods=["POST", "GET"])
def getSummaryResult():
    args, data = get_args_kwargs(request)
    initialize_call_log(data=data)
    try:
        result, status = get_summary_result(*args, **data)
        if result is None and status:
            return get_json_response(value=f"no result for {data}", status_code=400)
        if not status:
            return get_json_response(value=result, status_code=500)
        return get_json_response(value=result, status_code=200)
    except Exception as e:
        logger.exception("Error in summary_result")
        return get_json_response(value=str(e), status_code=500)
