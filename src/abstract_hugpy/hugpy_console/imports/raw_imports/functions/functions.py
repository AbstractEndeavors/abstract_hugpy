from .imports import *
from ..videoDownloader.manager_utils import get_video_info
from ...modules.whisperManager.src.manager_utils import get_whisper_text
def get_text(
    *args,
    text=None,
    url=None,
    file_path=None,
    audio_path=None,
    video_url=None,
    video_path=None,
    video_id=None,
    data_url=None,
    **kwargs
):
    # 1. Explicit text
    if text:
        if video_id:
            db_upsert(video_id, {"text": text})
        return text

    # 2. Audio/video → whisper
    if audio_path or video_url or video_path:
        text = get_whisper_text(
            *args,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path,
            **kwargs
        )
        if video_id and text:
            db_upsert(video_id, {"whisper": {"text": text}})
        return text

    # 3. URL / file ingestion
    if url or file_path:
        filename = None
        if file_path:
            basename = os.path.basename(file_path)
            filename, ext = os.path.splitext(basename)
        else:
            domain = urlManager(url).domain
            logger.info(domain)
            filename, ext = os.path.splitext(domain)
            basename = f"{filename}.json"

        save_file = os.path.join(DEFAULT_DOCUMENTS_ROOT, basename)
        text = get_soup_text(url) or read_file_as_text(file_path)

        # Mirror in registry as cache
        info = get_infoRegistry().add_file(
            file_path=file_path or save_file,
            url=url,
            video_id=video_id,
            data_url=data_url
        )
        safe_dump_to_file(data={filename: text}, file_path=save_file)

        # DB as single source of truth
        if video_id and text:
            db_upsert(video_id, {"info": info, "text": text})

        if text and info.get("text_path"):
            write_to_file(info["text_path"], text)
        return text

    # 4. Last chance → DB only
    if video_id:
        rec = db_get(video_id)
        if rec and rec.info:
            return rec.info.get("text")

    return None

def get_soup_text(url):
    soup = get_soup(url)
    return soup.text


def get_args_kwargs(req):
    """
    Extracts positional args and keyword args from a JSON request.
    Client should send:
        {
            "args": [arg1, arg2, ...],
            "kw1": "val1",
            "kw2": "val2"
        }
    Returns:
        tuple: (args_list, kwargs_dict)
    """
    data = req.get_json(force=True) or {}
    args = data.pop("args", [])   # safely remove and default to list
    if not isinstance(args, (list, tuple)):
        raise ValueError(f"Expected 'args' to be list/tuple, got {type(args)}")
    return args, data
def extract_audio_from_video(video_path: str, audio_path: str = None):
    """Extract audio from a video file using moviepy."""
    if audio_path == None:
        video_directory = os.path.dirname(video_path)
        audio_path = video_directory
    if os.path.isdir(audio_path):
        audio_path = os.path.join(audio_path,'audio.wav')
    try:
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} was not created.")
        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")
        return None
def get_schema_paths(
    *args,
    video_url=None,
    video_path=None,
    video_info=None,
    **kwargs
    ):
    if video_info and isinstance(video_info,dict) and video_info.get('schema_paths'):
        return video_info.get('schema_paths')
    schema_paths = get_video_info_spec(
        video_url=video_url,
        video_path=video_path,
        key = 'schema_paths'
        )
    return schema_paths
def get_whisper_video_info(
    *args,
    video_url=None,
    video_path=None,
    video_info=None,
    **kwargs
    ):
    if video_info and isinstance(video_info,dict) and video_info.get('schema_paths'):
        video_path = video_info.get('video_path')
    video_info = get_video_info(
        video_url=video_url,
        video_path=video_path,
        download=True
        )
    return video_info
def get_whisper_audio_path(
    *args,
    audio_path=None,
    video_url=None,
    video_path=None,
    video_info=None,
    **kwargs
    ):
    if audio_path and os.path.isfile(audio_path):
        return audio_path
    video_info = get_whisper_video_info(
            video_url=video_url,
            video_path=video_path,
            video_info=video_info
            )

    video_path = video_path or video_info.get('video_path')
    audio_path = audio_path or video_info.get('audio_path')
    extract_audio_from_video(
            video_path=video_path,
            audio_path=audio_path
            )
    return audio_path
def get_whisper_video_path(
        *args,
        video_url=None,
        video_path=None,
        **kwargs
    ):
    video_path = get_video_info_spec(
        video_url=video_url,
        video_path=video_path,
        download=True,
        key = 'video_path'
        )
    return video_path
def get_metadata_path(
        *args,
        video_url=None,
        video_path=None,
        **kwargs
    ):
    metadata_path = get_video_info_spec(
        video_url=video_url,
        video_path=video_path,
        download=True,
        key = 'metadata_path'
        )
    return metadata_path
def get_whisper_path(
        *args,
        video_url=None,
        video_path=None,
        **kwargs
    ):
    whisper_path = get_video_info_spec(
        video_url=video_url,
        video_path=video_path,
        download=True,
        key = 'whisper_path'
        )
    return whisper_path
