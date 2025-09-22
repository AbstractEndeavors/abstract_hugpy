import os, shutil, subprocess, uuid
from urllib.parse import urlparse, urlunparse
from abstract_utilities import get_any_value, make_list, safe_dump_to_json
from abstract_webs import dl_video, get_video_info_from_mgr
import logging

logger = logging.getLogger(__name__)

def transcode_to_mp4(input_path: str, output_path: str) -> str:
    """Force transcode to MP4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def for_dl_video(
    url: str,
    download_directory: str = None,
    output_filename: str = None,
    get_info: bool = True,
    download_video: bool = True,
    preferred_format: str = None,      # if set, will try to force
    force_transcode: bool = False,     # only if you *really* want mp4
):
    """
    Download a video and save metadata.

    - Will keep the native format unless `preferred_format` is given.
    - Records actual file extension & path in info.json.
    - Transcoding is optional.
    """
    download_directory = download_directory or os.getcwd()

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(download_directory, "%(id)s.%(ext)s"),
    }
    if preferred_format:
        ydl_opts["merge_output_format"] = preferred_format

    # Step 1: Download
    video_mgr = dl_video(
        url,
        download_directory=download_directory,
        output_filename=output_filename,
        get_info=get_info,
        download_video=download_video,
        ydl_opts=ydl_opts,
    )
    video_info = get_video_info_from_mgr(video_mgr)
    if not video_info:
        logger.error(f"dl_video produced no info for {url}")
        return None

    # Step 2: Collect metadata
    context = {}
    for key in ["file_path", "id"]:
        value = make_list(get_any_value(video_info, key) or None)[0]
        if isinstance(value, dict):
            context.update(value)
        else:
            context[key] = value

    file_id = context.get("id") or str(uuid.uuid4())
    file_path = video_info.get("file_path")

    if not file_path or not os.path.isfile(file_path):
        logger.critical(f"Downloaded file missing: {file_path}")
        return None

    ext = os.path.splitext(file_path)[-1].lstrip(".").lower()
    new_dir = os.path.join(download_directory, str(file_id))
    os.makedirs(new_dir, exist_ok=True)
    final_path = os.path.join(new_dir, f"{file_id}.{ext}")

    # Step 3: Move to stable folder
    if file_path != final_path:
        try:
            shutil.move(file_path, final_path)
            file_path = final_path
        except Exception as e:
            logger.error(f"Failed to move file: {e}")
            return None

    # Step 4: Optional transcode to mp4
    if preferred_format == "mp4" and (force_transcode or ext != "mp4"):
        try:
            trans_path = os.path.join(new_dir, f"{file_id}.mp4")
            logger.info(f"Transcoding {file_path} → {trans_path}")
            transcode_to_mp4(file_path, trans_path)
            file_path = trans_path
            ext = "mp4"
        except subprocess.CalledProcessError as e:
            logger.error(f"Transcode failed: {e}")
            # keep original

    # Step 5: Save metadata
    info_path = os.path.join(new_dir, "info.json")
    context.update({
        "file_path": file_path,
        "extension": ext,
        "id": file_id,
    })
    video_info["context"] = context
    safe_dump_to_json(data=video_info, file_path=info_path)

    logger.info(f"✅ Downloaded video to {file_path}")
    return video_info

