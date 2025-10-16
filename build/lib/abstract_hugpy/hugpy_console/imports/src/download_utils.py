from .constants import *
from .info_utils import *
from .schema_utils import *
from .directory_utils import *
def download_image(url, save_path=None):
    """
    Downloads an image from a URL and saves it to the specified path.
    
    Args:
        url (str): The URL of the image to download
        save_path (str, optional): Path to save the image. If None, uses the filename from URL
        
    Returns:
        str: Path where the image was saved, or None if download failed
    """
    try:
        # Send GET request to the URL
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Set decode_content=True to automatically handle Content-Encoding
            response.raw.decode_content = True
            
            # If no save_path provided, extract filename from URL
            if save_path is None:
                # Get filename from URL
                filename = url.split('/')[-1]
                save_path = filename
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Write the image content to file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Image successfully downloaded to {save_path}")
            return save_path
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {str(e)}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None
def get_thumbnails(directory,info):
    thumbnails_dir = os.path.join(directory,'thumbnails')
    os.makedirs(thumbnails_dir, exist_ok=True)
    thumbnails = info.get('thumbnails',[])
    for i,thumbnail_info in enumerate(thumbnails):
        thumbnail_url = thumbnail_info.get('url')
        thumbnail_base_url = thumbnail_url.split('?')[0]
        baseName = os.path.basename(thumbnail_base_url)
        fileName,ext = os.path.splitext(baseName)
        baseName = f"{fileName}{ext}"
        resolution = info['thumbnails'][i].get('resolution')
        if resolution:
            baseName = f"{resolution}_{baseName}"
        img_id = info['thumbnails'][i].get('id')
        if img_id:
            baseName = f"{img_id}_{baseName}"
        thumbnail_path = os.path.join(thumbnails_dir,baseName)
        info['thumbnails'][i]['path']=thumbnail_path
        download_image(thumbnail_url, save_path=thumbnail_path)
    return info

def optimize_video_for_safari(input_file, reencode=False):
    """
    Optimizes an MP4 file for Safari by moving the 'moov' atom to the beginning.
    Optionally, re-encodes the video for maximum compatibility.
    
    Args:
        input_file (str): Path to the original MP4 file.
        reencode (bool): If True, re-encode the video for Safari compatibility.
        
    Returns:
        str: Path to the optimized MP4 file.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        local_input = os.path.join(tmp_dir, os.path.basename(input_file))
        shutil.copy2(input_file, local_input)
        
        base, ext = os.path.splitext(local_input)
        local_output = f"{base}_optimized{ext}"
        
        if reencode:
            # Re-encoding command for maximum Safari compatibility
            command = [
                "ffmpeg", "-i", local_input,
                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "faststart",
                local_output
            ]
        else:
            # Simple faststart with stream copy
            command = [
                "ffmpeg", "-i", local_input,
                "-c", "copy", "-movflags", "faststart",
                local_output
            ]
        
        try:
            subprocess.run(command, check=True)
            shutil.copy2(local_output, input_file)
            print(f"Optimized video saved as {input_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error during optimization: {e}")
        return input_file
    finally:
        shutil.rmtree(tmp_dir)




def dl_video(url, download_directory=None, output_filename=None,
             get_info=None, download_video=None, ydl_opts=None):
    mgr = get_video_info(
        url,
        download_directory=download_directory,
        output_filename=output_filename,
        get_info=get_info,
        download_video=download_video,
        ydl_opts=ydl_opts,  # pass through
    )
    return get_video_info_from_mgr(mgr)
def for_dl_video(url,download_directory=None,output_filename=None,get_info=None,download_video=None):
    get_info = bool_or_default(get_info,default=True)
    download_video =bool_or_default(download_video,default=True)
    video_mgr = dl_video(url,download_directory=download_directory,output_filename=output_filename,get_info=get_info,download_video=download_video)
    if get_video_info_from_mgr(video_mgr):
        return video_mgr
    videos = soupManager(url).soup.find_all('video')
    for video in videos:
        src = video.get("src")
        video_mgr = dl_video(src,download_directory=download_directory,output_filename=output_filename,download_video=download_video)
        if get_video_info_from_mgr(video_mgr):
            return video_mgr
def downloadvideo(
    url,
    directory=None,
    output_filename=None,          # kept for API compat; not used for final name
    rename_display=None,           # kept for API compat; final file is always video.mp4
    thumbnails=None,
    audio=None,
    safari_optimize=None,
    download_video=None,
    flat_layout: bool = False,
    *args, **kwargs,
):
    """
    Download a video and save alongside info.json and other schema files.

    Rules:
      - Final file is ALWAYS <video_dir>/video.mp4
      - Only ONE canonical video_dir is used
      - No extra folders derived from 'display_id' are created
    """
    # normalize flags
    rename_display   = bool_or_default(rename_display)          # no-op for file name
    thumbnails       = bool_or_default(thumbnails)
    audio            = bool_or_default(audio, default=False)
    safari_optimize  = bool_or_default(safari_optimize, default=True)
    download_video   = bool_or_default(download_video, default=True)

    # Kick off fetch/download via your existing flow
    # (keeps your yt-dlp + registry logic intact)
    _tmp_name = output_filename or get_temp_file_name(url)
    video_mgr = for_dl_video(
        url,
        download_directory=directory,
        output_filename=_tmp_name,
        download_video=download_video,
    )
    info = get_video_info_from_mgr(video_mgr) or {}

    # Decide canonical id & base root
    video_id   = info.get("video_id") or info.get("id") or get_temp_id(url)
    # If caller passed a root directory, respect it; otherwise use what the registry established
    base_root  = directory or info.get("directory") or get_video_root(None)

    # Decide the single canonical directory to use
    video_dir  = base_root if flat_layout else os.path.join(base_root, video_id)
    logger.info(f"making this downloadvideo == {video_dir} == get_video_directory line 246")
    os.makedirs(video_dir, exist_ok=True)

    # Where the final file must live
    final_path = os.path.join(video_dir, "video.mp4")

    # If a file was downloaded somewhere else, move it into place
    cur_path = info.get("file_path") or info.get("video_path")
    if cur_path and os.path.isfile(cur_path):
        if os.path.abspath(cur_path) != os.path.abspath(final_path):
            final_dir = os.path.dirname(final_path)
            logger.info(f"making this downloadvideo == {final_dir} == get_video_directory line 251")
            os.makedirs(final_dir, exist_ok=True)
            try:
                shutil.move(cur_path, final_path)
            except Exception:
                # If move fails (e.g. cross-device), copy+remove
                shutil.copy2(cur_path, final_path)
                os.remove(cur_path)
    else:
        # If yt-dlp already wrote video.mp4 at the canonical spot, keep it.
        # Otherwise, try to find any video-like file in video_dir and rename it.
        if not os.path.isfile(final_path):
            candidates = []
            for name in os.listdir(video_dir):
                p = os.path.join(video_dir, name)
                if os.path.isfile(p) and os.path.splitext(name)[1].lower() in {".mp4", ".m4v", ".mov", ".webm", ".mkv"}:
                    candidates.append(p)
            if candidates:
                # choose the largest as the likely full file
                best = max(candidates, key=lambda p: os.path.getsize(p))
                if os.path.abspath(best) != os.path.abspath(final_path):
                    shutil.move(best, final_path)

    # Safari faststart (moov up front) if we have a proper mp4
    if os.path.isfile(final_path) and final_path.lower().endswith(".mp4") and safari_optimize:
        try:
            optimize_video_for_safari(final_path, reencode=safari_optimize)
        except Exception as e:
            print(f"Safari optimization skipped: {e}")

    # Update info with canonical paths
    info["video_id"]  = video_id
    info["directory"] = video_dir
    info["file_path"] = final_path
    info["video_path"]= final_path

    # Thumbnails / audio (optional)
    if thumbnails:
        try:
            info = get_thumbnails(video_dir, info)
        except Exception as e:
            print(f"Thumbnail fetch failed: {e}")
    if audio:
        try:
            info = download_audio(video_dir, info)  # your helper
        except Exception:
            info["audio_path"] = None

    # Write info.json next to the video
    info_path = os.path.join(video_dir, "info.json")
    info["info_path"] = info_path
    info["json_path"] = info_path
    safe_dump_to_file(info, info_path)

    # Attach the rest of the schema INSIDE the already-chosen video_dir
    info = ensure_standard_paths(info, video_dir, flat_layout=True)
    return info

    return info

