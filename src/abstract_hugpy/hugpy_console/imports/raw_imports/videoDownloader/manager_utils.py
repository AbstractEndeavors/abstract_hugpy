from ..imports import *
from .info_registry import *
from .videoDownloader import *
def get_info_from_mgr(mgr):
    if hasattr(mgr, 'info'):
        return mgr.info
    video_path, video_id, video_url = None, None, None
    if hasattr(mgr, 'video_path'):
        video_path = mgr.video_path
    if hasattr(mgr, 'video_id'):
        video_id = mgr.video_id
    elif hasattr(mgr, 'id'):
        video_id = mgr.id
    if hasattr(mgr, 'video_url'):
        video_url = mgr.video_url
    video_info = get_video_info(
        url=video_id,
        video_url=video_url,
        video_path=video_path
        )
    return video_info
def get_videoDownloader(url=None,
                   download_directory=None,
                   user_agent=None,
                   video_extention=None,
                   download_video=True,
                   video_path=None,
                   output_filename=None,
                   ydl_opts=None,
                   registry=None,
                   force_refresh=False,
                   flat_layout: bool = False,
                   video_url=None):
    video_url = get_video_url(url or video_url)
    videoDownload_mgr = VideoDownloader(
        video_url=video_url,
        download_directory=download_directory,
        user_agent=user_agent,
        video_extention=video_extention,
        download_video=download_video,
        video_path=video_path,
        output_filename=output_filename,
        ydl_opts=ydl_opts,
        registry=registry,
        flat_layout=flat_layout
        )
    return videoDownload_mgr
def download_video(url=None,
                   download_directory=None,
                   user_agent=None,
                   video_extention=None,
                   download_video=True,
                   video_path=None,
                   output_filename=None,
                   ydl_opts=None,
                   registry=None,
                   force_refresh=False,
                   flat_layout: bool = False,
                   video_url=None):
        video_url = get_video_url(url or video_url)
        videoDownload_mgr = get_videoDownloader(
            video_url=video_url,
            download_directory=download_directory,
            user_agent=user_agent,
            video_extention=video_extention,
            download_video=download_video,
            video_path=video_path,
            output_filename=output_filename,
            ydl_opts=ydl_opts,
            registry=registry,
            flat_layout=flat_layout
            )
        
        return get_info_from_mgr(videoDownload_mgr)

def get_infoRegistry(video_root=None, flat_layout=None):
    return infoRegistry(video_root=video_root, flat_layout=flat_layout)

def get_registry_video_root(video_root=None, flat_layout=None):
    registry_mgr = get_infoRegistry(video_root=video_root, flat_layout=flat_layout)
    return registry_mgr.video_root

def get_registry_path(video_root=None, flat_layout=None):
    registry_mgr = get_infoRegistry(video_root=video_root, flat_layout=flat_layout)
    return registry_mgr.registry_path

def get_video_info(url=None, video_url=None, video_id=None, 
                        video_path=None, video_root=None,
                        force_refresh=False, flat_layout=None, download=False):
    registry_mgr = get_infoRegistry(video_root=video_root, flat_layout=flat_layout)
    url = get_video_url(url or video_url)
    video_info = registry_mgr.get_video_info(
        url=url,
        video_id=video_id,
        video_path=video_path,
        force_refresh=force_refresh,
    )
    if download:
        video_info = download_video(
            video_url=video_url,
            download_directory=video_root,
            download_video=download,
            video_path=video_path,
            flat_layout=flat_layout
            )
    return video_info or {}

def get_video_info_spec(url=None, video_url=None, video_id=None, 
                        video_path=None, video_root=None, key=None,
                        force_refresh=False, flat_layout=None, download=False):
    video_url = url = get_video_url(url or video_url)
    video_info = get_video_info(
        url=url,
        video_url=video_url,
        video_id=video_id,
        video_path=video_path,
        force_refresh=force_refresh,
        video_root=video_root,
        flat_layout=flat_layout,
        download=download
    )
    if not key:
        return video_info
    keys = make_list(key)
    value = None
    for key in keys:
        value = video_info.get(key)
        if not value:
            values = make_list(get_any_value(video_info, key) or None)
            value = values[0]
        if value:
            break
    return value

def get_video_id(
        url=None,
        video_url=None,
        video_id=None,
        video_path=None,
        force_refresh=False,
        download=False,
        video_root=None,
        flat_layout=None
    ):
    video_url = url = get_video_url(url or video_url)
    info_spec = get_video_info_spec(
        url=url,
        video_url=video_url,
        video_id=video_id,
        video_path=video_path,
        force_refresh=force_refresh,
        download=download,
        video_root=video_root,
        flat_layout=flat_layout,
        key=['id','video_id']
        )
    return info_spec
def get_video_title(
        url=None,
        video_url=None,
        video_id=None,
        video_path=None,
        force_refresh=False,
        download=False,
        video_root=None,
        flat_layout=None
    ):
    info_spec = get_video_info_spec(
        url=url,
        video_url=video_url,
        video_id=video_id,
        video_path=video_path,
        force_refresh=force_refresh,
        download=download,
        video_root=video_root,
        flat_layout=flat_layout,
        key=['title']
        )
    return info_spec
def get_video_filepath(
        url=None,
        video_url=None,
        video_id=None,
        video_path=None,
        force_refresh=False,
        download=False,
        video_root=None,
        flat_layout=None
    ):
    info_spec = get_video_info_spec(
        url=url,
        video_url=video_url,
        video_id=video_id,
        video_path=video_path,
        force_refresh=force_refresh,
        download=download,
        video_root=video_root,
        flat_layout=flat_layout,
        key=['filepath','file_path','video_path','videopath']
        )
    return info_spec

def get_video_info_from_mgr(video_mgr):
    try:
        info = video_mgr.info
        return info
    except Exception as e:
        print(f"{e}")
        return None

