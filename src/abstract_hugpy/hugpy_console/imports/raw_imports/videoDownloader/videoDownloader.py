from .imports import *
from .info_registry import *
# ---------------- Downloader ---------------- #
class VideoDownloader:
    def __init__(self, url=None,download_directory=None, user_agent=None,
                 video_extention="mp4", download_video=True,
                 video_path=None,
                 output_filename=None, ydl_opts=None,
                 registry=None, force_refresh=False,
                 flat_layout: bool = False,video_url =None):

        self.video_url = get_video_url(url or video_url )
        self.video_urls = make_list(self.video_url)
        self.video_id =  generate_video_id(self.video_url)
        self.registry = registry or infoRegistry(video_root=download_directory,flat_layout=flat_layout)
        self.ydl_opts = ydl_opts or {}
        self.get_download = download_video
        self.user_agent = user_agent
        self.video_extention = video_extention
        self.download_directory = get_video_root(download_directory)
        self.output_filename = output_filename
        self.force_refresh = force_refresh
        self.flat_layout = flat_layout   # 🔑
        self.video_path=video_path
        self.info = self.registry.get_video_info(url=self.video_url,video_id=self.video_id,video_path=self.video_path, force_refresh=self.force_refresh) or {}
        self.video_path=self.info.get('video_path') or self.video_path
        
        self.monitoring = True
        self.pause_event = threading.Event()

        self._start()

    def _start(self):
        self.download_thread = threading.Thread(
            target=self._download_entrypoint, name="video-download", daemon=True
        )
        self.monitor_thread = threading.Thread(
            target=self._monitor, name="video-monitor", daemon=True
        )
        self.download_thread.start()
        self.monitor_thread.start()
        self.download_thread.join()

    def stop(self):
        self.monitoring = False
        self.pause_event.set()

    def _monitor(self, interval=30, max_minutes=15):
        start = time.time()
        while self.monitoring:
            logger.info("Monitoring...")
            if time.time() - start > max_minutes * 60:
                logger.info("Monitor: timeout reached, stopping.")
                break
            self.pause_event.wait(interval)
        logger.info("Monitor: exited.")

    def _build_ydl_opts(self, outtmpl, extractor_client=None):
        opts = {
            "quiet": True,
            "noprogress": True,
            # write under <download_directory>/<id>/<id>.<ext>
            "outtmpl": outtmpl,                      # see call-site below
            # pick best and merge; final container = mp4
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            # remux to mp4 even when streams are webm/opus
            "postprocessors": [
                {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
            ],
            "retries": 5,
            "fragment_retries": 5,
            "ignoreerrors": False,
            # optional: YouTube client to reduce 403s
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }
        if extractor_client:
            opts.setdefault("extractor_args", {}).setdefault("youtube", {})["player_client"] = [extractor_client]
        if self.user_agent:
            opts["http_headers"] = {"User-Agent": self.user_agent}
        opts.update(self.ydl_opts or {})
        return opts

    def _download_entrypoint(self):
        try:
            for url in self.video_urls:
                self._download_single(url)
        finally:
            self.stop()
    
    def _download_single(self, url=None,download_directory=None, user_agent=None,
                         video_extention="mp4", download_video=True,
                         video_path=None, output_filename=None, ydl_opts=None,
                         registry=None, force_refresh=False,
                         flat_layout: bool = False,video_url =None):
        video_url = get_video_url(url or video_url  or self.video_url)
        self.info = self.registry.get_video_info(url=video_url, force_refresh=self.force_refresh)
        info_video_path = self.info.get('video_path')
        if info_video_path and os.path.isfile(info_video_path) and not force_refresh:
            self.video_path = info_video_path
            self.video_id = self.info.get("id")
            self.video_url = video_url
            return self.info
        ydl_opts = ydl_opts or self.ydl_opts or {}
        user_agent = user_agent or self.user_agent
        force_refresh = force_refresh or self.force_refresh
        download_directory = download_directory or self.download_directory
        video_extention = video_extention or self.video_extention
        output_filename = output_filename or self.output_filename
        video_path = video_path or self.video_path
        if video_path:
             dirname = os.path.dirname(video_path)
             output_filename = dirname if dirname else download_directory or get_video_root(download_directory)
             basename = os.path.basename(video_path)
             filename,ext = os.path.splitext(basename)
             output_filename = filename if filename else output_filename or 'video'
             video_extention = ext if ext else video_extention or '.mp4'
        output_filename = output_filename or 'video'
        download_directory = download_directory  or get_video_root(download_directory)
        flat_layout = flat_layout or self.flat_layout
        registry = registry or self.registry or infoRegistry(
             video_root=download_directory,
             flat_layout=flat_layout
             )
        logger.info(f"[VideoDownloader] Processing: {video_url}")
        video_url = get_corrected_url(url or video_url or self.url or self.video_url) 
        if "youtube.com/watch" in video_url and "v=" not in video_url:
            logger.debug(f"[VideoDownloader] Skipping bare watch URL: {video_url}")
            return None

        info = self.registry.get_video_info(url=video_url, force_refresh=self.force_refresh)
        if not info:
            logger.error("[VideoDownloader] No info; cannot determine target directory")
            return None
        info = info.get('info')
        directory = info.get("directory") or os.path.join(self.download_directory, info.get("id", "video"))
        check_create_logs(f"making this directory == {directory} line 316")
        os.makedirs(directory, exist_ok=True)

        # force an actual yt-dlp field: %(ext)s (not %(video_extention)s)
        # this will become video.mp4 because merge_output_format='mp4'
        outtmpl = os.path.join(directory, "video.%(ext)s")

        with yt_dlp.YoutubeDL(self._build_ydl_opts(outtmpl)) as ydl:
            raw_info = ydl.extract_info(video_url, download=self.get_download)

        # compute final path (should already be video.mp4 with the template above)
        temp_path = ydl.prepare_filename(raw_info)
        final_path = os.path.join(directory, "video.mp4")
        if os.path.abspath(temp_path) != os.path.abspath(final_path) and os.path.isfile(temp_path):
            shutil.move(temp_path, final_path)

        # minimal info for registry
        video_id = raw_info.get("id") or generate_video_id(raw_info.get("title") or "video")
        minimal_info = {
            "id": raw_info.get("id"),
            "title": raw_info.get("title"),
            "ext": "mp4",
            "duration": raw_info.get("duration"),
            "upload_date": raw_info.get("upload_date"),
            "video_id": video_id,
            "video_path": final_path,
            "file_path": final_path,
        }
        self.registry.edit_info(minimal_info, url=video_url, video_id=video_id, video_path=final_path)


        info = self.registry.get_video_info(video_id=video_id)
        logger.info(f"[VideoDownloader] Stored in registry at {info.get('video_path')}")
        return info
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
