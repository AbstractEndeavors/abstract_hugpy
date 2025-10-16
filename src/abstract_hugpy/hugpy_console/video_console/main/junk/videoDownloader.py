import os, shutil, threading
import yt_dlp
from abstract_webtools import get_video_url, generate_video_id
from .registry import infoRegistry
import logging

logger = logging.getLogger("videoDownloader")

class VideoDownloader:
    def __init__(self, url=None, download_directory=None, video_extention="mp4",
                 download_video=True, video_path=None, force_refresh=False):
        self.video_url = get_video_url(url)
        self.video_id = generate_video_id(self.video_url)
        self.registry = infoRegistry()
        self.download_directory = download_directory or "./downloads"
        os.makedirs(self.download_directory, exist_ok=True)
        self.video_path = video_path
        self.force_refresh = force_refresh
        self.get_download = download_video
        self.info = self.registry.get_video_info(url=self.video_url, video_id=self.video_id,
                                                 video_path=self.video_path, force_refresh=self.force_refresh) or {}
        self._start()

    def _start(self):
        self.download_thread = threading.Thread(
            target=self._download_single, name="video-download", daemon=True
        )
        self.download_thread.start()
        self.download_thread.join()

    def _download_single(self):
        outtmpl = os.path.join(self.download_directory, "video.%(ext)s")
        opts = {
            "quiet": True,
            "noprogress": True,
            "outtmpl": outtmpl,
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            raw_info = ydl.extract_info(self.video_url, download=self.get_download)
        final_path = os.path.join(self.download_directory, "video.mp4")
        if os.path.isfile(final_path):
            minimal_info = {
                "id": raw_info.get("id"),
                "title": raw_info.get("title"),
                "duration": raw_info.get("duration"),
                "video_id": self.video_id,
                "video_path": final_path,
            }
            self.registry.edit_info(minimal_info, video_id=self.video_id, url=self.video_url)
        return final_path
