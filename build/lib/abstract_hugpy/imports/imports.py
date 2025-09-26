from yt_dlp import YoutubeDL
import os,pytesseract,cv2,pysrt
import numpy as np
from PIL import Image
from .constants import *
import moviepy.editor as mp
from moviepy.editor import *
from abstract_apis import *
from abstract_utilities import (
    get_logFile,
    safe_read_from_json,
    safe_dump_to_file,
    get_any_value,
    make_list,
    make_list,
    get_logFile,
    safe_dump_to_file,
    safe_load_from_file,
    safe_read_from_json,
    get_any_value,
    SingletonMeta
    )
from abstract_webtools.managers.videoDownloader import (
    get_video_filepath,
    get_video_id,
    get_video_info,
    ensure_standard_paths,
    VideoDownloader,
    get_video_info
    )

logger = get_logFile('videos_utils')
logger.info('started')
