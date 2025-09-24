
from yt_dlp import YoutubeDL
import os,os,pytesseract,cv2,pysrt
import numpy as np
from PIL import Image
import moviepy.editor as mp
from moviepy.editor import *
from abstract_utilities import get_logFile
from abstract_webtools import get_video_info, VideoDownloader
from abstract_apis import *
##from abstract_videos.text_tools.summarizer_utils.summarizer_services import get_summary
from abstract_utilities.abstract_classes import SingletonMeta
from abstract_utilities import make_list,get_logFile, safe_dump_to_file, safe_load_from_file, safe_read_from_json,get_any_value
from .constants import *
logger = get_logFile('videos_utils')
logger.info('started')
