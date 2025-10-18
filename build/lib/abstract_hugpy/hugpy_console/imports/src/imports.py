from __future__ import annotations
import threading,re,urllib.request,m3u8_To_MP4,subprocess,time,yt_dlp
import os,unicodedata,requests,shutil,tempfile, hashlib,logging,glob
import json,math,pytesseract,cv2,PyPDF2,argparse,sys,whisper,pytesseract
import os,pytesseract,cv2,pysrt, threading, logging
from m3u8 import M3U8  # Install: pip install m3u8
from urllib.parse import urljoin,quote
from pathlib import Path
from pdf2image import convert_from_path
from yt_dlp.postprocessor.ffmpeg import FFmpegFixupPostProcessor
from pydub.silence import detect_nonsilent,split_on_silence
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from yt_dlp import YoutubeDL
from functools import lru_cache
from datetime import datetime, timedelta 
import numpy as np
from PIL import Image
from typing import *
from .constants import *
import multiprocessing as mproc
import moviepy.editor as mp

from pathlib import Path
from tqdm import tqdm
import speech_recognition as sr
from moviepy.editor import *
from abstract_apis import *
from abstract_math import divide_it,add_it,multiply_it,subtract_it
from abstract_flask import *
from abstract_webtools.managers.urlManager import *
from abstract_webtools import get_corrected_url
from abstract_webtools.managers.soupManager import *
from abstract_ai.gpt_classes.prompt_selection.PromptBuilder import recursive_chunk
from abstract_utilities import SingletonMeta
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
    SingletonMeta,
    get_env_value,
    timestamp_to_milliseconds,
    format_timestamp,
    get_time_now_iso,
    parse_timestamp,
    get_logFile,
    url_join,
    make_dirs,
    safe_dump_to_file,
    safe_read_from_json,
    read_from_file,
    write_to_file,
    path_join,
    confirm_type,
    get_media_types,
    get_all_file_types,
    eatInner,
    eatOuter,
    eatAll,
    get_any_value,
    get_all_file_types,
    is_media_type,
    safe_load_from_json,
    get_file_map,
    get_logFile,
    safe_dump_to_file,
    get_time_stamp,
    SingletonMeta,
    is_number
    )

from abstract_webtools.managers.videoDownloader import (
    get_video_filepath,
    get_video_id,
    get_video_info,
    ensure_standard_paths,
    VideoDownloader,
    get_video_info
    )
from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Integer,
    String,
    JSON,
    TIMESTAMP,
    MetaData,
    text,
    select,
    LargeBinary,
    DateTime
)
from sqlalchemy.dialects.postgresql import JSONB, insert
def bool_or_default(obj,default=True):
    if obj == None:
        obj =  default
    return obj


def get_video_url(url=None, video_url=None):
    video_url = url or video_url
    if video_url:
        video_url = get_corrected_url(video_url)
    return video_url

def return_num_str(obj):
    return int(obj) if is_number(obj) else obj
logger = get_logFile('videos_console')












