from yt_dlp import YoutubeDL
import os
from .hugging_face_models.google_flan import *
from .hugging_face_models.keybert_model import *
from .hugging_face_models.summarizer_model import *
from .hugging_face_models.whisper_model import *
from abstract_webtools import get_video_info, VideoDownloader
from abstract_apis import *
from abstract_videos.text_tools.summarizer_utils.summarizer_services import get_summary
from abstract_utilities.abstract_classes import SingletonMeta
from abstract_utilities import make_list,get_logFile, safe_dump_to_file, safe_load_from_file, safe_read_from_json,get_any_value

VIDEOS_DIRECTORY = '/mnt/24T/hugging_face/videos'
logger = get_logFile('videos_utils')
logger.info('started')

def get_abs_videos_directory(directory=None):
    if not directory:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
    os.makedirs(directory, exist_ok=True)
    return directory

def get_video_id(video_url):
    return video_url.split('/')[-1].split('=')[-1]

def export_srt(segments, path):
    with open(path, 'w') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{str(seg['start']).replace('.', ',')} --> {str(seg['end']).replace('.', ',')}\n{seg['text']}\n\n")

def get_from_local_host(endpoint, **kwargs):
    return postRequest(f"https://abstractendeavors.com{endpoint}", data=kwargs)

def download_audio(youtube_url, audio_path, output_format='wav'):
    if audio_path.endswith(f'.{output_format}'):
        audio_path = audio_path[:-(len(output_format) + 1)]  # Strip .wav, .mp3, etc.

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,  # e.g. audio (yt-dlp will append .wav)
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': output_format,
            'preferredquality': '0',
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

class VideoDirectoryManager(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.videos_directory = get_abs_videos_directory(VIDEOS_DIRECTORY)
            self.url_data = {}

    def _init_data(self, video_url, video_id):
        dir_path = os.path.join(self.videos_directory, video_id)
        os.makedirs(dir_path, exist_ok=True)
        info_path = os.path.join(dir_path, 'video_info.json')
        video_info = VideoDownloader(video_url, download_directory=dir_path,download_video=False)
        video_id = video_info.info['id']
        video_basename = f"{video_id}.mp4"
        video_path = os.path.join(dir_path,video_basename)
        safe_dump_to_file(data=video_info.info,file_path= info_path)

        data = {
            'url': video_url,
            'video_id': video_id,
            'directory': dir_path,
            'info_path': info_path,
            'video_path': video_path,
            'audio_path': os.path.join(dir_path, 'audio.wav'),
            'whisper_path': os.path.join(dir_path, 'whisper_result.json'),
            'srt_path': os.path.join(dir_path, 'captions.srt'),
            'metadata_path': os.path.join(dir_path, 'video_metadata.json'),
            'info': video_info.info,
        }
        
        if os.path.isfile(data['whisper_path']):
            data['whisper'] = safe_read_from_json(data['whisper_path'])
        if os.path.isfile(data['metadata_path']):
            data['metadata'] = safe_load_from_file(data['metadata_path'])
        if os.path.isfile(data['srt_path']):
            data['captions'] = safe_load_from_file(data['srt_path'])
        
        self.url_data[video_id] = data
        return data

    def get_data(self, video_url=None, video_id=None):
        video_id = video_id or get_video_id(video_url)
        if video_id in self.url_data:
            return self.url_data[video_id]
        return self._init_data(video_url, video_id)

    def download_video(self, video_url):
        data = self.get_data(video_url)
        if not os.path.isfile(data['video_path']):
            video_info = VideoDownloader(video_url, download_directory=data['directory']).info
            safe_dump_to_file(video_info, data['info_path'])
            data['info'] = video_info
        return data['info']

    def extract_audio(self, video_url):
        data = self.get_data(video_url)
        audio_path = data.get('audio_path')
        if not os.path.isfile(audio_path):
            download_audio(video_url, audio_path)
        return audio_path

    def get_whisper_result(self, video_url):
        data = self.get_data(video_url)
        if not os.path.isfile(data['whisper_path']):
            audio = self.extract_audio(video_url)
            whisper = whisper_transcibe(audio)
            safe_dump_to_file(whisper, data['whisper_path'])
            data['whisper'] = whisper
        return data.get('whisper')
    def get_whisper_text(self, video_url):
        whisper_result = self.get_whisper_result(video_url)
        return whisper_result.get('text')
    def get_whisper_segments(self, video_url):
        whisper_result = self.get_whisper_result(video_url)
        return whisper_result.get('segments')
    def get_metadata(self, video_url):
        data = self.get_data(video_url)
        if not os.path.isfile(data['metadata_path']):
            whisper = self.get_whisper(video_url)
            summary = get_summary(whisper.get('text', ''), min_length=500, max_length=600)
            _, keywords_raw, _ = refine_keywords(whisper.get('text', ''))
            keywords = list(set(word for kw in keywords_raw for word in kw.split()))
            meta = {
                'title': data['info'].get('title'),
                'description': summary,
                'keywords': keywords
            }
            safe_dump_to_file(meta, data['metadata_path'])
            data['metadata'] = meta
        return data['metadata']

    def get_captions(self, video_url):
        data = self.get_data(video_url)
        if not os.path.isfile(data['srt_path']):
            whisper = self.get_whisper(video_url)
            export_srt(whisper.get('segments', []), data['srt_path'])
            data['captions'] = safe_load_from_file(data['srt_path'])
        return data['captions']

video_mgr = VideoDirectoryManager()

def download_video(video_url): return video_mgr.download_video(video_url)
def extract_video_audio(video_url): return video_mgr.extract_audio(video_url)
def get_video_whisper_result(video_url): return video_mgr.get_whisper_result(video_url)
def get_video_whisper_text(video_url): return video_mgr.get_whisper_text(video_url)
def get_video_whisper_segments(video_url): return video_mgr.get_whisper_segments(video_url)
def get_video_metadata(video_url): return video_mgr.get_metadata(video_url)
def get_video_captions(video_url): return video_mgr.get_captions(video_url)
def get_video_info(video_url): return video_mgr.get_data(video_url).get('info')
def get_video_directory(video_url): return video_mgr.get_data(video_url).get('directory')
def get_video_path(video_url): return video_mgr.get_data(video_url).get('video_path')
def get_audio_path(video_url): return video_mgr.get_data(video_url).get('audio_path')
def get_srt_path(video_url): return video_mgr.get_data(video_url).get('srt_path')
def get_metadata_path(video_url): return video_mgr.get_data(video_url).get('metadata_path')
