from yt_dlp import YoutubeDL
from .hugging_face_models.google_flan import *
from .hugging_face_models.keybert_model import *
from .hugging_face_models.summarizer_model import *
from .hugging_face_models.whisper_model import *
from .image_utils import *
from abstract_webtools import get_video_info, VideoDownloader
from abstract_apis import *
##from abstract_videos.text_tools.summarizer_utils.summarizer_services import get_summary
from abstract_utilities.abstract_classes import SingletonMeta
from abstract_utilities import make_list,get_logFile, safe_dump_to_file, safe_load_from_file, safe_read_from_json,get_any_value
REMOVE_PHRASES = ['Video Converter', 'eeso', 'Auseesott', 'Aeseesott', 'esoft']
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
            self.complete_key_map={
                "video_path":{"keys":True,"path":'video_path'},
                "audio_path":{"keys":True,"path":'audio_path'},
                "info":{"keys":True,"path":'info_path'},
                "whisper":{"keys":['segments','text'],"path":'whisper_path'},
                "metadata":{"keys":['keywords','description','title'],"path":'metadata_path'},
                "captions":{"keys":True,"path":'srt_path'},
                "thumbnails":{"keys":["thumbnail_texts","thumbnail_paths"],"path":'thumbnails_path'}
                }
            self.init_key_map={
                "video_path":False,
                "audio_path":False,
                "info":False,
                "whisper":False,
                "metadata":False,
                "captions":False,
                "thumbnails":False
                }
            self.complete_keys = list(self.complete_key_map.keys())
    def is_complete(self,key=None,video_url=None, video_id=None):
        data = self.get_data(video_url=video_url, video_id=video_id)
        if not os.path.isfile(data['total_info_path']):
            safe_dump_to_file(data=self.init_key_map,file_path=data['total_info_path'])
        total_info = safe_read_from_json(data['total_info_path'])
        keys = make_list(key or self.complete_keys)

        if total_info.get('total') == True:
            return True
        for key in keys:
            if total_info.get(key) != True:
                values = self.complete_key_map.get(key)
                value_keys = values.get("keys")
                path = data.get(values.get("path"))
                if os.path.isfile(path):
                    if value_keys == True:
                            total_info[key] = True
                    else:
                        key_data = safe_read_from_json(path)
                        if isinstance(key_data,dict):
                            total_info_key = True
                            for value_key in value_keys:
                                
                                key_value = key_data.get(value_key)
                                if not key_value:
                                    total_info_key = False
                                    break
                            if total_info_key:
                                total_info[key] = True
                        
        total_bools = list(set(total_info.keys()))
        if len(total_bools) == 1 and total_bools[0] == True:
            total_info['total'] = True
            total_data = self.get_data(video_url=video_url)
            safe_dump_to_file(data=total_info,file_path=data['total_info_path'])
            safe_dump_to_file(data=total_data,file_path=data['total_data_path'])
            return total_data
        safe_dump_to_file(data=total_info,file_path=data['total_info_path'])
    def _init_data(self, video_url, video_id):
        dir_path = os.path.join(self.videos_directory, video_id)
        os.makedirs(dir_path, exist_ok=True)
        info_path = os.path.join(dir_path, 'video_info.json')
        total_info_path = os.path.join(dir_path, 'total_info.json')
        total_data_path = os.path.join(dir_path, 'total_data.json') 
        thumbnails_dir = os.path.join(dir_path, 'thumbnails')
        os.makedirs(thumbnails_dir, exist_ok=True)
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
            'thumbnails_dir': thumbnails_dir,
            'total_info_path': total_info_path,
            'total_data_path':total_data_path,
            'thumbnails_path': os.path.join(dir_path, 'thumbnails.json'),
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
        if os.path.isfile(data['thumbnails_path']):
            data['thumbnails'] = safe_load_from_file(data['thumbnails_path'])
        if os.path.isfile(data['srt_path']):
            data['captions'] = safe_load_from_file(data['srt_path'])
        
        self.update_url_data(data,video_url=video_url, video_id=video_id)
        return data
    def update_url_data(self,data,video_url=None, video_id=None):
        video_id = video_id or get_video_id(video_url)
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
    def update_thumbnails_data(self,thumbnails,video_url,data=None):
        data = data or self.get_data(video_url)
        data['thumbnails'] = thumbnails
        self.update_url_data(data,video_url=video_url)
        safe_dump_to_file(thumbnails, data['thumbnails_path'])
        return data
    def get_thumbnail_data(self, video_url):
        data = self.get_data(video_url)
        thumbnails = data.get('thumbnails',{})
        if not os.path.isfile(data['thumbnails_path']):
            safe_dump_to_file(thumbnails, data['thumbnails_path'])
        data['thumbnails'] = safe_load_from_file(data['thumbnails_path'])
        return data['thumbnails']
    def get_thumbnails(self,video_url):
        data = self.get_data(video_url)
        thumbnails = self.get_thumbnail_data(video_url)
        thumbnail_paths = thumbnails.get('thumbnail_paths')
        if not thumbnail_paths:
            video_path =data.get('video_path')
            thumbnails_dir = data.get('thumbnails_dir')
            video_id = data.get('video_id')
            thumbnails['thumbnail_paths'] = extract_unique_via_capture(
                video_path = video_path,
                directory = thumbnails_dir,
                video_id = video_id
                )
            data = self.update_thumbnails_data(thumbnails,video_url)
        thumbnail_texts = thumbnails.get('thumbnail_texts')
        if not thumbnail_texts:
            thumbnails['thumbnail_texts'] = [ocr_image(frame) for frame in thumbnail_paths]
            data = self.update_thumbnails_data(thumbnails,video_url)
        self.is_complete(key='thumbnails',video_url=video_url)
        return data['thumbnails']
        
    def get_whisper_result(self, video_url):
        data = self.get_data(video_url)
        if not os.path.isfile(data['whisper_path']):
            audio = self.extract_audio(video_url)
            whisper = whisper_transcribe(audio)
            safe_dump_to_file(whisper, data['whisper_path'])
            data['whisper'] = whisper
            self.is_complete(key='whisper',video_url=video_url)
        return data.get('whisper')
    def get_metadata_data(self, video_url):
        data = self.get_data(video_url)
        metadata = data.get('metadata',{})
        if not os.path.isfile(data['metadata_path']):
            safe_dump_to_file(metadata, data['metadata_path'])
        data['metadata'] = safe_load_from_file(data['metadata_path'])
        return data['metadata']
    def get_whisper_text(self, video_url):
        whisper_result = self.get_whisper_result(video_url)
        return whisper_result.get('text')
    def get_whisper_segments(self, video_url):
        whisper_result = self.get_whisper_result(video_url)
        return whisper_result.get('segments')
    def get_video_summary(self,video_url):
        whisper = self.get_whisper_result(video_url)
        summary = get_summary(whisper.get('text', ''), min_length=500, max_length=600)
        return summary
    def get_video_keywords(self,video_url):
        whisper = self.get_whisper_result(video_url)
        keywords_raw = refine_keywords(whisper.get('text', ''))
        keywords = list(set(word for kw in keywords_raw['combined_keywords'] for word in kw.split()))
        return keywords
    def update_meta_data(self,metadata,video_url,data=None):
        data = data or self.get_data(video_url)
        data['metadata'] = metadata
        self.update_url_data(data,video_url=video_url)
        safe_dump_to_file(metadata, data['metadata_path'])
        return data

    def get_metadata(self, video_url):
        data = self.get_data(video_url)
        metadata = self.get_metadata_data(video_url)
        title = metadata.get('title')
        if not title:
            metadata['title'] = data['info'].get('title')
            data = self.update_meta_data(metadata,video_url)
        description = metadata.get('description')
        if not description:
            metadata['description'] = self.get_video_summary(video_url)
            data = self.update_meta_data(metadata,video_url)
        keywords = metadata.get('keywords')
        if not keywords:
            metadata['keywords'] = self.get_video_keywords(video_url)
            data = self.update_meta_data(metadata,video_url)
        self.is_complete(key='metadata',video_url=video_url)
        return data['metadata']
    
    def get_captions(self, video_url):
        data = self.get_data(video_url)
        if not os.path.isfile(data['srt_path']):
            whisper = self.get_whisper_result(video_url)
            export_srt(whisper.get('segments', []), data['srt_path'])
            data['captions'] = safe_load_from_file(data['srt_path'])
        self.is_complete(key='captions',video_url=video_url)
        return data['captions']

    def get_all_data(self, video_url):
        
        data = self.is_complete(video_url=video_url)
        if data:
            return data
        data = self.get_data(video_url)
        self.download_video(video_url)
        self.extract_audio(video_url)
        self.get_whisper_result(video_url)
        self.get_thumbnails(video_url)
        self.get_captions(video_url)
        self.get_metadata(video_url)
        video_id = get_video_id(video_url)
        return self.url_data[video_id]
def get_video_mgr():
    video_mgr = VideoDirectoryManager()
    return video_mgr
def download_video(video_url): return get_video_mgr().download_video(video_url)
def extract_video_audio(video_url): return get_video_mgr().extract_audio(video_url)
def get_video_whisper_result(video_url): return get_video_mgr().get_whisper_result(video_url)
def get_video_whisper_text(video_url): return get_video_mgr().get_whisper_text(video_url)
def get_video_whisper_segments(video_url): return get_video_mgr().get_whisper_segments(video_url)
def get_video_metadata(video_url): return get_video_mgr().get_metadata(video_url)
def get_video_captions(video_url): return get_video_mgr().get_captions(video_url)
def get_video_thumbnails(video_url): return get_video_mgr().get_thumbnails(video_url)
def get_video_info(video_url): return get_video_mgr().get_data(video_url).get('info')
def get_video_directory(video_url): return get_video_mgr().get_data(video_url).get('directory')
def get_video_path(video_url): return get_video_mgr().get_data(video_url).get('video_path')
def get_audio_path(video_url): return get_video_mgr().get_data(video_url).get('audio_path')
def get_thumbnail_dir(video_url): return get_video_mgr().get_data(video_url).get('thumbnail_dir')
def get_srt_path(video_url): return get_video_mgr().get_data(video_url).get('srt_path')
def get_metadata_path(video_url): return get_video_mgr().get_data(video_url).get('metadata_path')
def get_all_data(video_url): return get_video_mgr().get_all_data(video_url)
