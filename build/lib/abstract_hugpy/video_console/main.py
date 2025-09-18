from .imports import *
from .initFuncs import initFuncs
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
VideoDirectoryManager = initFuncs(VideoDirectoryManager)
def get_video_mgr():
    video_mgr = VideoDirectoryManager()
    return video_mgr


    
