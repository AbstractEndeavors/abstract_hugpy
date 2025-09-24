from ..imports import *
from abstract_utilities import eatAll
def get_audio_path(self, video_url, force_wav=True):
    data = self.get_data(video_url)
    video_id = data.get("video_id") or make_video_id(video_url)
    audio_path = data.get("audio_path")
    
    audio_dir = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    filename, ext = os.path.splitext(basename)
    if ext in [".wav",".webm"]:
        data["audio_dir"] = audio_dir
        self.update_url_data(data,video_url=video_url, video_id=video_id)
        return audio_path
    ext = "wav" if force_wav else "webm"
    final_path = os.path.join(audio_dir, f"{filename}.{ext}")
    data["audio_path"] = final_path
    data["audio_dir"] = audio_dir
    # update registry
    self.update_url_data(data,video_url=video_url, video_id=video_id)
    return final_path

def extract_audio(self, video_url, force_wav=True):
    audio_path = self.get_audio_path(video_url, force_wav=force_wav)
    base, ext = os.path.splitext(audio_path)
    ext = ext.lstrip(".")

    if not os.path.isfile(audio_path):
        audio_path = download_audio(video_url, base, output_format=ext)

    return audio_path

def get_whisper_result(self, video_url,force_wav=False):
    data = self.get_data(video_url)
    if not os.path.isfile(data['whisper_path']):
        audio = self.extract_audio(video_url, force_wav=force_wav)
        whisper = whisper_transcribe(audio)
        safe_dump_to_file(whisper, data['whisper_path'])
        data['whisper'] = whisper
        self.is_complete(key='whisper',video_url=video_url)
    return data.get('whisper')

def get_metadata_data(self, video_url=None, video_id=None):
    return self.get_spec_data(
        'metadata',
        'metadata_path',
        video_url=video_url,
        video_id=video_id
        )

def get_whisper_text(self, video_url):
    whisper_result = self.get_whisper_result(video_url)
    return whisper_result.get('text')

def get_whisper_segments(self, video_url):
    whisper_result = self.get_whisper_result(video_url)
    return whisper_result.get('segments')
