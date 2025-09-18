from ..imports import *
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
