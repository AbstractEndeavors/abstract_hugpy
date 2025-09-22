from ..imports import *


def extract_audio(self, video_url, force_wav=False):
    """Download and return path to audio for given video_url."""
    data = self.get_data(video_url)
    video_id = data.get("video_id") or make_video_id(video_url)  # use your own util
    audio_dir = data.get("audio_dir")

    # Prefer original format (.webm/.m4a) for Whisper
    ext = "wav" if force_wav else "webm"
    audio_path = os.path.join(audio_dir, f"{video_id}.{ext}")

    if not os.path.isfile(audio_path):
        if force_wav:
            # Download + convert to wav
            download_audio(video_url, audio_path, format="bestaudio", to_wav=True)
        else:
            # Download audio only (no conversion)
            download_audio(video_url, audio_path, format="251")  # opus webm
    return audio_path


def get_whisper_result(self, video_url):
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
