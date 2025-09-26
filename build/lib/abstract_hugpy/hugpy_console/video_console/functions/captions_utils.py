from ..imports import *
def get_captions(self, video_url):
    data = self.get_data(video_url)
    if not os.path.isfile(data['srt_path']):
        whisper = self.get_whisper_result(video_url)
        export_srt(whisper.get('segments', []), data['srt_path'])
        data['captions'] = safe_load_from_file(data['srt_path'])
    self.is_complete(key='captions',video_url=video_url)
    return data['captions']
