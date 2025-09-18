from ..imports import *
def get_video_summary(self,video_url):
    whisper = self.get_whisper_result(video_url)
    summary = get_summary(whisper.get('text', ''), min_length=500, max_length=600)
    return summary
def get_video_keywords(self,video_url):
    whisper = self.get_whisper_result(video_url)
    keywords_raw = refine_keywords(whisper.get('text', ''))
    keywords = list(set(word for kw in keywords_raw['combined_keywords'] for word in kw.split()))
    return keywords
def update_meta_data(self,metadata,video_url=None, video_id=None,data=None):
    return self.update_spec_data(
        metadata,
        'metadata',
        'metadata_path',
        video_url=video_url,
        video_id=video_id,
        data=data
        )
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
