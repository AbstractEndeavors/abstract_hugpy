from ..imports import *
def update_thumbnails_data(self,thumbnails,video_url=None, video_id=None,data=None):
    return self.update_spec_data(
        thumbnails,
        'thumbnails',
        'thumbnails_path',
        video_url=video_url,
        video_id=video_id,
        data=data
        )
def get_thumbnail_data(self, video_url=None,video_id=None):
    return self.get_spec_data(
        'thumbnails',
        'thumbnails_path',
        video_url=video_url,
        video_id=video_id
        )
def get_thumbnails(self,video_url):
    data = self.get_data(video_url)
    thumbnails = self.get_thumbnail_data(video_url)
    thumbnail_paths = thumbnails.get('thumbnail_paths',[])
    if not thumbnail_paths:
        video_path =data.get('video_path')
        thumbnails_dir = data.get('thumbnails_dir')
        video_id = data.get('video_id')
        thumbnail_paths = extract_video_frames_unique(
            video_path=video_path,
            directory=thumbnails_dir,
            video_id=video_id,
            method="dhash",
            dhash_thresh=4
        )
        thumbnails['thumbnail_paths'] = thumbnail_paths
        data = self.update_thumbnails_data(thumbnails,video_url)
    thumbnail_texts = thumbnails.get('thumbnail_texts',{})
    if not thumbnail_texts:
        thumbnail_texts =  [ocr_image(frame) for frame in thumbnail_paths]
        thumbnails['thumbnail_texts'] = thumbnail_texts
        data = self.update_thumbnails_data(thumbnails,video_url)
    self.is_complete(key='thumbnails',video_url=video_url)
    return data['thumbnails']

