from ..imports import *
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
def init_data(self, video_url, video_id):
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
        'video_basename': video_basename,
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
        data['whisper'] = safe_load_from_file(data['whisper_path'])
    if os.path.isfile(data['metadata_path']):
        data['metadata'] = safe_load_from_file(data['metadata_path'])
    if os.path.isfile(data['thumbnails_path']):
        data['thumbnails'] = safe_load_from_file(data['thumbnails_path'])
    if os.path.isfile(data['srt_path']):
        subs = pysrt.open(data['srt_path'])
        data['captions'] = [
            {"start": str(sub.start), "end": str(sub.end), "text": sub.text}
            for sub in subs
        ]
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
    return self.init_data(video_url, video_id)
def get_spec_data(self,key,path_str, video_url=None, video_id=None):
    data = self.get_data(video_url=video_url,video_id=video_id)
    values = data.get(key,{})
    path = data[path_str]
    if not os.path.isfile(path):
        safe_dump_to_file(values, path)
    return safe_load_from_file(path)
def update_spec_data(self,spec_data,key,path_key,video_url=None, video_id=None,data=None):
    data = data or self.get_data(video_url=video_url,video_id=video_id)
    data[key] = spec_data
    path = data[path_key]
    self.update_url_data(data,video_url=video_url,video_id=video_id)
    safe_dump_to_file(spec_data,path )
    return data
def download_video(self, video_url):
    data = self.get_data(video_url)
    if not os.path.isfile(data['video_path']):
        video_info = for_dl_video(url=video_url, preferred_format="mp4",download_directory=data['directory'],output_filename=data['video_basename'],download_video=True)
        safe_dump_to_file(data=video_info, file_path=data['info_path'])
        data['info'] = video_info
    return data['info']
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
