from ..imports import *
from abstract_utilities import get_any_value

def get_video_url_info(video_url=None, video_info=None, video_id=None, video_root=None, registry=None):
    registry = registry or infoRegistry(video_root)
    video_id = video_id or get_video_id(video_url)
    video_info = video_info or registry.get_video_info(url=video_url, video_id=video_id, force_refresh=False)
    video_info = ensure_standard_paths(video_info or {"video_id": video_id, "url": video_url}, video_root)
    return video_info or {}

def get_schema_paths(video_url=None, video_info=None, video_id=None, video_root=None, key=None, registry=None):
    video_info = get_video_url_info(video_url, video_info, video_id, video_root, registry=registry)
    schema_paths = video_info.get('schema_paths', {})
    if key:
        return schema_paths.get(key)
    return schema_paths

def is_complete(self, key=None, video_url=None, video_id=None):
    data = self.get_data(video_url=video_url, video_id=video_id)
    total_info_path = data.get("total_info_path")

    if not os.path.isfile(total_info_path):
        safe_dump_to_file(self.init_key_map, total_info_path)

    total_info = safe_read_from_json(total_info_path)
    keys = make_list(key or self.complete_keys)

    if total_info.get("total"):
        return True

    for k in keys:
        if not total_info.get(k):
            values = self.complete_key_map.get(k)
            
            path = data.get(values.get("path"))
            if os.path.isfile(path):
                if values.get("keys") is True:
                    total_info[k] = True
                else:
                    key_data = safe_read_from_json(path)
                    if isinstance(key_data, dict):
                        if all(key_data.get(vk) for vk in values.get("keys", [])):
                            total_info[k] = True

    if all(total_info.get(k) for k in keys):
        total_info["total"] = True
        safe_dump_to_file(total_info, total_info_path)
        total_aggregated_path = data.get("total_aggregated_path")
        aggregate = self.get_aggregated_data(video_url=video_url, video_id=video_id)
        safe_dump_to_file(aggregate, total_aggregated_path)
        return self.get_data(video_url=video_url, video_id=video_id)
         

    safe_dump_to_file(total_info, total_info_path)
    return total_info


    safe_dump_to_file(data=total_info,file_path=data['total_info_path'])
def init_data(self, video_url, video_id=None):
    video_id = video_id or get_video_id(video_url)

    # 1. Resolve metadata (registry first, then schema)
    video_info = self.registry.get_video_info(
        url=video_url, video_id=video_id, force_refresh=False
    )
    video_info = ensure_standard_paths(
        video_info or {"video_id": video_id, "url": video_url},
        self.video_root
    )

    dir_path = os.path.dirname(video_info["video_path"])
    os.makedirs(dir_path, exist_ok=True)

    # 2. Save info.json immediately
    safe_dump_to_file(data=video_info, file_path=video_info["info_path"])
    schema_paths = video_info.get("schema_paths", {})
    # 3. Build unified data dict
    data = {
        **schema_paths,
        "url": video_url,
        "video_id": video_id,
        "directory": dir_path,
        "video_basename": os.path.basename(video_info["video_path"]),
        "info": video_info,
    }

    # 4. Load optional sidecar files
    

    def load_if_exists(key, target=None, loader=safe_load_from_file):
        path = schema_paths.get(f"{key}_path")
        if path and os.path.isfile(path):
            data[target or key] = loader(path)

    load_if_exists("whisper")
    load_if_exists("metadata")
    load_if_exists("thumbnails")
    load_if_exists("total_aggregated")

    srt_path = schema_paths.get("captions_path")
    if srt_path and os.path.isfile(srt_path):
        subs = pysrt.open(srt_path)
        data["captions"] = [
            {"start": str(sub.start), "end": str(sub.end), "text": sub.text}
            for sub in subs
        ]

    # 5. Register
    self.update_url_data(data, video_url=video_url, video_id=video_id)
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
    safe_dump_to_file(spec_data,path)
    return data
def download_video(self, video_url, video_id=None):
    data = self.get_data(video_url, video_id=video_id)



    info_path =  data.get('info_path')
    directory =  data.get('directory')
    video_path =  data.get('video_path')

    basename = os.path.basename(video_path)
    info = data.get('info')
    # if already present, skip
    if os.path.isfile(video_path):
        return info
    
    # tell VideoDownloader to place the file exactly where schema says
    
    
    VideoDownloader(
        url=video_url,
        download_directory=directory,                # use canonical folder
        output_filename=basename,# force name "video.mp4"
        download_video=True,

    )
    
    
    # merge downloader info into our schema
    video_id= data.get('id') or data.get('video_id') or info.get('id') or info.get('video_id')
    video_info = self.registry.get_video_info(video_id=video_id)
    schema_paths = video_info.get('schema_paths', {})
    data["info"].update(video_info)
    
    data.update(schema_paths)
    self.update_url_data(data, video_url=video_url, video_id=video_id)
    # refresh registry entry too
    self.registry.edit_info(data["info"], url=video_url, video_id=video_id)
    return video_info

def get_aggregated_data(self,video_url=None, video_id=None):
    video_id = video_id or get_video_id(video_url=video_url)
    data = self.get_data(video_url=video_url,video_id=video_id)
    if data.get('aggregate_data') == None:
        directory= data.get('directory')
        aggregated_dir = data.get('aggregated_directory')
        aggregate_js = aggregate_from_base_dir(directory=directory,aggregated_dir=aggregated_dir)
        data['aggregate_data'] = aggregate_js
        self.update_url_data(data=data,video_url=video_url, video_id=video_id)
    return data.get('aggregate_data')
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
    self.get_aggregated_data(video_url)
    video_id = get_video_id(video_url)
    return self.url_data[video_id]
def get_all_aggregated_data(self, video_url):
    self.get_all_data(video_url)
    return self.get_aggregated_data(video_url)
