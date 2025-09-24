from ..imports import *
from abstract_webtools.managers.videoDownloader import get_video_filepath,get_video_id,get_video_info
from abstract_utilities import safe_read_from_json, safe_dump_to_file, get_any_value, make_list
from abstract_webtools.managers.videoDownloader.src.functions.info_utils import _ensure_standard_paths

import os
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
        aggregate = aggregate_from_base_dir(data.get('directory'))
        total_data.update(aggregate)
        safe_dump_to_file(data=total_data,file_path=data['total_data_path'])
        return total_data
    safe_dump_to_file(data=total_info,file_path=data['total_info_path'])
def init_data(self, video_url, video_id=None):
    """
    Initialize a full data record for a video.
    Always enforces schema so file_path, info_path, etc. exist.
    """

    # 1. Resolve ID
    video_id = video_id or get_video_id(video_url)

    # 2. Ask registry for info (metadata only, no download yet)
    video_info = self.registry.get_video_info(url=video_url, video_id=video_id, force_refresh=False)

    # 3. Ensure schema (guarantees file_path, info_path, etc.)
    video_info = _ensure_standard_paths(
        video_info or {"video_id": video_id, "url": video_url},
        self.video_root
    )

    # 4. Canonical base dir
    dir_path = os.path.dirname(video_info["file_path"])
    os.makedirs(dir_path, exist_ok=True)

    # 5. Standard sidecar files
    info_path        = video_info["info_path"]
    total_info_path  = os.path.join(dir_path, "total_info.json")
    total_data_path  = os.path.join(dir_path, "total_data.json")
    aggregated_dir   = os.path.join(dir_path, "aggregated")
    total_agg_path   = os.path.join(dir_path, "total_aggregated.json")
    os.makedirs(video_info["thumbnails_dir"], exist_ok=True)
    os.makedirs(aggregated_dir, exist_ok=True)

    # 6. Save canonical info.json immediately
    safe_dump_to_file(video_info, info_path)

    # 7. Build the unified data dict
    data = {
        "url": video_url,
        "video_id": video_id,
        "directory": dir_path,
        "info_path": info_path,
        "video_basename": os.path.basename(video_info["file_path"]),
        "video_path": video_info["file_path"],
        "thumbnails_dir": video_info["thumbnails_dir"],
        "thumbnails_path": video_info.get("thumbnails_path"),
        "audio_path": video_info.get("audio_path"),
        "whisper_path": video_info.get("whisper_path"),
        "srt_path": video_info.get("srt_path"),
        "metadata_path": video_info.get("metadata_path"),
        "total_info_path": total_info_path,
        "total_data_path": total_data_path,
        "aggregated_dir": aggregated_dir,
        "total_aggregated_path": total_agg_path,
        "info": video_info,
    }

    # 8. Load optional existing sidecar files
    if os.path.isfile(data["whisper_path"]):
        data["whisper"] = safe_load_from_file(data["whisper_path"])
    if os.path.isfile(data["metadata_path"]):
        data["metadata"] = safe_load_from_file(data["metadata_path"])
    if os.path.isfile(data["thumbnails_path"]):
        data["thumbnails"] = safe_load_from_file(data["thumbnails_path"])
    if os.path.isfile(total_agg_path):
        data["aggregate_data"] = safe_load_from_file(total_agg_path)
    if os.path.isfile(data["srt_path"]):
        subs = pysrt.open(data["srt_path"])
        data["captions"] = [
            {"start": str(sub.start), "end": str(sub.end), "text": sub.text}
            for sub in subs
        ]

    # 9. Register in memory
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

    # if already present, skip
    if os.path.isfile(data["video_path"]):
        return data["info"]

    # tell VideoDownloader to place the file exactly where schema says
    vd = VideoDownloader(
        url=video_url,
        download_directory=data["directory"],                # use canonical folder
        output_filename=os.path.basename(data["video_path"]),# force name "video.mp4"
        download_video=True,
        get_info=True,
    )

    # merge downloader info into our schema
    video_info = vd.info or {}
    video_info.update({
        "file_path": data["video_path"],
        "info_path": data["info_path"],
        "video_id": data["video_id"],
    })
    safe_dump_to_file(video_info, data["info_path"])
    data["info"] = video_info

    # refresh registry entry too
    self.registry.edit_info(video_info, url=video_url, video_id=data["video_id"])
    return video_info

def get_aggregated_data(self,video_url=None, video_id=None):
    video_id = video_id or get_video_id(video_url=video_url)
    data = self.get_data(video_url=video_url,video_id=video_id)
    if data.get('aggregate_data') == None:
        directory= data.get('directory')
        aggregated_dir = data.get('aggregated_dir')
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
def aggregate_key_maps(self, video_url=None, video_id=None) -> dict:
    """
    SEO-driven aggregator for video metadata.
    Priority: video_info.json → whisper_result.json → video_metadata.json

    Features:
    - Rolling merge with priority tiers.
    - Keyword dedupe + normalization.
    - Continuity fallbacks (title from keywords, desc from title).
    - Refinement with BigBird + GPT2 generator.
    - Heatmap highlights injection into description.
    - Thumbnail picked from top heatmap peaks.
    - Hashtag generation from keywords.
    - Category classification from keyword clusters.
    - Chapter generation from heatmap peaks + transcript.
    """

    data = self.get_data(video_url=video_url, video_id=video_id)
    files_priority = self.key_maps["file_tires"]
    key_map = self.key_maps["key_maps"]
    merged = {}

    # === Step 1: Rolling Merge ===
    for filename in files_priority:
        path = os.path.join(data["directory"], filename)
        if not os.path.isfile(path):
            continue
        content = safe_read_from_json(path) or {}
        for field, cfg in key_map.items():
            keys = cfg.get("keys", [])
            current_val = merged.get(field)
            if current_val:
                continue
            candidate = get_any_value(content, keys)
            if candidate:
                merged[field] = candidate

    # === Step 2: Continuity / Normalization ===
    if "keywords" in merged:
        kws = make_list(merged.get("keywords"))
        merged["keywords"] = sorted(set([kw.strip().lower() for kw in kws if kw]))

    if not merged.get("title") and merged.get("keywords"):
        merged["title"] = " ".join(merged["keywords"][:5]).title()

    if not merged.get("description") and merged.get("title"):
        merged["description"] = f"Video about {merged['title']}"

    # === Step 3: Refinement with BigBird + GPT2 ===
    transcript_text = merged.get("transcript") or merged.get("description") or ""
    generator = get_generator()

    if merged.get("title"):
        draft = refine_with_gpt(merged["title"], task="title", generator_fn=generator)
        if draft and len(draft.split()) > 3:
            merged["title"] = draft

    if merged.get("description"):
        draft = refine_with_gpt(transcript_text, task="description", generator_fn=generator)
        if draft and len(draft.split()) > 10:
            merged["description"] = draft

    # === Step 4: Heatmap Integration ===
    heatmap = merged.get("heatmap") or []
    highlights = []
    if isinstance(heatmap, list) and heatmap:
        top_segments = sorted(heatmap, key=lambda x: x["value"], reverse=True)[:3]
        for seg in top_segments:
            mins, secs = divmod(int(seg["start_time"]), 60)
            highlights.append(f"{mins}:{secs:02d}")
        if highlights:
            merged["description"] += "\n\nHighlights at: " + ", ".join(highlights)

    # === Step 5: Thumbnail from Heatmap Peaks ===
    try:
        if heatmap:
            top_peak = max(heatmap, key=lambda x: x["value"])
            peak_time = int((top_peak["start_time"] + top_peak["end_time"]) / 2)

            clip = VideoFileClip(data["video_path"])
            frame = clip.get_frame(peak_time)
            clip.close()

            thumb_path = os.path.join(data["directory"], "thumb.jpg")
            cv2.imwrite(thumb_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            thumbnail_url = generate_media_url(
                thumb_path,
                domain=data.get("domain", "https://example.com"),
                repository_dir=self.repository_dir
            )
            merged["thumbnail_url"] = thumbnail_url
    except Exception as e:
        print(f"Thumbnail selection failed: {e}")

    # === Step 6: Hashtag Generation ===
    hashtags = []
    for kw in merged.get("keywords", []):
        clean_kw = kw.replace(" ", "")
        if clean_kw.isalpha() and len(clean_kw) > 2:
            hashtags.append(f"#{clean_kw}")
    hashtags = hashtags[:10]
    if hashtags:
        merged["description"] += "\n\n" + " ".join(hashtags)
        merged["hashtags"] = hashtags

    # === Step 7: Category Classification ===
    def classify_category(keywords):
        kws = [kw.lower() for kw in keywords]
        if any(k in kws for k in ["comedy", "funny", "skit", "humor"]):
            return "Comedy"
        if any(k in kws for k in ["music", "song", "album", "concert"]):
            return "Music"
        if any(k in kws for k in ["news", "politics", "report", "debate"]):
            return "News & Politics"
        if any(k in kws for k in ["education", "tutorial", "lesson", "howto"]):
            return "Education"
        if any(k in kws for k in ["gaming", "playthrough", "walkthrough", "esports"]):
            return "Gaming"
        if any(k in kws for k in ["sports", "game", "match", "tournament"]):
            return "Sports"
        if any(k in kws for k in ["review", "tech", "product", "unboxing"]):
            return "Science & Technology"
        return "Entertainment"

    merged["category"] = classify_category(merged.get("keywords", []))

    # === Step 8: Chapters (YouTube-style) ===
    chapters = []
    if isinstance(heatmap, list) and heatmap:
        top_segments = sorted(heatmap, key=lambda x: x["value"], reverse=True)[:5]
        for seg in top_segments:
            start_time = int(seg["start_time"])
            mins, secs = divmod(start_time, 60)
            timestamp = f"{mins}:{secs:02d}"

            # short label from transcript or keywords
            snippet = transcript_text[:120] if transcript_text else "Segment"
            label = refine_with_gpt(
                snippet,
                task="title",
                generator_fn=generator
            ) or "Chapter"
            chapters.append({"time": timestamp, "title": label})

    if chapters:
        merged["chapters"] = chapters
        # also embed chapter list into description for YouTube
        chapter_lines = [f"{c['time']} {c['title']}" for c in chapters]
        merged["description"] += "\n\nChapters:\n" + "\n".join(chapter_lines)

    # === Step 9: Save Final ===
    total_data_path = os.path.join(data["directory"], "total_data.json")
    safe_dump_to_file(merged, total_data_path)

    return merged
