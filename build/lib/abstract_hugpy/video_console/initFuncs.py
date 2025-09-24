from .functions import (get_all_aggregated_data,get_aggregated_data,get_audio_path,get_meta_data,aggregate_key_maps,get_captions, is_complete, init_data, update_url_data, get_data, get_spec_data, update_spec_data, download_video, get_all_data, _merge_sources, _summarize_text, _extract_keywords, _generate_title, get_video_summary, get_video_keywords, get_video_title, update_meta_data, get_metadata, extract_audio, get_whisper_result, get_metadata_data, get_whisper_text, get_whisper_segments, update_thumbnails_data, get_thumbnail_data, get_thumbnails)
def initFuncs(self):
    try:
        for f in (get_all_aggregated_data,get_aggregated_data,get_audio_path,get_meta_data,aggregate_key_maps,get_captions, is_complete, init_data, update_url_data, get_data, get_spec_data, update_spec_data, download_video, get_all_data, _merge_sources, _summarize_text, _extract_keywords, _generate_title, get_video_summary, get_video_keywords, get_video_title, update_meta_data, get_metadata, extract_audio, get_whisper_result, get_metadata_data, get_whisper_text, get_whisper_segments, update_thumbnails_data, get_thumbnail_data, get_thumbnails):
            setattr(self, f.__name__, f)
    except Exception as e:
        logger.info(f"{e}")
    return self
