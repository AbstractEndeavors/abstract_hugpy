import os
import os,re
VIDEO_ENV_KEY = "DATA_DIRECTORY"
VIDEOS_ROOT_DEFAULT = "/mnt/24T/media/DATA/videos"
DOCS_ROOT_DEFAULT   = "/mnt/24T/media/DATA/documents"
VIDEOS_TEMP_DEFAULT   = "/mnt/24T/media/DATA/downloads"
# near your helpers
YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
DATA_SCHEMA = {
    "data_id": None,
    "url": None,
    "file_path": None,
    "info_path": "info.json",

    # text + metadata
    "text_path": "document.txt",
    "metadata_path": "metadata.json",
    "summary_path": "summary.txt",
    "keywords_path": "keywords.json",

    # optional audio/video derivatives
    "audio_path": "audio.wav",
    "speech_path": "speech.json",
    "preview_image": "preview.jpg",

    # aggregations
    "total_info_path": "total_info.json",
    "total_aggregated_path": "total_aggregated.json",

    # analysis
    "embeddings_path": "embeddings.npy",
    "entities_path": "entities.json",
    "topics_path": "topics.json",
}



# Full schema
VIDEO_SCHEMA = {
    "video_path": "video.mp4",
    "info_path": "info.json",
    "audio_path": "audio.wav",
    "whisper_path": "whisper.json",
    "captions_path": "captions.srt",
    "metadata_path": "metadata.json",
    "thumbnail_path": "thumb.jpg",
    "thumbnails_path": "thumbnails.json",
    "total_info_path": "total_info.json",
    "total_aggregated_path": "total_aggregated.json",
    "aggregated_directory": "aggregated",
    "aggregated_dir": {
        "aggregated_json_path": "aggregated.json",
        "aggregated_metadata_path": "aggregated_metadata.json",
        "best_clip_path": "best_clip.txt",
        "hashtags_path": "hashtags.txt",
    },
    "thumbnails_directory": "thumbnails",
    "thumbnails_dir": {
        "frames": "{video_id}_frame_{i}.jpg",  # pattern
    }
}
REMOVE_PHRASES = ['Video Converter', 'eeso', 'Auseesott', 'Aeseesott', 'esoft']
MODULE_DEFAULTS = {
    "whisper": {
        "path": "/mnt/24T/hugging_face/modules/whisper_base",
        "id": "openai/whisper-base",
        "name":"whisper"
    },
    "keybert": {
        "path": "/mnt/24T/hugging_face/modules/all_minilm_l6_v2",
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "keybert"
    },
    "summarizer": {
        "path": "/mnt/24T/hugging_face/modules/text_summarization",
        "id": "Falconsai/text_summarization",
        "name": "summarizer"
    },
    "flan": {
        "path": "/mnt/24T/hugging_face/modules/flan_t5_xl",
        "id": "google/flan-t5-xl",
        "name": "flan"
    },
    "bigbird": {
        "path": "/mnt/24T/hugging_face/modules/led_large_16384",
        "id": "allenai/led-large-16384",
        "name": "bigbird"
    },
    "deepcoder": {
        "path": "/mnt/24T/hugging_face/modules/DeepCoder-14B",
        "id": "agentica-org/DeepCoder-14B-Preview",
        "name": "deepcoder"
    },
    "huggingface": {
        "path": "/mnt/24T/hugging_face/modules/hugging_face_models",
        "id": "huggingface/hub",
        "name": "hugging_face_models"
    },
    "zerosearch": {
        "path": "/mnt/24T/hugging_face/modules/ZeroSearch_dataset",
        "id": "ZeroSearch/dataset",
        "name": "ZeroSearch"
    }
}
DEFAULT_PATHS = {
    "whisper": MODULE_DEFAULTS.get("whisper"),
    "keybert": MODULE_DEFAULTS.get("keybert"),
    "summarizer": MODULE_DEFAULTS.get("summarizer"),
    "flan": MODULE_DEFAULTS.get("flan"),
    "bigbird": MODULE_DEFAULTS.get("bigbird"),
    "deepcoder": MODULE_DEFAULTS.get("deepcoder"),
    "huggingface": MODULE_DEFAULTS.get("huggingface"),
    "zerosearch": MODULE_DEFAULTS.get("zerosearch")
}
