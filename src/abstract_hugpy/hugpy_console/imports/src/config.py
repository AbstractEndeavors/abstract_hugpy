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
