logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deepcoder_server.log"),
        logging.StreamHandler()
    ]
)
logger = get_logFile(__name__)

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
        "name": "DeepCoder"
    },
    "huggingface": {
        "path": "/mnt/24T/hugging_face/modules/hugging_face_models",
        "id": "huggingface/hub",
        "name": "hugging_face_models"
    },
    "zerosearch": {
        "path": "/mnt/24T/hugging_face/modules/all_minilm_l6_v2",
        "id": "ZeroSearch/dataset",
        "name": "ZeroSearch"
    }
}
DEFAULT_PATHS = {
    "whisper": MODULE_DEFAULTS.get("whisper").get('path'),
    "keybert": MODULE_DEFAULTS.get("keybert").get('path'),
    "summarizer": MODULE_DEFAULTS.get("summarizer").get('path'),
    "flan": MODULE_DEFAULTS.get("flan").get('path'),
    "bigbird": MODULE_DEFAULTS.get("bigbird").get('path'),
    "deepcoder": MODULE_DEFAULTS.get("deepcoder").get('path'),
    "huggingface": MODULE_DEFAULTS.get("huggingface").get('path'),
    "zerosearch": MODULE_DEFAULTS.get("zerosearch").get('path')
}
def resolve_model_path(entry):
    """Return a valid model path or HF repo id from DEFAULT_PATHS entry."""
    if entry is None:
        logger.error("{entry}: DEFAULT_PATHS entry missing.")
        return None
    if isinstance(entry,str) and entry in MODULE_DEFAULTS:
       return MODULE_DEFAULTS.get(entry)

    if isinstance(entry, dict):
        local_path = entry.get("path")
        repo_id = entry.get("id")
        name = entry.get("name")

        if local_path and os.path.exists(local_path):
            logger.info(f"{name} resolved local model path: {local_path}")
            return local_path

        if repo_id:
            logger.info(f"{name} resolved remote repo id: {repo_id}")
            return repo_id

        logger.error(f"{name}: malformed entry: {entry}")
        return None

    if isinstance(entry, str):
        logger.info(f"{name} using direct model string: {entry}")
        return entry

    logger.error(f"{name}: invalid model path type: {type(entry)}")
    return None
