from .imports import *
from .constants import *
# pipeline_tag -> task vocabulary. Every value is HF-derivable.
HF_TASK_TO_TASKS = {
    "text-generation": ["text-generation"],
    "image-text-to-text": ["image-text-to-text", "text-generation"],
    "automatic-speech-recognition": ["automatic-speech-recognition"],
    "text-summarization": ["text-summarization"],
    "text2text-generation": ["text-summarization", "text2text-generation"],
    "feature-extraction": ["feature-extraction", "sentence-similarity"],
    "sentence-similarity": ["feature-extraction", "sentence-similarity"],
}
RUNNER_PAIRS = {
    ("transformers", "text-generation"), ("llama_cpp", "text-generation"),
    ("transformers", "image-text-to-text"), ("transformers", "automatic-speech-recognition"),
    ("transformers", "text-summarization"), ("transformers", "text2text-generation"),
    ("transformers", "feature-extraction"), ("transformers", "sentence-similarity"),
}
MEDIA_DEFAULTS: Dict[str, str] = {
    "document": DEFAULT_CHAT_MODEL,
    "code":     DEFAULT_CHAT_MODEL,
    "text":     DEFAULT_CHAT_MODEL,
    "image":    DEFAULT_VISION_MODEL,
    "audio":    DEFAULT_WHISPER_MODEL,
    "video":    DEFAULT_WHISPER_MODEL,
}

TASK_DEFAULTS: Dict[str, str] = {
    "text-generation":              DEFAULT_CHAT_MODEL,
    "image-text-to-text":           DEFAULT_VISION_MODEL,
    "automatic-speech-recognition": DEFAULT_WHISPER_MODEL,
    "text-summarization":           DEFAULT_SUMMARIZE_MODEL,
    "text2text-generation":         DEFAULT_SUMMARIZE_MODEL,
    "feature-extraction":           DEFAULT_EMBED_MODEL,
    "sentence-similarity":          DEFAULT_EMBED_MODEL,
}
