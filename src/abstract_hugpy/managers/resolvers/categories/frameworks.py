from .imports import *
FRAMEWORK_RUNNERS: Dict[Tuple[str, str], Type[Runner]] = {
    ("transformers", "text-generation"):              DeepCoderChatRunner,
    ("llama_cpp",    "text-generation"):              LlamaCppChatRunner,
    ("transformers", "image-text-to-text"):           VisionRunner,
    ("transformers", "automatic-speech-recognition"): WhisperRunner,
    ("transformers", "text-summarization"):                SummarizeRunner,
    ("transformers", "text2text-generation"):         SummarizeRunner,
    ("transformers", "feature-extraction"):           FeatureExtractionRunner,
    ("transformers", "sentence-similarity"):          FeatureExtractionRunner,
}

# Derived from FRAMEWORK_RUNNERS so it can't drift.
KNOWN_TASKS_REGISTRY: frozenset[str] = frozenset(task for _, task in FRAMEWORK_RUNNERS.keys())

