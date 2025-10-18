# What you pass to from_pretrained(...)
MODEL_SOURCES = {
    # Local path (already downloaded, contains config.json & weights)
    "whisper":     "/mnt/24T/hugging_face/modules/whisper_base",
    "keybert":     "sentence-transformers/all-MiniLM-L6-v2",  # pure repo id
    "summarizer":  "Falconsai/text_summarization",
    "flan":        "/mnt/24T/hugging_face/modules/flan_t5_xl",
    "bigbird":     "allenai/led-large-16384",
    "deepcoder":   "agentica-org/DeepCoder-14B-Preview",
    # not a model – dataset; don't use from_pretrained for this one
    # "zerosearch": "ZeroSearch/dataset",
}

# Optional cache destinations to control where HF stores/reads files.
# Pass these via cache_dir=... when calling from_pretrained(...).
MODEL_CACHE_DIRS = {
    "whisper":    "/mnt/24T/hugging_face/modules/whisper_base",
    "keybert":    "/mnt/24T/hugging_face/modules/all_minilm_l6_v2",
    "summarizer": "/mnt/24T/hugging_face/modules/text_summarization",
    "flan":       "/mnt/24T/hugging_face/modules/flan_t5_xl",
    "bigbird":    "/mnt/24T/hugging_face/modules/led_large_16384",
    "deepcoder":  "/mnt/24T/hugging_face/modules/DeepCoder-14B",
    # datasets use snapshot_download(repo_type="dataset") instead
    # "zerosearch": "/mnt/24T/hugging_face/modules/ZeroSearch_dataset",
}
