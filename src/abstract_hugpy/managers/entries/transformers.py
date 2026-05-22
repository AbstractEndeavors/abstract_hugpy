# managers/models/entries/transformers.py — additions
register_transformers(
    name="flan-t5-xl",
    task="text2text-generation",
    runner="summarize",
    model_max_length=1024,
    hub_id="google/flan-t5-xl",
    folder="google/flan-t5-xl",
)

register_transformers(
    name="t5-summarizer",                 # whatever DEFAULT_PATHS["summarizer"] points at
    task="summarization",
    runner="summarize",
    model_max_length=512,
    hub_id="...",                          # fill in
    folder="...",
)

register_transformers(
    name="falconsai-text-summarization",
    task="summarization",
    runner="summarize",
    model_max_length=512,
    hub_id="Falconsai/text_summarization",
    folder="Falconsai/text_summarization",
)

register_transformers(
    name="led-large-16384",
    task="summarization",
    runner="summarize",
    model_max_length=16384,
    hub_id="allenai/led-large-16384",
    folder="allenai/led-large-16384",
)

register_transformers(
    name="all-minilm-l6-v2",
    task="sentence-similarity",
    runner="keyword",
    model_max_length=512,
    hub_id="sentence-transformers/all-minilm-l6-v2",
    folder="sentence-transformers/all-minilm-l6-v2",
)
