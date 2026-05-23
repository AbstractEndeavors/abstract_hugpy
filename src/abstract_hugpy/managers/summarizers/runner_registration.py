# managers/summarizers/runner_registration.py
from ..models.runners import register_runner
from .summarize_runner import SummarizeRunner

@register_runner("transformers", "summarize")
def _summarize_inprocess(entry):
    return SummarizeRunner(model_key=entry.name)

# managers/keywords/runner_registration.py
from ..models.runners import register_runner
from .keywords_runner import KeywordRunner

@register_runner("transformers", "keyword")
def _keyword_inprocess(entry):
    return KeywordRunner(model_key=entry.name)
