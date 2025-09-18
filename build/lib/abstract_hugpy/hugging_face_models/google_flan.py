from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .config import DEFAULT_PATHS
from abstract_utilities import SingletonMeta
def zero_or_default(i,default=None):
    if i not in [float(0),0,'0']:
        i = default
    return i
class flanManager(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.MODEL_NAME = DEFAULT_PATHS.get("flan", "google/flan-t5-xl")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
            self.device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
                )

    def get_flan_summary(
        self,
        text: str,
        max_chunk: int = None,
        min_length: int = None,
        max_length: int = None,
        do_sample: bool = None
    ) -> str:
        prompt = f"""
You are a highly observant assistant tasked with summarizing long, unscripted video monologues.

TEXT:
{text}

INSTRUCTIONS:
Summarize the speaker’s core points and tone as if describing the monologue to someone who hasn’t heard it.
Group related ideas together. Highlight interesting or unusual claims. Use descriptive language.
Output a full narrative paragraph (or two), not bullet points.
"""
        max_chunk = zero_or_default(max_chunk,default=512)
        min_length = zero_or_default(min_length,default=100)
        max_length = zero_or_default(max_length,default=512)
        do_sample = do_sample or False
        return self.summarizer(prompt,
                          max_length=max_length,
                          min_length=min_length,
                          do_sample=do_sample)[0]['generated_text']
def get_flan_manager():
    return flanManager().summarizer
def get_flan_summary(
    text: str,
    max_chunk: int = None,
    min_length: int = None,
    max_length: int = None,
    do_sample: bool = None
    ):
    return get_flan_manager().get_flan_summary(
        text=text,
        max_chunk=max_chunk,
        min_length=min_length,
        max_length=max_length,
        do_sample=do_sample
        )
