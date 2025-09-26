import os
import re
from urllib.parse import quote
from collections import Counter
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LEDTokenizer,
    LEDForConditionalGeneration
)
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import cos_sim
from transformers import BitsAndBytesConfig
from .config import DEFAULT_PATHS
from typing import *
SUMMARIZER_DIR = DEFAULT_PATHS["summarizer_t5"]
FLAN_MODEL_NAME = DEFAULT_PATHS["flan"]
# ------------------------------------------------------------------------------
# 3. LONGFORM SUMMARIZATION (T5 / FLAN / LED)
# ------------------------------------------------------------------------------

# 3.1. FLAN-BASED (google/flan-t5-xl) SUMMARIZER
class flanManager:
    def __init__(self,model_name:str=None):
        self.model_name = model or FLAN_MODEL_NAME
        self.flan_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.flan_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.flan_device = 0 if torch.cuda.is_available() else -1
        self.flan_summarizer = pipeline(
            "text2text-generation",
            model=self.flan_model,
            tokenizer=self.flan_tokenizer,
            device=self.flan_device
        )

def get_flan_sumarizer(model_name:str=None):
    flan_mgr = flanManager(model_name=model_name)
    return flan_mgr.flan_summarizer
def get_flan_summary(
    text: str,
    max_chunk: int = 512,
    max_length: int | None = None,
    min_length: int | None = None,
    do_sample: bool = False,
    model_name: str=None
) -> str:
    """
    Use google/flan-t5-xl to generate a human-like summary of an input text.

    Args:
        text (str): Input text to summarize.
        max_chunk (int): Maximum tokens for each chunk. Defaults to 512.
        max_length (int | None): Max tokens for generated summary. Defaults to 512.
        min_length (int | None): Min tokens for generated summary. Defaults to 100.
        do_sample (bool): Whether to sample. Defaults to False.

    Returns:
        str: The generated summary.
    """
    max_length = max_length or 512
    min_length = min_length or 100
    prompt = (
        f"Summarize the following text in a coherent, concise paragraph:\n\n{text}"
    )
    flan_sumarizer = get_flan_sumarizer(model_name=model_name)
    output = flan_summarizer(
        prompt,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample
    )[0]["generated_text"]
    return output.strip()



class t5Manager:
    def __init__(self,model_directory:str=None):
        self.model_directory = model_directory or SUMMARIZER_DIR
        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_directory)
def get_t5_manager(model_directory:str=None):
    t5_mgr = t5Manager(model_directory=model_directory)
    return t5_mgr
def get_t5__tokenizer(model_directory:str=None):
    t5_mgr = get_t5_manager(model_directory=model_directory)
    return t5_mgr.t5_tokenizer
def get_t5_model(model_directory:str=None):
    t5_mgr = get_t5_manager(model_directory=model_directory)
    return t5_mgr.t5_model
# 3.2. CHUNK-BASED T5 SUMMARIZER (for arbitrarily long text)



def split_to_chunk(full_text: str, max_words: int = 300) -> list[str]:
    """
    Break a long text into smaller chunks (approximately max_words per chunk),
    splitting on sentence-like boundaries (". ").

    Args:
        full_text (str): The entire document text.
        max_words (int): Maximum words allowed in each chunk. Defaults to 300.

    Returns:
        list[str]: List of text chunks.
    """
    sentences = full_text.split(". ")
    chunks = []
    buffer = ""
    for sent in sentences:
        candidate = (buffer + sent).strip()
        if len(candidate.split()) <= max_words:
            buffer = candidate + ". "
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = sent + ". "
    if buffer:
        chunks.append(buffer.strip())
    return chunks


def chunk_summaries(
    chunks: list[str],
    max_length: int = 160,
    min_length: int = 40,
    truncation: bool = False,
    model_directory:str=None
) -> list[str]:
    """
    Summarize each chunk individually using a T5-based summarizer,
    then return a list of summary strings.

    Args:
        chunks (list[str]): List of text chunks.
        max_length (int): Max output tokens per chunk summary.
        min_length (int): Min output tokens per chunk summary.
        truncation (bool): If True, enforce truncation of inputs that exceed model length.

    Returns:
        list[str]: Summaries for each chunk.
    """
    summaries = []
    t5_tokenizer = get_t5__tokenizer(model_directory=model_directory)
    t5_model = get_t5__tokenizer(model_directory=model_directory)
    for chunk in chunks:
        inputs = t5_tokenizer(
            chunk,
            return_tensors="pt",
            truncation=truncation,
            max_length=512
        )
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
        summary_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary_text.strip())
    return summaries


def get_summary(
    full_text: str,
    max_words: int = 300,
    max_length: int = 160,
    min_length: int = 40,
    truncation: bool = False
) -> str:
    """
    Create a summary of arbitrarily long text by:
      1) Splitting into ~300-word chunks
      2) Summarizing each chunk individually (T5)
      3) Stitching chunk summaries back together into one paragraph

    Args:
        full_text (str): The entire document text.
        max_words (int): Maximum words per chunk. Defaults to 300.
        max_length (int): Max tokens per chunk summary. Defaults to 160.
        min_length (int): Min tokens per chunk summary. Defaults to 40.
        truncation (bool): If True, force-truncate over-length inputs.

    Returns:
        str: Full stitched summary.
    """
    if not full_text:
        return ""

    chunks = split_to_chunk(full_text, max_words=max_words)
    summaries = chunk_summaries(
        chunks,
        max_length=max_length,
        min_length=min_length,
        truncation=truncation
    )
    return " ".join(summaries).strip()

class falconsManager:
    def __init__(self):
        self.falcons_summarizer = pipeline("summarization",model="Falconsai/text_summarization",device=0 if torch.cuda.is_available() else -1)
def get_falcons_sumarizer():
    falcons_mgr = falconsManager()
    return falcons_mgr.falcons_summarizer

def chunk_falcons_summaries(chunks, max_length=160, min_length=40, truncation=True):
    """
    Summarize each chunk using Falconsai/text_summarization.
    Returns a list of summary strings.
    """
    summaries = []
    falcons_sumarizer = get_falcons_sumarizer()
    for chunk in chunks:
        # The pipeline returns a list of dicts; [0]["summary_text"] is our text.
        out = falcons_summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            truncation=truncation
        )
        summaries.append(out[0]["summary_text"].strip())
    return summaries
