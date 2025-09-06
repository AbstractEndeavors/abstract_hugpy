from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .config import DEFAULT_PATHS

MODEL_NAME = DEFAULT_PATHS.get("flan", "google/flan-t5-xl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

def get_flan_summary(
    text: str,
    max_chunk: int = 512,
    min_length: int = 100,
    max_length: int = 512,
    do_sample: bool = False
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
    return summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=do_sample)[0]['generated_text']