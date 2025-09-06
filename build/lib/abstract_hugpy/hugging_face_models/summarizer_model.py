import os
import unicodedata
import re
from dataclasses import asdict
from typing import List, Literal, Optional, Tuple
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast, GenerationConfig
from .config import DEFAULT_PATHS
from abstract_utilities import safe_read_from_json
from abstract_ai.gpt_classes.prompt_selection.PromptBuilder import recursive_chunk

DEFAULT_DIR = DEFAULT_PATHS["summarizer_t5"]
MODEL_NAME = "gpt-4"
CHUNK_OVERLAP = 30
DEFAULT_CHUNK_TOK = 450
SHORTCUT_THRESHOLD = 200

tokenizer = T5TokenizerFast.from_pretrained(DEFAULT_DIR)

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text
        

def clean_output_text(text: str) -> str:
    text = re.sub(r'["]{2,}', '"', text)
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r"[^\w\s\.,;:?!\-'\"()]+", "", text)
    return text.strip()

def chunk_text(text: str, max_tokens: int) -> List[str]:
    return recursive_chunk(
        text=text,
        desired_tokens=max_tokens,
        model_name=MODEL_NAME,
        separators=["\n\n","\n", r"(?<=[\.?\!])\s", ", ", " "],
        overlap=CHUNK_OVERLAP
    )

def scale_lengths(mode: str, tokens: int) -> Tuple[int, int]:
    m = mode.lower()
    if m == "short":
        return max(16, int(tokens*0.1)), max(40, int(tokens*0.25))
    if m == "medium":
        return max(32, int(tokens*0.25)), max(80, int(tokens*0.5))
    if m == "long":
        return max(64, int(tokens*0.35)), max(150, int(tokens*0.7))
    return max(32, int(tokens*0.2)), max(120, int(tokens*0.6))

def load_gen_config() -> GenerationConfig:
    config = safe_read_from_json(os.path.join(DEFAULT_DIR, "generation_config.json"))
    return GenerationConfig(**config)

def run_t5_inference(text: str, min_length: int, max_length: int) -> str:
    inputs = tokenizer("summarize: " + normalize_text(text), return_tensors="pt", truncation=True, max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(DEFAULT_DIR)
    gen_cfg = load_gen_config()
    gen_cfg.min_length = min_length
    gen_cfg.max_length = max_length
    with torch.no_grad():
        out = model.generate(inputs.input_ids, **asdict(gen_cfg))
    return tokenizer.decode(out[0], skip_special_tokens=True)

def summarize(
    text: str,
    summary_mode: Literal["short","medium","long","auto"] = "medium",
    max_chunk_tokens: int = DEFAULT_CHUNK_TOK,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> str:
    txt = normalize_text(text)
    tok_cnt = len(tokenizer.tokenize(txt))
    if tok_cnt <= SHORTCUT_THRESHOLD:
        mn = min_length or int(tok_cnt * 0.25)
        mx = max_length or int(tok_cnt * 0.6)
        return clean_output_text(run_t5_inference(txt, mn, mx))
    chunks = chunk_text(txt, max_chunk_tokens)
    parts = []
    for chunk in chunks:
        cnt = len(tokenizer.tokenize(chunk))
        mn, mx = (min_length, max_length) if min_length and max_length else scale_lengths(summary_mode, cnt)
        parts.append(clean_output_text(run_t5_inference(chunk, mn, mx)))
    merged = " ".join(parts)
    cnt2 = len(tokenizer.tokenize(merged))
    mn2, mx2 = (min_length, max_length) if min_length and max_length else scale_lengths(summary_mode, cnt2)
    return clean_output_text(run_t5_inference(merged, mn2, mx2))
