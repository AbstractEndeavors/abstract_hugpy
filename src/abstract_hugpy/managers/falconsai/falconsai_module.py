"""
Unified summarization registry.

Every back-end exposes the same contract via SummarizerBackend.
Model dirs resolve through ensure_model(key) — the same single source
of truth the embed runner uses. No MODELS_ROOT, no entry.folder, no
module-level path resolution.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

from .imports import *


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, SummaryPreset] = {}


def register_preset(key: str, preset: SummaryPreset) -> None:
    if key in _PRESETS:
        raise KeyError(f"Preset {key!r} already registered")
    _PRESETS[key] = preset


def available_presets() -> List[str]:
    return sorted(_PRESETS)


def get_preset(key: str) -> SummaryPreset:
    if key not in _PRESETS:
        raise KeyError(f"Unknown preset {key!r}. Available: {available_presets()}")
    return _PRESETS[key]


register_preset("default", SummaryPreset())
register_preset("article", SummaryPreset(
    max_chunk_tokens=500, min_length=120, max_length=600, summary_mode="long",
    consolidation_min_length=120, consolidation_max_length=300, max_output_words=350,
))
register_preset("brief", SummaryPreset(
    max_chunk_tokens=350, min_length=30, max_length=200, summary_mode="short",
    consolidation_min_length=40, consolidation_max_length=100, max_output_words=80,
))
register_preset("headline", SummaryPreset(
    max_chunk_tokens=300, min_length=8, max_length=60, summary_mode="short",
    consolidation_min_length=8, consolidation_max_length=40, max_output_words=25,
))


@runtime_checkable
class SummarizerBackend(Protocol):
    def summarize(self, req: SummaryRequest) -> str: ...


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, type] = {}


def register_backend(key: str):
    def decorator(cls):
        if key in _BACKENDS:
            raise KeyError(f"Summarizer back-end {key!r} already registered")
        _BACKENDS[key] = cls
        return cls
    return decorator


def available_backends() -> List[str]:
    return sorted(_BACKENDS)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def clean_output(text: str) -> str:
    text = re.sub(r'["]{2,}', '"', text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"[^\w\s\.,;:?!\-'\"()]+", "", text)
    return text.strip()


def split_sentences(full_text: str, max_words: int = 300) -> List[str]:
    sentences = full_text.split(". ")
    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        candidate = (buf + sent).strip()
        if len(candidate.split()) <= max_words:
            buf = candidate + ". "
        else:
            if buf:
                chunks.append(buf.strip())
            buf = sent + ". "
    if buf:
        chunks.append(buf.strip())
    return chunks


def scale_lengths(mode: str, token_count: int) -> Tuple[int, int]:
    m = (mode or "auto").lower()
    if m == "short":
        return max(16, int(token_count * 0.1)), max(40, int(token_count * 0.25))
    if m == "medium":
        return max(32, int(token_count * 0.25)), max(80, int(token_count * 0.5))
    if m == "long":
        return max(64, int(token_count * 0.35)), max(150, int(token_count * 0.7))
    return max(32, int(token_count * 0.2)), max(120, int(token_count * 0.6))


MODEL_NAME_CHUNK = "gpt-4"
CHUNK_OVERLAP = 30


# ---------------------------------------------------------------------------
# Backend: Flan-T5 (key: flan-t5-xl)
# ---------------------------------------------------------------------------

@register_backend("flan")
class FlanBackend(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, "_ready"):
            model_dir = ensure_model("flan-t5-xl")   # was DEFAULT_PATHS["flan"] -> KeyError
            self._tokenizer = get_transformers("AutoTokenizer").from_pretrained(model_dir)
            self._model = get_transformers("AutoModelForSeq2SeqLM").from_pretrained(model_dir)
            device = 0 if get_torch().cuda.is_available() else -1
            self._pipeline = get_transformers("pipeline")(
                "text2text-generation",        # was "text-generation" — wrong head for T5
                model=self._model, tokenizer=self._tokenizer, device=device,
            )
            self._ready = True

    def summarize(self, req: SummaryRequest) -> str:
        prompt = "Summarize the following text in a coherent, concise paragraph:\n\n" + req.text
        out = self._pipeline(
            prompt, max_length=req.max_length, min_length=req.min_length,
            do_sample=req.do_sample,
        )
        return out[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Backend: seq2seq chunked (any seq2seq summarizer by model_key)
# ---------------------------------------------------------------------------

@register_backend("seq2seq_chunked")
class Seq2SeqChunkedBackend(metaclass=SingletonMeta):
    """Chunk → summarize → consolidate."""
    def __init__(self, model_key: str):
        if not hasattr(self, "_ready"):
            model_dir = ensure_model(model_key)   # was os.path.join(MODELS_ROOT, entry.folder)
            self._tokenizer = get_transformers("AutoTokenizer").from_pretrained(model_dir)
            self._model = get_transformers("AutoModelForSeq2SeqLM").from_pretrained(model_dir)
            self._model_key = model_key
            self._ready = True

    def _infer(self, text: str, min_len: int, max_len: int) -> str:
        torch = get_torch()
        inputs = self._tokenizer(
            "summarize: " + normalize_text(text),
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            ids = self._model.generate(
                inputs.input_ids,
                min_length=int(min_len), max_length=int(max_len),
                num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
            )
        return self._tokenizer.decode(ids[0], skip_special_tokens=True)

    def summarize(self, req: SummaryRequest) -> str:
        txt = normalize_text(req.text)
        chunks = recursive_chunk(
            text=txt, desired_tokens=req.max_chunk_tokens, model_name=MODEL_NAME_CHUNK,
            separators=["\n\n", "\n", r"(?<=[\.?\!])\s", ", ", " "], overlap=CHUNK_OVERLAP,
        )
        summaries: List[str] = []
        for chunk in chunks:
            cnt = len(self._tokenizer.tokenize(chunk))
            mn, mx = scale_lengths(req.summary_mode, cnt)
            summaries.append(clean_output(self._infer(chunk, mn, mx)))
        merged = " ".join(summaries)

        try:
            merged_chunks = recursive_chunk(
                text=merged, desired_tokens=300, model_name=MODEL_NAME_CHUNK, overlap=20,
            )
            final_parts = [
                clean_output(self._infer(c, req.consolidation_min_length, req.consolidation_max_length))
                for c in merged_chunks
            ]
            consolidated = " ".join(final_parts)
        except Exception:
            consolidated = merged

        words = consolidated.split()
        if len(words) > req.max_output_words:
            consolidated = " ".join(words[: req.max_output_words]) + "..."
        return consolidated


# ---------------------------------------------------------------------------
# Backend: pipeline chunked (Falconsai-style, no consolidation)
# ---------------------------------------------------------------------------

@register_backend("pipeline_chunked")
class PipelineChunkedBackend(metaclass=SingletonMeta):
    """Sentence-split → HF summarization pipeline → join."""
    def __init__(self, model_key: str):
        if not hasattr(self, "_ready"):
            model_dir = ensure_model(model_key)   # was os.path.join(MODELS_ROOT, entry.folder)
            device = 0 if get_torch().cuda.is_available() else -1
            self._pipeline = get_transformers("pipeline")(
                "summarization", model=model_dir, device=device,
            )
            self._ready = True

    def summarize(self, req: SummaryRequest) -> str:
        if not req.text:
            return ""
        chunks = split_sentences(req.text, max_words=300)
        parts: List[str] = []
        for chunk in chunks:
            out = self._pipeline(
                chunk, max_length=req.max_length, min_length=req.min_length, truncation=True,
            )
            parts.append(out[0]["summary_text"].strip())
        return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_backend(key: str) -> SummarizerBackend:
    if key not in _BACKENDS:
        raise KeyError(f"Unknown summarizer {key!r}. Available: {available_backends()}")
    return _BACKENDS[key]()


def summarize(
    text: str = None,
    backend: str = "seq2seq_chunked",
    *,
    request: Optional[SummaryRequest] = None,
    preset: Optional[str] = None,
    max_chunk_tokens: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    summary_mode: Optional[Literal["short", "medium", "long", "auto"]] = None,
    input_policy: Optional[InputPolicy] = None,
    min_input_words: Optional[int] = None,
    consolidation_min_length: Optional[int] = None,
    consolidation_max_length: Optional[int] = None,
    max_output_words: Optional[int] = None,
) -> str:
    if request is not None:
        if text is not None:
            raise ValueError("Cannot pass both `text` and `request`")
        return get_backend(backend).summarize(request)

    if text is None:
        raise ValueError("Must pass either `text` or `request`")

    p = get_preset(preset) if preset else SummaryPreset()

    def _resolve(explicit, from_preset, schema_default):
        if explicit is not None:
            return explicit
        if from_preset is not None:
            return from_preset
        return schema_default

    _d = {f.name: f.default for f in SummaryRequest.__dataclass_fields__.values()}

    req = SummaryRequest(
        text=text,
        max_chunk_tokens=_resolve(max_chunk_tokens, p.max_chunk_tokens, _d["max_chunk_tokens"]),
        min_length=_resolve(min_length, p.min_length, _d["min_length"]),
        max_length=_resolve(max_length, p.max_length, _d["max_length"]),
        do_sample=_resolve(do_sample, p.do_sample, _d["do_sample"]),
        summary_mode=_resolve(summary_mode, p.summary_mode, _d["summary_mode"]),
        input_policy=_resolve(input_policy, p.input_policy, _d["input_policy"]),
        min_input_words=_resolve(min_input_words, p.min_input_words, _d["min_input_words"]),
        consolidation_min_length=_resolve(consolidation_min_length, p.consolidation_min_length, _d["consolidation_min_length"]),
        consolidation_max_length=_resolve(consolidation_max_length, p.consolidation_max_length, _d["consolidation_max_length"]),
        max_output_words=_resolve(max_output_words, p.max_output_words, _d["max_output_words"]),
    )
    return get_backend(backend).summarize(req)
