"""
Unified summarization registry.

Every back-end exposes the same contract via SummarizerBackend.
Callers pick a strategy + model, call `summarize(text, backend=..., model_key=...)`.

Model dirs resolve through ensure_model(model_key) — the same single source
of truth the embed runner uses. No MODELS_ROOT, no entry.folder, no
module-level path resolution.

Back-ends are strategy classes; the heavy model/tokenizer/pipeline is cached
per model_key in a class-level dict (same shape as FeatureExtractionRunner's
_MODELS), so many calls for one model share one load and one strategy can
serve several models.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

from .imports import *


# ---------------------------------------------------------------------------
# JSON convenience entry point
# ---------------------------------------------------------------------------

def summarize_from_json(json_str: str, backend: str = "seq2seq_chunked", **kwargs) -> str:
    """Deserialize JSON and call summarize.

    Expected JSON:
        {"text": "...", "max_length": 300, "input_policy": "warn", "preset": "article", ...}
    """
    data = json.loads(json_str)
    if "input_policy" in data and isinstance(data["input_policy"], str):
        data["input_policy"] = InputPolicy(data["input_policy"])
    text = data.pop("text")
    return summarize(text, backend=backend, **data, **kwargs)


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


MODEL_NAME_CHUNK = "gpt-4"   # used only by recursive_chunk's token counter
CHUNK_OVERLAP = 30


# ---------------------------------------------------------------------------
# Backend: Flan-T5 (text2text-generation pipeline, no chunking)
# ---------------------------------------------------------------------------

@register_backend("flan")
class FlanBackend:
    """One-shot Flan-T5 summary: prompt -> coherent paragraph."""

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, model_key: str = "flan-t5-xl"):
        self.model_key = model_key

    @property
    def _pipeline(self):
        cached = self._PIPELINES.get(self.model_key)
        if cached is not None:
            return cached
        with self._LOCK:
            cached = self._PIPELINES.get(self.model_key)
            if cached is not None:
                return cached
            model_dir = ensure_model(self.model_key)   # was DEFAULT_PATHS["flan"] -> KeyError
            tokenizer = get_transformers("AutoTokenizer").from_pretrained(model_dir)
            model = get_transformers("AutoModelForSeq2SeqLM").from_pretrained(model_dir)
            device = 0 if get_torch().cuda.is_available() else -1
            pipe = get_transformers("pipeline")(
                "text2text-generation",        # was "text-generation" — wrong head for T5
                model=model, tokenizer=tokenizer, device=device,
            )
            self._PIPELINES[self.model_key] = pipe
            return pipe

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
class Seq2SeqChunkedBackend:
    """Chunk → summarize → consolidate."""

    _MODELS: Dict[str, Tuple[Any, Any]] = {}
    _LOCK = threading.Lock()

    def __init__(self, model_key: str = "flan-t5-xl"):
        self.model_key = model_key

    def _load(self) -> Tuple[Any, Any]:
        cached = self._MODELS.get(self.model_key)
        if cached is not None:
            return cached
        with self._LOCK:
            cached = self._MODELS.get(self.model_key)
            if cached is not None:
                return cached
            model_dir = ensure_model(self.model_key)   # was os.path.join(MODELS_ROOT, entry.folder)
            tokenizer = get_transformers("AutoTokenizer").from_pretrained(model_dir)
            model = get_transformers("AutoModelForSeq2SeqLM").from_pretrained(model_dir)
            self._MODELS[self.model_key] = (tokenizer, model)
            return tokenizer, model

    @property
    def _tokenizer(self):
        return self._load()[0]

    @property
    def _model(self):
        return self._load()[1]

    def _infer(self, text: str, min_len: int, max_len: int) -> str:
        torch = get_torch()
        tokenizer, model = self._load()
        inputs = tokenizer(
            "summarize: " + normalize_text(text),
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            ids = model.generate(
                inputs.input_ids,
                min_length=int(min_len), max_length=int(max_len),
                num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True)

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
class PipelineChunkedBackend:
    """Sentence-split → HF summarization pipeline → join."""

    _PIPELINES: Dict[str, Any] = {}
    _LOCK = threading.Lock()

    def __init__(self, model_key: str = "text_summarization"):
        self.model_key = model_key

    @property
    def _pipeline(self):
        cached = self._PIPELINES.get(self.model_key)
        if cached is not None:
            return cached
        with self._LOCK:
            cached = self._PIPELINES.get(self.model_key)
            if cached is not None:
                return cached
            model_dir = ensure_model(self.model_key)   # was os.path.join(MODELS_ROOT, entry.folder)
            device = 0 if get_torch().cuda.is_available() else -1
            pipe = get_transformers("pipeline")(
                "summarization", model=model_dir, device=device,
            )
            self._PIPELINES[self.model_key] = pipe
            return pipe

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

def get_backend(key: str, model_key: Optional[str] = None) -> SummarizerBackend:
    """Return a backend instance for *key*, loading weights for *model_key*.

    Every registered backend takes an optional model_key in __init__ and
    defaults to a sensible model when none is given, so direct callers can
    do `summarize(text, backend="flan")` and the runner can pass an explicit
    model_key off the resolved cfg.
    """
    if key not in _BACKENDS:
        raise KeyError(
            f"Unknown summarizer {key!r}. Available: {available_backends()}"
        )
    cls = _BACKENDS[key]
    return cls(model_key) if model_key is not None else cls()


def _dispatch(backend: str, model_key: Optional[str], req: SummaryRequest) -> str:
    """Apply input_policy, then run the backend."""
    be = get_backend(backend, model_key)

    problem = req.check_input()
    if problem is not None:
        if req.input_policy is InputPolicy.STRICT:
            raise ValueError(problem)
        if req.input_policy is InputPolicy.WARN:
            return f"[WARNING: {problem}] {be.summarize(req)}"
        # InputPolicy.ALLOW — run it, no questions asked.

    return be.summarize(req)


def summarize(
    text: str = None,
    backend: str = "seq2seq_chunked",
    *,
    model_key: Optional[str] = None,
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
    """One call, any back-end, optional preset.

    Two entry points:
        1. Traditional: summarize(text, backend="seq2seq_chunked", preset="article", ...)
        2. From request: summarize(request=SummaryRequest(...))

    Resolution order (highest wins):
        1. Explicit kwarg passed by the caller
        2. Preset value (if a preset is named)
        3. SummaryRequest field default
    """
    if isinstance(input_policy, str):
        input_policy = InputPolicy(input_policy)

    # -- entry point 1: SummaryRequest directly ----------------------------
    if request is not None:
        if text is not None:
            raise ValueError("Cannot pass both `text` and `request`")
        return _dispatch(backend, model_key, request)

    # -- entry point 2: traditional kwargs ---------------------------------
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
        consolidation_min_length=_resolve(
            consolidation_min_length, p.consolidation_min_length, _d["consolidation_min_length"]
        ),
        consolidation_max_length=_resolve(
            consolidation_max_length, p.consolidation_max_length, _d["consolidation_max_length"]
        ),
        max_output_words=_resolve(max_output_words, p.max_output_words, _d["max_output_words"]),
    )
    return _dispatch(backend, model_key, req)


# Back-compat alias — older callers imported summarize_t5.
summarize_t5 = summarize
