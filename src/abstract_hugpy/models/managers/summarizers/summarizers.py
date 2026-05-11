"""
Unified summarization registry.

Every back-end exposes the same contract via SummarizerBackend.
Callers pick a key, call `summarize(key, text, **kw)` — done.

Back-ends are lazy-loaded singletons: the model only touches GPU/RAM
on the first call, and only for the back-end you actually request.
"""

from __future__ import annotations

import os
import re,json
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

from .imports import (
    DEFAULT_PATHS,
    SingletonMeta,
    get_torch,
    get_transformers,
    recursive_chunk,
    safe_read_from_json,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class InputPolicy(Enum):
    """
    Controls what happens when input text is too short to summarize
    meaningfully.

    STRICT  — raise ValueError.  The safe default.
    WARN    — return a prefixed warning + whatever the model produces.
    ALLOW   — pass it straight through.  You asked for fiction, you get fiction.
    """

    STRICT = "strict"
    WARN = "warn"
    ALLOW = "allow"


MIN_INPUT_WORDS_DEFAULT = 10
def summarize_from_json(json_str: str, backend: str = "t5") -> str:
    """
    Deserialize JSON and call summarize.
    
    Expected JSON:
    {
        "text": "...",
        "max_length": 300,
        "input_policy": "warn",
        "preset": "article",
        ...
    }
    """
    import json
    
    data = json.loads(json_str)
    
    # Convert string policy to enum
    if "input_policy" in data and isinstance(data["input_policy"], str):
        data["input_policy"] = InputPolicy(data["input_policy"])
    
    text = data.pop("text")
    
    return summarize(text, backend=backend, **data)

@dataclass(frozen=True)
class SummaryRequest:
    """Immutable bag of parameters every back-end understands."""

    text: str
    max_chunk_tokens: int = 450
    min_length: int = 100
    max_length: int = 512
    do_sample: bool = False
    summary_mode: Literal["short", "medium", "long", "auto"] = "medium"
    input_policy: InputPolicy = InputPolicy.STRICT
    min_input_words: int = MIN_INPUT_WORDS_DEFAULT

    # Consolidation pass (T5Backend merges chunk summaries into one).
    # These were hardcoded as (80, 160) — now the caller/preset decides.
    consolidation_min_length: int = 80
    consolidation_max_length: int = 160
    max_output_words: int = 150

    def check_input(self) -> Optional[str]:
        """
        Return a human-readable problem string if the input is suspect,
        or None if everything looks fine.
        """
        word_count = len(self.text.split())
        if word_count < self.min_input_words:
            return (
                f"Input has {word_count} word(s); need at least "
                f"{self.min_input_words} for a meaningful summary."
            )
        return None


# ---------------------------------------------------------------------------
# Presets — named parameter bundles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SummaryPreset:
    """
    A frozen bag of defaults that a preset name resolves to.
    Every field here maps 1:1 to a SummaryRequest field.
    Only non-None values override the caller's explicit kwargs.
    """

    max_chunk_tokens: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    do_sample: Optional[bool] = None
    summary_mode: Optional[Literal["short", "medium", "long", "auto"]] = None
    input_policy: Optional[InputPolicy] = None
    min_input_words: Optional[int] = None
    consolidation_min_length: Optional[int] = None
    consolidation_max_length: Optional[int] = None
    max_output_words: Optional[int] = None


_PRESETS: Dict[str, SummaryPreset] = {}


def register_preset(key: str, preset: SummaryPreset) -> None:
    if key in _PRESETS:
        raise KeyError(f"Preset {key!r} already registered")
    _PRESETS[key] = preset


def available_presets() -> List[str]:
    return sorted(_PRESETS)


def get_preset(key: str) -> SummaryPreset:
    if key not in _PRESETS:
        raise KeyError(
            f"Unknown preset {key!r}. Available: {available_presets()}"
        )
    return _PRESETS[key]


# ---- built-in presets ----------------------------------------------------

register_preset("default", SummaryPreset())  # everything at SummaryRequest defaults

register_preset(
    "article",
    SummaryPreset(
        max_chunk_tokens=500,       # wider context window per chunk
        min_length=120,             # don't let per-chunk output collapse
        max_length=600,             # room for dense source material
        summary_mode="long",        # scale_lengths uses generous ratios
        consolidation_min_length=120,  # final merge pass gets breathing room
        consolidation_max_length=300,  # ~2 solid paragraphs out
        max_output_words=350,       # don't guillotine a 3000-word article summary at 150 words
    ),
)

register_preset(
    "brief",
    SummaryPreset(
        max_chunk_tokens=350,
        min_length=30,
        max_length=200,
        summary_mode="short",
        consolidation_min_length=40,
        consolidation_max_length=100,
        max_output_words=80,
    ),
)

register_preset(
    "headline",
    SummaryPreset(
        max_chunk_tokens=300,
        min_length=8,
        max_length=60,
        summary_mode="short",
        consolidation_min_length=8,
        consolidation_max_length=40,
        max_output_words=25,
    ),
)


@runtime_checkable
class SummarizerBackend(Protocol):
    """What every back-end must look like."""

    def summarize(self, req: SummaryRequest) -> str: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, type] = {}


def register_backend(key: str):
    """Class decorator — registers a back-end under *key*."""

    def decorator(cls):
        if key in _BACKENDS:
            raise KeyError(f"Summarizer back-end {key!r} already registered")
        _BACKENDS[key] = cls
        return cls

    return decorator


def available_backends() -> List[str]:
    return sorted(_BACKENDS)


# ---------------------------------------------------------------------------
# Text utilities (shared across back-ends)
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
    """Split on '. ' boundaries, keeping chunks ≤ max_words."""
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
    m = mode.lower()
    if m == "short":
        return max(16, int(token_count * 0.1)), max(40, int(token_count * 0.25))
    if m == "medium":
        return max(32, int(token_count * 0.25)), max(80, int(token_count * 0.5))
    if m == "long":
        return max(64, int(token_count * 0.35)), max(150, int(token_count * 0.7))
    return max(32, int(token_count * 0.2)), max(120, int(token_count * 0.6))


# ---------------------------------------------------------------------------
# Back-end: Flan-T5  (google/flan-t5-xl or override)
# ---------------------------------------------------------------------------

MODEL_NAME_CHUNK = "gpt-4"   # used only by recursive_chunk's token counter
CHUNK_OVERLAP = 30


@register_backend("flan")
class FlanBackend(metaclass=SingletonMeta):

    def __init__(self):
        if not hasattr(self, "_ready"):
            name = DEFAULT_PATHS["flan"]
            tf = get_transformers()
            self._tokenizer = get_transformers("AutoTokenizer").from_pretrained(name)
            self._model = get_transformers("AutoModelForSeq2SeqLM").from_pretrained(name)
            device = 0 if get_torch().cuda.is_available() else -1
            self._pipeline = get_transformers("pipeline")(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
            )
            self._ready = True

    # -- contract ----------------------------------------------------------

    def summarize(self, req: SummaryRequest) -> str:
        prompt = (
            "Summarize the following text in a coherent, concise paragraph:\n\n"
            + req.text
        )
        out = self._pipeline(
            prompt,
            max_length=req.max_length,
            min_length=req.min_length,
            do_sample=req.do_sample,
        )
        return out[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Back-end: Local T5 (chunked → merged → consolidated)
# ---------------------------------------------------------------------------

@register_backend("t5")
class T5Backend(metaclass=SingletonMeta):

    def __init__(self):
        if not hasattr(self, "_ready"):
            model_dir = DEFAULT_PATHS["summarizer"]
            tf = get_transformers()
            self._tokenizer = get_transformers("T5TokenizerFast").from_pretrained(model_dir)
            self._model = get_transformers("T5ForConditionalGeneration").from_pretrained(model_dir)
            self._gen_config_raw = safe_read_from_json(
                os.path.join(model_dir, "generation_config.json")
            ) or {}
            self._ready = True

    # -- internals ---------------------------------------------------------

    def _infer(self, text: str, min_len: int, max_len: int) -> str:
        torch = get_torch()
        inputs = self._tokenizer(
            "summarize: " + normalize_text(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            ids = self._model.generate(
                inputs.input_ids,
                min_length=int(min_len),
                max_length=int(max_len),
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.decode(ids[0], skip_special_tokens=True)

    # -- contract ----------------------------------------------------------

    def summarize(self, req: SummaryRequest) -> str:
        txt = normalize_text(req.text)

        chunks = recursive_chunk(
            text=txt,
            desired_tokens=req.max_chunk_tokens,
            model_name=MODEL_NAME_CHUNK,
            separators=["\n\n", "\n", r"(?<=[\.?\!])\s", ", ", " "],
            overlap=CHUNK_OVERLAP,
        )

        summaries: List[str] = []
        for chunk in chunks:
            cnt = len(self._tokenizer.tokenize(chunk))
            mn, mx = scale_lengths(req.summary_mode, cnt)
            summaries.append(clean_output(self._infer(chunk, mn, mx)))

        merged = " ".join(summaries)

        try:
            merged_chunks = recursive_chunk(
                text=merged,
                desired_tokens=300,
                model_name=MODEL_NAME_CHUNK,
                overlap=20,
            )

            final_parts = []

            for chunk in merged_chunks:
                final_parts.append(
                    clean_output(
                        self._infer(
                            chunk,
                            req.consolidation_min_length,
                            req.consolidation_max_length,
                        )
                    )
                )

            consolidated = " ".join(final_parts)
        except Exception:
            consolidated = merged

        words = consolidated.split()
        if len(words) > req.max_output_words:
            consolidated = " ".join(words[: req.max_output_words]) + "..."
        return consolidated


# ---------------------------------------------------------------------------
# Back-end: Falconsai  (Falconsai/text_summarization)
# ---------------------------------------------------------------------------

@register_backend("falconsai")
class FalconsaiBackend(metaclass=SingletonMeta):

    def __init__(self):
        if not hasattr(self, "_ready"):
            tf = get_transformers()
            device = 0 if get_torch().cuda.is_available() else -1
            self._pipeline = get_transformers("pipeline")(
                "summarization",
                model="Falconsai/text_summarization",
                device=device,
            )
            self._ready = True

    # -- contract ----------------------------------------------------------

    def summarize(self, req: SummaryRequest) -> str:
        if not req.text:
            return ""
        chunks = split_sentences(req.text, max_words=300)
        parts: List[str] = []
        for chunk in chunks:
            out = self._pipeline(
                chunk,
                max_length=req.max_length,
                min_length=req.min_length,
                truncation=True,
            )
            parts.append(out[0]["summary_text"].strip())
        return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_backend(key: str) -> SummarizerBackend:
    """Return the singleton instance for *key*. Raises KeyError if unknown."""
    if key not in _BACKENDS:
        raise KeyError(
            f"Unknown summarizer {key!r}. "
            f"Available: {available_backends()}"
        )
    return _BACKENDS[key]()


def summarize(
    text: str = None,
    backend: str = "t5",
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
    """
    One call, any back-end, optional preset.
    
    Two entry points:
        1. Traditional: summarize(text, backend="t5", preset="article", ...)
        2. From request: summarize(request=SummaryRequest(...))
    
    Resolution order (highest wins):
        1. Explicit kwarg passed by the caller
        2. Preset value (if a preset is named)
        3. SummaryRequest field default
    """
    
    # -- entry point 1: SummaryRequest directly ---------------------------
    if request is not None:
        if text is not None:
            raise ValueError("Cannot pass both `text` and `request`")
        
        problem = request.check_input()
        if problem is not None:
            if request.input_policy is InputPolicy.STRICT:
                print(problem)
            if request.input_policy is InputPolicy.WARN:
                raw = get_backend(backend).summarize(request)
                return f"[WARNING: {problem}] {raw}"
        
        return get_backend(backend).summarize(request)
    
    # -- entry point 2: traditional kwargs --------------------------------
    if text is None:
        raise ValueError("Must pass either `text` or `request`")
    
    # -- layer 1: preset defaults (all None if no preset) ------------------
    p = get_preset(preset) if preset else SummaryPreset()

    def _resolve(explicit, from_preset, schema_default):
        """First non-None wins."""
        if explicit is not None:
            return explicit
        if from_preset is not None:
            return from_preset
        return schema_default

    # SummaryRequest.__dataclass_fields__ gives us the schema defaults
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

    problem = req.check_input()
    if problem is not None:
        if req.input_policy is InputPolicy.STRICT:
            print(problem)
        if req.input_policy is InputPolicy.WARN:
            raw = get_backend(backend).summarize(req)
            return f"[WARNING: {problem}] {raw}"
        # InputPolicy.ALLOW — run it, no questions asked

    return get_backend(backend).summarize(req)
