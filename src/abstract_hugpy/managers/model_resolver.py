"""Model resolution — single source of truth.

Everything in dispatch reads from `Resolution`, which is built exactly
once per request by `resolve()`. No downstream layer is allowed to
re-derive task, framework, builder, or runner_cls from kwargs — if it
needs any of those, it reads them off the Resolution object.

Adding a new (framework, task) pair:
    1. Implement a runner class conforming to the Runner protocol.
    2. Add a row to _RUNNERS.
    3. Add a row to _REQUEST_BUILDERS.
    4. (Optional) Add a row to _TASK_DEFAULTS if there's a sensible
       default model for "task only" callers.

Adding a new model:
    Add a row to MODEL_REGISTRY (in models_dict.py). validate_registry()
    will fail at import time if (framework, primary_task) or any
    (framework, task) in cfg.tasks isn't registered.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict

from .imports import *
from .generate import DeepCoderChatRunner
from .vision import VisionRunner
from .vision.schemas import VisionRequest
from .llama import LlamaCppChatRunner
from .whisper_model import WhisperRunner, TranscribeRequest
from .summarizers import SummarizeRunner
from .embed import FeatureExtractionRunner, EmbedRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution — the contract between resolution and execution.
# ---------------------------------------------------------------------------

class Resolution(BaseModel):
    """Frozen decision object. Everything downstream reads from here."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model_key: str
    framework: str
    task: str                 # effective task, NOT cfg.primary_task
    cfg: Any                  # ModelConfig
    builder: Callable[[Dict[str, Any], str], BaseModel]
    runner_cls: Type          # Runner subclass
    cache_key: Tuple[str, str]   # (model_key, task)


def _make_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Request builders — one per (framework, task).
# ---------------------------------------------------------------------------

def _build_chat_request(kwargs: Dict[str, Any], model_key: str) -> ChatRequest:
    out: Dict[str, Any] = {"model_key": model_key}

    if "messages" in kwargs:
        out["messages"] = kwargs["messages"]
    elif "prompt" in kwargs:
        out["messages"] = [{"role": "user", "content": kwargs["prompt"]}]
    else:
        raise ValueError(
            "chat request needs either 'messages' or 'prompt'; "
            f"got keys: {sorted(kwargs)}"
        )

    for k in ("max_new_tokens", "temperature", "top_p", "do_sample", "request_id"):
        if k in kwargs:
            out[k] = kwargs[k]

    out.setdefault("request_id", _make_request_id())
    return ChatRequest(**out)


def _build_vision_request(kwargs: Dict[str, Any], model_key: str) -> VisionRequest:
    image_path = kwargs.get("image_path") or kwargs.get("file")
    if image_path is None:
        raise ValueError(
            "vision request needs 'image_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "image_path": image_path,
        "request_id": kwargs.get("request_id", _make_request_id()),
    }
    for k in ("prompt", "max_new_tokens", "max_tokens"):
        if k in kwargs:
            out[k] = kwargs[k]
    return VisionRequest(**out)


def _build_whisper_request(kwargs: Dict[str, Any], model_key: str) -> TranscribeRequest:
    file_path = kwargs.get("audio_path") or kwargs.get("file")
    if file_path is None:
        raise ValueError(
            "whisper request needs 'audio_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    return TranscribeRequest(
        model_key=model_key,
        file_path=file_path,
        capture_frames=kwargs.get("capture_frames", False),
        request_id=kwargs.get("request_id", _make_request_id()),
    )


def _build_summarize_request(kwargs: Dict[str, Any], model_key: str) -> "SummarizeRequest":
    text = kwargs.get("text") or kwargs.get("prompt")
    if text is None and kwargs.get("file"):
        text = read_from_file(kwargs["file"])
    if text is None:
        raise ValueError(
            "summarize request needs 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )

    out: Dict[str, Any] = {
        "model_key": model_key,
        "text": text,
        "request_id": kwargs.get("request_id", _make_request_id()),
    }
    for k in (
        "preset", "summary_mode", "input_policy",
        "max_chunk_tokens", "min_length", "max_length",
        "do_sample", "min_input_words",
        "consolidation_min_length", "consolidation_max_length",
        "max_output_words",
    ):
        if k in kwargs:
            out[k] = kwargs[k]
    return SummarizeRequest(**out)


def _texts_from_kwargs(kwargs: Dict[str, Any]) -> list[str]:
    """Shared text extraction: texts | text | prompt | file -> list[str].

    Used by both embed builders. Returns a list even for single-string
    input so the runner doesn't have to branch.
    """
    raw = kwargs.get("texts") or kwargs.get("text") or kwargs.get("prompt")
    if raw is None and kwargs.get("file"):
        raw = read_from_file(kwargs["file"])
    if raw is None:
        raise ValueError(
            "embed request needs 'texts', 'text', 'prompt', or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        return list(raw)
    raise TypeError(
        f"embed input must be str or list[str], got {type(raw).__name__}"
    )


def _build_embed_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", _make_request_id()),
        texts=_texts_from_kwargs(kwargs),
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


def _build_similarity_request(kwargs: Dict[str, Any], model_key: str) -> EmbedRequest:
    """sentence-similarity needs a second set of texts to compare against."""
    other_raw = (
        kwargs.get("other_texts")
        or kwargs.get("other_text")
        or kwargs.get("compare_to")
    )
    if other_raw is None:
        raise ValueError(
            "sentence-similarity needs 'other_texts', 'other_text', or 'compare_to' "
            f"in addition to 'texts'/'text'/'prompt'/'file'; got keys: {sorted(kwargs)}"
        )
    if isinstance(other_raw, str):
        other_texts = [other_raw]
    elif isinstance(other_raw, list) and all(isinstance(t, str) for t in other_raw):
        other_texts = list(other_raw)
    else:
        raise TypeError(
            f"other_texts must be str or list[str], got {type(other_raw).__name__}"
        )

    return EmbedRequest(
        model_key=model_key,
        request_id=kwargs.get("request_id", _make_request_id()),
        texts=_texts_from_kwargs(kwargs),
        other_texts=other_texts,
        normalize=kwargs.get("normalize", True),
        batch_size=kwargs.get("batch_size", 32),
    )


# ---------------------------------------------------------------------------
# Registries — single source of truth.
# ---------------------------------------------------------------------------

_REQUEST_BUILDERS: Dict[Tuple[str, str], Callable[[Dict[str, Any], str], BaseModel]] = {
    ("transformers", "code-generation"):              _build_chat_request,
    ("transformers", "text-generation"):              _build_chat_request,
    ("llama_cpp",    "code-generation"):              _build_chat_request,
    ("llama_cpp",    "text-generation"):              _build_chat_request,
    ("transformers", "image-text-to-text"):           _build_vision_request,
    ("transformers", "automatic-speech-recognition"): _build_whisper_request,
    ("transformers", "summarization"):                _build_summarize_request,
    ("transformers", "text2text-generation"):         _build_summarize_request,
    ("transformers", "feature-extraction"):           _build_embed_request,
    ("transformers", "sentence-similarity"):          _build_similarity_request,
}

_RUNNERS: Dict[Tuple[str, str], Type[Runner]] = {
    ("transformers", "code-generation"):              DeepCoderChatRunner,
    ("transformers", "text-generation"):              DeepCoderChatRunner,
    ("llama_cpp",    "code-generation"):              LlamaCppChatRunner,
    ("llama_cpp",    "text-generation"):              LlamaCppChatRunner,
    ("transformers", "image-text-to-text"):           VisionRunner,
    ("transformers", "automatic-speech-recognition"): WhisperRunner,
    ("transformers", "summarization"):                SummarizeRunner,
    ("transformers", "text2text-generation"):         SummarizeRunner,
    ("transformers", "feature-extraction"):           FeatureExtractionRunner,
    ("transformers", "sentence-similarity"):          FeatureExtractionRunner,
}

_MEDIA_DEFAULTS: Dict[str, str] = {
    "document": DEFAULT_CHAT_MODEL,
    "code":     DEFAULT_CHAT_MODEL,
    "text":     DEFAULT_CHAT_MODEL,
    "image":    DEFAULT_VISION_MODEL,
    "audio":    DEFAULT_WHISPER_MODEL,
    "video":    DEFAULT_WHISPER_MODEL,
}

_TASK_DEFAULTS: Dict[str, str] = {
    "code-generation":              DEFAULT_CHAT_MODEL,
    "text-generation":              DEFAULT_CHAT_MODEL,
    "image-text-to-text":           DEFAULT_VISION_MODEL,
    "automatic-speech-recognition": DEFAULT_WHISPER_MODEL,
    "summarization":                DEFAULT_SUMMARIZE_MODEL,
    "text2text-generation":         DEFAULT_SUMMARIZE_MODEL,
    "feature-extraction":           DEFAULT_EMBED_MODEL,
    "sentence-similarity":          DEFAULT_EMBED_MODEL,
}

# Derived from _RUNNERS so it can't drift.
_KNOWN_TASKS: frozenset[str] = frozenset(task for _, task in _RUNNERS.keys())


# ---------------------------------------------------------------------------
# resolve_model_key — picks the model. Default-resolution chain only.
# Does NOT pick task; that's resolve()'s job.
# ---------------------------------------------------------------------------

def resolve_model_key(
    *,
    model_key: Optional[str] = None,
    file: Optional[str] = None,
    media_type: Optional[str] = None,
    task: Optional[str] = None,
) -> str:
    """Pick a model_key via explicit resolution chain.

    Order: explicit model_key > explicit task > explicit media_type
           > file -> media_type > chat default.

    `task`, when given alongside `model_key`, is validated against
    cfg.tasks. When given alone, it picks _TASK_DEFAULTS[task].
    """
    if task is not None and task not in _KNOWN_TASKS:
        raise KeyError(
            f"Unknown task={task!r}; known: {sorted(_KNOWN_TASKS)}"
        )

    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model_key={model_key!r}; "
                f"known: {sorted(MODEL_REGISTRY.keys())}"
            )
        if task is not None and task not in MODEL_REGISTRY[model_key].tasks:
            raise ValueError(
                f"Model {model_key!r} does not support task={task!r}; "
                f"supported: {sorted(MODEL_REGISTRY[model_key].tasks)}"
            )
        logger.debug("resolve_model_key: explicit key=%s task=%s", model_key, task)
        return model_key

    if task is not None:
        inferred = _TASK_DEFAULTS.get(task)
        if inferred is None:
            raise KeyError(
                f"No default model for task={task!r}; "
                f"tasks with defaults: {sorted(_TASK_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Task default {inferred!r} for {task!r} not in MODEL_REGISTRY"
            )
        if task not in MODEL_REGISTRY[inferred].tasks:
            raise ValueError(
                f"Task default {inferred!r} for {task!r} does not list "
                f"{task!r} in cfg.tasks={sorted(MODEL_REGISTRY[inferred].tasks)!r}"
            )
        logger.debug("resolve_model_key: task=%s -> key=%s", task, inferred)
        return inferred

    if media_type is None and file is not None:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"resolve_model_key: file does not exist: {file!r}"
            )
        media_type = derive_media_type(file)
        logger.debug("resolve_model_key: file=%s -> media=%s", file, media_type)

    if media_type is not None:
        inferred = _MEDIA_DEFAULTS.get(media_type)
        if inferred is None:
            raise KeyError(
                f"No default model for media_type={media_type!r}; "
                f"known: {sorted(_MEDIA_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Media default {inferred!r} for {media_type!r} "
                f"not in MODEL_REGISTRY"
            )
        logger.debug("resolve_model_key: media=%s -> key=%s", media_type, inferred)
        return inferred

    if DEFAULT_CHAT_MODEL not in MODEL_REGISTRY:
        raise KeyError(
            f"DEFAULT_CHAT_MODEL={DEFAULT_CHAT_MODEL!r} not in MODEL_REGISTRY"
        )
    logger.debug("resolve_model_key: fallback to chat default=%s", DEFAULT_CHAT_MODEL)
    return DEFAULT_CHAT_MODEL


# ---------------------------------------------------------------------------
# resolve — the only function that maps kwargs -> Resolution.
# ---------------------------------------------------------------------------

def resolve(prompt_kwargs: Dict[str, Any]) -> Resolution:
    """Build a Resolution from request kwargs. One call site for all routing.

    `task`, if given by the caller, wins over cfg.primary_task. This is the
    single rule that the old dispatch broke in three different places.
    """
    requested_task = prompt_kwargs.get("task")

    model_key = resolve_model_key(
        model_key=prompt_kwargs.get("model_key"),
        file=prompt_kwargs.get("file"),
        media_type=prompt_kwargs.get("media_type"),
        task=requested_task,
    )

    cfg = MODEL_REGISTRY[model_key]
    task = requested_task or cfg.primary_task

    if task not in cfg.tasks:
        raise ValueError(
            f"Model {model_key!r} does not support task={task!r}; "
            f"supported: {sorted(cfg.tasks)}"
        )

    key = (cfg.framework, task)

    builder = _REQUEST_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"No request builder for {key!r}; model={model_key!r}, "
            f"known: {sorted(_REQUEST_BUILDERS)}"
        )

    runner_cls = _RUNNERS.get(key)
    if runner_cls is None:
        raise KeyError(
            f"No runner for {key!r}; model={model_key!r}, "
            f"known: {sorted(_RUNNERS)}"
        )

    logger.debug(
        "resolve: model=%s framework=%s task=%s (requested=%s primary=%s)",
        model_key, cfg.framework, task, requested_task, cfg.primary_task,
    )

    return Resolution(
        model_key=model_key,
        framework=cfg.framework,
        task=task,
        cfg=cfg,
        builder=builder,
        runner_cls=runner_cls,
        cache_key=(model_key, task),
    )


# ---------------------------------------------------------------------------
# validate_registry — fail at import time, not on first request.
# ---------------------------------------------------------------------------

def validate_registry() -> None:
    """Walk MODEL_REGISTRY and assert every entry can actually be served.

    Two checks per model:
      1. (framework, primary_task) has a runner registered.
      2. Every task in cfg.tasks has a runner AND a builder registered.

    Raises RuntimeError listing ALL broken entries — not just the first —
    so a single import gives you the full list of registry bugs to fix.
    """
    errors: list[str] = []

    for model_key, cfg in MODEL_REGISTRY.items():
        primary_key = (cfg.framework, cfg.primary_task)
        if primary_key not in _RUNNERS:
            errors.append(
                f"  {model_key}: primary_task={cfg.primary_task!r} on "
                f"framework={cfg.framework!r} has no runner registered"
            )

        for task in cfg.tasks:
            task_key = (cfg.framework, task)
            if task_key not in _RUNNERS:
                errors.append(
                    f"  {model_key}: task={task!r} in cfg.tasks on "
                    f"framework={cfg.framework!r} has no runner registered"
                )
            if task_key not in _REQUEST_BUILDERS:
                errors.append(
                    f"  {model_key}: task={task!r} in cfg.tasks on "
                    f"framework={cfg.framework!r} has no request builder registered"
                )

    if errors:
        raise RuntimeError(
            f"MODEL_REGISTRY validation failed ({len(errors)} issues):\n"
            + "\n".join(errors)
            + f"\n\nRegistered runners:  {sorted(_RUNNERS)}"
            + f"\nRegistered builders: {sorted(_REQUEST_BUILDERS)}"
        )

    logger.info(
        "validate_registry: ok — %d models, %d runner pairs, %d builder pairs",
        len(MODEL_REGISTRY), len(_RUNNERS), len(_REQUEST_BUILDERS),
    )


# Run at import time. If the registry is bad, fail loudly here — not
# halfway through a user's request.
validate_registry()
