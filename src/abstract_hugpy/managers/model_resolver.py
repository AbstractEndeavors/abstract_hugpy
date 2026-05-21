from typing import Any, Optional, Dict, Tuple, Type, Callable
import os
import logging
import threading
from pydantic import BaseModel

from .imports import (
    Runner, ChatRequest, MODEL_REGISTRY,TranscribeRequest,
    DEFAULT_CHAT_MODEL, DEFAULT_VISION_MODEL, DEFAULT_WHISPER_MODEL,derive_media_type
)
from .generate import DeepCoderChatRunner
from .vision import VisionRunner
from .llama import LlamaCppChatRunner
from .whisper_model import WhisperRunner

logger = logging.getLogger(__name__)
import uuid
from .vision.schemas import VisionRequest


def _make_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


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
    audio_path = kwargs.get("audio_path") or kwargs.get("file")
    if audio_path is None:
        raise ValueError(
            "whisper request needs 'audio_path' or 'file'; "
            f"got keys: {sorted(kwargs)}"
        )
    return TranscribeRequest(
        model_key=model_key,
        audio_path=audio_path,
        request_id=kwargs.get("request_id", _make_request_id()),
    )


_REQUEST_BUILDERS: Dict[Tuple[str, str], Callable[[Dict[str, Any], str], BaseModel]] = {
    ("transformers", "code-generation"):    _build_chat_request,
    ("transformers", "text-generation"):    _build_chat_request,
    ("llama_cpp",    "code-generation"):    _build_chat_request,
    ("llama_cpp",    "text-generation"):    _build_chat_request,
    ("transformers", "vision-language"):    _build_vision_request,
    ("transformers", "speech-recognition"): _build_whisper_request,
}

_RUNNERS: Dict[Tuple[str, str], Type[Runner]] = {
    ("transformers", "code-generation"): DeepCoderChatRunner,
    ("transformers", "text-generation"): DeepCoderChatRunner,
    ("llama_cpp",    "code-generation"): LlamaCppChatRunner,
    ("llama_cpp",    "text-generation"): LlamaCppChatRunner,
    ("transformers", "vision-language"): VisionRunner,
    ("transformers", "speech-recognition"): WhisperRunner,
}


_MEDIA_DEFAULTS: Dict[str, str] = {
    "document": DEFAULT_CHAT_MODEL,
    "code":     DEFAULT_CHAT_MODEL,
    "text":     DEFAULT_CHAT_MODEL,
    "image":    DEFAULT_VISION_MODEL,
    "audio":    DEFAULT_WHISPER_MODEL,
    "video":    DEFAULT_WHISPER_MODEL,
}


def resolve_model_key(
    *,
    model_key: Optional[str] = None,
    file: Optional[str] = None,
    media_type: Optional[str] = None,
) -> str:
    """Pick a model_key via explicit resolution chain.

    Order: explicit model_key > explicit media_type > file -> media_type > chat default.
    Every step validates against MODEL_REGISTRY. Errors carry the chain so
    the caller knows which step failed.
    """
    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model_key={model_key!r}; "
                f"known: {sorted(MODEL_REGISTRY.keys())}"
            )
        logger.debug("resolve_model_key: explicit key=%s", model_key)
        return model_key

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
                f"No default model registered for media_type={media_type!r}; "
                f"known media types: {sorted(_MEDIA_DEFAULTS)}"
            )
        if inferred not in MODEL_REGISTRY:
            raise KeyError(
                f"Media default {inferred!r} for {media_type!r} "
                f"is not in MODEL_REGISTRY; check the DEFAULT_* constants."
            )
        logger.debug("resolve_model_key: media=%s -> key=%s", media_type, inferred)
        return inferred

    if DEFAULT_CHAT_MODEL not in MODEL_REGISTRY:
        raise KeyError(
            f"DEFAULT_CHAT_MODEL={DEFAULT_CHAT_MODEL!r} not in MODEL_REGISTRY"
        )
    logger.debug("resolve_model_key: fallback to chat default=%s", DEFAULT_CHAT_MODEL)
    return DEFAULT_CHAT_MODEL
