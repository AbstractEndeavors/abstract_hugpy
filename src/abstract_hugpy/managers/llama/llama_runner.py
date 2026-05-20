"""LlamaCpp runners (HTTP and in-process Python).

Two classes, same surface:
    - LlamaCppRunner         : talks to a llama-server over HTTP
    - LlamaCppPythonRunner   : loads a GGUF in-process via llama_cpp

Both expose:
    stream_chat(req, cancel_event)            -> AsyncIterator[StreamEvent]
    stream_chat_unbounded(req, cancel_event)  -> AsyncIterator[StreamEvent]   (Python only)
    generate_text(messages, **kw)             -> str
    generate_text_unbounded(messages, **kw)   -> str                          (Python only)

Design notes:
    - Streaming and non-streaming both go through the GGUF's embedded chat
      template (create_chat_completion), not a hand-rolled User:/Assistant:
      formatter. That formatter exists only as a fallback for raw-completion
      paths.
    - finish_reason is mapped from llama.cpp's vocabulary ('length', 'stop')
      to the schema's vocabulary ('max_tokens', 'stop') in one place.
    - Defaults live in DEFAULT_MAX_TOKENS at the top of the file, not as
      magic numbers buried four levels deep in method bodies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import AsyncIterator, Dict, Optional
from ..message_utils import messages_to_dicts
import httpx
from abstract_security import *

from .imports import (
    ensure_model,
    get_model_config,
    get_gguf_file,
    ChatRequest,
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TokenEvent,
)

logger = logging.getLogger(__name__)







# ---------------------------------------------------------------------------
# Prompt formatting — only used as a fallback when the chat template can't
# be applied (raw completion path on the HTTP runner).
# ---------------------------------------------------------------------------

def messages_to_prompt_from_dicts(messages: list[dict]) -> str:
    """Hand-rolled User:/Assistant: scaffolding for raw completion endpoints.

    Prefer the model's embedded chat template over this when possible —
    GGUFs from Qwen/Llama/etc ship with proper templates that match what
    they were trained on. This fallback exists for legacy /completion calls.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def messages_to_prompt(req: ChatRequest) -> str:
    """ChatRequest variant of the above. One definition, not two."""
    return messages_to_prompt_from_dicts(messages_to_dicts(req.messages))

# ---------------------------------------------------------------------------
# Helpers — finish reason mapping, defaulted resolvers
# ---------------------------------------------------------------------------




def _map_finish_reason(raw: Optional[str]) -> str:
    return FINISH_REASON_MAP.get(raw, "stop")


def _resolve_max_tokens(requested: Optional[int]) -> int:
    if not requested or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return requested


def _resolve_temperature(requested: Optional[float], do_sample: bool) -> float:
    if not do_sample:
        return 0.0
    if requested is None or requested < 0:
        return DEFAULT_TEMPERATURE
    return min(requested, 2.0)


def _resolve_top_p(requested: Optional[float]) -> float:
    if requested is None or requested <= 0 or requested > 1:
        return DEFAULT_TOP_P
    return requested









