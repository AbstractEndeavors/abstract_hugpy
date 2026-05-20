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
from abc import ABC, abstractmethod
import httpx
from abstract_security import *

from ...imports import *

logger = logging.getLogger(__name__)
