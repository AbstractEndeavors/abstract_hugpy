"""Llama.cpp chat runner.

Adapter over the existing LlamaCppPythonRunner (in llama/llama_runner.py).
That class already does all the real work — chat-template handling,
streaming, finish_reason mapping, the unbounded continue loop. This file
just bolts the Runner protocol onto it.

Why two runners (deepcoder + llama.cpp) instead of one with a switch:
    They genuinely have different lifecycles (DeepCoder is built from a
    DeepCoderConfig and cached by cache_key; LlamaCppPythonRunner is keyed
    by model_key and built directly). Keeping them as separate classes
    means the dispatch table reads as a 1:1 mapping of (framework, task)
    -> class, and neither runner has to know about the other's loader.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from .protocol import TaskRequest
from .chat_schemas import ChatRequest, ChatResult

# Existing module — adjust dotted path to wherever this file lives.
from ..llama.llama_runner import (
    LlamaCppPythonRunner,
    get_llama_runner,
)
from ..llama.imports import StreamEvent

logger = logging.getLogger(__name__)


class LlamaCppChatRunner:
    """Runner for GGUF models loaded in-process via llama_cpp.

    Uses the existing get_llama_runner() singleton cache so multiple
    runners for the same model_key share a single LlamaCppPythonRunner
    (which itself holds the loaded GGUF + KV cache + generate_lock).
    """

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, model_key: str, **runtime_kwargs):
        self.model_key = model_key
        # runtime_kwargs (n_ctx override, n_threads, ...) are forwarded
        # only on first construction; subsequent runners with the same
        # model_key get the cached instance regardless. This matches the
        # existing singleton behavior of get_llama_runner().
        self._runtime_kwargs = runtime_kwargs

    @property
    def runner(self) -> LlamaCppPythonRunner:
        # Lazy resolution. First access triggers the GGUF load (which can
        # take seconds for a 14B model), subsequent accesses are dict lookups.
        return get_llama_runner(self.model_key)

    # --- non-streaming -----------------------------------------------------

    async def run(self, req: ChatRequest,messages=None) -> ChatResult:
        try:
            messages = messages or [m.model_dump() for m in req.messages]

            if req.unbounded:
                # Auto-continue past 'length' until EOS or max_chunks.
                # Returns concatenated text only — no finish_reason
                # surfaced today, so we report 'stop' on success.
                text = await asyncio.to_thread(
                    self.runner.generate_text_unbounded,
                    messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=req.do_sample,
                )
                return ChatResult(
                    request_id=req.request_id, model_key=req.model_key,
                    ok=True, text=text, finish_reason="stop",
                )

            # Standard single-shot generation.
            text = await asyncio.to_thread(
                self.runner.generate_text,
                messages,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                use_chat_template=True,
                return_full_text=False,
            )
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=True, text=text, finish_reason="stop",
            )

        except Exception as exc:
            logger.exception(
                "LlamaCppChatRunner.run failed: model=%s req=%s",
                self.model_key, req.request_id,
            )
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=False, error=f"{type(exc).__name__}: {exc}",
                text="", finish_reason="error",
            )

    # --- streaming ---------------------------------------------------------

    async def stream(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Pick stream_chat or stream_chat_unbounded based on req.unbounded.

        Both methods already conform to the StreamEvent contract
        (TokenEvent stream + one terminal DoneEvent/ErrorEvent), so the
        adapter is a straight passthrough.
        """
        streamer = (
            self.runner.stream_chat_unbounded(req, cancel_event=cancel_event)
            if req.unbounded
            else self.runner.stream_chat(req, cancel_event=cancel_event)
        )
        async for event in streamer:
            yield event
