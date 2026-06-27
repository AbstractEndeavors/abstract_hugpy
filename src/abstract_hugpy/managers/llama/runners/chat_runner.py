"""Llama.cpp chat runner.

Adapter over the existing LlamaCppPythonRunner / LlamaCppRunner (HTTP).
Both are kept in their per-process singleton cache (get_llama_runner),
so the heavy GGUF load happens once per model_key regardless of how
many adapter wrappers exist.

Constructor signature is uniform with the other runners in _RUNNERS:
    __init__(self, cfg: ModelConfig, **runtime_kwargs)

This matches what dispatch._get_or_build_runner expects, so the
dispatch table can stay a flat (framework, task) -> class mapping
without any per-runner adapter logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from .src import *
from .imports import *
from .get import get_llama_runner

logger = logging.getLogger(__name__)


class LlamaCppChatRunner:
    """Runner for GGUF models loaded in-process via llama_cpp or via HTTP.

    Uses the get_llama_runner() singleton cache so multiple adapter
    wrappers for the same model_key share a single underlying runner
    (which itself holds the loaded GGUF + KV cache + generate_lock).
    """

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, cfg, **runtime_kwargs):
        self.cfg = cfg
        # model_key is whatever the registry uses as its key; ModelConfig
        # exposes it as model_key (set by get_models_dict in models_config).
        self.model_key = cfg.model_key
        # runtime_kwargs (n_ctx override, n_threads, ...) are forwarded
        # only on first construction; subsequent runners with the same
        # model_key get the cached instance regardless. Singleton behavior
        # is intentional — we don't want two GGUF loads for one model.
        self._runtime_kwargs = runtime_kwargs

    @property
    def runner(self):
        # Lazy resolution. First access triggers the GGUF load (which can
        # take seconds for a 14B model); subsequent accesses are dict lookups.
        return get_llama_runner(self.model_key)

    # --- non-streaming -----------------------------------------------------

    async def run(self, req) -> ChatResult:
        req = ChatRequest.coerce(req, model_key=self.model_key)
        messages = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in req.messages
        ]

        if req.unbounded:
            text = await self.runner.generate_text_async(
                messages,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
            )
        else:
            text = await self.runner.generate_text_async(
                messages,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                do_sample=req.do_sample,
                use_chat_template=True,
                return_full_text=False,
            )

        return ChatResult(
            request_id=req.request_id,
            model_key=req.model_key,
            ok=True,
            text=text,
            finish_reason="stop",
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
