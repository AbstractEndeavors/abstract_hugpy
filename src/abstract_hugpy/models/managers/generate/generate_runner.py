"""DeepCoder chat runner.

Thin adapter that wraps the existing DeepCoder + REGISTRY + build_deepcoder_runtime
machinery (in deepcoder/coder.py and deepcoder/config.py) and exposes them
behind the Runner protocol.

Construction is cheap — just stores the model_key. The actual model load
happens lazily on first .run() / .stream() call, which is when REGISTRY.get()
fires. That matches DeepCoder's existing 'one instance per cfg.cache_key()'
caching, so multiple runners for the same model still share weights.

Sync->async: DeepCoder.generate_text is sync (it does PyTorch inference on
the calling thread). We wrap it in asyncio.to_thread so the FastAPI event
loop stays free during long generations. Without this, one /generate call
blocks every other request the worker is serving — including SSE heartbeats
on /chat for other clients.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from .imports import TaskRequest,ChatRequest, ChatResult,StreamEvent

# These imports go through the existing module layout. Adjust the dotted
# paths to match wherever you wire this file in — the runner doesn't care
# about the path, only that these names resolve.
from .coder import REGISTRY, DeepCoder
from .config import (
    DeepCoderConfig,
    build_deepcoder_runtime,
)

logger = logging.getLogger(__name__)


class DeepCoderChatRunner:
    """Runner for transformers-based causal LMs (DeepCoder, DAN-Qwen3, etc).

    The model_key -> DeepCoderConfig translation happens once in __init__
    so the cache_key is stable and REGISTRY.get() returns the same DeepCoder
    instance across requests.
    """

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, model_key: str, **runtime_kwargs):
        self.model_key = model_key
        self._cfg: DeepCoderConfig = build_deepcoder_runtime(
            model_key=model_key,
            **runtime_kwargs,
        )
        # Note: not calling REGISTRY.get(self._cfg) here — that would force
        # the model load at runner construction time. Defer to first use.

    @property
    def coder(self) -> DeepCoder:
        """Resolve the underlying DeepCoder instance. Loads on first access."""
        return REGISTRY.get(self._cfg)

    # --- non-streaming -----------------------------------------------------

    async def run(self, req: ChatInput) -> ChatResult:
        req = ChatRequest.coerce(req, model_key=self.model_key)
        messages = [m.model_dump() for m in req.messages]

        def _do() -> GenerationOutcome:
            if req.unbounded:
                return run_unbounded(
                    self._inner_generate_once,
                    messages,
                    chunk_tokens=req.max_new_tokens or 1024,
                    max_chunks=req.max_chunks or 8,
                )
            return self._inner_generate_once(messages, req.max_new_tokens)

        try:
            outcome = await asyncio.to_thread(_do)
            return ChatResult(
                request_id=req.request_id, model_key=req.model_key,
                ok=True, text=outcome.text,
                finish_reason=_map_finish_reason(outcome.finish_reason),
            )
        except Exception as exc:
            logger.exception("run failed: model=%s req=%s",
                             self.model_key, req.request_id)
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
        """Delegate to DeepCoder.stream_chat — it already implements the
        StreamEvent protocol. We don't re-wrap the events; pass them through.

        The route layer is responsible for SSE-encoding (`data: {...}\\n\\n`).
        Runners just yield typed events.
        """
        # DeepCoder.stream_chat already accepts the project's ChatRequest
        # type (from .imports), which has the same fields as ours. If the
        # two ChatRequest classes ever diverge, this is the line that
        # breaks first — and that's a feature, not a bug.
        async for event in self.coder.stream_chat(req, cancel_event=cancel_event):
            yield event
