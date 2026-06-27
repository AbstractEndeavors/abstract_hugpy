"""DeepCoder chat runner.

Thin adapter that wraps the existing DeepCoder + REGISTRY + build_deepcoder_runtime
machinery behind the Runner protocol.

Construction is cheap — just stores the model_key. The model load happens
lazily on first .run()/.stream(), when REGISTRY.get() fires, matching
DeepCoder's 'one instance per cfg.cache_key()' caching so multiple runners
for the same model share weights.

Sync->async: DeepCoder.generate_text is sync (PyTorch inference on the
calling thread). We wrap it in asyncio.to_thread so the event loop stays
free during long generations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from .imports import TaskRequest, ChatRequest, ChatResult, StreamEvent,attempt
from .coder import REGISTRY, DeepCoder
from .config import DeepCoderConfig, build_deepcoder_runtime

logger = logging.getLogger(__name__)


class DeepCoderChatRunner:
    """Runner for transformers-based causal LMs (DeepCoder, DAN-Qwen3, etc)."""

    request_type = ChatRequest
    result_type = ChatResult

    def __init__(self, cfg, **runtime_kwargs):
        self.model_key = cfg.model_key
        self._cfg = build_deepcoder_runtime(model_key=cfg.model_key, **runtime_kwargs)
        # Not calling REGISTRY.get(self._cfg) here — that would force the
        # model load at construction. Defer to first use.

    @property
    def coder(self) -> DeepCoder:
        """Resolve the underlying DeepCoder. Loads on first access."""
        return REGISTRY.get(self._cfg)

    # --- result helpers ----------------------------------------------------

    def _error_result(self, req: ChatRequest, error: str) -> ChatResult:
        """Single construction site for ok=False results — so the failure
        shape can't drift between the two error paths."""
        return ChatResult(
            request_id=req.request_id,
            model_key=req.model_key,
            ok=False,
            error=error,
            text="",
            finish_reason="error",
        )

    # --- non-streaming -----------------------------------------------------

    async def run(self, req: ChatRequest) -> ChatResult:
        messages = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in req.messages
        ]

        # generate_text is sync; offload so the event loop keeps running.
        # attempt() logs the full traceback and hands back the exception so
        # we can branch on its type below.
        ok, text, exc = await asyncio.to_thread(
            attempt,
            self.coder.generate_text,
            messages,
            label=f"DeepCoderChatRunner.run model={self.model_key} req={req.request_id}",
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            use_chat_template=True,
            return_full_text=False,
        )

        if ok:
            return ChatResult(
                request_id=req.request_id,
                model_key=req.model_key,
                ok=True,
                text=text,
                # generate_text doesn't surface a finish_reason; 'stop' is
                # the honest default. Caller detects truncation via length.
                finish_reason="stop",
            )

        # _resolve_max_new_tokens raises ValueError on cap violation — a
        # request-side problem, not a model failure. attempt() already
        # logged it at exception level; downgrade the *meaning* here.
        if isinstance(exc, ValueError):
            logger.warning("DeepCoderChatRunner.run rejected: %s", exc)
            return self._error_result(req, str(exc))

        return self._error_result(req, f"{type(exc).__name__}: {exc}")

    # --- streaming ---------------------------------------------------------

    async def stream(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Delegate to DeepCoder.stream_chat — it already implements the
        StreamEvent protocol. Pass events through unwrapped.

        The route layer SSE-encodes (`data: {...}\\n\\n`); runners just
        yield typed events.

        Not wrapped in attempt(): this is a generator, and attempt() runs a
        callable to completion — it can't drive a stream. Errors mid-stream
        belong in DeepCoder.stream_chat as ErrorEvents, not swallowed here.
        """
        async for event in self.coder.stream_chat(req, cancel_event=cancel_event):
            yield event
