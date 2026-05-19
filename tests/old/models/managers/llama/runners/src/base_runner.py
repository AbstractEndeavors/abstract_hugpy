from .imports import *
# base_runner.py
class LlamaCppBaseRunner(ABC):
    """Shared scaffolding — event loop, unbounded loop, logging.
    Subclasses implement only the raw I/O."""

    model_key: str

    # --- abstract I/O hooks ------------------------------------------------

    @abstractmethod
    async def _iter_stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
    ) -> AsyncIterator[tuple[str, Optional[str]]]:
        """Yield (text_chunk, finish_reason_or_None) pairs from the backend."""
        ...
    @abstractmethod
    def _chat_complete(
        self,
        messages: list[dict],
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
    ) -> tuple[str, str]:
        """Chat-template path. Return (text, finish_reason)."""
        ...

    @abstractmethod
    def _raw_complete(
        self,
        prompt: str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        return_full_text: bool,
    ) -> tuple[str, str]:
        """Raw-prompt fallback path. Return (text, finish_reason)."""
        ...
    # _blocking_complete is now FINAL — no override needed in subclasses
    def _blocking_complete(
        self,
        messages: list[dict] | str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        use_chat_template: bool,
        return_full_text: bool,
    ) -> tuple[str, str]:
        if use_chat_template and isinstance(messages, list):
            return self._chat_complete(messages, max_tokens, temp, top_p, stop)

        prompt = (
            messages
            if isinstance(messages, str)
            else messages_to_prompt_from_dicts(messages)
        )
        return self._raw_complete(prompt, max_tokens, temp, top_p, stop, return_full_text)
    # --- shared streaming --------------------------------------------------

    async def stream_chat(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        max_tokens = resolve_max_tokens(req.max_new_tokens)
        temp      = resolve_temperature(req.temperature, req.do_sample)
        top_p     = resolve_top_p(req.top_p)
        messages  = messages_to_dicts(req.messages)
        output_chunks = 0
        last_finish: Optional[str] = None

        try:
            async for text, fr in self._iter_stream(messages, max_tokens, temp, top_p):
                if cancel_event and cancel_event.is_set():
                    self._log_done(req, "cancelled", output_chunks, max_tokens)
                    yield DoneEvent(request_id=req.request_id, input_tokens=0,
                                   output_chunks=output_chunks, finish_reason="cancelled")
                    return
                if text:
                    output_chunks += 1
                    yield TokenEvent(request_id=req.request_id, text=text)
                if fr is not None:
                    last_finish = fr

            mapped = map_finish_reason(last_finish)
            self._log_done(req, mapped, output_chunks, max_tokens)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                           output_chunks=output_chunks, finish_reason=mapped)
        except Exception as exc:
            logger.exception("stream_chat failed: model=%s req=%s", self.model_key, req.request_id)
            yield ErrorEvent(request_id=req.request_id, message=f"{type(exc).__name__}: {exc}")

    # --- shared unbounded streaming ----------------------------------------

    async def stream_chat_unbounded(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
        *,
        chunk_tokens: int = 1024,
        max_chunks: int = 8,
    ) -> AsyncIterator[StreamEvent]:
        temp     = resolve_temperature(req.temperature, req.do_sample)
        top_p    = resolve_top_p(req.top_p)
        convo    = messages_to_dicts(req.messages)
        output_chunks = 0
        last_finish = "stop"

        try:
            for _ in range(max_chunks):
                if cancel_event and cancel_event.is_set():
                    self._log_done(req, "cancelled", output_chunks, chunk_tokens)
                    yield DoneEvent(request_id=req.request_id, input_tokens=0,
                                   output_chunks=output_chunks, finish_reason="cancelled")
                    return

                piece_text = ""
                chunk_finish: Optional[str] = None

                async for text, fr in self._iter_stream(convo, chunk_tokens, temp, top_p):
                    if cancel_event and cancel_event.is_set():
                        self._log_done(req, "cancelled", output_chunks, chunk_tokens)
                        yield DoneEvent(request_id=req.request_id, input_tokens=0,
                                       output_chunks=output_chunks, finish_reason="cancelled")
                        return
                    if text:
                        output_chunks += 1
                        piece_text += text
                        yield TokenEvent(request_id=req.request_id, text=text)
                    if fr is not None:
                        chunk_finish = fr

                last_finish = chunk_finish or "stop"
                if last_finish != "length" or not piece_text:
                    break

                convo.append({"role": "assistant", "content": piece_text})
                convo.append({"role": "user", "content": "continue"})

            mapped = map_finish_reason(last_finish)
            self._log_done(req, mapped, output_chunks, chunk_tokens)
            yield DoneEvent(request_id=req.request_id, input_tokens=0,
                           output_chunks=output_chunks, finish_reason=mapped)
        except Exception as exc:
            logger.exception("stream_chat_unbounded failed: model=%s req=%s", self.model_key, req.request_id)
            yield ErrorEvent(request_id=req.request_id, message=f"{type(exc).__name__}: {exc}")

    # --- shared non-streaming ----------------------------------------------

    async def generate_text_async(self, messages, **kw) -> str:
        return await asyncio.to_thread(self.generate_text, messages, **kw)

    def generate_text(self, messages, *, max_new_tokens=0, temperature=0.0,
                      top_p=1.0, do_sample=False, use_chat_template=True,
                      return_full_text=False, stop=None, **_) -> str:
        max_tokens = resolve_max_tokens(max_new_tokens)
        temp       = resolve_temperature(temperature, do_sample)
        top_p_val  = resolve_top_p(top_p)
        text, finish = self._blocking_complete(
            messages, max_tokens, temp, top_p_val, stop, use_chat_template, return_full_text
        )
        logger.info("generate_text done: model=%s finish=%s cap=%s", self.model_key, finish, max_tokens)
        return text

    def generate_text_unbounded(self, messages, *, chunk_tokens=1024,
                                max_chunks=8, temperature=0.0, top_p=1.0,
                                do_sample=False, stop=None, **_) -> str:
        temp      = resolve_temperature(temperature, do_sample)
        top_p_val = resolve_top_p(top_p)
        accumulated = ""
        convo = list(messages)

        for chunk_idx in range(max_chunks):
            text, finish = self._blocking_complete(
                convo, chunk_tokens, temp, top_p_val, stop,
                use_chat_template=True, return_full_text=False
            )
            accumulated += text
            logger.info("generate_text_unbounded chunk=%s model=%s finish=%s",
                       chunk_idx, self.model_key, finish)
            if finish != "length" or not text:
                break
            convo = convo + [{"role": "assistant", "content": text},
                             {"role": "user", "content": "continue"}]

        return accumulated

    # --- shared internals --------------------------------------------------

    def _log_done(self, req: ChatRequest, finish: str, chunks: int, cap: int) -> None:
        logger.info("stream_chat done: model=%s req=%s finish=%s chunks=%s cap=%s",
                   self.model_key, req.request_id, finish, chunks, cap)
