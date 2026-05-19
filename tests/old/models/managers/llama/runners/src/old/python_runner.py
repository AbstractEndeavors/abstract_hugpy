from .imports import *
# ===========================================================================
# In-process Python runner — loads a GGUF via llama_cpp directly
# ===========================================================================

class LlamaCppPythonRunner:
    def __init__(
        self,
        model_key: str,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: Optional[int] = None,
    ):
        from llama_cpp import Llama

        self.model_key = model_key
        self.cfg = get_model_config(model_key)

        model_dir = ensure_model(model_key)
        # No pathlib — get_gguf_file accepts strings via os.fspath internally
        model_path = get_gguf_file(model_dir, self.cfg)

        if not model_path:
            raise FileNotFoundError(f"No GGUF file found for model_key={model_key}")

        self.model_path = os.fspath(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self.generate_lock = threading.Lock()

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )

        logger.info(
            "LlamaCppPythonRunner ready: model=%s n_ctx=%s n_threads=%s path=%s",
            model_key, self.n_ctx, self.n_threads, self.model_path,
        )

    # --- streaming ---------------------------------------------------------

    async def stream_chat(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream via the GGUF's embedded chat template (create_chat_completion).

        No hand-rolled prompt scaffolding, no User:/Assistant: stop strings.
        Streaming chunks come back in OpenAI shape: choices[0].delta.content.
        """
        max_tokens = _resolve_max_tokens(req.max_new_tokens)
        temp = _resolve_temperature(req.temperature, req.do_sample)
        top_p = _resolve_top_p(req.top_p)

        messages = messages_to_dicts(req.messages)
        output_chunks = 0
        last_finish: Optional[str] = None

        try:
            def run_stream():
                with self.generate_lock:
                    return self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temp,
                        top_p=top_p,
                        stream=True,
                        stop=None,  # let the chat template's EOS handle it
                    )

            stream = await asyncio.to_thread(run_stream)

            for raw in stream:
                if cancel_event is not None and cancel_event.is_set():
                    self._log_done(req, "cancelled", output_chunks, max_tokens)
                    yield DoneEvent(
                        request_id=req.request_id,
                        input_tokens=0,
                        output_chunks=output_chunks,
                        finish_reason="cancelled",
                    )
                    return

                try:
                    choice = raw["choices"][0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content", "") or ""
                    fr = choice.get("finish_reason")
                except Exception:
                    text, fr = "", None

                if text:
                    output_chunks += 1
                    yield TokenEvent(request_id=req.request_id, text=text)

                if fr is not None:
                    last_finish = fr

                await asyncio.sleep(0)

            mapped = _map_finish_reason(last_finish)
            self._log_done(req, mapped, output_chunks, max_tokens)
            yield DoneEvent(
                request_id=req.request_id,
                input_tokens=0,
                output_chunks=output_chunks,
                finish_reason=mapped,
            )

        except Exception as exc:
            logger.exception("stream_chat failed: model=%s req=%s", self.model_key, req.request_id)
            yield ErrorEvent(
                request_id=req.request_id,
                message=f"{type(exc).__name__}: {exc}",
            )

    async def stream_chat_unbounded(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
        *,
        chunk_tokens: int = 1024,
        max_chunks: int = 8,
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat that auto-continues when the model hits 'length'.

        Same event contract as stream_chat (TokenEvent stream + one terminal
        DoneEvent/ErrorEvent). Internally re-issues with the partial output
        appended as an assistant turn + 'continue' user turn whenever
        finish_reason='length'. Stops on EOS, max_chunks, or cancel.

        Final DoneEvent.finish_reason:
            'stop'       — model reached natural end
            'max_tokens' — hit max_chunks ceiling, response truncated
            'cancelled'  — cancel_event was set
        """
        temp = _resolve_temperature(req.temperature, req.do_sample)
        top_p = _resolve_top_p(req.top_p)

        convo: list[dict] = messages_to_dicts(req.messages)
        logger.info(
            "llama unbounded context: model=%s req=%s messages=%s roles=%s chars=%s",
            self.model_key,
            req.request_id,
            len(convo),
            [m.get("role") for m in convo],
            sum(len(m.get("content", "")) for m in convo),
        )
        output_chunks = 0
        last_finish_reason = "stop"

        try:
            for chunk_idx in range(max_chunks):
                if cancel_event is not None and cancel_event.is_set():
                    self._log_done(req, "cancelled", output_chunks, chunk_tokens)
                    yield DoneEvent(
                        request_id=req.request_id,
                        input_tokens=0,
                        output_chunks=output_chunks,
                        finish_reason="cancelled",
                    )
                    return

                piece_text = ""
                chunk_finish: Optional[str] = None

                # Bind convo into the closure so the worker thread sees the
                # current state, not whatever convo points to later.
                def run_stream(current_messages=list(convo)):
                    with self.generate_lock:
                        return self.llm.create_chat_completion(
                            messages=current_messages,
                            max_tokens=chunk_tokens,
                            temperature=temp,
                            top_p=top_p,
                            stream=True,
                            stop=None,
                        )

                stream = await asyncio.to_thread(run_stream)

                for raw in stream:
                    if cancel_event is not None and cancel_event.is_set():
                        self._log_done(req, "cancelled", output_chunks, chunk_tokens)
                        yield DoneEvent(
                            request_id=req.request_id,
                            input_tokens=0,
                            output_chunks=output_chunks,
                            finish_reason="cancelled",
                        )
                        return

                    try:
                        choice = raw["choices"][0]
                        delta = choice.get("delta") or {}
                        text = delta.get("content", "") or ""
                        fr = choice.get("finish_reason")
                    except Exception:
                        text, fr = "", None

                    if text:
                        output_chunks += 1
                        piece_text += text
                        yield TokenEvent(request_id=req.request_id, text=text)

                    if fr is not None:
                        chunk_finish = fr

                    await asyncio.sleep(0)

                last_finish_reason = chunk_finish or "stop"

                # Natural stop OR no text produced -> done
                if last_finish_reason != "length" or not piece_text:
                    break

                # Roll forward for the next pass
                convo.append({"role": "assistant", "content": piece_text})
                convo.append({"role": "user", "content": "continue"})

            mapped = _map_finish_reason(last_finish_reason)
            self._log_done(req, mapped, output_chunks, chunk_tokens)
            yield DoneEvent(
                request_id=req.request_id,
                input_tokens=0,
                output_chunks=output_chunks,
                finish_reason=mapped,
            )

        except Exception as exc:
            logger.exception("stream_chat_unbounded failed: model=%s req=%s",
                             self.model_key, req.request_id)
            yield ErrorEvent(
                request_id=req.request_id,
                message=f"{type(exc).__name__}: {exc}",
            )

    # --- non-streaming -----------------------------------------------------

    def generate_text(
        self,
        messages: list[dict] | str,
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        use_chat_template: bool = True,
        return_full_text: bool = False,
        stop: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        """Blocking, non-streaming completion.

        messages: list of {'role','content'} dicts OR a raw prompt string.
        use_chat_template=True -> create_chat_completion (uses GGUF's template)
        use_chat_template=False -> raw self.llm() with messages_to_prompt_from_dicts
        """
        max_tokens = _resolve_max_tokens(max_new_tokens)
        temp = _resolve_temperature(temperature, do_sample)
        top_p_val = _resolve_top_p(top_p)

        with self.generate_lock:
            if use_chat_template and isinstance(messages, list):
                out = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p_val,
                    stop=stop,
                    stream=False,
                )
                choice = out["choices"][0]
                text = choice["message"]["content"] or ""
                logger.info(
                    "generate_text done: model=%s finish=%s usage=%s cap=%s",
                    self.model_key, choice.get("finish_reason"),
                    out.get("usage"), max_tokens,
                )
                return text

            prompt = (
                messages
                if isinstance(messages, str)
                else messages_to_prompt_from_dicts(messages)
            )
            out = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p_val,
                stop=stop,  # no implicit User:/\\nUser: stops anymore
                stream=False,
                echo=return_full_text,
            )
            choice = out["choices"][0]
            text = choice.get("text", "")
            logger.info(
                "generate_text(raw) done: model=%s finish=%s cap=%s",
                self.model_key, choice.get("finish_reason"), max_tokens,
            )
            return text

    def generate_text_unbounded(
        self,
        messages: list[dict],
        *,
        chunk_tokens: int = 1024,
        max_chunks: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        stop: Optional[list[str]] = None,
        **kwargs
    ) -> str:
        """Non-streaming counterpart to stream_chat_unbounded.

        Re-issues with 'continue' nudges until EOS or max_chunks.
        Returns the concatenated text.
        """
        temp = _resolve_temperature(temperature, do_sample)
        top_p_val = _resolve_top_p(top_p)

        accumulated = ""
        convo = list(messages)
        last_finish = "stop"

        for chunk_idx in range(max_chunks):
            with self.generate_lock:
                out = self.llm.create_chat_completion(
                    messages=convo,
                    max_tokens=chunk_tokens,
                    temperature=temp,
                    top_p=top_p_val,
                    stop=stop,
                    stream=False,
                )
            choice = out["choices"][0]
            piece = choice["message"]["content"] or ""
            last_finish = choice.get("finish_reason") or "stop"
            accumulated += piece

            logger.info(
                "generate_text_unbounded chunk=%s model=%s finish=%s usage=%s",
                chunk_idx, self.model_key, last_finish, out.get("usage"),
            )

            if last_finish != "length" or not piece:
                break

            convo = convo + [
                {"role": "assistant", "content": piece},
                {"role": "user", "content": "continue"},
            ]

        return accumulated

    # --- internals ---------------------------------------------------------

    def _log_done(self, req: ChatRequest, finish: str, chunks: int, cap: int) -> None:
        logger.info(
            "stream_chat done: model=%s req=%s finish=%s chunks=%s cap=%s",
            self.model_key, req.request_id, finish, chunks, cap,
        )
