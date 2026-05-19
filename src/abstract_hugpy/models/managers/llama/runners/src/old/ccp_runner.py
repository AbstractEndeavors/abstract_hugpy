from .imports import *
# ===========================================================================
# HTTP runner — talks to a running llama-server process
# ===========================================================================

class LlamaCppRunner:
    def __init__(self, model_key: str, *, env_path: Optional[str] = None):

        if model_key not in LLAMA_MODEL_PORTS:
            raise KeyError(
                f"Unknown model_key={model_key!r}; "
                f"known: {sorted(LLAMA_MODEL_PORTS)}"
            )

        cfg = _load_llama_config(env_path=env_path)
        self.model_key = model_key
        self.llama_host: str = cfg["LLAMA_HOST"]
        self.port: int = cfg[model_key]
        self.base_url = f"{self.llama_host}:{self.port}"

    async def stream_chat(
        self,
        req: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream via llama-server's OpenAI-compatible /v1/chat/completions.

        Goes through the GGUF's chat template server-side. No hand-rolled
        User:/Assistant: scaffolding, no User:/\\nUser: stop strings.
        """
        max_tokens = _resolve_max_tokens(req.max_new_tokens)
        temp = _resolve_temperature(req.temperature, req.do_sample)
        top_p = _resolve_top_p(req.top_p)

        messages = messages_to_dicts(req.messages)
        logger.info(
            "llama stream context: model=%s req=%s messages=%s roles=%s chars=%s",
            self.model_key,
            req.request_id,
            len(messages),
            [m.get("role") for m in messages],
            sum(len(m.get("content", "")) for m in messages),
        )
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "stream": True,
        }

        output_chunks = 0
        last_finish: Optional[str] = None

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if cancel_event is not None and cancel_event.is_set():
                            self._log_done(req, "cancelled", output_chunks, max_tokens)
                            yield DoneEvent(
                                request_id=req.request_id,
                                input_tokens=0,
                                output_chunks=output_chunks,
                                finish_reason="cancelled",
                            )
                            return

                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        if line.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # OpenAI streaming shape: choices[0].delta.content
                        try:
                            choice = data["choices"][0]
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

    def generate_text(
        self,
        messages: list[dict] | str,
        prompt:str=None,
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        use_chat_template: bool = True,
        return_full_text: bool = False,
        stop: Optional[list[str]] = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> str:
        """Blocking, non-streaming completion via llama-server HTTP.

        sync method, sync HTTP. From async code call via asyncio.to_thread.
        """
        messages = messages or get_messages(prompt)
        max_tokens = _resolve_max_tokens(max_new_tokens)
        temp = _resolve_temperature(temperature, do_sample)
        top_p_val = _resolve_top_p(top_p)

        if use_chat_template and isinstance(messages, list):
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": top_p_val,
                "stream": False,
            }
            if stop:
                payload["stop"] = stop

            with httpx.Client(timeout=timeout) as client:
                r = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()

            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            logger.info(
                "generate_text done: model=%s finish=%s usage=%s cap=%s",
                self.model_key, choice.get("finish_reason"),
                data.get("usage"), max_tokens,
            )
            return text

        # Raw-prompt fallback: /completion (no chat template available)
        prompt = (
            messages
            if isinstance(messages, str)
            else messages_to_prompt_from_dicts(messages)
        )
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temp,
            "top_p": top_p_val,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{self.base_url}/completion", json=payload)
            r.raise_for_status()
            data = r.json()

        text = data.get("content") or data.get("text") or ""
        if return_full_text:
            text = prompt + text
        return text

    # --- internals ---------------------------------------------------------

    def _log_done(self, req: ChatRequest, finish: str, chunks: int, cap: int) -> None:
        logger.info(
            "stream_chat done: model=%s req=%s finish=%s chunks=%s cap=%s",
            self.model_key, req.request_id, finish, chunks, cap,
        )
