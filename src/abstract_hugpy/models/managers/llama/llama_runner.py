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
# Defaults — single source of truth for "what does the runner do when the
# request omits a value." Override per-runner via constructor or per-request
# via ChatRequest.
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 2048      # was 512 / 256 / scattered; 2048 is the floor for useful coder outputs
DEFAULT_N_CTX = 16384          # was 4096; small ctx silently truncated long outputs
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_HTTP_TIMEOUT = 120.0   # non-streaming HTTP only; streaming uses None


# ---------------------------------------------------------------------------
# Env / port wiring (host:port discovery for the HTTP runner)
# ---------------------------------------------------------------------------

LLAMA_HOST_DEFAULT = "http://127.0.0.1"

LLAMA_MODEL_PORTS: Dict[str, int] = {
    "Qwen2.5-Coder-1.5B-GGUF":6008,
    "Qwen3-Coder-Next-Q4_K_M":6009,
    "DAN-L3-R1-8B-i1-GGUF":6090,
    "Qwen2.5-Coder-3B-GGUF":6091,
    "flux":6092,

}


def _load_llama_config(env_path: Optional[str] = None) -> Dict[str, str | int]:
    """Resolve host + per-model ports from env, with defaults as fallback.

    Env keys are the uppercased model_key:
        LLAMA_HOST=http://127.0.0.1
        Qwen2.5-Coder-1.5B-GGUF=6008
        Qwen3-Coder-Next-Q4_K_M=6009
        DAN-L3-R1-8B-i1-GGUF=6090
        Qwen2.5-Coder-3B-GGUF=6091

    """
    cfg: Dict[str, str | int] = {}
    cfg["LLAMA_HOST"] = get_env_value("LLAMA_HOST", path=env_path) or LLAMA_HOST_DEFAULT

    for model_key, default_port in LLAMA_MODEL_PORTS.items():
        raw = get_env_value(model_key.upper(), path=env_path)
        cfg[model_key] = int(raw) if raw else default_port

    return cfg


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
    return messages_to_prompt_from_dicts([m.model_dump() for m in req.messages])


# ---------------------------------------------------------------------------
# Helpers — finish reason mapping, defaulted resolvers
# ---------------------------------------------------------------------------

# llama.cpp says 'length' / 'stop'; schema says 'max_tokens' / 'stop'.
_FINISH_REASON_MAP = {
    "length": "max_tokens",
    "stop": "stop",
    None: "stop",
}


def _map_finish_reason(raw: Optional[str]) -> str:
    return _FINISH_REASON_MAP.get(raw, "stop")


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

        messages = [m.model_dump() for m in req.messages]
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

        messages = [m.model_dump() for m in req.messages]
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

        convo: list[dict] = [m.model_dump() for m in req.messages]
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


# ---------------------------------------------------------------------------
# Process-local registry (kept as-is; survives across HTTP requests)
# ---------------------------------------------------------------------------

_LLAMA_INSTANCES: Dict[str, "LlamaCppPythonRunner"] = {}
_LLAMA_LOCK = threading.Lock()


def get_llama_runner(model_key: str) -> LlamaCppPythonRunner:
    with _LLAMA_LOCK:
        runner = _LLAMA_INSTANCES.get(model_key)
        if runner is None:
            runner = LlamaCppPythonRunner(model_key)
            _LLAMA_INSTANCES[model_key] = runner
        return runner


