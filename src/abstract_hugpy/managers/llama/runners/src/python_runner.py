from .base_runner import LlamaCppBaseRunner
from .imports import *
# ===========================================================================
# In-process Python runner — loads a GGUF via llama_cpp directly
# ===========================================================================

# python_runner.py  (in-process)
class LlamaCppPythonRunner(LlamaCppBaseRunner):
    def __init__(
        self,
        model_key: str,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
    ):
        from llama_cpp import Llama
        from ....spill import llama_kwargs

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

        # GPU/CPU spill. The resolver/dispatch path doesn't pass n_gpu_layers,
        # so by default we derive it from the spill module (env + autofit).
        # An explicit constructor arg always wins. Without this, llama.cpp ran
        # CPU-only because n_gpu_layers was never set.
        gpu_kwargs = llama_kwargs(self.model_path)
        if n_gpu_layers is not None:
            gpu_kwargs["n_gpu_layers"] = n_gpu_layers
        self.n_gpu_layers = gpu_kwargs.get("n_gpu_layers", 0)

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
            **gpu_kwargs,
        )

        logger.info(
            "LlamaCppPythonRunner ready: model=%s n_ctx=%s n_threads=%s "
            "n_gpu_layers=%s path=%s",
            model_key, self.n_ctx, self.n_threads, self.n_gpu_layers, self.model_path,
        )

    # ----- context-window fitting -----------------------------------------
    # Extra tokens the chat template / BOS / role wrappers add on top of the
    # raw message text we measure. Kept as headroom so we never tip over n_ctx.
    _CTX_SAFETY_MARGIN = 64

    def _count_tokens(self, text: str) -> int:
        """Exact token count via the loaded model's own tokenizer."""
        if not text:
            return 0
        try:
            return len(self.llm.tokenize(text.encode("utf-8"),
                                         add_bos=False, special=False))
        except Exception:
            return max(1, len(text) // 4)

    def _fit_chat(self, messages, max_tokens):
        """Compact messages + clamp max_tokens so prompt+output fit n_ctx.

        This is what stops llama.cpp's "Requested tokens (N) exceed context
        window of M" error: the conversation is trimmed (oldest turns first,
        then the middle of the newest message) to the real, tokenizer-measured
        input budget, and the output cap is shrunk to whatever room is left.
        """
        from ....chat_context.context_budget import (
            ContextBudget, compact_messages_to_budget,
        )
        n_ctx = int(self.n_ctx)
        out = max(16, min(int(max_tokens) if max_tokens else n_ctx // 2,
                          n_ctx - 256))
        budget = ContextBudget(
            max_context_tokens=n_ctx,
            reserved_output_tokens=out,
            reserved_system_tokens=0,
        )
        fitted = compact_messages_to_budget(
            messages, budget, token_counter=self._count_tokens,
        )
        prompt_tokens = sum(
            self._count_tokens(str(m.get("content", ""))) + 8 for m in fitted
        )
        room = n_ctx - prompt_tokens - self._CTX_SAFETY_MARGIN
        out = max(16, min(out, room))
        original = len([m for m in messages if str(m.get("content", "")).strip()])
        if len(fitted) != original:
            logger.warning(
                "context fit: model=%s trimmed messages %s -> %s "
                "(n_ctx=%s, prompt~%s tok, out cap=%s)",
                self.model_key, original, len(fitted), n_ctx, prompt_tokens, out,
            )
        return fitted, out

    def _fit_raw(self, prompt, max_tokens):
        """Trim a raw prompt (head+tail) to fit n_ctx and clamp max_tokens."""
        n_ctx = int(self.n_ctx)
        out = max(16, min(int(max_tokens) if max_tokens else n_ctx // 2,
                          n_ctx - 256))
        budget_tokens = max(64, n_ctx - out - self._CTX_SAFETY_MARGIN)
        try:
            toks = self.llm.tokenize(prompt.encode("utf-8"),
                                     add_bos=True, special=True)
        except Exception:
            return prompt, out
        if len(toks) <= budget_tokens:
            return prompt, out
        keep_head = budget_tokens // 3
        keep_tail = budget_tokens - keep_head
        kept = toks[:keep_head] + toks[-keep_tail:]
        try:
            new_prompt = self.llm.detokenize(kept).decode("utf-8", errors="ignore")
        except Exception:
            approx = budget_tokens * 4
            new_prompt = prompt[: approx // 3] + prompt[-(approx - approx // 3):]
        logger.warning(
            "context fit: model=%s trimmed raw prompt %s -> %s tokens (n_ctx=%s)",
            self.model_key, len(toks), len(kept), n_ctx,
        )
        return new_prompt, out

    async def _iter_stream(self, messages, max_tokens, temp, top_p):
        messages, max_tokens = self._fit_chat(messages, max_tokens)

        def run():
            with self.generate_lock:
                return self.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temp, top_p=top_p, stream=True, stop=None)
        stream = await asyncio.to_thread(run)
        for raw in stream:
            try:
                choice = raw["choices"][0]
                text = (choice.get("delta") or {}).get("content") or ""
                fr   = choice.get("finish_reason")
            except Exception:
                text, fr = "", None
            yield text, fr
            await asyncio.sleep(0)
    def _chat_complete(self, messages, max_tokens, temp, top_p, stop):
        messages, max_tokens = self._fit_chat(messages, max_tokens)
        with self.generate_lock:
            out = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temp, top_p=top_p, stop=stop, stream=False)
        choice = out["choices"][0]
        logger.info("_chat_complete: model=%s finish=%s usage=%s cap=%s",
                    self.model_key, choice.get("finish_reason"), out.get("usage"), max_tokens)
        return choice["message"]["content"] or "", choice.get("finish_reason") or "stop"

    def _raw_complete(self, prompt, max_tokens, temp, top_p, stop, return_full_text):
        prompt, max_tokens = self._fit_raw(prompt, max_tokens)
        with self.generate_lock:
            out = self.llm(prompt, max_tokens=max_tokens, temperature=temp,
                           top_p=top_p, stop=stop, stream=False, echo=return_full_text)
        choice = out["choices"][0]
        logger.info("_raw_complete: model=%s finish=%s cap=%s",
                    self.model_key, choice.get("finish_reason"), max_tokens)
        return choice.get("text", ""), choice.get("finish_reason") or "stop"
    def _blocking_complete(
        self,
        messages: list[dict] | str,
        max_tokens: int,
        temp: float,
        top_p: float,
        stop: Optional[list[str]],
        use_chat_template: bool,
        return_full_text: bool,
    ) -> str:
        with self.generate_lock:
            if use_chat_template and isinstance(messages, list):
                messages, max_tokens = self._fit_chat(messages, max_tokens)
                out = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                    stop=stop,
                    stream=False,
                )

                choice = out["choices"][0]
                text = choice["message"]["content"] or ""

                logger.info(
                    "_blocking_complete done: model=%s finish=%s usage=%s cap=%s",
                    self.model_key,
                    choice.get("finish_reason"),
                    out.get("usage"),
                    max_tokens,
                )

                return text

            prompt = (
                messages
                if isinstance(messages, str)
                else messages_to_prompt_from_dicts(messages)
            )
            prompt, max_tokens = self._fit_raw(prompt, max_tokens)

            out = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                stop=stop,
                stream=False,
                echo=return_full_text,
            )

            choice = out["choices"][0]
            text = choice.get("text", "")

            logger.info(
                "_blocking_complete(raw) done: model=%s finish=%s cap=%s",
                self.model_key,
                choice.get("finish_reason"),
                max_tokens,
            )

            return text


