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

    async def _iter_stream(self, messages, max_tokens, temp, top_p):
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
        with self.generate_lock:
            out = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temp, top_p=top_p, stop=stop, stream=False)
        choice = out["choices"][0]
        logger.info("_chat_complete: model=%s finish=%s usage=%s cap=%s",
                    self.model_key, choice.get("finish_reason"), out.get("usage"), max_tokens)
        return choice["message"]["content"] or "", choice.get("finish_reason") or "stop"

    def _raw_complete(self, prompt, max_tokens, temp, top_p, stop, return_full_text):
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


