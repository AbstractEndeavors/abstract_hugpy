
# deepcoder/coder.py
import os
import asyncio
import threading
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
)

from .imports import (
    ChatRequest,
    get_torch,
    get_transformers,
    get_logFile,
    require,
)

from .config import (
    DeepCoderConfig,
    build_deepcoder_config,
    CancelStoppingCriteria,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    StreamEvent,
    _SENTINEL,
)

logger = get_logFile("deepcoder")


class DeepCoder:
    """Loaded model + tokenizer + generation_config. One instance per config."""

    def __init__(self, cfg: DeepCoderConfig):
        require("transformers", reason="DeepCoder requires HuggingFace transformers")

        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_init = threading.Lock()

        self._configure_cpu_runtime()
        self._load_tokenizer()
        self._load_model()
        self._load_generation_config()

        logger.info(
            "DeepCoder ready: device=%s dtype=%s concurrency=%d max_tokens_cap=%d",
            cfg.device,
            cfg.torch_dtype,
            cfg.max_concurrent_generations,
            cfg.max_new_tokens_cap,
        )

    def _configure_cpu_runtime(self) -> None:
        """
        Configure PyTorch CPU thread usage.

        For maximum CPU assigned to a single generation:
          - cpu_threads should usually be os.cpu_count()
          - cpu_interop_threads should usually be 1
          - max_concurrent_generations should usually be 1
        """
        if self.cfg.device == "cuda":
            return

        torch = get_torch()

        cpu_threads = self.cfg.cpu_threads or os.cpu_count() or 1
        interop_threads = self.cfg.cpu_interop_threads or 1

        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))

        try:
            torch.set_num_threads(cpu_threads)
        except Exception as exc:
            logger.warning("failed to set torch num threads: %s", exc)

        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            logger.warning("torch interop threads already initialized")
        except Exception as exc:
            logger.warning("failed to set torch interop threads: %s", exc)

        logger.info(
            "CPU runtime configured: cpu_threads=%s interop_threads=%s",
            cpu_threads,
            interop_threads,
        )

    def _load_tokenizer(self) -> None:
        AutoTokenizer = get_transformers("AutoTokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_dir,
            trust_remote_code=True,
            local_files_only=self.cfg.local_files_only,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self) -> None:
        AutoModelForCausalLM = get_transformers("AutoModelForCausalLM")

        kwargs: Dict[str, Any] = {
            "torch_dtype": self.cfg.torch_dtype,
            "local_files_only": self.cfg.local_files_only,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if self.cfg.device == "cuda":
            kwargs["device_map"] = "auto"

        if self.cfg.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        if self.cfg.use_quantization:
            BitsAndBytesConfig = get_transformers("BitsAndBytesConfig")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.cfg.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_dir,
            **kwargs,
        )

        if self.cfg.device != "cuda":
            self.model = self.model.to(self.cfg.device)

        self.model.eval()

    def _load_generation_config(self) -> None:
        GenerationConfig = get_transformers("GenerationConfig")

        try:
            generation_config = GenerationConfig.from_pretrained(
                self.cfg.model_dir,
                local_files_only=self.cfg.local_files_only,
            )
        except Exception:
            generation_config = GenerationConfig()

        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.use_cache = True

        self.generation_config = generation_config

    def _semaphore_for_loop(self) -> asyncio.Semaphore:
        with self._semaphore_init:
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(
                    self.cfg.max_concurrent_generations,
                )

            return self._semaphore

    def _resolve_max_new_tokens(self, max_new_tokens: Optional[int]) -> int:
        """
        max_new_tokens is output budget, not CPU power.

        None / 0 / negative means:
            use this runtime's configured max_new_tokens_cap.
        """
        requested = max_new_tokens or self.cfg.max_new_tokens_cap

        if requested <= 0:
            requested = self.cfg.max_new_tokens_cap

        if requested > self.cfg.max_new_tokens_cap:
            raise ValueError(
                f"max_new_tokens={requested} exceeds cap="
                f"{self.cfg.max_new_tokens_cap}; raise the cap explicitly via "
                "DeepCoderRuntime if intentional."
            )

        return requested

    async def stream_chat(
        self,
        request: ChatRequest,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield zero-or-more TokenEvent then exactly one DoneEvent or ErrorEvent."""
        torch = get_torch()
        TextIteratorStreamer = get_transformers("TextIteratorStreamer")
        StoppingCriteriaList = get_transformers("StoppingCriteriaList")

        try:
            requested_max_new_tokens = self._resolve_max_new_tokens(
                request.max_new_tokens,
            )
        except ValueError as exc:
            yield ErrorEvent(
                request_id=request.request_id,
                message=str(exc),
            )
            return

        sem = self._semaphore_for_loop()

        async with sem:
            messages = [message.model_dump() for message in request.messages]

            template_out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if hasattr(template_out, "shape"):
                input_ids = template_out.to(self.cfg.device)
                model_inputs = {"input_ids": input_ids}
            else:
                model_inputs = {
                    key: value.to(self.cfg.device) if hasattr(value, "to") else value
                    for key, value in template_out.items()
                }
                input_ids = model_inputs["input_ids"]

            input_len = int(input_ids.shape[-1])

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            thread_cancel = threading.Event()
            stopping = StoppingCriteriaList(
                [CancelStoppingCriteria(thread_cancel)],
            )

            gen_kwargs: Dict[str, Any] = {
                **model_inputs,
                "max_new_tokens": requested_max_new_tokens,
                "do_sample": bool(request.do_sample),
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "stopping_criteria": stopping,
            }

            if request.do_sample:
                gen_kwargs["temperature"] = request.temperature
                gen_kwargs["top_p"] = request.top_p

            generation_error: Optional[BaseException] = None

            def _run_generate() -> None:
                nonlocal generation_error

                try:
                    with torch.inference_mode():
                        self.model.generate(**gen_kwargs)
                except BaseException as exc:
                    generation_error = exc
                    streamer.end()

            gen_task = asyncio.create_task(asyncio.to_thread(_run_generate))

            cancel_task: Optional[asyncio.Task] = None

            if cancel_event is not None:

                async def _watch_cancel() -> None:
                    await cancel_event.wait()
                    thread_cancel.set()

                cancel_task = asyncio.create_task(_watch_cancel())

            loop = asyncio.get_running_loop()
            output_chunks = 0

            try:
                streamer_iter = iter(streamer)

                while True:
                    chunk = await loop.run_in_executor(
                        None,
                        next,
                        streamer_iter,
                        _SENTINEL,
                    )

                    if chunk is _SENTINEL:
                        break

                    if chunk == "":
                        continue

                    output_chunks += 1

                    yield TokenEvent(
                        request_id=request.request_id,
                        text=chunk,
                    )

                await gen_task

            finally:
                if cancel_task is not None and not cancel_task.done():
                    cancel_task.cancel()

                    try:
                        await cancel_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            if generation_error is not None:
                logger.error(
                    "generation failed for %s",
                    request.request_id,
                    exc_info=generation_error,
                )

                yield ErrorEvent(
                    request_id=request.request_id,
                    message=f"{type(generation_error).__name__}: {generation_error}",
                )
                return

            if thread_cancel.is_set():
                finish_reason = "cancelled"
            elif output_chunks >= requested_max_new_tokens:
                finish_reason = "max_tokens"
            else:
                finish_reason = "stop"

            yield DoneEvent(
                request_id=request.request_id,
                input_tokens=input_len,
                output_chunks=output_chunks,
                finish_reason=finish_reason,
            )

    def generate_text(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        max_new_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        use_chat_template: bool = False,
        return_full_text: bool = False,
    ) -> str:
        """
        Sync, non-streaming generation.

        Use stream_chat for the chat host. This is for eval scripts and batch
        jobs that do not need streaming/backpressure.
        """
        torch = get_torch()
        requested_max_new_tokens = self._resolve_max_new_tokens(max_new_tokens)

        if use_chat_template:
            messages = prompt if isinstance(prompt, list) else None

            if not messages:
                raise ValueError("use_chat_template=True requires list-form prompt")

            template_out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if hasattr(template_out, "shape"):
                input_ids = template_out.to(self.cfg.device)
                model_inputs = {"input_ids": input_ids}
            else:
                model_inputs = {
                    key: value.to(self.cfg.device) if hasattr(value, "to") else value
                    for key, value in template_out.items()
                }
                input_ids = model_inputs["input_ids"]

        else:
            tokenized = self.tokenizer(
                str(prompt),
                return_tensors="pt",
                padding=False,
                truncation=True,
            )

            model_inputs = {
                key: value.to(self.cfg.device)
                for key, value in tokenized.items()
            }

            input_ids = model_inputs["input_ids"]

        input_len = int(input_ids.shape[-1])

        gen_kwargs: Dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": requested_max_new_tokens,
            "do_sample": bool(do_sample),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        ids = outputs[0] if return_full_text else outputs[0][input_len:]

        return self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()


class _Registry:
    """
    Cache of DeepCoder instances keyed by DeepCoderConfig.cache_key().

    Module-level singleton, but inspectable:
      REGISTRY.keys()
      REGISTRY.evict(cfg)
      REGISTRY.clear()
    """

    def __init__(self):
        self._instances: Dict[tuple, DeepCoder] = {}
        self._lock = threading.Lock()

    def get(self, cfg: DeepCoderConfig) -> DeepCoder:
        key = cfg.cache_key()

        with self._lock:
            instance = self._instances.get(key)

            if instance is None:
                instance = DeepCoder(cfg)
                self._instances[key] = instance

            return instance

    def evict(self, cfg: DeepCoderConfig) -> bool:
        key = cfg.cache_key()

        with self._lock:
            return self._instances.pop(key, None) is not None

    def keys(self) -> List[tuple]:
        with self._lock:
            return list(self._instances.keys())

    def clear(self) -> None:
        with self._lock:
            self._instances.clear()


REGISTRY = _Registry()


def get_deep_coder(
    cfg: Optional[DeepCoderConfig] = None,
    **build_kwargs,
) -> DeepCoder:
    """Boot-time get. Pass a cfg or build one. Same key returns same instance."""
    if cfg is None:
        cfg = build_deepcoder_config(**build_kwargs)

    return REGISTRY.get(cfg)


def deep_coder_generate(prompt, **kwargs) -> str:
    """Convenience wrapper with the same surface as before."""
    gen_keys = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "use_chat_template",
        "messages",
        "do_sample",
        "return_full_text",
    }

    gen_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in gen_keys
    }

    coder = get_deep_coder(**kwargs)
    
    return coder.generate_text(
        prompt=prompt,
        **gen_kwargs,
    )

