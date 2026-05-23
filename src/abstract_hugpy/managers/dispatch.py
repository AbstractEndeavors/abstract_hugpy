"""Runner dispatch — registry-driven.

The route layer calls runner_for(model_key) and gets back something
implementing the Runner protocol. Routes don't know which class they
got, only that they can call .run() / .stream() on it.

Resolution:
    1. Look up model_key in MODEL_REGISTRY -> get (framework, task)
    2. Look up (framework, task) in _RUNNERS -> get a runner class
    3. Instantiate (cached per model_key) and return

Adding a new model:
    Add a row to MODEL_REGISTRY. Done — if its (framework, task) pair
    already has a runner class registered.

Adding a new task family:
    1. Define new TaskRequest / TaskResult types
    2. Write a runner class implementing the Runner protocol
    3. Add a row to _RUNNERS for the (framework, task) keys it handles
    4. Add a row to MODEL_REGISTRY for the model
    No route changes needed.

Why a per-process cache instead of recreating runners every call:
    Loading a 14B model takes seconds; doing it on every request is
    obviously wrong. Per-key caching ensures one model = one loaded
    instance per worker process. Inner singletons (REGISTRY for DeepCoder,
    get_llama_runner for llama.cpp) handle further deduplication.
"""
from __future__ import annotations
import logging,threading,pydantic,os
from typing import Dict, Tuple, Type
from .imports import Runner,ChatRequest,MODEL_REGISTRY
from .generate import DeepCoderChatRunner
from .vision import VisionRunner
from .llama import LlamaCppChatRunner
from .whisper_model import WhisperRunner
from .model_resolver import resolve_model_key,_REQUEST_BUILDERS,_RUNNERS
def infer_arg_name(arg: Any) -> str | None:
    if arg is None:
        return None
    if isinstance(arg, bool):
        return "do_sample"
    if isinstance(arg, int):
        return "max_new_tokens"
    if isinstance(arg, float):
        return "temperature"
    if isinstance(arg, list):
        return "messages"
    if isinstance(arg, str):
        if os.path.exists(arg):
            return "file"
        lowered = arg.lower()
        looks_like_model = (
            "/" in arg
            or "_gguf" in lowered
            or any(tag in lowered for tag in ("qwen", "llama", "mistral", "gpt"))
        )
        return "model_key" if looks_like_model else "messages"
    return None


def normalize_prompt_kwargs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Converts flexible input into ChatRequest-compatible kwargs.

    Explicit kwargs win over inferred positional args.
    """

    prompt_kwargs = dict(kwargs)

    for arg in args:
        guessed_key = infer_arg_name(arg)

        if guessed_key is None:
            raise TypeError(f"Could not infer argument type for positional arg: {arg!r}")

        if guessed_key in prompt_kwargs:
            continue

        # Special handling for a second float:
        # execute_prompt("hello", 0.7, 0.95)
        # -> temperature=0.7, top_p=0.95
        if guessed_key == "temperature" and "temperature" in prompt_kwargs:
            if "top_p" not in prompt_kwargs:
                prompt_kwargs["top_p"] = arg
            continue

        prompt_kwargs[guessed_key] = arg

    return prompt_kwargs
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch table — single source of truth for (framework, task) -> Runner.
#
# Currently handles chat-family models (transformers causal LMs + llama.cpp
# GGUFs). Other task families plug in here as they're implemented:
#   ("transformers", "summarization")      -> SummarizeRunner
#   ("transformers", "speech-recognition") -> WhisperRunner
#   ("transformers", "vision-language")    -> VisionRunner
#   ("transformers", "embeddings")         -> KeywordRunner / EmbeddingRunner
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Per-process instance cache
# ---------------------------------------------------------------------------

_INSTANCES: Dict[str, Runner] = {}
_INSTANCES_LOCK = threading.Lock()


def runner_for(model_key: str) -> Runner:
    cached = _INSTANCES.get(model_key)
    if cached is not None:
        return cached

    with _INSTANCES_LOCK:
        cached = _INSTANCES.get(model_key)
        if cached is not None:
            return cached

        cfg = MODEL_REGISTRY.get(model_key)
        if cfg is None:
            raise KeyError(
                f"Unknown model_key={model_key!r}; "
                f"known: {sorted(MODEL_REGISTRY.keys())}"
            )

        key = (cfg.framework, cfg.primary_task)
        cls = _RUNNERS.get(key)
        if cls is None:
            raise KeyError(
                f"No runner registered for {model_key!r} "
                f"(framework={cfg.framework!r}, tasks={list(cfg.tasks)!r}, "
                f"primary={cfg.primary_task!r}); "
                f"known runner keys: {sorted(_RUNNERS.keys())}"
            )

        logger.info(
            "instantiating runner: model=%s class=%s framework=%s primary_task=%s",
            model_key, cls.__name__, cfg.framework, cfg.primary_task,
        )
        instance = cls(cfg)
        _INSTANCES[model_key] = instance
        return instance




# ---------------------------------------------------------------------------
# Inspection / lifecycle helpers — useful for tests, ops, and debugging
# ---------------------------------------------------------------------------

def loaded_model_keys() -> list[str]:
    """Which model_keys currently have a runner instantiated."""
    with _INSTANCES_LOCK:
        return sorted(_INSTANCES.keys())


def evict(model_key: str) -> bool:
    """Drop a runner from the cache. Returns True if something was evicted.

    The underlying model may still be loaded if the inner singleton
    (REGISTRY for DeepCoder, _LLAMA_INSTANCES for llama.cpp) holds it.
    Eviction here only releases the runner wrapper.
    """
    with _INSTANCES_LOCK:
        return _INSTANCES.pop(model_key, None) is not None


def clear() -> None:
    """Drop all cached runners. Tests use this; production probably shouldn't."""
    with _INSTANCES_LOCK:
        _INSTANCES.clear()


def supported_task_keys() -> list[Tuple[str, str]]:
    """List the (framework, task) pairs the dispatch table currently handles."""
    return sorted(_RUNNERS.keys())


def execute_prompt(*args: Any, **kwargs: Any):
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)

    model_key = resolve_model_key(
        model_key=prompt_kwargs.get("model_key"),
        file=prompt_kwargs.get("file"),
        media_type=prompt_kwargs.get("media_type"),
    )

    cfg = MODEL_REGISTRY[model_key]
    task_key = (cfg.framework, cfg.primary_task)

    builder = _REQUEST_BUILDERS.get(task_key)
    if builder is None:
        raise KeyError(
            f"No request builder registered for {model_key!r} "
            f"(framework={cfg.framework!r}, tasks={list(cfg.tasks)!r}, "
            f"primary={cfg.primary_task!r}); "
            f"known builder keys: {sorted(_REQUEST_BUILDERS)}"
        )

    req = builder(prompt_kwargs, model_key)
    runner = runner_for(model_key)
    return runner.run(req=req)

