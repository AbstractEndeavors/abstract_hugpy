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
import logging,threading,pydantic
from typing import Dict, Tuple, Type
from .imports import Runner,ChatRequest,MODEL_REGISTRY
from .generate import DeepCoderChatRunner
from .vision import VisionRunner
from .llama import LlamaCppChatRunner

def infer_arg_name(arg: Any) -> str | None:
    """
    Makes an educated guess about what positional arg represents.
    """

    if isinstance(arg, str):
        # Could be model_key or prompt.
        # Prefer treating unknown strings as prompt/messages.
        # Model keys usually look like known model identifiers.
        if (
            "/" in arg
            or "_gguf" in arg
            or "qwen" in arg.lower()
            or "llama" in arg.lower()
            or "mistral" in arg.lower()
            or "gpt" in arg.lower()
        ):
            return "model_key"

        return "messages"

    if isinstance(arg, list):
        return "messages"

    if isinstance(arg, int):
        return "max_new_tokens"

    if isinstance(arg, float):
        # First float usually means temperature.
        return "temperature"

    if isinstance(arg, bool):
        # bool is also an int subclass, so this must be checked before int
        return "do_sample"

    if arg is None:
        return None

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

_RUNNERS: Dict[Tuple[str, str], Type[Runner]] = {
    ("transformers", "code-generation"): DeepCoderChatRunner,
    ("transformers", "text-generation"): DeepCoderChatRunner,
    ("llama_cpp",    "code-generation"): LlamaCppChatRunner,
    ("llama_cpp",    "text-generation"): LlamaCppChatRunner,
    ("transformers", "vision-language"): VisionRunner

}


# ---------------------------------------------------------------------------
# Per-process instance cache
# ---------------------------------------------------------------------------

_INSTANCES: Dict[str, Runner] = {}
_INSTANCES_LOCK = threading.Lock()


def runner_for(model_key: str) -> Runner:
    """Return the cached Runner for `model_key`, building it on first call.

    Raises KeyError if the model isn't in MODEL_REGISTRY, or if no runner
    class is registered for its (framework, task) pair. Both errors carry
    enough info for the caller to surface a useful 4xx.
    """
    # Fast path: already built.
    cached = _INSTANCES.get(model_key)
    if cached is not None:
        return cached

    with _INSTANCES_LOCK:
        # Re-check after acquiring the lock — another thread may have
        # built it while we were waiting.
        cached = _INSTANCES.get(model_key)
        if cached is not None:
            return cached

        cfg = MODEL_REGISTRY.get(model_key)
        if cfg is None:
            raise KeyError(
                f"Unknown model_key={model_key!r}; "
                f"known: {sorted(MODEL_REGISTRY.keys())}"
            )

        key = (cfg.framework, cfg.task)
        cls = _RUNNERS.get(key)
        if cls is None:
            raise KeyError(
                f"No runner registered for {model_key!r} "
                f"(framework={cfg.framework!r}, task={cfg.task!r}); "
                f"known runner keys: {sorted(_RUNNERS.keys())}"
            )

        logger.info(
            "instantiating runner: model=%s class=%s framework=%s task=%s",
            model_key, cls.__name__, cfg.framework, cfg.task,
        )
        instance = cls(model_key)
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


def execute_prompt(*args: Any, **kwargs: Any) -> ChatRequest:
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)
    req = ChatRequest(**prompt_kwargs)
    runner = runner_for(req.model_key)
    return runner.run(req=req)
