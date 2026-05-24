"""Runner dispatch — dumb consumer of Resolution.

All routing logic lives in model_resolver.resolve(). This module owns
two things and only two things:

    1. A per-process instance cache keyed by (model_key, task).
    2. An execute_prompt entry point that turns request kwargs into
       a result by handing off to resolve() and the runner.

It does not:
    - Decide which builder to call.
    - Decide which runner class to instantiate.
    - Validate that model+task are compatible.
    - Default task to cfg.primary_task.

If you find yourself adding any of that here, stop and add it to
model_resolver.resolve() instead. That's the whole point.

Why a per-process cache:
    Loading a 14B model takes seconds; doing it on every request is
    obviously wrong. Per-(model_key, task) caching means the same
    model can host two task-runners (e.g. text-generation + code-
    generation on one llama.cpp instance) and each gets its own
    runner wrapper, but inner singletons (REGISTRY for DeepCoder,
    get_llama_runner for llama.cpp) still de-dup the heavy state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from .imports import Runner
from .model_resolver import Resolution, resolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process instance cache — keyed by (model_key, task) per the contract
# in Resolution.cache_key.
# ---------------------------------------------------------------------------

_INSTANCES: Dict[Tuple[str, str], Runner] = {}
_INSTANCES_LOCK = threading.Lock()


def _get_or_build_runner(res: Resolution) -> Runner:
    """Cache-coherent runner lookup. Double-checked locking under the cache lock."""
    cached = _INSTANCES.get(res.cache_key)
    if cached is not None:
        return cached

    with _INSTANCES_LOCK:
        cached = _INSTANCES.get(res.cache_key)
        if cached is not None:
            return cached

        logger.info(
            "instantiating runner: model=%s task=%s class=%s framework=%s",
            res.model_key, res.task, res.runner_cls.__name__, res.framework,
        )
        instance = res.runner_cls(res.cfg)
        _INSTANCES[res.cache_key] = instance
        return instance


# ---------------------------------------------------------------------------
# Argument normalization — flexible positional input -> kwargs dict.
# ---------------------------------------------------------------------------

def infer_arg_name(arg: Any) -> Optional[str]:
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


def normalize_prompt_kwargs(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convert flexible input into builder-compatible kwargs.

    Explicit kwargs win over inferred positional args. A second float
    becomes top_p (since temperature is already set).
    """
    prompt_kwargs = dict(kwargs)

    for arg in args:
        guessed_key = infer_arg_name(arg)
        if guessed_key is None:
            raise TypeError(f"Could not infer argument type for positional arg: {arg!r}")

        if guessed_key in prompt_kwargs:
            if guessed_key == "temperature" and "top_p" not in prompt_kwargs:
                prompt_kwargs["top_p"] = arg
            continue

        prompt_kwargs[guessed_key] = arg

    return prompt_kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def runner_for(
    model_key: Optional[str] = None,
    *,
    task: Optional[str] = None,
) -> Runner:
    """Get a runner by model_key, task, or both.

    Both are passed through resolve() — so the same (model_key, task)
    pair always lands on the same cached runner, whether you came in
    here or through execute_prompt.
    """
    if model_key is None and task is None:
        raise ValueError("runner_for requires at least one of model_key or task")

    res = resolve({"model_key": model_key, "task": task})
    return _get_or_build_runner(res)


def execute_prompt(*args: Any, **kwargs: Any):
    """One-shot request -> result. Sync entrypoint; awaits inside if needed."""
    prompt_kwargs = normalize_prompt_kwargs(*args, **kwargs)
    res = resolve(prompt_kwargs)
    req = res.builder(prompt_kwargs, res.model_key)
    runner = _get_or_build_runner(res)
    return runner.run(req=req)


# ---------------------------------------------------------------------------
# Inspection / lifecycle — single definition each, no duplicates.
# ---------------------------------------------------------------------------

def loaded_model_keys() -> List[Tuple[str, str]]:
    """Which (model_key, task) pairs currently have a runner instantiated."""
    with _INSTANCES_LOCK:
        return sorted(_INSTANCES.keys())


def evict(model_key: str, task: Optional[str] = None) -> bool:
    """Drop runner(s) from the cache.

    If task is None, all task-variants for that model_key are dropped.
    Returns True if anything was evicted.

    The underlying model may still be loaded if the inner singleton
    (REGISTRY for DeepCoder, _LLAMA_INSTANCES for llama.cpp) holds it.
    Eviction here only releases the runner wrapper.
    """
    with _INSTANCES_LOCK:
        if task is not None:
            return _INSTANCES.pop((model_key, task), None) is not None
        to_drop = [k for k in list(_INSTANCES) if k[0] == model_key]
        for k in to_drop:
            _INSTANCES.pop(k, None)
        return bool(to_drop)


def clear() -> None:
    """Drop all cached runners. Tests use this; production probably shouldn't."""
    with _INSTANCES_LOCK:
        _INSTANCES.clear()


def supported_task_keys() -> List[Tuple[str, str]]:
    """List the (framework, task) pairs that have runners registered."""
    from .model_resolver import _RUNNERS
    return sorted(_RUNNERS.keys())
