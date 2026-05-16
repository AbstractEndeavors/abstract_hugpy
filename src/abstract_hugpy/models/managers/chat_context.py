from __future__ import annotations

from dataclasses import replace

from .context_budget import ContextBudget, compact_messages_to_budget


DEFAULT_CONTEXT_TOKENS_BY_MODEL: dict[str, int] = {
    "deepcoder": 16384,
    "Qwen2.5-Coder-1.5B-GGUF": 16384,
    "Qwen2.5-Coder-3B-GGUF": 16384,
    "Qwen3-Coder-Next-Q4_K_M": 16384,
    "DAN-L3-R1-8B-i1-GGUF": 16384,
}


def default_context_tokens_for_model(model_key: str) -> int:
    return DEFAULT_CONTEXT_TOKENS_BY_MODEL.get(model_key, 8192)


def compact_chat_request(req):
    """
    Return a copy of ChatRequest with token-safe messages.

    This assumes ChatRequest is a Pydantic model. If yours is a dataclass,
    replace req.model_copy(...) with dataclasses.replace(...).
    """
    max_context_tokens = default_context_tokens_for_model(req.model_key)
    reserved_output_tokens = req.max_new_tokens or 2048

    budget = ContextBudget(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    messages = [m.model_dump() for m in req.messages]
    compacted = compact_messages_to_budget(messages, budget)

    return req.model_copy(update={"messages": compacted})
