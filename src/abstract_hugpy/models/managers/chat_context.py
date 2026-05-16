from __future__ import annotations

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


def _message_to_dict(message) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()

    return {
        "role": str(message.get("role", "user")),
        "content": str(message.get("content", "")),
    }


def compact_chat_request(req):
    """
    Return a copy of ChatRequest with token-safe messages.

    Preserves the original message model type so downstream runners can still
    call m.model_dump().
    """
    max_context_tokens = default_context_tokens_for_model(req.model_key)
    reserved_output_tokens = req.max_new_tokens or 2048

    budget = ContextBudget(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    raw_messages = [_message_to_dict(message) for message in req.messages]
    compacted_dicts = compact_messages_to_budget(raw_messages, budget)

    if req.messages:
        message_type = type(req.messages[0])
        compacted_messages = [message_type(**message) for message in compacted_dicts]
    else:
        compacted_messages = []

    return req.model_copy(update={"messages": compacted_messages})
