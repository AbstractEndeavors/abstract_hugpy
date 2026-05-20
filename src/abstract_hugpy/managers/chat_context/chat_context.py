from __future__ import annotations
from .imports import default_context_tokens_for_model,message_to_dict,DEFAULT_MAX_TOKENS
from .context_budget import ContextBudget, compact_messages_to_budget

def compact_chat_request(req):
    """
    Return a copy of ChatRequest with token-safe messages.

    Preserves the original message model type so downstream runners can still
    call m.model_dump().
    """
    max_context_tokens = default_context_tokens_for_model(req.model_key)

    requested_output_tokens = req.max_new_tokens or DEFAULT_MAX_TOKENS

    # Do not let output reservation consume the whole context window.
    # This keeps room for previous turns.
    reserved_output_tokens = min(
        requested_output_tokens,
        max(4096, max_context_tokens // 3),
    )

    budget = ContextBudget(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    raw_messages = [message_to_dict(message) for message in req.messages]
    compacted_dicts = compact_messages_to_budget(raw_messages, budget)

    if req.messages:
        message_type = type(req.messages[0])
        compacted_messages = [message_type(**message) for message in compacted_dicts]
    else:
        compacted_messages = []

    return req.model_copy(update={"messages": compacted_messages})
